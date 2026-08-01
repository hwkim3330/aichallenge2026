import argparse
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl

# Autoware's own message types are not in rosbags' built-in typestore. An MCAP bag
# embeds its schemas so it decodes anyway, but a sqlite3 (.db3) bag does not --
# there, deserialising /control/command/control_cmd raises KeyError, and the
# per-message `except` below used to swallow it, so the run reported "insufficient
# data" instead of a missing type. The IDL files in msgdefs/ were taken from
# ghcr.io/automotiveaichallenge/autoware-universe:humble-latest
# (/autoware/install/<pkg>/share/<pkg>/msg/*.idl).
MSGDEFS_DIR = Path(__file__).resolve().parent / 'msgdefs'


def build_typestore() -> 'object':
    """Humble typestore plus Autoware's IDL-defined messages."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if MSGDEFS_DIR.is_dir():
        extra = {}
        for idl in sorted(MSGDEFS_DIR.rglob('*.idl')):
            extra.update(get_types_from_idl(idl.read_text()))
        if extra:
            typestore.register(extra)
    return typestore


@dataclass
class ExtractionConfig:
    """Configuration parameters for data extraction."""
    control_topic: str
    scan_topic: str
    speed_topic: str = '/vehicle/status/velocity_status'
    control_msg_type: str = 'autoware_auto_control_msgs/msg/AckermannControlCommand'
    scan_msg_type: str = 'sensor_msgs/msg/LaserScan'
    speed_msg_type: str = 'autoware_auto_vehicle_msgs/msg/VelocityReport'
    max_scan_range: float = 30.0
    # Drop samples whose measured speed is below this, in m/s. Race bags are
    # mostly the car standing still -- the npc-with-* bags are 68% below
    # 0.5 m/s -- and cloning those frames teaches the net to stop. Commanded
    # speed is not a usable filter for this: it is non-zero through most of the
    # stall (7% zero commanded against 68% actually stopped), because the
    # controller keeps asking for motion it cannot achieve. 0.0 keeps everything.
    min_speed: float = 0.0


def worker_init(debug_mode: bool) -> None:
    """
    Initializes the logging configuration for worker processes.

    Args:
        debug_mode: If True, sets logging level to DEBUG.
    """
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] [PID:%(process)d] %(message)s',
        force=True
    )


def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Sets up the main process logger.

    Args:
        debug: If True, enables debug logging.

    Returns:
        Configured Logger instance.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] [PID:%(process)d] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def clean_scan_array(scan_array: np.ndarray, max_range: float) -> np.ndarray:
    """
    Sanitizes LiDAR scan data by handling non-finite values.

    Operations:
        - NaN -> 0.0
        - Positive Inf -> max_range
        - Negative Inf -> 0.0
        - Values > max_range -> clipped to max_range

    Args:
        scan_array: Raw input array from LaserScan message.
        max_range: The maximum valid range distance.

    Returns:
        A float32 numpy array with cleaned values.
    """
    if not isinstance(scan_array, np.ndarray):
        scan_array = np.array(scan_array, dtype=np.float32)

    # Replace NaN with 0.0 and positive infinity with max_range
    cleaned = np.nan_to_num(scan_array, nan=0.0, posinf=max_range, neginf=0.0)
    
    # Clip values to ensure they fall within the valid range [0.0, max_range]
    cleaned = np.clip(cleaned, 0.0, max_range)
    
    return cleaned.astype(np.float32)


def synchronize_data(src_times: np.ndarray, target_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronizes two time series using Nearest Neighbor search.
    Optimized with np.searchsorted for O(N log M) complexity.

    Args:
        src_times: Timestamps of the source data (e.g., Scan times).
        target_times: Reference timestamps to match against (e.g., Control times).

    Returns:
        A tuple containing:
            - indices: Indices of target_times that are closest to src_times.
            - deltas: Absolute time differences between matched timestamps.
    """
    if len(target_times) == 0:
        return np.array([]), np.array([])
        
    # Find insertion points for source times in target times
    idx_sorted = np.searchsorted(target_times, src_times)
    
    # Clip indices to stay within valid bounds
    idx_sorted = np.clip(idx_sorted, 0, len(target_times) - 1)
    prev_idx = np.clip(idx_sorted - 1, 0, len(target_times) - 1)
    
    # Calculate time differences for current and previous indices
    time_diff_curr = np.abs(target_times[idx_sorted] - src_times)
    time_diff_prev = np.abs(target_times[prev_idx] - src_times)
    
    # Select the index with the smaller time difference
    use_prev = time_diff_prev < time_diff_curr
    final_indices = np.where(use_prev, prev_idx, idx_sorted)
    final_deltas = np.where(use_prev, time_diff_prev, time_diff_curr)
    
    return final_indices, final_deltas


def process_bag(
    bag_path: Path,
    output_root: Path,
    config: ExtractionConfig,
    debug: bool = False,
    label: str = ''
) -> None:
    """
    Worker function to process a single ROS bag file.
    Reads, cleans, synchronizes, and saves the data.
    """
    logger = logging.getLogger(__name__)
    bag_name = label or bag_path.name
    out_dir = output_root / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start_total = time.perf_counter()

    cmd_data: List[List[float]] = []
    cmd_times: List[int] = []
    scan_data: List[np.ndarray] = []
    scan_times: List[int] = []
    speed_data: List[float] = []
    speed_times: List[int] = []
    decode_failures: dict = {}
    first_failure: dict = {}

    # --- 1. Read Bag File ---
    t_start_read = time.perf_counter()
    try:
        with AnyReader([bag_path], default_typestore=build_typestore()) as reader:
            target_topics = [config.control_topic, config.scan_topic, config.speed_topic]
            connections = [c for c in reader.connections if c.topic in target_topics]
            
            if not connections:
                if debug: logger.warning(f"{bag_name}: No relevant topics found.")
                return

            for conn, timestamp, raw in reader.messages(connections=connections):
                try:
                    msg = reader.deserialize(raw, conn.msgtype)
                    
                    # Extract Control Command
                    if conn.topic == config.control_topic:
                        if conn.msgtype == config.control_msg_type:
                            accel = msg.longitudinal.acceleration
                            steer = msg.lateral.steering_tire_angle
                            cmd_data.append([steer, accel])
                            cmd_times.append(timestamp)
                    
                    # Extract LiDAR Scan
                    elif conn.topic == config.scan_topic:
                        if conn.msgtype == config.scan_msg_type:
                            ranges = np.array(msg.ranges, dtype=np.float32)
                            scan_vec = clean_scan_array(ranges, config.max_scan_range)
                            scan_data.append(scan_vec)
                            scan_times.append(timestamp)

                    # Extract measured speed: a model input in its own right (an
                    # acceleration target is unlearnable without the current
                    # speed) and the only reliable stall detector.
                    elif conn.topic == config.speed_topic:
                        if conn.msgtype == config.speed_msg_type:
                            speed_data.append(float(msg.longitudinal_velocity))
                            speed_times.append(timestamp)
                except Exception as exc:
                    # Count and report instead of silently dropping: a wrong or
                    # unregistered message type fails on every single message, and
                    # a bare `continue` turned that into an empty result with no
                    # explanation.
                    decode_failures[type(exc).__name__] = (
                        decode_failures.get(type(exc).__name__, 0) + 1
                    )
                    first_failure.setdefault(type(exc).__name__, f"{conn.topic}: {exc}")
                    continue
        if decode_failures:
            for name, count in sorted(decode_failures.items(), key=lambda kv: -kv[1]):
                logger.error(
                    f"{bag_name}: {count} messages failed to decode with {name} "
                    f"-- first was {first_failure[name]}"
                )
    except Exception as e:
        logger.error(f"Failed to read {bag_name}: {e}")
        return

    t_end_read = time.perf_counter()

    if not cmd_data or not scan_data:
        if debug: logger.warning(f"Skipping {bag_name}: Insufficient data.")
        return

    # Convert lists to NumPy arrays for efficient processing
    np_cmd_data = np.array(cmd_data, dtype=np.float32)
    np_cmd_times = np.array(cmd_times, dtype=np.int64)
    np_scan_data = np.array(scan_data, dtype=np.float32)
    np_scan_times = np.array(scan_times, dtype=np.int64)

    # --- 2. Synchronize Data ---
    t_start_sync = time.perf_counter()
    
    # searchsorted requires the target array to be sorted
    sort_idx = np.argsort(np_cmd_times)
    np_cmd_times = np_cmd_times[sort_idx]
    np_cmd_data = np_cmd_data[sort_idx]

    indices, deltas = synchronize_data(np_scan_times, np_cmd_times)

    synced_cmds = np_cmd_data[indices]
    synced_steers = synced_cmds[:, 0]
    synced_accels = synced_cmds[:, 1]

    # Measured speed, synchronised onto the same scan timestamps.
    if speed_data:
        np_speed_times = np.array(speed_times, dtype=np.int64)
        np_speed_data = np.array(speed_data, dtype=np.float32)
        s_sort = np.argsort(np_speed_times)
        np_speed_times = np_speed_times[s_sort]
        np_speed_data = np_speed_data[s_sort]
        s_indices, _ = synchronize_data(np_scan_times, np_speed_times)
        synced_speeds = np_speed_data[s_indices]
    else:
        logger.warning(
            f"{bag_name}: no {config.speed_topic} messages; speeds.npy will not be "
            f"written and --min-speed cannot be applied"
        )
        synced_speeds = None

    if config.min_speed > 0.0:
        if synced_speeds is None:
            logger.error(
                f"{bag_name}: --min-speed requires {config.speed_topic}, which this "
                f"bag does not contain. Refusing to write an unfiltered dataset."
            )
            return
        keep = synced_speeds >= config.min_speed
        kept, total = int(keep.sum()), len(keep)
        if not kept:
            logger.error(f"{bag_name}: every sample is below --min-speed; nothing to save")
            return
        logger.info(
            f"{bag_name}: speed filter >= {config.min_speed} m/s keeps "
            f"{kept}/{total} ({100.0 * kept / total:.1f}%)"
        )
        np_scan_data = np_scan_data[keep]
        synced_steers = synced_steers[keep]
        synced_accels = synced_accels[keep]
        synced_speeds = synced_speeds[keep]
        deltas = deltas[keep]

    t_end_sync = time.perf_counter()

    # --- 3. Save Results ---
    t_start_save = time.perf_counter()
    
    np.save(out_dir / 'scans.npy', np_scan_data)
    np.save(out_dir / 'steers.npy', synced_steers)
    np.save(out_dir / 'accelerations.npy', synced_accels)
    if synced_speeds is not None:
        np.save(out_dir / 'speeds.npy', synced_speeds)
    
    # Save delta times only when debugging to save disk space/IO
    if debug:
        delta_seconds = deltas / 1e9
        np.save(out_dir / 'delta_times.npy', delta_seconds)
    
    t_end_save = time.perf_counter()
    duration_total = t_end_save - t_start_total

    # Log successful processing
    logger.info(f"Saved {bag_name}: {len(np_scan_data)} samples (Total: {duration_total:.2f}s)")

    if debug:
        duration_read = t_end_read - t_start_read
        duration_sync = t_end_sync - t_start_sync
        duration_save = t_end_save - t_start_save
        delta_seconds = deltas / 1e9
        
        logger.debug(
            f"  [Performance {bag_name}]\n"
            f"    - Read : {duration_read:.4f}s\n"
            f"    - Sync : {duration_sync:.4f}s\n"
            f"    - Save : {duration_save:.4f}s\n"
            f"  [Sync Stats]\n"
            f"    - Δt Mean: {delta_seconds.mean():.6f}s\n"
            f"    - Δt Max : {delta_seconds.max():.6f}s"
        )


def _unique_labels(bag_dirs: List[Path]) -> List[str]:
    """One distinct output-directory name per bag.

    Starts at the bag's own basename and prepends parent directories only as far
    as needed to disambiguate, so the common case stays readable ("rosbag2_bc")
    and the colliding case becomes "npc-with-1__bag".
    """
    depth = 1
    while depth <= 6:
        labels = ['__'.join(p.parts[-depth:]) for p in bag_dirs]
        if len(set(labels)) == len(labels):
            return labels
        depth += 1
    # Pathological: fall back to index suffixes rather than overwrite.
    return [f"{p.name}__{i}" for i, p in enumerate(bag_dirs)]


def main():
    parser = argparse.ArgumentParser(
        description='Extract and synchronize scan and control data from ROS 2 bags.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bags-dir', type=Path, help='Path to directory containing rosbag folders (recursive search).')
    group.add_argument('--seq-dirs', type=Path, nargs='+', help='List of specific sequence directories to process.')
    parser.add_argument('--outdir', type=Path, required=True, help='Root directory for output files.')
    
    # Topic configuration
    parser.add_argument('--control-topic', type=str, default='/control/command/control_cmd', help='Topic name for control commands.')
    parser.add_argument('--scan-topic', type=str, default='/sensing/lidar/scan', help='Topic name for LiDAR scans.')
    parser.add_argument('--speed-topic', type=str, default='/vehicle/status/velocity_status', help='Topic name for measured vehicle speed.')
    parser.add_argument('--min-speed', type=float, default=0.0,
                        help='Drop samples whose measured speed is below this (m/s). '
                             'Race bags are mostly stalled frames; cloning them teaches '
                             'the model to stand still. 0 keeps everything.')
    
    # Performance arguments
    default_workers = min(os.cpu_count() or 1, 8)
    parser.add_argument('--workers', type=int, default=default_workers, help='Number of parallel workers.')
    parser.add_argument('--debug', action='store_true', help='Enable detailed performance and debug logging.')

    args = parser.parse_args()
    setup_logger(args.debug)
    logger = logging.getLogger(__name__)

    # --- Discovery Phase ---
    bag_dirs = []
    if args.bags_dir:
        p = args.bags_dir.expanduser().resolve()
        # Find directories containing metadata.yaml
        bag_dirs = [x.parent for x in p.rglob("metadata.yaml")]
        # Handle case where bags-dir itself is a bag
        if not bag_dirs and (p / "metadata.yaml").exists():
            bag_dirs = [p]
    elif args.seq_dirs:
        for p in args.seq_dirs:
            p = p.expanduser().resolve()
            if (p / "metadata.yaml").exists():
                bag_dirs.append(p)
    
    bag_dirs = sorted(list(set(bag_dirs)))
    if not bag_dirs:
        logger.error("No valid ROS 2 bag directories found.")
        return

    # Intelligent worker sizing: don't create more workers than tasks
    num_workers = min(max(1, args.workers), len(bag_dirs))
    logger.info(f"Found {len(bag_dirs)} bags. Starting processing with {num_workers} workers.")

    # --- Processing Phase ---
    config = ExtractionConfig(control_topic=args.control_topic, scan_topic=args.scan_topic,
                              speed_topic=args.speed_topic, min_speed=args.min_speed)
    # Output directory names must be unique per bag. Deriving them from
    # bag_path.name alone silently overwrote results whenever two bags shared a
    # basename -- e.g. output/npc-with-{1,2,3}/bag are all called "bag", so a
    # 3-bag run produced one bag's worth of data and reported success. Fall back
    # to more of the path until the labels are distinct.
    labels = _unique_labels(bag_dirs)
    for p, label in zip(bag_dirs, labels):
        if label != p.name:
            logger.info(f"{p}: writing as '{label}' to keep output names unique")
    tasks = [(p, args.outdir, config, args.debug, label)
             for p, label in zip(bag_dirs, labels)]

    start_time = time.time()
    
    # Use 'spawn' method for compatibility with various environments (especially if CUDA is involved later)
    with multiprocessing.Pool(processes=num_workers, initializer=worker_init, initargs=(args.debug,)) as pool:
        pool.starmap(process_bag, tasks)
        
    logger.info(f"All processing finished in {time.time() - start_time:.2f} seconds.")


if __name__ == '__main__':
    # Ensure multiprocessing works correctly on all platforms
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
