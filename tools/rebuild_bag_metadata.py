#!/usr/bin/env python3
"""Rebuild a missing rosbag2 metadata.yaml from the .db3 beside it.

The 2026-07-31 npc-with-{1,2,3} recordings were killed without a clean
shutdown, so `ros2 bag record` never wrote metadata.yaml. The sqlite files are
intact -- topics, QoS profiles, ros_distro, timestamps and every message are all
in there -- but without the sidecar nothing in the ROS toolchain will open them,
including `extract_data_from_bag.py` (via rosbags' AnyReader) and `ros2 bag play`.

Everything metadata.yaml needs is recoverable from the db3, so this reconstructs
it rather than teaching one consumer to read db3 directly. Writes nothing if a
metadata.yaml already exists unless --force is passed.

Usage:
  python3 tools/rebuild_bag_metadata.py output/npc-with-*/bag
  python3 tools/rebuild_bag_metadata.py --check output/npc-with-1/bag
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

# rosbag2 humble writes version 5; the reader accepts <= its own version.
METADATA_VERSION = 5


def collect(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        distro_row = con.execute("select ros_distro from schema").fetchone()
        ros_distro = distro_row[0] if distro_row else "humble"

        topics = []
        total = 0
        starts, ends = [], []
        for tid, name, ttype, fmt, qos in con.execute(
            "select id, name, type, serialization_format, offered_qos_profiles from topics"
        ):
            count, tmin, tmax = con.execute(
                "select count(*), min(timestamp), max(timestamp) from messages where topic_id = ?",
                (tid,),
            ).fetchone()
            topics.append(
                {
                    "topic_metadata": {
                        "name": name,
                        "type": ttype,
                        "serialization_format": fmt,
                        # Round-tripped verbatim: rosbag2 stores this as an opaque
                        # YAML string, and re-serialising it would change it.
                        "offered_qos_profiles": qos,
                    },
                    "message_count": count,
                }
            )
            total += count
            if count:
                starts.append(tmin)
                ends.append(tmax)
    finally:
        con.close()

    if not total:
        raise ValueError(f"{db} contains no messages")

    start, end = min(starts), max(ends)
    return {
        "rosbag2_bagfile_information": {
            "version": METADATA_VERSION,
            "storage_identifier": "sqlite3",
            "relative_file_paths": [db.name],
            "duration": {"nanoseconds": end - start},
            "starting_time": {"nanoseconds_since_epoch": start},
            "message_count": total,
            "topics_with_message_count": topics,
            "compression_format": "",
            "compression_mode": "",
            "ros_distro": ros_distro,
        }
    }


def process(bag_dir: Path, force: bool, check_only: bool) -> bool:
    db3s = sorted(bag_dir.glob("*.db3"))
    if not db3s:
        print(f"SKIP  {bag_dir}: no .db3")
        return True
    if len(db3s) > 1:
        # Multi-file bags need the split order preserved; refuse rather than guess.
        print(f"SKIP  {bag_dir}: {len(db3s)} .db3 files, split bags are not handled")
        return False

    meta_path = bag_dir / "metadata.yaml"
    if meta_path.exists() and not force:
        print(f"OK    {bag_dir}: metadata.yaml already present")
        return True

    try:
        meta = collect(db3s[0])
    except Exception as exc:
        print(f"FAIL  {bag_dir}: {exc}")
        return False

    info = meta["rosbag2_bagfile_information"]
    summary = (
        f"{info['message_count']} msgs, "
        f"{info['duration']['nanoseconds'] / 1e9:.0f}s, "
        f"{len(info['topics_with_message_count'])} topics"
    )
    if check_only:
        print(f"WOULD WRITE  {meta_path}  ({summary})")
        return True

    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"WROTE {meta_path}  ({summary})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dirs", nargs="+", type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite an existing metadata.yaml")
    ap.add_argument("--check", action="store_true", help="report what would be written")
    args = ap.parse_args()

    ok = True
    for bag_dir in args.bag_dirs:
        if not bag_dir.is_dir():
            print(f"FAIL  {bag_dir}: not a directory")
            ok = False
            continue
        ok &= process(bag_dir, args.force, args.check)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
