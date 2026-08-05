#!/usr/bin/env python3
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand

from tiny_lidar_net_controller_core import TinyLidarNetCore


class TinyLidarNetNode(Node):
    """ROS 2 Node for TinyLidarNet autonomous driving control.

    This node subscribes to LaserScan messages, processes them using the
    TinyLidarNetCore logic, and publishes AckermannControlCommand messages.
    """

    def __init__(self):
        super().__init__('tiny_lidar_net_node')

        # --- Parameter Declaration ---
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.output_dim', 2)
        self.declare_parameter('model.architecture', 'large')
        self.declare_parameter('model.ckpt_path', '')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('acceleration', 0.1)
        # The network's second head is an acceleration, and this node used to publish only
        # that, leaving AckermannControlCommand.longitudinal.speed at its default 0.0. The
        # vehicle follows speed, so the car never moved: measured live at 20 Hz with
        # steering alive at -0.216 rad, speed 0.0, and longitudinal_velocity 0.0 throughout.
        # Zero laps, and it was read as the model having failed to learn speed.
        # Integrate the commanded acceleration into a speed target instead.
        # 9.2 m/s, not 8.33. The expert demonstrations recorded from clean MPC runs command up to
        # 9.167 m/s and 48.4 percent of their frames exceed 8.33, so normalising against 8.33 would
        # squash nearly half the target range against the top of the tanh and teach the network that
        # every fast section is the same speed. Must equal TLN_SPEED_VMAX used when training.
        self.declare_parameter('v_max', 9.2)
        self.declare_parameter('accel_scale', 1.0)
        self.declare_parameter('control_mode', 'ai')
        self.declare_parameter('debug', False)

        # --- Initialization ---
        input_dim = self.get_parameter('model.input_dim').value
        output_dim = self.get_parameter('model.output_dim').value
        architecture = self.get_parameter('model.architecture').value
        ckpt_path = self.get_parameter('model.ckpt_path').value
        max_range = self.get_parameter('max_range').value
        acceleration = self.get_parameter('acceleration').value
        self._v_max = float(self.get_parameter('v_max').value)
        self._accel_scale = float(self.get_parameter('accel_scale').value)

        # Set TLN_DIRECT_SPEED=1 only with a checkpoint trained on the commanded
        # speed; the default keeps the acceleration convention the shipped weights use.
        self._direct_speed = os.environ.get("TLN_DIRECT_SPEED", "0") not in ("0", "false", "")
        self._target_v = 0.0
        self._last_t = None
        control_mode = self.get_parameter('control_mode').value
        
        self.debug = self.get_parameter('debug').value
        self.log_interval = self.get_parameter('log_interval_sec').value

        try:
            self.core = TinyLidarNetCore(
                input_dim=input_dim,
                output_dim=output_dim,
                architecture=architecture,
                ckpt_path=ckpt_path,
                acceleration=acceleration,
                control_mode=control_mode,
                max_range=max_range
            )
            self.get_logger().info(
                f"Core initialized. Arch: {architecture}, MaxRange: {max_range}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize core logic: {e}")
            raise e

        # --- Communication Setup ---
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos
        )
        self.pub_control = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1
        )

        self.get_logger().info("TinyLidarNetNode is ready.")

    def scan_callback(self, msg: LaserScan):
        """Callback for LaserScan subscription.

        Processes the scan data via the core logic and publishes a control command.

        Args:
            msg (LaserScan): The incoming ROS 2 LaserScan message.
        """
        start_time = time.monotonic()

        # 1. Convert ROS message to Numpy
        # We pass the raw array; the core logic handles NaN/Inf and normalization.
        ranges = np.array(msg.ranges, dtype=np.float32)

        # 2. Process via Core Logic
        accel, steer = self.core.process(ranges)

        # 3. Publish Command
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = float(accel)
        now_s = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.05 if self._last_t is None else min(max(now_s - self._last_t, 0.0), 0.2)
        self._last_t = now_s
# Two target conventions, chosen explicitly. A checkpoint trained against one is
        # meaningless under the other, and the failure would look like bad driving rather than a
        # configuration mismatch, so it is never inferred.
        if self._direct_speed:
            # head[0] is a normalised speed. No integration, so no accumulated bias from a
            # quantity the 750-point scan cannot determine on its own.
            self._target_v = min(max((float(accel) + 1.0) * 0.5 * self._v_max, 0.0), self._v_max)
        else:
            self._target_v = min(
                max(self._target_v + float(accel) * self._accel_scale * dt, 0.0), self._v_max)
        cmd.longitudinal.speed = float(self._target_v)
        cmd.lateral.steering_tire_angle = float(steer)
        self.pub_control.publish(cmd)

        # 4. Debug Logging
        if self.debug:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            self.inference_times.append(duration_ms)
            self._log_performance_metrics()

    def _log_performance_metrics(self):
        """Logs internal performance metrics at fixed intervals."""
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_log_time).nanoseconds / 1e9

        if elapsed_sec > self.log_interval:
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                max_time = np.max(self.inference_times)
                fps = 1000.0 / avg_time if avg_time > 0 else 0.0

                self.get_logger().info(
                    f"DEBUG: Avg Inference: {avg_time:.2f}ms ({fps:.2f}Hz) | "
                    f"Max: {max_time:.2f}ms"
                )
                self.inference_times.clear()
            
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = TinyLidarNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
