#!/usr/bin/env python3
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport

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
        self.declare_parameter('control_mode', 'ai')
        self.declare_parameter('debug', False)
        self.declare_parameter('max_speed_mps', 10.0)
        self.declare_parameter('min_speed_mps', 0.0)

        # --- Initialization ---
        input_dim = self.get_parameter('model.input_dim').value
        output_dim = self.get_parameter('model.output_dim').value
        architecture = self.get_parameter('model.architecture').value
        ckpt_path = self.get_parameter('model.ckpt_path').value
        max_range = self.get_parameter('max_range').value
        acceleration = self.get_parameter('acceleration').value
        control_mode = self.get_parameter('control_mode').value
        # Override so the July design and any alternative can be compared without file edits.
        control_mode = os.environ.get('TLN_CONTROL_MODE', '') or control_mode
        
        self.debug = self.get_parameter('debug').value
        self.log_interval = self.get_parameter('log_interval_sec').value
        self.launch_speed_mps = float(os.environ.get('TLN_LAUNCH_SPEED', '') or '2.0')
        self.lead_margin_mps = float(os.environ.get('TLN_LEAD_MARGIN', '') or '1.5')
        self.max_speed_mps = float(os.environ.get('TLN_MAX_SPEED', '')
                                   or self.get_parameter('max_speed_mps').value)
        self.min_speed_mps = self.get_parameter('min_speed_mps').value

        # AckermannControlCommand.longitudinal.speed is the setpoint AWSIM's
        # vehicle bridge actually tracks; .acceleration alone never moves the
        # vehicle. No planned trajectory exists here to read a target speed
        # from like MPC does, so integrate one from the model's own accel
        # output against the last known real velocity.
        self._last_velocity_mps = 0.0
        self._last_callback_time = None

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
        self.sub_velocity = self.create_subscription(
            VelocityReport, "/vehicle/status/velocity_status",
            self._velocity_callback, qos
        )
        self.pub_control = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1
        )

        self.get_logger().info("TinyLidarNetNode is ready.")

    def _velocity_callback(self, msg: VelocityReport):
        """Tracks the vehicle's real current speed for speed-setpoint integration."""
        self._last_velocity_mps = msg.longitudinal_velocity

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

        # 2b. Integrate a speed setpoint from the model's own acceleration
        # (see note above __init__ on why .acceleration alone can't move
        # the vehicle).
        now = self.get_clock().now()
        if self._last_callback_time is not None:
            dt = (now - self._last_callback_time).nanoseconds / 1e9
        else:
            dt = 0.1  # first callback: assume the nominal ~10Hz scan rate
        self._last_callback_time = now
        dt = max(0.0, min(dt, 0.5))  # guard against a stale/huge gap

        # July's closed loop, plus a floor. Integrating from the MEASURED velocity is what makes the
        # setpoint track reality instead of climbing to the cap while the car is stuck against a wall.
        # But at a standstill it traps itself: v_meas 0.00 gives 0 + 0.6*0.05 = 0.03 m/s forever, which is
        # exactly what the restored node did -- the car never left the grid. The floor only applies while
        # the vehicle is essentially stopped, so it restores launch without weakening the tracking.
        # Both pure forms fail, in opposite directions, and this is measured rather than argued:
        #   open loop  (setpoint += accel*dt)         climbs to the cap while the car sits against a wall,
        #                                            so the command bears no relation to reality
        #   closed loop (setpoint = v_meas + accel*dt) is July's, and it plateaus: the vehicle tracks the
        #                                            setpoint with a steady-state error, so the two hold
        #                                            each other at an equilibrium -- measured 2.5 m/s here
        # So ratchet the setpoint up like the open loop, but never let it lead the measured speed by more
        # than lead_margin. That accelerates properly and still collapses back to reality when the car is
        # held up, which is what the closed loop was protecting.
        self._setpoint = getattr(self, "_setpoint", 0.0) + float(accel) * dt
        self._setpoint = min(self._setpoint, self._last_velocity_mps + self.lead_margin_mps)
        if self._last_velocity_mps < self.launch_speed_mps:
            self._setpoint = max(self._setpoint, self.launch_speed_mps)
        target_speed = self._setpoint
        target_speed = float(np.clip(target_speed, self.min_speed_mps, self.max_speed_mps))

        # 3. Publish Command
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.speed = target_speed
        cmd.longitudinal.acceleration = float(accel)
        cmd.lateral.steering_tire_angle = float(steer)
        self.pub_control.publish(cmd)
        # Report the head, the measured speed it was integrated from, and the setpoint, once a
        # second. Without this the node runs silently and a constant output looks like driving.
        if getattr(self, '_last_report', None) is None or \
                (now.nanoseconds * 1e-9) - self._last_report >= 1.0:
            self._last_report = now.nanoseconds * 1e-9
            self.get_logger().info(
                f'[tln] head=({float(accel):+.3f}, {float(steer):+.3f}) '
                f'v_meas={self._last_velocity_mps:.2f} setpoint={target_speed:.2f} '
                f'lead={self.lead_margin_mps:.1f} '
                f'cap={self.max_speed_mps:.2f} mode={self.core.control_mode}')

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
