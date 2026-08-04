#!/usr/bin/env python3
"""Small LiDAR safety layer for the high-speed MPC controller.

The MPC remains the primary controller.  This node only intervenes when a
wall is immediately ahead or on the side toward which the kart is steering.
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand


class LidarGuard(Node):
    def __init__(self) -> None:
        super().__init__("lidar_guard")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.front_limit = float(self.declare_parameter("front_limit", 1.25).value)
        self.side_limit = float(self.declare_parameter("side_limit", 0.72).value)
        self.guard_speed = float(self.declare_parameter("guard_speed", 5.5).value)
        self._scan: Optional[LaserScan] = None
        self._cmd: Optional[AckermannControlCommand] = None
        self.create_subscription(LaserScan, "/sensing/lidar/scan", self._scan_cb, qos)
        self.create_subscription(
            AckermannControlCommand, "/control/command/mpc_cmd", self._cmd_cb, qos
        )
        self._pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        self._scan = msg

    def _cmd_cb(self, msg: AckermannControlCommand) -> None:
        self._cmd = msg
        # Publish on arrival rather than on a timer. This node is always in the MPC's
        # command chain, so with no LiDAR it has to be an exact passthrough -- a timer
        # would both add latency and republish stale commands at a rate unrelated to
        # the controller's own.
        self._publish()

    @staticmethod
    def _sector(scan: LaserScan, lo: float, hi: float) -> float:
        vals = []
        for i, value in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            if lo <= angle <= hi and math.isfinite(value) and value > 0.05:
                vals.append(min(value, scan.range_max))
        return min(vals) if vals else scan.range_max

    def _publish(self) -> None:
        if self._cmd is None:
            return
        out = self._cmd
        if self._scan is not None:
            front = self._sector(self._scan, -0.44, 0.44)
            left = self._sector(self._scan, 0.44, 1.40)
            right = self._sector(self._scan, -1.40, -0.44)
            steer = out.lateral.steering_tire_angle
            toward_side = right if steer < 0.0 else left
            imminent = front < self.front_limit or toward_side < self.side_limit
            if imminent:
                out.longitudinal.speed = min(out.longitudinal.speed, self.guard_speed)
                out.longitudinal.acceleration = min(out.longitudinal.acceleration, 0.45)
            if front < 0.90:
                # Bias toward the side with more free space only at imminent range.
                escape = 0.28 if left > right else -0.28
                out.lateral.steering_tire_angle = 0.55 * steer + 0.45 * escape
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarGuard()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
