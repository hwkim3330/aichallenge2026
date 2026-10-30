#!/usr/bin/env python3
"""Small LiDAR safety layer for the high-speed MPC controller.

The MPC remains the primary controller.  This node only intervenes when a
wall is immediately ahead or on the side toward which the kart is steering.

Race mode (SIM_MODE=race*, the human vs AI race of tools/race.sh): it also keeps clear of whatever the LiDAR sees on
the path ahead -- the human's kart, which the MPC knows nothing of. The corridor follows the arc the MPC is steering
(so a wall round a bend is not "ahead"); when something is in it within AVOID_RANGE the kart is pulled toward the
lateral offset with the most room (pure pursuit to that offset), and its speed is held to what stops it AVOID_GAP short
of whatever is still in its way. Commanded acceleration stays under the kart's cap (AWSIM flags more as OVER).
"""

import math
import os
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand


RACE = os.environ.get("SIM_MODE", "").startswith("race")
WB = 1.087                 # m, the kart's wheelbase
HALF = 0.85                # m, half the corridor: the kart's half width and a margin
AVOID_RANGE = 14.0         # m, start avoiding something in the corridor this far ahead
AVOID_GAP = 2.5            # m, stop this far short of it
DECEL = 1.2                # m/s2, the braking the speed cap plans with (under the 1.37 cap)
OFFSETS = (0.0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8)


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

    @staticmethod
    def _objects(scan: LaserScan):
        """scan points of object-sized clusters only: consecutive returns closer than 0.4 m form a cluster, and one
        under 2.2 m across is a kart (or a person), not the long run of a wall or a barrier"""
        pts = []
        for i, r in enumerate(scan.ranges):
            a = scan.angle_min + i * scan.angle_increment
            if math.isfinite(r) and 0.3 < r < AVOID_RANGE + 5.0 and abs(a) < 1.3:
                pts.append((r * math.cos(a), r * math.sin(a)))
            else:
                pts.append(None)
        out, cur = [], []

        def flush():
            if len(cur) >= 2 and math.dist(cur[0], cur[-1]) < 2.2:
                out.extend(cur)
        for p in pts:
            if p is None or (cur and math.dist(cur[-1], p) > 0.4):
                flush()
                cur = []
            if p is not None:
                cur.append(p)
        flush()
        return out

    def _clear(self, scan: LaserScan, steer: float):
        """free distance ahead along the steered arc, for each lateral offset of the corridor"""
        k = math.tan(steer) / WB
        free = {o: AVOID_RANGE + 5.0 for o in OFFSETS}
        for x, y in self._objects(scan):
            if x < 0.3:
                continue
            lat = y - 0.5 * k * x * x          # off the arc
            for o in OFFSETS:
                if abs(lat - o) < HALF and x < free[o]:
                    free[o] = x
        return free

    def _avoid(self, out: AckermannControlCommand) -> None:
        steer = out.lateral.steering_tire_angle
        free = self._clear(self._scan, steer)
        ahead = free[0.0]
        if ahead >= AVOID_RANGE:
            return
        # the offset with the most room; a smaller swerve wins a near tie
        best = max(OFFSETS, key=lambda o: (free[o] - 1.5 * abs(o), -abs(o)))
        room = ahead
        if best != 0.0 and free[best] > ahead + 2.0:
            look = max(4.0, min(ahead, 8.0))
            out.lateral.steering_tire_angle = max(-0.64, min(0.64, steer + math.atan(WB * 2.0 * best / (look * look))))
            room = free[best]
        cap = math.sqrt(max(0.0, 2.0 * DECEL * (room - AVOID_GAP)))
        if out.longitudinal.speed > cap:
            out.longitudinal.speed = cap
            out.longitudinal.acceleration = min(out.longitudinal.acceleration, 0.0)

    def _publish(self) -> None:
        if self._cmd is None:
            return
        out = self._cmd
        if self._scan is not None and RACE:
            self._avoid(out)
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
