"""Record where the car actually is when it stops, so the wedges can be located.

Five solo runs of one unchanged configuration totalled 231.9, 249.2, 293.6, 303.6 and
232.5 s. Three of the five lost 20-70 s to a wedge. That spread is larger than any
parameter effect measured in this project, so the thing to fix is the wedge, and the first
question is where it happens -- not which weight to nudge.

Writes a CSV of pose, speed and steering at 20 Hz plus a marked list of stop events.
"""
import csv
import math
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from autoware_auto_control_msgs.msg import AckermannControlCommand


class Probe(Node):
    def __init__(self, path):
                # BEST_EFFORT to match the publishers; a RELIABLE subscriber silently
        # gets nothing from them.
        super().__init__("wedge_probe")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "x", "y", "v", "cmd_v", "steer"])
        self.odo = None
        self.cmd = None
        self.create_subscription(Odometry, "/localization/kinematic_state",
                                 self.on_odo, qos)
        self.create_subscription(AckermannControlCommand,
                                 "/control/command/control_cmd", self.on_cmd, qos)
        self.create_timer(0.05, self.tick)
        self.n = 0

    def on_odo(self, m):
        self.odo = m

    def on_cmd(self, m):
        self.cmd = m

    def tick(self):
        if self.odo is None:
            return
        t = self.get_clock().now().nanoseconds / 1e9
        p = self.odo.pose.pose.position
        v = self.odo.twist.twist.linear.x
        cv = self.cmd.longitudinal.speed if self.cmd else float("nan")
        st = self.cmd.lateral.steering_tire_angle if self.cmd else float("nan")
        self.w.writerow([f"{t:.3f}", f"{p.x:.3f}", f"{p.y:.3f}",
                         f"{v:.3f}", f"{cv:.3f}", f"{st:.4f}"])
        self.n += 1
        if self.n % 200 == 0:
            self.f.flush()


rclpy.init()
node = Probe(sys.argv[1])
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.f.flush()
    node.f.close()
