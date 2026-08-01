#!/usr/bin/env python3
"""Wraps a plain AckermannControlCommand into an AckermannControlBoostCommand
with boost_mode always on, so it can feed multi_purpose_mpc_ros's existing
boost_commander node (which republishes the latest command at ~1700Hz while
boost_mode is true, vs ~50Hz normally). This is a one-off test to see whether
the same high-rate-republish trick used for MPC also lets tiny_lidar_net's
fixed-acceleration output exceed the official ~1.0 m/s^2 acceleration cap.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from autoware_auto_control_msgs.msg import AckermannControlCommand
from multi_purpose_mpc_ros_msgs.msg import AckermannControlBoostCommand


class LidarBoostBridge(Node):
    def __init__(self):
        super().__init__('lidar_boost_bridge')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub = self.create_subscription(
            AckermannControlCommand, 'input/control_cmd', self._callback, qos
        )
        self.pub = self.create_publisher(
            AckermannControlBoostCommand, 'output/boost_command', 10
        )

    def _callback(self, msg: AckermannControlCommand):
        out = AckermannControlBoostCommand()
        out.command = msg
        out.boost_mode = True
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LidarBoostBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
