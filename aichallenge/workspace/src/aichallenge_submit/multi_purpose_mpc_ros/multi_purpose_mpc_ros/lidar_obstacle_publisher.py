#!/usr/bin/env python3
"""Publish lidar-detected dynamic obstacles on /aichallenge/objects.

Why this exists: the MPC's obstacle-avoidance path is fully wired -- config flags,
constraint computation, the lot -- but the obstacle set has always been empty because
nothing publishes /aichallenge/objects (tools/GOAL.md, 2026-08-02). Everything fixed in
that path so far is on the cost side; this is the only thing that can put anything on
the benefit side.

It matters because the arithmetic leaves no alternative: the NPC laps in about 149 s,
so following it for the whole 480 s window yields 3.2 laps against the 6 required.
Overtaking is not an optimisation, and overtaking needs the NPC in the planner's
constraints. The NPC never appears on v2x -- measured -- so lidar is the only sensor
that sees it.

Telling the NPC apart from the track is the actual work. The approach here is that the
static track is already known: the occupancy grid the MPC plans against contains every
wall and cone. So a return that lands where the map says free space is, by elimination,
something that was not there when the map was made.

Contract (path_constraints_provider._obstacles_callback and mpc_controller alike):
Float64MultiArray, stride 4, data[i] and data[i+1] are map-frame x and y. The consumer
attaches obstacles.radius itself, so the last two slots are padding.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.common import convert_to_namedtuple

import yaml


class LidarObstaclePublisher(Node):

    def __init__(self) -> None:
        super().__init__("lidar_obstacle_publisher")

        self.declare_parameter("config_path", "")
        self.declare_parameter("scan_topic", "/sensing/lidar/scan")
        self.declare_parameter("odom_topic", "/localization/kinematic_state")
        self.declare_parameter("objects_topic", "/aichallenge/objects")
        # A return this far from the nearest occupied cell is not explained by the static
        # map. One cell of slack is not enough: localisation error and the grid's own
        # discretisation both smear wall returns outward.
        self.declare_parameter("free_space_margin_m", 0.60)
        self.declare_parameter("max_range_m", 25.0)
        self.declare_parameter("min_range_m", 0.5)
        # Returns closer together than this belong to the same object.
        self.declare_parameter("cluster_gap_m", 0.8)
        # A vehicle subtends several beams; a single stray return does not.
        self.declare_parameter("min_cluster_points", 3)
        self.declare_parameter("publish_empty", True)

        pkg = get_package_share_directory("multi_purpose_mpc_ros") + "/"
        cfg_path = self.get_parameter("config_path").value or (pkg + "config/config.yaml")
        cfg = convert_to_namedtuple(yaml.safe_load(open(cfg_path)))
        self._map = Map(pkg + cfg.map.yaml_path)

        # Distance from every cell to the nearest occupied cell, in metres. Computed once;
        # the static map does not change. Without scipy this would be a per-return search.
        from scipy import ndimage
        free = np.asarray(self._map.data, dtype=bool)
        # Map.data is True/1 where drivable, so occupied is its complement.
        self._dist_to_wall_m = ndimage.distance_transform_edt(free) * self._map.resolution

        self._free_margin = float(self.get_parameter("free_space_margin_m").value)
        self._max_range = float(self.get_parameter("max_range_m").value)
        self._min_range = float(self.get_parameter("min_range_m").value)
        self._gap = float(self.get_parameter("cluster_gap_m").value)
        self._min_pts = int(self.get_parameter("min_cluster_points").value)
        self._publish_empty = bool(self.get_parameter("publish_empty").value)

        self._pose: Tuple[float, float, float] | None = None
        self._last_n = -1

        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value,
                                 self._on_odom, 1)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value,
                                 self._on_scan, sensor_qos)
        self._pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("objects_topic").value, 1)

        self.get_logger().info(
            f"lidar_obstacle_publisher up: free_margin={self._free_margin} m, "
            f"gap={self._gap} m, min_points={self._min_pts}")

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def _on_scan(self, msg: LaserScan) -> None:
        if self._pose is None:
            return
        x0, y0, yaw = self._pose

        rng = np.asarray(msg.ranges, dtype=np.float64)
        ok = np.isfinite(rng) & (rng >= self._min_range) & (rng <= self._max_range)
        if not ok.any():
            self._publish([])
            return

        idx = np.nonzero(ok)[0]
        ang = msg.angle_min + idx * msg.angle_increment + yaw
        px = x0 + rng[idx] * np.cos(ang)
        py = y0 + rng[idx] * np.sin(ang)

        # Keep only returns the static map cannot account for.
        gx = np.clip(((px - self._map.origin[0]) / self._map.resolution + 0.5).astype(int),
                     0, self._map.width - 1)
        gy = np.clip(((self._map.height - 1)
                      - (py - self._map.origin[1]) / self._map.resolution + 0.5).astype(int),
                     0, self._map.height - 1)
        free_by = self._dist_to_wall_m[gy, gx]
        dynamic = free_by > self._free_margin
        if not dynamic.any():
            self._publish([])
            return

        self._publish(self._cluster(px[dynamic], py[dynamic]))

    def _cluster(self, px: np.ndarray, py: np.ndarray) -> List[Tuple[float, float]]:
        """Split consecutive returns wherever the gap between them exceeds cluster_gap_m.

        Beams arrive in angular order, so consecutive returns on one object are adjacent
        in the array. That makes a single pass enough and avoids pulling in a clustering
        dependency.
        """
        out: List[Tuple[float, float]] = []
        start = 0
        n = len(px)
        for i in range(1, n + 1):
            split = i == n
            if not split:
                split = math.hypot(px[i] - px[i - 1], py[i] - py[i - 1]) > self._gap
            if split:
                if i - start >= self._min_pts:
                    out.append((float(px[start:i].mean()), float(py[start:i].mean())))
                start = i
        return out

    def _publish(self, centroids: List[Tuple[float, float]]) -> None:
        if not centroids and not self._publish_empty:
            return
        msg = Float64MultiArray()
        data: List[float] = []
        for cx, cy in centroids:
            data.extend([cx, cy, 0.0, 0.0])   # slots 2 and 3 unused; radius is the consumer's
        msg.data = data
        self._pub.publish(msg)
        if len(centroids) != self._last_n:
            self._last_n = len(centroids)
            self.get_logger().info(f"obstacles: {len(centroids)}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarObstaclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
