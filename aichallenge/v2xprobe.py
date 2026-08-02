"""Count what /v2x/vehicle_positions actually carries, and whether a LiDAR scan exists.

The question this answers: with one Autoware car and one AWSIM NPC on track, does v2x
report one entity or two? Our MPC sees rivals only through v2x, so if the NPC is absent
there it is invisible to us -- and the scored battles always include one.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class Probe(Node):
    def __init__(self):
        super().__init__("v2x_probe")
        self.v2x_msgs = 0
        self.v2x_counts = set()
        self.scan_msgs = 0
        self.scan_ranges = 0
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        try:
            from v2x_msgs.msg import V2XVehiclePositionArray
            self.create_subscription(V2XVehiclePositionArray,
                                     "/v2x/vehicle_positions", self.on_v2x, qos)
            self.get_logger().info("v2x_msgs 사용 가능")
        except Exception as e:
            self.get_logger().error(f"v2x_msgs 임포트 실패: {e}")
        from sensor_msgs.msg import LaserScan
        for topic in ("/sensing/lidar/scan", "/scan", "/sensing/lidar/top/scan"):
            self.create_subscription(LaserScan, topic,
                                     lambda m, t=topic: self.on_scan(m, t), qos)
        self.topics = {}
        self.create_timer(2.0, self.tick)
        self.ticks = 0

    def on_v2x(self, msg):
        self.v2x_msgs += 1
        self.v2x_counts.add(len(msg.vehicles) if hasattr(msg, "vehicles")
                            else len(getattr(msg, "positions", [])))

    def on_scan(self, msg, topic):
        self.scan_msgs += 1
        self.scan_ranges = len(msg.ranges)
        self.topics[topic] = self.topics.get(topic, 0) + 1

    def tick(self):
        self.ticks += 1
        names = [n for n, ts in self.get_topic_names_and_types()
                 if any(k in n for k in ("v2x", "scan", "lidar", "points",
                                         "objects", "obstacle"))]
        if self.ticks >= 6:
            print("\n===== 결과 =====")
            print(f"v2x 메시지 {self.v2x_msgs}건, 보고된 차량 수 집합 {sorted(self.v2x_counts)}")
            print(f"LaserScan 메시지 {self.scan_msgs}건, 빔 {self.scan_ranges}개, "
                  f"토픽별 {self.topics}")
            print("관련 토픽:")
            for n in sorted(names):
                print(f"  {n}")
            raise SystemExit(0)


rclpy.init()
try:
    rclpy.spin(Probe())
except SystemExit:
    pass
