"""How many vehicles does v2x actually list, and where is the leader fast?"""
import sys
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import Odometry
from autoware_auto_control_msgs.msg import AckermannControlCommand
from v2x_msgs.msg import V2XVehiclePositionArray

r = SequentialReader()
r.open(StorageOptions(uri=sys.argv[1], storage_id="mcap"), ConverterOptions("", ""))
counts, P, C = {}, [], []
while r.has_next():
    topic, data, ts = r.read_next()
    if topic == "/v2x/vehicle_positions":
        m = deserialize_message(data, V2XVehiclePositionArray)
        n = len(m.vehicles) if hasattr(m, "vehicles") else len(m.positions)
        counts[n] = counts.get(n, 0) + 1
    elif topic == "/localization/kinematic_state":
        m = deserialize_message(data, Odometry)
        P.append((ts / 1e9, m.pose.pose.position.x, m.pose.pose.position.y,
                  m.twist.twist.linear.x))
    elif topic == "/control/command/control_cmd":
        m = deserialize_message(data, AckermannControlCommand)
        C.append((ts / 1e9, m.longitudinal.speed, m.lateral.steering_tire_angle))

print(f"v2x 메시지당 차량 수 분포: {dict(sorted(counts.items()))}")
A = np.array(P)
t, x, y, v = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
if C:
    D = np.array(C)
    print(f"명령 속도 범위 {D[:,1].min():.2f} ~ {D[:,1].max():.2f} m/s "
          f"({D[:,1].max()*3.6:.1f} km/h), 조향 |max| {np.abs(D[:,2]).max():.3f}")

d0 = np.hypot(x - x[0], y - y[0])
cross = [i for i in range(40, len(d0)-1)
         if d0[i] < 8.0 and d0[i] <= d0[i-1] and d0[i] < d0[i+1]]
cross = [c for i, c in enumerate(cross) if i == 0 or t[c] - t[cross[i-1]] > 20]
laps = [(t[b]-t[a], a, b) for a, b in zip(cross, cross[1:])]
dur, a, b = min(laps)
s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x[a:b]), np.diff(y[a:b])))])
print(f"\n최고 랩 {dur:.2f}초, 길이 {s[-1]:.1f} m — 20 m 구간별 속도")
grid = np.arange(0, s[-1], 20.0)
vi = np.interp(grid, s, v[a:b])
xi = np.interp(grid, s, x[a:b]); yi = np.interp(grid, s, y[a:b])
for g, sv, px, py in zip(grid, vi, xi, yi):
    print(f"  {g:5.0f} m  {sv:5.2f} m/s ({sv*3.6:5.1f} km/h)  위치({px:.0f},{py:.0f})")
print(f"\n최저 {vi.min():.2f} m/s ({vi.min()*3.6:.1f} km/h)  최고 {vi.max():.2f} ({vi.max()*3.6:.1f})")
