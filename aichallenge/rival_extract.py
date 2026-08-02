"""Pull the line and speed profile out of a rival's downloaded rosbag.

Each public submission on the board exposes vehicle 1's bag at
  https://d3al8lo5i04x19.cloudfront.net/<build_id>/1/rosbag2_autoware.mcap
and the board's own live payload carries the build_id for every recent battle, so the
leaders' actual driven lines are obtainable. This is how traj_top36_fitted was produced
earlier in the project; the same treatment applied to the current leaders gives a
reference measured under the scored conditions rather than inferred from lap times.
"""
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import Odometry

r = SequentialReader()
r.open(StorageOptions(uri=sys.argv[1], storage_id="mcap"), ConverterOptions("", ""))
P = []
topics = {}
while r.has_next():
    topic, data, ts = r.read_next()
    topics[topic] = topics.get(topic, 0) + 1
    if topic != "/localization/kinematic_state":
        continue
    m = deserialize_message(data, Odometry)
    P.append((ts / 1e9, m.pose.pose.position.x, m.pose.pose.position.y,
              m.twist.twist.linear.x))

print(f"토픽 {len(topics)}종")
for t, c in sorted(topics.items(), key=lambda x: -x[1])[:8]:
    print(f"  {c:7d}  {t}")

if not P:
    print("kinematic_state 없음")
    raise SystemExit(1)

A = np.array(P)
t, x, y, v = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
print(f"\n주행 {t[-1]-t[0]:.0f}초, {len(A)}샘플")
print(f"좌표 x {x.min():.1f}~{x.max():.1f}  y {y.min():.1f}~{y.max():.1f}")
print(f"속도 {v.min():.2f} ~ {v.max():.2f} m/s  (평균 {v.mean():.2f})")

# lap split on return to the start point
d0 = np.hypot(x - x[0], y - y[0])
cross = []
for i in range(40, len(d0) - 1):
    if d0[i] < 8.0 and d0[i] <= d0[i-1] and d0[i] < d0[i+1] and \
            (not cross or t[i] - t[cross[-1]] > 20):
        cross.append(i)
print(f"랩 경계 {len(cross)}개")
laps = []
for a, b in zip(cross, cross[1:]):
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x[a:b]), np.diff(y[a:b])))])
    laps.append((t[b] - t[a], s[-1], v[a:b], a, b))
for i, (dur, L, vv, _, _) in enumerate(laps):
    print(f"  랩{i+1}: {dur:6.2f}초  길이 {L:6.1f} m  속도 {vv.min():5.2f}~{vv.max():5.2f}")

if laps:
    best = min(laps, key=lambda L: L[0])
    a, b = best[3], best[4]
    out = sys.argv[2]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x[a:b]), np.diff(y[a:b])))])
    psi = np.arctan2(np.gradient(y[a:b]), np.gradient(x[a:b]))
    with open(out, "w") as f:
        f.write("s_m,x_m,y_m,psi_rad,vx_mps\n")
        for i in range(b - a):
            f.write(f"{s[i]:.6f},{x[a+i]:.6f},{y[a+i]:.6f},{psi[i]:.6f},{v[a+i]:.6f}\n")
    print(f"\n최고 랩 {best[0]:.2f}초 ({best[1]:.1f} m) -> {out}")
