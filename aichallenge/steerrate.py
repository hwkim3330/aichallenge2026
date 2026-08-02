"""Is the MPC's steering-rate constraint the thing holding lateral tracking back?

The constraint is built in MPC.py as max_delta_change = max_steering_rate * Ts, applied
between consecutive horizon steps. But the horizon is SPATIAL -- step n advances by
delta_s (0.6 m) -- while Ts is 1/control_rate = 0.025 s. Those are not the same clock.
The real time to cover one step is delta_s / v, which at 8 m/s is 0.075 s, so the
constraint is roughly 3x tighter than the configured 0.35 rad/s intends, and it does not
loosen as the car slows down (it should).

If that is binding, the commanded steer angle will sit on a straight ramp through the
corner entries rather than curving to meet the reference.
"""
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from autoware_auto_control_msgs.msg import AckermannControlCommand
from nav_msgs.msg import Odometry

r = SequentialReader()
r.open(StorageOptions(uri=sys.argv[1], storage_id="mcap"), ConverterOptions("", ""))

cmd, odo = [], []
while r.has_next():
    topic, data, ts = r.read_next()
    if topic == "/control/command/control_cmd":
        m = deserialize_message(data, AckermannControlCommand)
        cmd.append((ts / 1e9, m.lateral.steering_tire_angle, m.longitudinal.speed))
    elif topic == "/localization/kinematic_state":
        m = deserialize_message(data, Odometry)
        odo.append((ts / 1e9, m.pose.pose.position.x, m.pose.pose.position.y,
                    m.twist.twist.linear.x))

if not cmd:
    print("no control_cmd in bag")
    raise SystemExit(1)

C = np.array(cmd)
t, d = C[:, 0], C[:, 1]
dt = np.diff(t)
rate = np.abs(np.diff(d)) / np.maximum(dt, 1e-6)
ok = dt > 1e-4
rate, dt = rate[ok], dt[ok]

O = np.array(odo)
v_at = np.interp(t[1:][ok], O[:, 0], O[:, 3]) if len(odo) else np.full(len(rate), np.nan)

GAIN = 1.639
CFG = 0.35
print(f"제어 명령 {len(C)}개, {t[-1]-t[0]:.0f}초, 평균 {1/np.median(dt):.1f} Hz")
print(f"조향각 범위 {d.min():+.3f} ~ {d.max():+.3f} rad  (한계 ±{np.arctan(0.668*1.087):.3f})")
print()
print("실측 조향 변화율 |ddelta/dt| (rad/s)")
for q in (50, 90, 95, 99, 99.9, 100):
    print(f"  p{q:<5} {np.percentile(rate, q):6.3f}")

# what the spatial-vs-temporal mismatch predicts as the ceiling, per speed
print()
print("속도별: 실측 상위 변화율 vs 두 가지 이론 한계")
print("  (A) 현재 코드 = rate_max/gain * Ts / (delta_s/v)  ← 속도에 비례해 더 빡세짐")
print("  (B) 의도한 값 = rate_max/gain")
edges = [0, 4, 6, 8, 10, 12, 14, 99]
for lo, hi in zip(edges, edges[1:]):
    m = (v_at >= lo) & (v_at < hi)
    if m.sum() < 20:
        continue
    # the code allows max_delta_change per 0.6 m step; convert to rad/s at this speed
    vmid = np.nanmedian(v_at[m])
    allowed_A = (CFG / GAIN) * 0.025 / (0.6 / max(vmid, 0.5))
    print(f"  {lo:2d}-{hi:2d} m/s  n={m.sum():5d}  p99 {np.percentile(rate[m], 99):6.3f}  "
          f"최대 {rate[m].max():6.3f}   (A) {allowed_A:.3f}   (B) {CFG/GAIN:.3f}")
