"""Would the SHIPPED lidar_guard defaults have cancelled the AI's pace? Answer offline.

Every AI number I have was measured with GUARD_FRONT_LIMIT=0 and GUARD_SIDE_LIMIT=0, i.e. with the guard
inert. The tarball a grader runs ships front_limit 1.25, side_limit 0.72, guard_speed 5.5, and when the
guard fires it does three things: caps speed at 5.5, caps ACCELERATION at 0.45, and replaces 45% of the
steering with a fixed escape bias. The acceleration cap is the dangerous one, because the whole 42.75 s
result comes from commanding up to +1.35.

Rather than burn a simulator run to find out, this replays the recorded scans and the recorded steering
through the guard's exact predicate:

    front       = min range over [-0.44, +0.44] rad
    left/right  = min range over [+0.44, +1.40] / [-1.40, -0.44]
    toward_side = right if steer < 0 else left
    imminent    = front < front_limit or toward_side < side_limit

and reports the fraction of frames that would have been throttled, plus the fraction that would also have
had their steering overridden (front < 0.90). Sectors are computed from the scan's own angle_min and
angle_increment, not assumed.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/kim/aichallenge2026/aichallenge/ml_workspace/tiny_lidar_net")
from rosbags.highlevel import AnyReader  # noqa: E402
from extract_data_from_bag import build_typestore  # noqa: E402

FRONT_LIMIT, SIDE_LIMIT, GUARD_SPEED = 1.25, 0.72, 5.5


def sector_min(rng, ang, lo, hi, rmax):
    m = (ang >= lo) & (ang <= hi) & np.isfinite(rng) & (rng > 0.05)
    if not m.any():
        return rmax
    return float(np.minimum(rng[m], rmax).min())


for bag in sys.argv[1:]:
    b = pathlib.Path(bag)
    scans, angmin, anginc, rmax = [], None, None, None
    cmd_t, cmd_steer, cmd_accel = [], [], []
    scan_t = []
    with AnyReader([b], default_typestore=build_typestore()) as r:
        want = {"/sensing/lidar/scan", "/control/command/control_cmd"}
        for c, t, raw in r.messages(connections=[c for c in r.connections if c.topic in want]):
            m = r.deserialize(raw, c.msgtype)
            if c.topic.endswith("scan"):
                scans.append(np.asarray(m.ranges, dtype=float))
                scan_t.append(t)
                if angmin is None:
                    angmin, anginc, rmax = m.angle_min, m.angle_increment, m.range_max
            else:
                cmd_t.append(t)
                cmd_steer.append(m.lateral.steering_tire_angle)
                cmd_accel.append(m.longitudinal.acceleration)
    if not scans:
        print(f"{b.parts[-3]}: no scans")
        continue
    ang = angmin + np.arange(len(scans[0])) * anginc
    cmd_t = np.asarray(cmd_t)
    cmd_steer = np.asarray(cmd_steer)
    cmd_accel = np.asarray(cmd_accel)

    imminent = np.zeros(len(scans), dtype=bool)
    steer_override = np.zeros(len(scans), dtype=bool)
    accel_cut = []
    for i, rng in enumerate(scans):
        f = sector_min(rng, ang, -0.44, 0.44, rmax)
        l = sector_min(rng, ang, 0.44, 1.40, rmax)
        rt = sector_min(rng, ang, -1.40, -0.44, rmax)
        j = min(max(np.searchsorted(cmd_t, scan_t[i]) - 1, 0), len(cmd_t) - 1)
        st = cmd_steer[j]
        toward = rt if st < 0.0 else l
        imminent[i] = (f < FRONT_LIMIT) or (toward < SIDE_LIMIT)
        steer_override[i] = f < 0.90
        if imminent[i]:
            accel_cut.append(cmd_accel[j])

    n = len(scans)
    print(f"\n{b.parts[-3]}: {n} scan frames")
    print(f"  would be THROTTLED (speed<=5.5, accel<=0.45): {imminent.mean():.1%}")
    print(f"  would have STEERING overridden (front<0.90):  {steer_override.mean():.1%}")
    if accel_cut:
        a = np.asarray(accel_cut)
        print(f"  commanded accel on those frames: mean {a.mean():+.3f} max {a.max():+.3f}; "
              f"{np.mean(a > 0.45):.1%} of them exceed the 0.45 cap")
