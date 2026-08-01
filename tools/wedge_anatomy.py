"""What is in front of the car during its long stalls?

Distinguishes two failure modes with completely different fixes:
  - queued behind the NPC: front sector blocked at a few metres, sides clear
  - wedged off-line on a wall: something very close on a side, front may be clear
"""
import sys
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl

ts = get_typestore(Stores.ROS2_HUMBLE); extra = {}
for idl in sorted(Path('aichallenge/ml_workspace/tiny_lidar_net/msgdefs').rglob('*.idl')):
    extra.update(get_types_from_idl(idl.read_text()))
ts.register(extra)

def sector_min(rng, amin, ainc, lo, hi):
    n = len(rng)
    ang = amin + np.arange(n) * ainc
    m = (ang >= lo) & (ang <= hi)
    v = rng[m]
    v = v[np.isfinite(v) & (v > 0.05)]
    return float(v.min()) if len(v) else np.inf

def run(bag, min_stall_s=20.0):
    with AnyReader([Path(bag)], default_typestore=ts) as r:
        vt, vv, st, sc, ct, cs = [], [], [], [], [], []
        for con, t, raw in r.messages():
            m = r.deserialize(raw, con.msgtype)
            if con.topic.endswith('velocity_status'):
                vt.append(t); vv.append(m.longitudinal_velocity)
            elif con.topic.endswith('lidar/scan'):
                a = np.asarray(m.ranges, dtype=np.float32)
                st.append(t)
                sc.append((sector_min(a, m.angle_min, m.angle_increment, -0.35, 0.35),
                           sector_min(a, m.angle_min, m.angle_increment, 0.35, 1.6),
                           sector_min(a, m.angle_min, m.angle_increment, -1.6, -0.35)))
            elif con.topic.endswith('control_cmd'):
                ct.append(t); cs.append((m.longitudinal.acceleration, m.lateral.steering_tire_angle))
    vt = np.array(vt, dtype=np.float64); vv = np.array(vv)
    o = np.argsort(vt); vt, vv = vt[o], vv[o]
    st = np.array(st, dtype=np.float64); sc = np.array(sc)
    o = np.argsort(st); st, sc = st[o], sc[o]
    ct = np.array(ct, dtype=np.float64); cs = np.array(cs)
    o = np.argsort(ct); ct, cs = ct[o], cs[o]

    hz = len(vt) / ((vt[-1] - vt[0]) / 1e9)
    stalled = vv < 0.5
    # contiguous stall runs
    runs, start = [], None
    for i, x in enumerate(stalled):
        if x and start is None: start = i
        elif not x and start is not None:
            runs.append((start, i)); start = None
    if start is not None: runs.append((start, len(stalled)))
    runs = [(a, b) for a, b in runs if (b - a) / hz >= min_stall_s]

    print(f"\n=== {bag}  ({len(runs)} stalls >= {min_stall_s:.0f}s)")
    print(f"{'t0(s)':>7} {'len':>6} {'front':>7} {'left':>7} {'right':>7} {'|steer|':>8} {'accel':>7}")
    for a, b in runs:
        t0, t1 = vt[a], vt[b - 1]
        sm = (st >= t0) & (st <= t1)
        cm = (ct >= t0) & (ct <= t1)
        f, l, rr = (np.median(sc[sm, i]) if sm.any() else np.nan for i in range(3))
        stq = np.median(np.abs(cs[cm, 1])) if cm.any() else np.nan
        ac = np.median(cs[cm, 0]) if cm.any() else np.nan
        print(f"{(t0-vt[0])/1e9:7.0f} {(b-a)/hz:5.0f}s {f:7.2f} {l:7.2f} {rr:7.2f} {stq:8.3f} {ac:7.2f}")

for bag in sys.argv[1:]:
    run(bag)
