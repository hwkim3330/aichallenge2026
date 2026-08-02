"""Time from each recovery trigger until the car is moving again.

Per-event rather than per-run: a 480 s run yields one lap-time number but three to
ten recovery events, which is the difference between n=2 and n=10 for the same
simulator time.
"""
import re, sys
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl

ts = get_typestore(Stores.ROS2_HUMBLE); ex = {}
for i in sorted(Path('aichallenge/ml_workspace/tiny_lidar_net/msgdefs').rglob('*.idl')):
    ex.update(get_types_from_idl(i.read_text()))
ts.register(ex)

RECOVERED_MPS = 2.0      # sustained speed that counts as back under way
SUSTAIN_S = 1.0
CAP_S = 200.0            # give up rather than report an unbounded number

def events(logpath):
    out = []
    pat = re.compile(r"\[(\d+\.\d+)\] \[stuck_recovery_controller\]: (stuck detected|yield deadlock)")
    for line in Path(logpath).read_text(errors='ignore').splitlines():
        m = pat.search(re.sub(r'\x1b\[[0-9;]*m', '', line))
        if m:
            out.append(float(m.group(1)))
    return out

def velocity(bagdir):
    with AnyReader([Path(bagdir)], default_typestore=ts) as r:
        cons = [c for c in r.connections if c.topic.endswith('velocity_status')]
        rows = [(t, r.deserialize(raw, c.msgtype).longitudinal_velocity)
                for c, t, raw in r.messages(connections=cons)]
    t = np.array([x[0] for x in rows], dtype=np.float64) / 1e9
    v = np.array([x[1] for x in rows])
    o = np.argsort(t)
    return t[o], v[o]

def costs(logpath, bagdir):
    ev = events(logpath)
    t, v = velocity(bagdir)
    hz = len(t) / (t[-1] - t[0])
    need = max(1, int(SUSTAIN_S * hz))
    out = []
    for e in ev:
        i = np.searchsorted(t, e)
        if i >= len(t):
            continue
        moving = v[i:] >= RECOVERED_MPS
        # first index where `need` consecutive samples are moving
        k = None
        run = 0
        for j, mv in enumerate(moving):
            run = run + 1 if mv else 0
            if run >= need:
                k = j - need + 1
                break
            if j / hz > CAP_S:
                break
        out.append(min(k / hz, CAP_S) if k is not None else CAP_S)
    return out

runs = [
    ("baseline OFF  solo_01", "output/teacher_solo_01/d1/autoware.log", "aichallenge/ml_workspace/rawdata/20260801-235202"),
    ("baseline OFF  solo_02", "output/teacher_solo_02/d1/autoware.log", "aichallenge/ml_workspace/rawdata/20260802-000837"),
    ("avoidance ON  recorded", "output/avoid_recorded/d1/autoware.log", "aichallenge/ml_workspace/avoidtest/20260802-102443"),
]
allc = {}
for name, log, bag in runs:
    try:
        c = costs(log, bag)
    except Exception as exc:
        print(f"{name:24s} FAILED {type(exc).__name__}: {exc}"); continue
    allc.setdefault(name.strip(), []).extend(c)
    print(f"{name:24s} n={len(c):2d}  " + (", ".join(f"{x:.1f}s" for x in c) if c else "(no events)"))
print()
for k, v in allc.items():
    if v:
        print(f"{k:16s} n={len(v):2d}  median={np.median(v):6.1f}s  mean={np.mean(v):6.1f}s  max={max(v):6.1f}s")
