#!/usr/bin/env python3
"""Lap progress and stall statistics from a bag's velocity_status alone.

Written because the per-run result-summary.json does not survive: AWSIM writes it
to aichallenge/result-summary.json and the next run overwrites it, so the
2026-07-31 npc-with / npc-without races left no scores behind -- only bags.

Everything here comes from /vehicle/status/velocity_status on purpose.
/localization/kinematic_state is the obvious source for lap counting but it
teleports by hundreds of metres (see tools/GOAL.md), so distance is integrated
from forward speed instead. Reverse is clipped away: backing out of a wedge covers
ground but does not advance the lap.

Usage: tools/analyse_stall_from_bag.py output/npc-*/bag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl

MSGDEFS = Path(__file__).resolve().parent.parent / 'aichallenge/ml_workspace/tiny_lidar_net/msgdefs'
TRAPZ = getattr(np, 'trapezoid', None) or np.trapz


def typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    extra = {}
    for idl in sorted(MSGDEFS.rglob('*.idl')):
        extra.update(get_types_from_idl(idl.read_text()))
    if extra:
        ts.register(extra)
    return ts


def analyse(bag: Path, ts, lap_m: float, window_s: float, stall_below: float):
    with AnyReader([bag], default_typestore=ts) as reader:
        cons = [c for c in reader.connections if c.topic == '/vehicle/status/velocity_status']
        if not cons:
            return None
        rows = [(t, reader.deserialize(raw, c.msgtype).longitudinal_velocity)
                for c, t, raw in reader.messages(connections=cons)]
    if not rows:
        return None
    t = np.array([r[0] for r in rows], dtype=np.float64) / 1e9
    v = np.array([r[1] for r in rows])
    order = np.argsort(t)
    t, v = t[order], v[order]
    fwd = np.clip(v, 0.0, None)
    inside = (t - t[0]) <= window_s
    hz = len(t) / (t[-1] - t[0])

    stalled = v < stall_below
    runs, cur = [], 0
    for x in stalled:
        if x:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)

    return {
        'duration': t[-1] - t[0],
        'laps': TRAPZ(fwd[inside], t[inside]) / lap_m,
        'distance': TRAPZ(fwd, t),
        'stopped_pct': stalled.mean() * 100.0,
        'stall_events': len(runs),
        'longest_stall_s': max(runs) / hz if runs else 0.0,
        'mean_v': v.mean(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', type=Path)
    ap.add_argument('--lap-metres', type=float, default=336.1,
                    help='racing line length; default is traj_top36_blend45.csv')
    ap.add_argument('--window', type=float, default=480.0,
                    help='scored window in seconds; AWSIM keeps driving past it')
    ap.add_argument('--stall-below', type=float, default=0.5, help='m/s')
    args = ap.parse_args()

    ts = typestore()
    print(f"{'bag':26s} {'dur':>6s} {'laps@w':>7s} {'dist(m)':>8s} "
          f"{'stop%':>6s} {'stalls':>7s} {'longest':>8s} {'mean v':>7s}")
    ok = True
    for bag in args.bags:
        st = analyse(bag, ts, args.lap_metres, args.window, args.stall_below)
        label = '/'.join(bag.parts[-2:])
        if st is None:
            print(f"{label:26s} no velocity_status")
            ok = False
            continue
        print(f"{label:26s} {st['duration']:6.0f} {st['laps']:7.2f} {st['distance']:8.0f} "
              f"{st['stopped_pct']:6.1f} {st['stall_events']:7d} {st['longest_stall_s']:7.1f}s "
              f"{st['mean_v']:7.2f}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
