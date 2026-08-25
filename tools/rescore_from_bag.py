#!/usr/bin/env python3
"""Score a solo run from its bag, because the log-based scorer miscounts here.

`race_outcome` infers the unlogged sixth lap from the gap between the last "Lap N
completed" line and the orchestrator's finalize, accepting it when the gap is 0.6-1.4
typical laps. That window was calibrated on 2026-08-02 runs whose finalize took 46.0 s.
With ROSBAG=1 the finalize also closes an mcap and writes motion analytics, and the gap
becomes 68-79 s -- outside the window -- so every run in the shift A/B was scored as five
laps and "did not finish", and the whole comparison read as truncated. It was not: the
cars finished. The metric was wrong, which is the same trap as the "0 laps" reading in
2026-08-07.

So laps are counted here from the geometry instead: the nearest reference waypoint index
wraps once per lap, and /localization/kinematic_state is in the bag at ~50 Hz. Lap times
measured this way are not AWSIM's official ones -- they start at the first wrap rather
than the green light -- but every arm is measured identically, which is what an A/B needs.

Usage: tools/rescore_from_bag.py output/shifta-*/
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import numpy as np

ENV = pathlib.Path(__file__).resolve().parent.parent / (
    "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/final_ver3")
WP_LO, WP_HI = 128, 142


def read(bag: pathlib.Path):
    from rosbags.highlevel import AnyReader
    with AnyReader([bag]) as r:
        conns = [c for c in r.connections if c.topic == "/localization/kinematic_state"]
        rows = []
        for con, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, con.msgtype)
            p = m.pose.pose.position
            v = m.twist.twist.linear
            rows.append((t * 1e-9, p.x, p.y, float(np.hypot(v.x, v.y))))
    return np.array(rows) if rows else None


def score(bag: pathlib.Path) -> dict | None:
    P = read(bag)
    if P is None or len(P) < 500:
        return None
    traj = np.loadtxt(ENV / "traj_top36_blend45.csv", delimiter=",", skiprows=1,
                      usecols=(1, 2))
    d = np.hypot(P[:, 1][:, None] - traj[None, :, 0], P[:, 2][:, None] - traj[None, :, 1])
    wp = d.argmin(1)
    tan = np.roll(traj, -1, 0) - traj
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    off = P[:, 1:3] - traj[wp]
    lat = d.min(1) * np.sign(tan[wp, 0] * off[:, 1] - tan[wp, 1] * off[:, 0])

    wraps = np.nonzero(np.diff(wp.astype(int)) < -200)[0] + 1
    lap_id = np.zeros(len(P), dtype=int)
    for w in wraps:
        lap_id[w:] += 1
    # The car sits on the grid before the first wrap; that partial segment is not a lap.
    lap_t = [P[w, 0] for w in wraps]
    laps = [lap_t[i + 1] - lap_t[i] for i in range(len(lap_t) - 1)]

    peaks = []
    for L in range(lap_id.max() + 1):
        sel = (lap_id == L) & (wp >= WP_LO) & (wp <= WP_HI)
        if sel.sum() >= 3:
            peaks.append(float(lat[sel].max()))

    # A wall contact shows as a deceleration no brake can produce (limit 1.6 m/s^2).
    # Counted as EVENTS, not samples: at ~50 Hz a single impact trips the test for about
    # a second, which is why the first version of this reported 50 contacts a race. And
    # only inside the racing window -- the car also decelerates hard when the race ends.
    step = max(1, int(round(1.0 / max(np.median(np.diff(P[:, 0])), 1e-6))))
    dv = np.zeros(len(P))
    dv[step:] = P[step:, 3] - P[:-step, 3]
    win = np.zeros(len(P), dtype=bool)
    if len(wraps) >= 2:
        win[wraps[0]:wraps[-1]] = True
    bad = (dv < -5.0) & win
    hits = int(np.sum(bad[1:] & ~bad[:-1])) + int(bad[0])
    return dict(laps=laps, n_laps=len(laps), total=sum(laps) if laps else None,
                peaks=peaks, worst_decel=float(dv[win].min()) if win.any() else 0.0,
                hits=hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        p = pathlib.Path(d)
        bag = p / "d1" / "rosbag2_autoware"
        if not bag.exists():
            print(f"{p.name}: no bag, skipped")
            continue
        s = score(bag)
        if s is None:
            print(f"{p.name}: bag too short, skipped")
            continue
        s["tag"] = p.name.rstrip("/")
        rows.append(s)
        laps = " ".join(f"{x:5.2f}" for x in s["laps"])
        tot = f"{s['total']:7.2f}" if s["total"] else "      -"
        print(f"{s['tag']:24s} laps {s['n_laps']}  total {tot}  [{laps}]  "
              f"peak {max(s['peaks']):5.2f}  contacts {s['hits']}")

    arms: dict[str, list] = {}
    for r in rows:
        arm = r["tag"].split("-")[1]
        arms.setdefault(arm, []).append(r)
    print("\narm        runs  laps 2-6 total  peak offset (median / max)  contacts")
    for arm, sel in arms.items():
        tot = [r["total"] for r in sel if r["total"]]
        allp = [p for r in sel for p in r["peaks"]]
        print(f"{arm:10s} {len(sel):3d}  "
              f"{statistics.mean(tot) if tot else float('nan'):10.2f}  "
              f"{statistics.median(allp):16.2f} / {max(allp):.2f}  "
              f"{sum(r['hits'] for r in sel):8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
