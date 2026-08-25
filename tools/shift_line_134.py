#!/usr/bin/env python3
"""Write a racing line with a local lateral bump, to buy margin at the wp 131-139 wall.

Measured problem (2026-08-23, one 600 s eval race, all six laps in the bag). Through
wp 128-142 the car sits on the LEFT of the line every single lap -- +1.0 to +1.6 m -- and
the wall on that side is 3.7-3.9 m out. One lap in six wanders to +2.7 m, where the body
half-width plus yaw eats the last metre, and the car hits: 8.10 -> 0.01 m/s in one second,
which is -8.1 m/s^2 against a 1.6 m/s^2 brake. Result: a 6.5 s wall penalty and a 54.9 s
lap against a 44.2 s clean one. It has happened in all three races measured so far
(lap 5, lap 5, lap 4), always 17.0-17.3 s into the lap.

So the fix is not to stop a rare event; it is to move a distribution whose median already
sits two thirds of the way to the wall. Shifting the line to the RIGHT there costs nothing
that has been measured: the right-side clearance at the same waypoints is 2.26-2.56 m and
the car never uses it, and the shifted line's curvature is LOWER (0.118 -> 0.106 at 0.6 m)
because it cuts across a bend that the current line takes wide.

Sign convention is the left normal of the polyline, matching the bag analysis, so a
negative amplitude moves the line away from the wall the car actually hits.

Only x and y are perturbed; s, psi and kappa are recomputed from the new geometry so the
csv stays self-consistent. vx and ax are copied unchanged -- the speed the controller uses
comes from ref_vel.yaml, and section speed cuts are on the do-not-retest list anyway
(2026-08-03: they break recovery, twice observed).

Usage:
  tools/shift_line_134.py --amp -0.6 --lo 124 --hi 144 --out traj_top36_blend45_r06.csv
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ENV = pathlib.Path(__file__).resolve().parent.parent / (
    "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/final_ver3")


def recompute(x: np.ndarray, y: np.ndarray):
    n = len(x)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    seg = np.hypot(dx, dy)
    s = np.r_[0.0, np.cumsum(seg)[:-1]]
    # central differences on a closed loop
    tx = (np.roll(x, -1) - np.roll(x, 1))
    ty = (np.roll(y, -1) - np.roll(y, 1))
    psi = np.arctan2(ty, tx)
    # curvature from consecutive heading change over arc length
    dpsi = np.angle(np.exp(1j * (np.roll(psi, -1) - np.roll(psi, 1))))
    ds2 = seg + np.roll(seg, 1)
    kappa = dpsi / np.maximum(ds2, 1e-9)
    return s, psi, kappa, seg.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="traj_top36_blend45.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--amp", type=float, required=True,
                    help="metres along the LEFT normal; negative moves away from the wall "
                         "the car hits at wp 131-139")
    ap.add_argument("--lo", type=int, default=124)
    ap.add_argument("--hi", type=int, default=144)
    args = ap.parse_args()

    df = pd.read_csv(ENV / args.src)
    x = df["x_m"].to_numpy(float)
    y = df["y_m"].to_numpy(float)
    n = len(x)

    dx, dy = np.gradient(x), np.gradient(y)
    nrm = np.hypot(dx, dy)
    nx, ny = -dy / nrm, dx / nrm

    w = np.zeros(n)
    span = np.arange(args.lo, args.hi + 1) % n
    w[span] = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(span)) / max(len(span) - 1, 1)))

    x2 = x + args.amp * w * nx
    y2 = y + args.amp * w * ny

    s0, psi0, k0, len0 = recompute(x, y)
    s2, psi2, k2, len2 = recompute(x2, y2)

    out = df.copy()
    out["s_m"], out["x_m"], out["y_m"] = s2, x2, y2
    out["psi_rad"], out["kappa_radpm"] = psi2, k2
    dst = ENV / args.out
    out.to_csv(dst, index=False, float_format="%.7f")

    moved = np.nonzero(np.abs(w) > 1e-6)[0]
    print(f"{args.src} -> {args.out}")
    print(f"  bump {args.amp:+.2f} m over csv {args.lo}-{args.hi} "
          f"({len(moved)} of {n} points moved, peak {np.abs(args.amp):.2f} m)")
    print(f"  length {len0:.2f} -> {len2:.2f} m ({len2-len0:+.2f})")
    print(f"  |kappa| max over the window {np.abs(k0[args.lo:args.hi+1]).max():.4f} -> "
          f"{np.abs(k2[args.lo:args.hi+1]).max():.4f}")
    print(f"  |kappa| max whole lap        {np.abs(k0).max():.4f} -> {np.abs(k2).max():.4f}")
    print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
