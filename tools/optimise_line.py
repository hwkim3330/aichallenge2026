#!/usr/bin/env python3
"""Blend the fast line toward the official one per waypoint, not uniformly.

Why
---
traj_top36_blend45 is a flat 45 % mix of the 36.319 s run's fitted line and the
official mincurv line, and it works: 44.54 s against the 47.28 s baseline. But 45 %
everywhere is a compromise set by the tightest point on the lap. Most of the lap has
room to spare, so most of the lap is giving away length for no reason.

This chooses the mix per waypoint: as much of the short line as the clearance allows,
falling back toward the official line only where it does not.

Two constraints, both measured rather than assumed:

  clearance   the occupancy grid eroded by what the car actually sweeps. The path is
              the rear-axle centre, so the outer front corner runs wider: with
              half-width 0.65 m and rear-axle-to-nose 1.554 m, a 0.236 1/m corner
              needs 0.89 m rather than 0.65. Required margin on top of that is the
              --margin argument.

  curvature   <= --kappa-max. This is the constraint the first attempt at widening
              missed: pushing points outward along the clearance gradient bought
              margin but took peak curvature from 0.231 to 0.245, and the lateral
              demand went to 17.0 m/s^2 against the 15.41 the 36 s run actually held.
              x1.15 later confirmed the ceiling by failing at 20.93.

Method
------
  1. Per waypoint, find the smallest blend alpha whose swept clearance clears the
     margin. Small alpha means more of the short line.
  2. Smooth alpha along the path. A per-point choice puts curvature exactly where the
     track is narrowest, which is where it hurts.
  3. Raise alpha globally until the curvature ceiling holds. Report what it cost.

Usage
-----
    python3 tools/optimise_line.py <out.csv> [--margin 0.35] [--kappa-max 0.231]
"""
import argparse
import math
import pathlib
import sys

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lap_time_model import construct_path, construct_waypoints, load_xy  # noqa: E402
from widen_line import load_grid  # noqa: E402

ENV = (pathlib.Path(__file__).resolve().parent.parent /
       "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/final_ver3")

# racing_kart_description/config/vehicle_info.param.yaml
HALF_WIDTH = (1.12 + 0.09 + 0.09) / 2      # 0.65 m
AXLE_TO_NOSE = 1.087 + 0.467               # 1.554 m
AXLE_TO_TAIL = 0.510


def swept_need(kappa):
    """Outward room the rear-axle path needs at this curvature."""
    k = np.abs(kappa)
    R = np.where(k > 1e-4, 1.0 / np.maximum(k, 1e-4), 1e6)
    return np.maximum(np.hypot(R + HALF_WIDTH, AXLE_TO_NOSE),
                      np.hypot(R + HALF_WIDTH, AXLE_TO_TAIL)) - R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dst")
    ap.add_argument("--margin", type=float, default=0.35,
                    help="clearance to keep beyond what the car sweeps, in metres")
    ap.add_argument("--kappa-max", type=float, default=0.231)
    ap.add_argument("--smooth", type=float, default=6.0)
    args = ap.parse_args()

    clear, res, gx, gy, W, H = load_grid()

    def clearance(px, py):
        cx = np.clip(((np.asarray(px) - gx) / res).astype(int), 0, W - 1)
        cy = np.clip(((np.asarray(py) - gy) / res).astype(int), 0, H - 1)
        return clear[H - 1 - cy, cx]

    fx, fy = load_xy(ENV / "traj_top36_fitted.csv")
    ox, oy = load_xy(ENV / "traj_mincurv.csv")
    fx, fy, ox, oy = map(np.asarray, (fx, fy, ox, oy))
    rows = np.genfromtxt(ENV / "traj_top36_fitted.csv", delimiter=",", names=True)

    # official point paired with each fast-line point
    pair = np.array([int(np.argmin((ox - fx[i]) ** 2 + (oy - fy[i]) ** 2))
                     for i in range(len(fx))])
    tx, ty = ox[pair], oy[pair]

    def line(alpha):
        return fx + (tx - fx) * alpha, fy + (ty - fy) * alpha

    def measure(alpha):
        px, py = line(alpha)
        w = construct_waypoints(*construct_path(list(px), list(py)))
        seg = np.array([q[4] for q in w])
        kap = np.array([q[3] for q in w])
        wx = np.array([q[0] for q in w])
        wy = np.array([q[1] for q in w])
        c = clearance(wx, wy)
        deficit = swept_need(kap) + args.margin - c
        return seg.sum(), np.abs(kap).max(), deficit, len(w)

    # 1. smallest alpha per point that clears the margin
    grid = np.linspace(0.0, 1.0, 21)
    per = np.zeros(len(fx))
    for a in grid:
        px, py = line(np.full(len(fx), a))
        c = clearance(px, py)
        # curvature of the source line is a good enough proxy for this pass
        need = swept_need(rows["kappa_radpm"]) + args.margin
        per = np.where((per == 0) & (c >= need), a, per)
    per[per == 0] = 1.0
    print(f"per-point alpha: min {per.min():.2f} mean {per.mean():.2f} max {per.max():.2f}")

    # 2. smooth, so the correction does not add curvature where the track is tight
    alpha = gaussian_filter1d(per, args.smooth, mode="wrap")

    # 3. lift until the curvature ceiling holds
    lift = 0.0
    while lift <= 1.0:
        a = np.clip(alpha + lift, 0.0, 1.0)
        L, kmax, deficit, n = measure(a)
        bad = int((deficit > 0).sum())
        print(f"  lift {lift:.2f}: length {L:6.1f} m  |kappa|max {kmax:.3f}  "
              f"margin short at {bad:3d} wp  worst {deficit.max():+.2f} m")
        if kmax <= args.kappa_max and bad == 0:
            break
        lift += 0.05
    else:
        print("could not satisfy both constraints; fall back to the uniform blend")
        return 1

    px, py = line(np.clip(alpha + lift, 0.0, 1.0))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    with open(args.dst, "w") as f:
        f.write("s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2\n")
        for i in range(len(px)):
            f.write(f"{s[i]:.7f},{px[i]:.7f},{py[i]:.7f},"
                    f"{rows['psi_rad'][i]:.7f},{rows['kappa_radpm'][i]:.7f},"
                    f"{rows['vx_mps'][i]:.7f},{rows['ax_mps2'][i]:.7f}\n")

    # what the uniform blend gives, for comparison
    L0, k0, d0, _ = measure(np.full(len(fx), 0.45))
    L1, k1, d1, _ = measure(np.clip(alpha + lift, 0.0, 1.0))
    print(f"\nuniform 0.45 : {L0:6.1f} m  |kappa|max {k0:.3f}  short at "
          f"{int((d0>0).sum())} wp")
    print(f"per-waypoint : {L1:6.1f} m  |kappa|max {k1:.3f}  short at "
          f"{int((d1>0).sum())} wp")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
