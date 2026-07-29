#!/usr/bin/env python3
"""Push a racing line away from walls wherever it leaves the car too little room.

Why
---
traj_top36_fitted.csv is genuinely faster: on a clean lap it ran 45.43 s against
the baseline line's 46.96 s. But it is only barely drivable. Measured against the
occupancy grid inflated by the vehicle half-width (0.75 m), its tightest point has
0.76 m of clearance -- one centimetre of margin -- and over six laps the car wedged
43 times, ruining two of them (325.89 s and 66.31 s).

Clearance is consumed by more than the body outline: steering angle swings the
front corners out, yaw during a turn does the same, and the controller's own
tracking error moves the whole car. One centimetre does not cover any of that.

So this keeps the short line's shape and only moves the points that are too close
to something, along the direction that gains clearance fastest (the gradient of the
distance-to-obstacle field). Points with enough room are left exactly where they
are, which is what preserves the line's speed.

Method
------
  1. Build the clearance field: distance transform of the free space, in metres.
  2. For each waypoint below the target, step along the clearance gradient until it
     reaches the target or hits --max-shift.
  3. Smooth the resulting shifts along the path so the line stays continuous --
     a per-point correction would add curvature exactly where the track is
     narrowest, which is the last place that helps.
  4. Report the resulting length, curvature and clearance so the trade is visible.

Usage
-----
    python3 tools/widen_line.py <in.csv> <out.csv> [--target 1.2] [--max-shift 1.0]
"""
import argparse
import math
import pathlib

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

REPO = pathlib.Path(__file__).resolve().parent.parent
ENV = REPO / "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/final_ver3"


def load_grid():
    cfg = yaml.safe_load((ENV / "occupancy_grid_map.yaml").read_text())
    raw = (ENV / "occupancy_grid_map.pgm").read_bytes()
    parts, i = [], 0
    while len(parts) < 4:
        while raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b"#":
            while raw[i:i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while not raw[j:j + 1].isspace():
            j += 1
        parts.append(raw[i:j])
        i = j
    i += 1
    W, H, maxv = int(parts[1]), int(parts[2]), int(parts[3])
    img = np.frombuffer(raw[i:i + W * H], dtype=np.uint8).reshape(H, W)
    blocked = (1.0 - img / maxv) > cfg["occupied_thresh"]
    clear = distance_transform_edt(~blocked) * cfg["resolution"]
    return clear, cfg["resolution"], cfg["origin"][0], cfg["origin"][1], W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--target", type=float, default=1.2,
                    help="clearance to reach in metres (half-width 0.75 plus margin)")
    ap.add_argument("--max-shift", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=3.0,
                    help="gaussian sigma over path index for the shift field")
    args = ap.parse_args()

    clear, res, ox, oy, W, H = load_grid()
    rows = np.genfromtxt(args.src, delimiter=",", names=True)
    x, y = rows["x_m"].astype(float), rows["y_m"].astype(float)

    def sample(px, py):
        cx = np.clip(((px - ox) / res).astype(int), 0, W - 1)
        cy = np.clip(((py - oy) / res).astype(int), 0, H - 1)
        return clear[H - 1 - cy, cx]

    gy, gx = np.gradient(clear)          # rows (north), cols (east)

    def grad(px, py):
        cx = np.clip(((px - ox) / res).astype(int), 0, W - 1)
        cy = np.clip(((py - oy) / res).astype(int), 0, H - 1)
        r = H - 1 - cy
        # image row increases southward, so flip the north component
        return gx[r, cx], -gy[r, cx]

    before = sample(x, y)
    print(f"in : {len(x)} pts   clearance min {before.min():.2f} p5 {np.percentile(before,5):.2f} "
          f"mean {before.mean():.2f}")

    # how far each point wants to move
    need = np.maximum(args.target - before, 0.0)
    dx, dy = grad(x, y)
    n = np.hypot(dx, dy)
    n[n < 1e-9] = 1.0
    shift = np.minimum(need, args.max_shift)
    shift = gaussian_filter1d(shift, args.smooth, mode="wrap")

    nx, ny = x + dx / n * shift, y + dy / n * shift
    after = sample(nx, ny)

    def geom(px, py):
        d = np.hypot(np.diff(px, append=px[0]), np.diff(py, append=py[0]))
        psi = np.arctan2(np.diff(py, append=py[0]), np.diff(px, append=px[0]))
        dpsi = (np.diff(psi, append=psi[0]) + math.pi) % (2 * math.pi) - math.pi
        return d.sum(), np.abs(dpsi / np.maximum(d, 1e-6))

    L0, k0 = geom(x, y)
    L1, k1 = geom(nx, ny)
    moved = (shift > 0.01).sum()
    print(f"out: moved {moved}/{len(x)} pts (max {shift.max():.2f} m)")
    print(f"     clearance min {after.min():.2f} p5 {np.percentile(after,5):.2f} "
          f"mean {after.mean():.2f}")
    print(f"     length {L0:.1f} -> {L1:.1f} m   |kappa| p95 {np.percentile(k0,95):.3f} -> "
          f"{np.percentile(k1,95):.3f}   max {k0.max():.3f} -> {k1.max():.3f}")
    half = 0.75
    print(f"     below half-width: {(before<half).sum()} -> {(after<half).sum()}")
    print(f"     below target {args.target}: {(before<args.target).sum()} -> {(after<args.target).sum()}")

    # keep the source's column layout; psi/kappa are recomputed by the controller
    # from x/y anyway (create_ref_path discards them), so carry them across as-is.
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(nx), np.diff(ny)))])
    with open(args.dst, "w") as f:
        f.write("s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2\n")
        for i in range(len(nx)):
            f.write(f"{s[i]:.7f},{nx[i]:.7f},{ny[i]:.7f},"
                    f"{rows['psi_rad'][i]:.7f},{rows['kappa_radpm'][i]:.7f},"
                    f"{rows['vx_mps'][i]:.7f},{rows['ax_mps2'][i]:.7f}\n")
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
