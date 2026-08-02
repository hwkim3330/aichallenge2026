#!/usr/bin/env python3
"""Turn a rival's driven trajectory into a reference line in our format.

Why bother
----------
The board publishes each submission's vehicle-1 rosbag, so the leaders' actual driven
lines are obtainable rather than inferred. Measured from KSK (rating 865, 1st overall,
6 laps, P1) against our submission:

                  best lap   line length   min speed     max speed
    KSK            43.81 s     363.0 m     26.3 km/h     35.4 km/h
    ours           45.85 s     337.8 m     22.0 km/h     30.0 km/h

They run 25 m FURTHER and are 2 s faster, and they never drop below 26.3 km/h. This
project spent its effort in the opposite direction: shortening the line (336.1 m) and
slowing the corners (two sections at 22 km/h). Their slowest stretch is the same corner
where our runs wedge, and they hold a steady ~26 km/h through it while we step 22 -> 30
across it.

A driven line also comes with a guarantee no synthesised line has: it was completed six
times under the scored conditions.

What this does
--------------
The raw bag samples are ~0.09 m apart, so per-sample heading is dominated by localisation
noise (consecutive psi values of -0.55 and +2.20 rad in the extract). Resample to uniform
spacing first, smooth, then differentiate. Curvature from a noisy path is meaningless and
curvature is exactly what the MPC consumes.

Then check the geometry the car actually needs: the path is the rear-axle centre, so at
curvature k the outer front corner sweeps wider by
    sqrt((1/k + half_width)^2 + axle_to_nose^2) - 1/k
which at k=0.236 and our dimensions is 0.89 m against the 0.65 m half-width.

Usage
-----
    python3 tools/port_rival_line.py aichallenge/rivalbags/ksk_bestlap.csv out.csv
"""
import argparse
import pathlib
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from widen_line import load_grid  # noqa: E402

# racing_kart_description/config/vehicle_info.param.yaml
HALF_WIDTH = (1.12 + 0.09 + 0.09) / 2      # 0.65 m
AXLE_TO_NOSE = 1.087 + 0.467               # 1.554 m
AXLE_TO_TAIL = 0.510


def swept_need(kappa):
    k = np.abs(kappa)
    R = np.where(k > 1e-4, 1.0 / np.maximum(k, 1e-4), 1e6)
    return np.maximum(np.hypot(R + HALF_WIDTH, AXLE_TO_NOSE),
                      np.hypot(R + HALF_WIDTH, AXLE_TO_TAIL)) - R


def resample_closed(x, y, step):
    """Uniform arc-length resampling of a closed loop."""
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])
    seg = np.hypot(np.diff(xc), np.diff(yc))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(round(s[-1] / step)))
    su = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.interp(su, s, xc), np.interp(su, s, yc), s[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--step", type=float, default=0.6,
                    help="match reference_path.resolution so the MPC sees what we check")
    ap.add_argument("--smooth", type=float, default=4.0,
                    help="gaussian sigma in samples, applied to x/y before differentiating")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="clearance to keep beyond what the car sweeps")
    a = ap.parse_args()

    raw = np.genfromtxt(a.src, delimiter=",", names=True)
    x, y, v = raw["x_m"], raw["y_m"], raw["vx_mps"]

    xr, yr, length = resample_closed(x, y, a.step)
    # smooth on the loop, then differentiate on the loop, so the seam is not a corner
    xs = gaussian_filter1d(xr, a.smooth, mode="wrap")
    ys = gaussian_filter1d(yr, a.smooth, mode="wrap")

    dx = np.gradient(xs); dy = np.gradient(ys)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    psi = np.arctan2(dy, dx)
    denom = (dx * dx + dy * dy) ** 1.5
    kappa = np.where(denom > 1e-9, (dx * ddy - dy * ddx) / np.maximum(denom, 1e-9), 0.0)

    seg = np.hypot(np.diff(np.append(xs, xs[0])), np.diff(np.append(ys, ys[0])))
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])])

    # speed carried over from the rival's own run, resampled onto the new stations
    sx = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    vr = np.interp(s, sx, v)

    clear, res, gx, gy, W, H = load_grid()
    cx = np.clip(((xs - gx) / res).astype(int), 0, W - 1)
    cy = np.clip(((ys - gy) / res).astype(int), 0, H - 1)
    c = clear[H - 1 - cy, cx]
    need = swept_need(kappa) + a.margin
    deficit = need - c

    print(f"원본 {len(x)}샘플 -> 재표본 {len(xs)}점 ({a.step} m 간격)")
    print(f"길이 {length:.1f} m")
    print(f"|kappa| 최대 {np.abs(kappa).max():.3f}  p99 {np.percentile(np.abs(kappa), 99):.3f}")
    print(f"여유 최소 {c.min():.2f} m  중앙 {np.median(c):.2f} m")
    print(f"스윕 요구 대비 부족: {int((deficit > 0).sum())}점 / {len(xs)}  "
          f"최악 {deficit.max():+.2f} m")
    print(f"속도 {vr.min():.2f} ~ {vr.max():.2f} m/s "
          f"({vr.min()*3.6:.1f} ~ {vr.max()*3.6:.1f} km/h)")
    ay = vr ** 2 * np.abs(kappa)
    print(f"이 속도와 이 곡률에서의 횡가속 최대 {ay.max():.2f} m/s^2  "
          f"p95 {np.percentile(ay, 95):.2f}   (차량 실측 상한 18.60)")

    worst = np.argsort(deficit)[-5:][::-1]
    if deficit.max() > 0:
        print("가장 부족한 지점:")
        for i in worst:
            if deficit[i] <= 0:
                continue
            print(f"  wp {i:4d}  s={s[i]:6.1f} m  ({xs[i]:.1f},{ys[i]:.1f})  "
                  f"kappa {kappa[i]:+.3f}  여유 {c[i]:.2f}  요구 {need[i]:.2f}")

    ax_prof = np.gradient(vr) * vr / np.maximum(np.gradient(s, edge_order=1), 1e-6)
    with open(a.dst, "w") as f:
        f.write("s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2\n")
        for i in range(len(xs)):
            f.write(f"{s[i]:.7f},{xs[i]:.7f},{ys[i]:.7f},{psi[i]:.7f},"
                    f"{kappa[i]:.7f},{vr[i]:.7f},{ax_prof[i]:.7f}\n")
    print(f"\n{a.dst} 작성 ({len(xs)}점)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
