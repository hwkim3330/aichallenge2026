#!/usr/bin/env python3
"""Predict lap time from a path CSV and a speed policy, without the simulator.

Why this exists
---------------
Every configuration idea so far has cost a full `create_submit_file` ->
`docker_build eval` -> `make eval` cycle to evaluate, and most of them lose. This
reproduces the parts of the controller that actually decide lap time, so a
candidate can be scored in a second and only the survivors go to the simulator.

What it reproduces, and what it deliberately does not
-----------------------------------------------------
Faithful to `multi_purpose_mpc_ros`:

  * `ReferencePath._construct_path` -- circular padding, per-segment resampling at
    `resolution`, then the moving-average smoothing with its window of
    `2 * smoothing_distance + 1` and the trimmed ends.
  * `ReferencePath._construct_waypoints` -- psi from the forward difference,
    `kappa = angle_dif / (dist_ahead + eps)`, and the last coordinate dropped.
  * `ReferenceVelocityConfigulator.get_ref_vel` -- section lookup by waypoint id
    with wrap-around, which matters because the sections are indices into the
    RESAMPLED path, not into the CSV.

Structural point worth stating, because it changes what is worth tuning: when the
ref_vel configurator is active the controller sets

    v_ref = [section_speed] * len(waypoints)

so the curvature-based `compute_speed_profile` output is overwritten on every
control cycle and `ay_max` stops having any effect. Section speeds are the knob.

Not modelled: the MPC itself, tyre grip, and therefore whether a section speed is
actually survivable. This gives an OPTIMISTIC bound -- a lower bound on lap time
for a given speed policy. Its value is comparative: if a candidate cannot beat
the baseline even here, it cannot beat it in the simulator either.

Usage
-----
    python3 tools/lap_time_model.py                       # baseline, official path
    python3 tools/lap_time_model.py --speed-csv <csv>      # telemetry speed policy
    python3 tools/lap_time_model.py --scale 1.15           # all sections +15 %
"""
import argparse
import math
import pathlib

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
PKG = REPO / "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros"
EPS = 1e-12


def load_xy(csv_path):
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    return list(rows["x_m"]), list(rows["y_m"])


def construct_path(wp_x, wp_y, resolution=0.6, smoothing_distance=2, circular=True):
    """Mirror of ReferencePath._construct_path."""
    if circular:
        wp_x = wp_x + wp_x[:smoothing_distance * 3]
        wp_y = wp_y + wp_y[:smoothing_distance * 3]

    n_wp = [max(1, int(math.hypot(wp_x[i + 1] - wp_x[i], wp_y[i + 1] - wp_y[i])
                       / resolution))
            for i in range(len(wp_x) - 1)]

    gp_x, gp_y = wp_x[-1], wp_y[-1]
    xs = [v for i in range(len(wp_x) - 1)
          for v in np.linspace(wp_x[i], wp_x[i + 1], n_wp[i], endpoint=False)] + [gp_x]
    ys = [v for i in range(len(wp_y) - 1)
          for v in np.linspace(wp_y[i], wp_y[i + 1], n_wp[i], endpoint=False)] + [gp_y]

    sd = smoothing_distance
    sx = [float(np.mean(xs[i - sd:i + sd + 1])) for i in range(sd, len(xs) - sd)]
    sy = [float(np.mean(ys[i - sd:i + sd + 1])) for i in range(sd, len(ys) - sd)]
    return sx, sy


def construct_waypoints(sx, sy):
    """Mirror of ReferencePath._construct_waypoints: psi, kappa, segment length."""
    out = []
    for i in range(len(sx) - 1):
        dx, dy = sx[i + 1] - sx[i], sy[i + 1] - sy[i]
        psi = math.atan2(dy, dx)
        dist = math.hypot(dx, dy)
        if i == 0:
            kappa = 0.0
        else:
            pdx, pdy = sx[i] - sx[i - 1], sy[i] - sy[i - 1]
            behind = math.atan2(pdy, pdx)
            d = (psi - behind + math.pi) % (2 * math.pi) - math.pi
            kappa = d / (dist + EPS)
        out.append((sx[i], sy[i], psi, kappa, dist))
    return out


def section_speeds(n_wp, ref_vel_yaml, scale=1.0):
    """Mirror of ReferenceVelocityConfigulator.get_ref_vel, vectorised over ids."""
    cfg = yaml.safe_load(open(ref_vel_yaml))["ref_vel_configulator"]
    pairs = sorted((int(v["wp_id"]), float(v["ref_vel"])) for v in cfg.values())
    keys = [p[0] for p in pairs]
    speed_of = dict(pairs)

    v = np.zeros(n_wp)
    for i in range(n_wp):
        for k in range(len(keys)):
            start = keys[k]
            end = keys[(k + 1) % len(keys)]
            if start <= end:
                if start <= i < end:
                    v[i] = speed_of[start]
                    break
            else:
                if i >= start or i < end:
                    v[i] = speed_of[start]
                    break
        else:
            v[i] = speed_of[keys[-1]]
    return v * scale / 3.6          # km/h -> m/s


def speeds_from_csv(csv_path, waypoints):
    """Mirror of the CSV branch: spatial nearest telemetry sample per waypoint."""
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    tx, ty, tv = rows["x_m"], rows["y_m"], rows["vx_mps"]
    out = np.empty(len(waypoints))
    for i, (x, y, *_rest) in enumerate(waypoints):
        j = int(np.argmin((tx - x) ** 2 + (ty - y) ** 2))
        out[i] = tv[j]
    return out


def limit_by_acceleration(v_target, seg_len, a_max, a_min, laps=3):
    """Forward/backward passes so the profile is reachable.

    Circular, so it is iterated: a braking zone can propagate back past the start
    line, and one pass would silently leave that violated.
    """
    v = v_target.astype(float).copy()
    n = len(v)
    for _ in range(laps):
        for i in range(n):                       # forward: acceleration limit
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(max(v[i] ** 2 + 2 * a_max * seg_len[i], 0.0)))
        for i in range(n - 1, -1, -1):           # backward: braking limit
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(max(v[j] ** 2 + 2 * abs(a_min) * seg_len[i], 0.0)))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-csv", default=str(PKG / "env/final_ver3/traj_mincurv.csv"))
    ap.add_argument("--ref-vel", default=str(PKG / "config/ref_vel.yaml"))
    ap.add_argument("--speed-csv", default=None,
                    help="use a telemetry CSV's vx_mps instead of the ref_vel sections")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every section speed by this factor")
    ap.add_argument("--a-max", type=float, default=1.35)
    ap.add_argument("--a-min", type=float, default=-1.6)
    ap.add_argument("--resolution", type=float, default=0.6)
    ap.add_argument("--smoothing-distance", type=int, default=2)
    args = ap.parse_args()

    wp_x, wp_y = load_xy(args.path_csv)
    sx, sy = construct_path(wp_x, wp_y, args.resolution, args.smoothing_distance)
    wps = construct_waypoints(sx, sy)
    seg = np.array([w[4] for w in wps])
    kappa = np.array([w[3] for w in wps])

    print(f"path      : {pathlib.Path(args.path_csv).name}")
    print(f"waypoints : {len(wps)}   length {seg.sum():.1f} m   "
          f"|kappa| max {np.abs(kappa).max():.3f} 1/m")

    if args.speed_csv:
        v_t = speeds_from_csv(args.speed_csv, wps)
        label = f"telemetry {pathlib.Path(args.speed_csv).name}"
    else:
        v_t = section_speeds(len(wps), args.ref_vel, args.scale)
        label = f"ref_vel sections x{args.scale:g}"

    v = limit_by_acceleration(v_t, seg, args.a_max, args.a_min)
    t_target = float(np.sum(seg / np.maximum(v_t, 1e-3)))
    t_real = float(np.sum(seg / np.maximum(v, 1e-3)))

    print(f"speed     : {label}")
    print(f"            target  {v_t.min()*3.6:5.1f}..{v_t.max()*3.6:5.1f} km/h  "
          f"-> lap {t_target:6.2f} s  (ignoring a_max)")
    print(f"            limited {v.min()*3.6:5.1f}..{v.max()*3.6:5.1f} km/h  "
          f"-> lap {t_real:6.2f} s  (a_max {args.a_max}, a_min {args.a_min})")
    print(f"            mean {seg.sum()/t_real*3.6:.1f} km/h")
    print(f"a_max cost: {t_real - t_target:+.2f} s")


if __name__ == "__main__":
    main()
