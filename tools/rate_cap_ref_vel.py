#!/usr/bin/env python3
"""Lower the shipped ref_vel only where the racing line demands more steering RATE than the
actuator can deliver.

Why this and not another generated profile. Three regenerated profiles have already failed
(40 km/h wholesale: 16 stalls, 355.9 s; two make_ref_vel variants: 60.63 s laps against a predicted
40.90). Each replaced a validated profile wholesale. This one starts FROM the shipped profile and
only ever lowers it, at waypoints picked by measurement.

Which waypoints, and why they are not the ones I kept investigating. The corridor exits cluster at
wp 284-287 and 133-134, and I spent several rounds asking what is special about that geometry --
finding nothing, since wp 117 is sharper (|kappa| 0.228) and wp 201 narrower (4.258 m). The
measurement that explains it ranks waypoints by demanded steering RATE rather than steering angle:

    delta = atan(kappa * L)  ->  d(delta)/dt = v * (d kappa/ds) * L / (1 + (kappa*L)^2)

against the ceiling scaled_steer_rate_max = steer_rate_max / steering_tire_angle_gain_var
= 0.35 / 1.639 = 0.2135 rad/s. That ranking puts wp 275, 276, 278 OVER the ceiling and wp 280-281 at
0.8-0.9x -- a sustained stretch the car cannot track, ending exactly where the excursions peak. The
failure is at 284-287; the cause is six waypoints upstream. Same shape before 133-134.

So the excursion is not a cornering-grip problem and not a corridor-width problem. It is the line
asking for steering faster than the actuator turns, with lateral error accumulating downstream until
it reaches the corridor edge, where the QP goes infeasible as a consequence.

The fix follows from the same equation: cap v so the demanded rate stays under the ceiling.

    v_allow = ceiling * (1 + (kappa*L)^2) / (|d kappa/ds| * L)

A backward pass then makes the cap reachable under the braking limit, since arriving at a capped
waypoint requires having started to slow before it.
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

from multi_purpose_mpc_ros.common import convert_to_namedtuple
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.utils import load_ref_path

SRC = pathlib.Path(
    "/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/config")


def shipped_profile(pkg: pathlib.Path, n: int) -> np.ndarray:
    """Expand ref_vel.yaml's coarse sections to one value per waypoint, the way get_ref_vel reads
    it: a sorted lookup that holds each value until the next key."""
    sec = yaml.safe_load((pkg / "config/ref_vel.yaml").open())["ref_vel_configulator"]
    pts = sorted((v["wp_id"], float(v["ref_vel"])) for v in sec.values())
    out = np.empty(n)
    for i in range(n):
        kmh = pts[0][1]
        for wp0, sp in pts:
            if i >= wp0:
                kmh = sp
        out[i] = kmh
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wheelbase", type=float, default=1.087)
    ap.add_argument("--rate", type=float, default=0.35 / 1.639,
                    help="steering rate ceiling, rad/s")
    ap.add_argument("--margin", type=float, default=0.85,
                    help="fraction of the ceiling to aim for, leaving the MPC headroom to correct "
                         "rather than merely track")
    ap.add_argument("--v-floor", type=float, default=18.0,
                    help="km/h below which the cap is ignored; a curvature spike between two "
                         "waypoints can otherwise ask for a near-stop")
    ap.add_argument("--a-brake", type=float, default=1.6, help="m/s^2 for the reachability pass")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--windows",
                    help="restrict the cap to these waypoint ranges, e.g. 120-140,265-292. "
                         "The global cap costs 3.15 s/lap predicted across 139 of 351 waypoints, "
                         "which is a whole-lap slowdown rather than a fix for the two places the "
                         "excursions actually happen. Narrowing it first tests the mechanism for "
                         "a fraction of the lap time.")
    args = ap.parse_args()

    pkg = pathlib.Path(get_package_share_directory("multi_purpose_mpc_ros"))
    cfg = convert_to_namedtuple(yaml.safe_load((pkg / "config/config.yaml").open()))
    m = Map(str(pkg / cfg.map.yaml_path))
    wx, wy, _, _ = load_ref_path(str(pkg / cfg.reference_path.csv_path))
    rp = ReferencePath(m, wx, wy, cfg.reference_path.resolution,
                       cfg.reference_path.smoothing_distance,
                       cfg.reference_path.max_width, cfg.reference_path.circular)
    W = rp.waypoints
    n = len(W)
    L = args.wheelbase
    k = np.array([float(getattr(w, "kappa", 0.0)) for w in W])
    P = np.array([[w.x, w.y] for w in W])
    ds = np.array([float(np.hypot(*(P[(i + 1) % n] - P[i]))) for i in range(n)])
    dk = np.array([(k[(i + 1) % n] - k[i - 1]) / (ds[i - 1] + ds[i]) for i in range(n)])

    ship = shipped_profile(pkg, n)
    v = ship / 3.6

    ceiling = args.rate * args.margin
    denom = np.abs(dk) * L
    v_allow = np.where(denom > 1e-9,
                       ceiling * (1.0 + (k * L) ** 2) / np.maximum(denom, 1e-9), np.inf)
    v_allow = np.maximum(v_allow, args.v_floor / 3.6)

    if args.windows:
        keep = np.zeros(n, dtype=bool)
        for w in args.windows.split(","):
            lo, hi = (int(x) for x in w.split("-"))
            keep[lo:hi + 1] = True
        v_allow = np.where(keep, v_allow, np.inf)

    capped = np.minimum(v, v_allow)
    touched = np.nonzero(capped < v - 1e-9)[0]

    # Reachability: a capped waypoint only helps if braking can get there from upstream.
    # Two wrapped passes, as for a closed circuit.
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            capped[i] = min(capped[i],
                            math.sqrt(capped[(i + 1) % n] ** 2 + 2 * args.a_brake * ds[i]))

    kmh = np.round(capped * 3.6 / args.step) * args.step
    changed = np.nonzero(np.abs(kmh - ship) > 1e-9)[0]

    def lap(vv):
        return sum(ds[i] / max((vv[i] + vv[(i + 1) % n]) / 2, 0.1) for i in range(n))

    print(f"waypoints {n}  ceiling {args.rate:.4f} rad/s, aiming at {ceiling:.4f} "
          f"({args.margin:.2f}x)")
    print(f"directly over the cap: {len(touched)} waypoints -> "
          f"{', '.join(str(int(i)) for i in touched)}")
    print(f"changed after the braking pass: {len(changed)} of {n}")
    print(f"predicted lap  shipped {lap(ship / 3.6):.2f} s  ->  capped {lap(capped):.2f} s  "
          f"({lap(capped) - lap(ship / 3.6):+.2f} s)")
    for lo, hi in ((124, 138), (270, 292)):
        print(f"  wp {lo}-{hi}")
        print("    shipped " + " ".join(f"{ship[i]:4.0f}" for i in range(lo, hi)))
        print("    capped  " + " ".join(f"{kmh[i]:4.0f}" for i in range(lo, hi)))

    lines = ["# Generated by tools/rate_cap_ref_vel.py -- shipped ref_vel, lowered only where the",
             "# line demands more steering rate than the actuator delivers.",
             f"# ceiling {args.rate:.4f} rad/s x margin {args.margin}, wheelbase {L}, "
             f"floor {args.v_floor} km/h",
             f"# {len(touched)} waypoints over the cap, {len(changed)} changed after braking",
             f"# predicted lap {lap(capped):.2f} s vs shipped {lap(ship / 3.6):.2f} s",
             "ref_vel_configulator:"]
    prev = None
    kept = 0
    for i in range(n):
        if kmh[i] == prev:
            continue
        prev = kmh[i]
        kept += 1
        lines.append(f"  w{i:04d}:\n    ref_vel: {kmh[i]:.1f}\n    wp_id: {i}")
    (SRC / args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {SRC / args.out}  ({kept} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
