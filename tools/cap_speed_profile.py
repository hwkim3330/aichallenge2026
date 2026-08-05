#!/usr/bin/env python3
"""Cap the continuous speed-profile CSV over a waypoint range, without introducing a step.

Why here and why now. The promoted profile's single remaining failure is located by two independent
measurements from the same official evaluation run: the wall penalty fired 17.05 s into lap 5, which
at clean-lap pace is about waypoint 134, and that lap's peak |e_y| was 2.819 m at wp 133 -- the
largest of the run, on the one penalised lap. 2.819 m is past the roughly 1.94 m at which the corridor
constraint goes infeasible, and the penalty confirms contact.

And the profile is fastest on the whole lap exactly there:

    wp        124   127   130   133   136
    ramps    33.0  32.0  31.0  29.8  28.6 km/h
    shipped    30    30    30    30    30

So the promoted profile is 3 km/h quicker than the shipped one at the place it now fails. Capping that
back costs about 0.3 s a lap against a 6.43 s penalty plus roughly 11 s of ruined lap.

The cap must not become a step. The step is the mechanism this whole line of work identified: the
shipped profile jumps 22 -> 30 km/h at wp 290, the MPC horizon sees it from wp 282 and begins
accelerating while |kappa| is still 0.17, and the excursions cluster immediately before it. Reproducing
that error here would trade one hotspot for another, so the cap is followed by wrapped braking and
acceleration passes on the CSV's own arc-length grid, which leaves the profile continuous.
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

SRC = pathlib.Path("/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="env/final_ver3/traj_blend45_s110L_ramps.csv")
    ap.add_argument("--out", required=True, help="path under the package, e.g. env/final_ver3/x.csv")
    ap.add_argument("--range", required=True, help="waypoint range to cap, e.g. 105-140")
    ap.add_argument("--kmh", type=float, required=True, help="ceiling over that range")
    ap.add_argument("--a-max", type=float, default=0.6, help="m/s^2, the profile's own value")
    ap.add_argument("--a-min", type=float, default=1.4, help="m/s^2 braking, the profile's own value")
    args = ap.parse_args()

    pkg = pathlib.Path(get_package_share_directory("multi_purpose_mpc_ros"))
    cfg = convert_to_namedtuple(yaml.safe_load((pkg / "config/config.yaml").open()))
    m = Map(str(pkg / cfg.map.yaml_path))
    wx, wy, _, _ = load_ref_path(str(pkg / cfg.reference_path.csv_path))
    rp = ReferencePath(m, wx, wy, cfg.reference_path.resolution,
                       cfg.reference_path.smoothing_distance,
                       cfg.reference_path.max_width, cfg.reference_path.circular)
    WP = np.array([[w.x, w.y] for w in rp.waypoints])

    raw = (pkg / args.src).read_text().splitlines()
    header, rows = raw[0], [r for r in raw[1:] if r.strip()]
    cols = [c.strip() for c in header.split(",")]
    data = np.array([[float(x) for x in r.split(",")] for r in rows])
    ix, iy = cols.index("x_m"), cols.index("y_m")
    iv, isx = cols.index("vx_mps"), cols.index("s_m")
    P, v, s = data[:, [ix, iy]], data[:, iv].copy(), data[:, isx]
    n = len(P)

    lo, hi = (int(x) for x in args.range.split("-"))
    # Waypoints and trajectory points are different grids with different lengths, so map by nearest
    # neighbour rather than by index.
    targets = {int(np.argmin(np.hypot(*(P - WP[i]).T))) for i in range(lo, hi + 1)}
    idx = np.array(sorted(targets))

    v_before = v.copy()
    ceiling = args.kmh / 3.6
    v[idx] = np.minimum(v[idx], ceiling)
    capped_n = int((v_before[idx] > ceiling).sum())

    # Wrapped braking then acceleration passes on the trajectory's own spacing, so the cap is
    # reachable and the result stays continuous -- no step for the MPC horizon to accelerate into.
    ds = np.array([np.hypot(*(P[(i + 1) % n] - P[i])) for i in range(n)])
    for _ in range(3):
        for i in range(n - 1, -1, -1):
            v[i] = min(v[i], math.sqrt(v[(i + 1) % n] ** 2 + 2 * args.a_min * ds[i]))
        for i in range(n):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * args.a_max * ds[i - 1]))

    changed = np.nonzero(np.abs(v - v_before) > 1e-9)[0]
    jump = np.abs(np.diff(np.concatenate([v, v[:1]]))).max()
    jump_before = np.abs(np.diff(np.concatenate([v_before, v_before[:1]]))).max()

    def lap(vv):
        return sum(ds[i] / max((vv[i] + vv[(i + 1) % n]) / 2, 0.1) for i in range(n))

    print(f"trajectory points {n}, waypoints {len(WP)}")
    print(f"waypoints {lo}-{hi} map to {len(idx)} trajectory points; "
          f"{capped_n} were above {args.kmh} km/h")
    print(f"changed after the smoothing passes: {len(changed)} of {n}")
    print(f"largest point-to-point speed jump: {jump_before*3.6:.2f} -> {jump*3.6:.2f} km/h "
          f"(must not grow -- a step is the failure mode this avoids)")
    print(f"predicted lap {lap(v_before):.2f} -> {lap(v):.2f} s ({lap(v) - lap(v_before):+.2f} s)")
    if len(changed):
        a, b = changed.min(), changed.max()
        print(f"  affected arc length s_m {s[a]:.1f}-{s[b]:.1f} of {s[-1]:.1f}")
        print("  before " + " ".join(f"{v_before[i]*3.6:.1f}" for i in changed[::max(1, len(changed)//14)]))
        print("  after  " + " ".join(f"{v[i]*3.6:.1f}" for i in changed[::max(1, len(changed)//14)]))

    data[:, iv] = v
    out = SRC / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n" + "\n".join(
        ", ".join(f"{x:.7f}" for x in row) for row in data) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
