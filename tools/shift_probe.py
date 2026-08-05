"""Would shifting the line away from the wall at wp 112-130 help, and what does it cost in curvature?

The clearance there is 1.36-1.55 m on the ub side against at least 3 m on the lb side, at the lap's
highest speed. Moving the line toward lb buys margin at no lap-time cost, which the 0.27 s/lap speed cap
does not. But a shifted line is a different line: it can raise curvature, and curvature is what sets
both the cornering demand and the MPC's own speed limit. So compute before building anything.

A smooth raised-cosine bump is used rather than a step, for the same reason the speed cap avoids one.
Nothing is written -- this only reports.
"""
import numpy as np, yaml
from ament_index_python.packages import get_package_share_directory
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.utils import load_ref_path
from multi_purpose_mpc_ros.common import convert_to_namedtuple

PKG = get_package_share_directory("multi_purpose_mpc_ros") + "/"
cfg = convert_to_namedtuple(yaml.safe_load(open(PKG + "config/config.yaml")))
m = Map(PKG + cfg.map.yaml_path)
wx, wy, _, _ = load_ref_path(PKG + cfg.reference_path.csv_path)
wx, wy = np.array(wx, float), np.array(wy, float)
N = len(wx)
L, LIMIT = 1.087, 0.35 / 1.639

def build(x, y):
    rp = ReferencePath(m, list(x), list(y), cfg.reference_path.resolution,
                       cfg.reference_path.smoothing_distance,
                       cfg.reference_path.max_width, cfg.reference_path.circular)
    W = rp.waypoints
    n = len(W)
    k = np.array([float(w.kappa) for w in W])
    lb = np.array([float(w.lb) for w in W]); ub = np.array([float(w.ub) for w in W])
    P = np.array([[w.x, w.y] for w in W])
    ds = np.array([np.hypot(*(P[(i+1) % n] - P[i])) for i in range(n)])
    return n, k, lb, ub, ds

n0, k0, lb0, ub0, ds0 = build(wx, wy)
# The csv index range matching waypoints 112-130: the two grids differ in length.
def wp_to_csv(i):
    return int(round(i * N / n0))
c_lo, c_hi = wp_to_csv(110), wp_to_csv(132)
print(f"csv points {N}, waypoints {n0}; wp 110-132 -> csv {c_lo}-{c_hi}")
print(f"baseline: min ub over wp 112-130 = {ub0[112:131].min():+.2f}, "
      f"max |kappa| = {np.abs(k0[112:131]).max():.4f}, length {ds0.sum():.1f} m")

# Lateral normal from the csv polyline itself.
dx = np.gradient(wx); dy = np.gradient(wy)
nrm = np.hypot(dx, dy); nx, ny = -dy / nrm, dx / nrm    # left normal

for amp in (0.4, 0.6, 0.8):
    for sign, label in ((-1.0, "toward lb"), (+1.0, "toward ub")):
        w = np.zeros(N)
        span = np.arange(c_lo, c_hi + 1)
        w[span] = 0.5 * (1 - np.cos(2 * np.pi * (span - c_lo) / max(len(span) - 1, 1)))
        sx, sy = wx + sign * amp * w * nx, wy + sign * amp * w * ny
        try:
            n1, k1, lb1, ub1, ds1 = build(sx, sy)
        except Exception as e:
            print(f"  {amp:.1f} m {label}: build failed ({type(e).__name__})"); continue
        a, b = 112 * n1 // n0, 131 * n1 // n0
        print(f"  {amp:.1f} m {label}: n={n1} min ub {ub1[a:b].min():+.2f} "
              f"(was {ub0[112:131].min():+.2f})  min width {(ub1-lb1)[a:b].min():.2f} "
              f"max |kappa| {np.abs(k1[a:b]).max():.4f} (was {np.abs(k0[112:131]).max():.4f})  "
              f"len {ds1.sum():.1f} m")
