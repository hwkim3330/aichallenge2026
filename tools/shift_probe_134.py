"""Would shifting the line away from the wall at wp 126-142 help, and what does it cost?

Measured cause of the once-per-race wall penalty (2026-08-23, three races). The car exits
the wp 116-123 corner and drifts to the ub side of the line, reaching +1.0 to +1.6 m by
wp 134 on EVERY lap. On the lap that hits the wall it reaches +2.24 m and the QP goes
infeasible; the bag shows 7.9 -> 0.05 m/s in one second, which is -7.9 m/s^2 against a
1.6 m/s^2 brake, i.e. an impact rather than braking. So the crash is the tail of a
distribution whose median already sits most of the way to the wall.

    lap |  wp128  wp131  wp133  wp134  wp136  wp139 | outcome
      1 |  +0.29  +0.80  +1.07  +1.15  +1.20  +0.97 | clean
      2 |  +0.39  +1.17  +1.50  +1.58  +1.54  +1.11 | clean
      3 |  +0.45  +0.87  +1.04  +1.06  +0.96  +0.65 | clean
      4 |  +0.45  +1.39  +1.98  +2.24  +2.69  +2.21 | WALL, 6.5 s penalty
      5 |  +0.43  +0.88  +1.03  +1.04  +0.90  +0.56 | clean
      6 |  +0.66  +1.31  +1.54  +1.57  +1.45  +0.98 | clean

This is the wp 129-134 hotspot that config.yaml:51 named and that 2026-08-03 called the
next target before the work moved to the AI track. It was never dissected until now.

Everything else tried in this region is spent: max_width=4.0 is indistinguishable from
baseline (10 vs 4 runs, +4.3 +/- 11.1 s), s6=18.0 is a section-speed cut and those are
banned outright for breaking recovery, steer_rate_max=0.50 and wp_id_offset=3 are far
worse, smoothing_distance=5 is far worse. A LOCAL lateral shift of the line is none of
those and has not been tried here -- shift_probe.py did the same computation for wp
112-130 and was never followed up either.

Reports only, as shift_probe.py does. A shifted line is a different line: it can raise
curvature, and curvature sets both the cornering demand and the MPC's own speed limit.
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

WP_LO, WP_HI = 126, 142


def build(x, y):
    rp = ReferencePath(m, list(x), list(y), cfg.reference_path.resolution,
                       cfg.reference_path.smoothing_distance,
                       cfg.reference_path.max_width, cfg.reference_path.circular)
    W = rp.waypoints
    n = len(W)
    k = np.array([float(w.kappa) for w in W])
    lb = np.array([float(w.lb) for w in W])
    ub = np.array([float(w.ub) for w in W])
    P = np.array([[w.x, w.y] for w in W])
    ds = np.array([np.hypot(*(P[(i + 1) % n] - P[i])) for i in range(n)])
    return n, k, lb, ub, ds


n0, k0, lb0, ub0, ds0 = build(wx, wy)
print(f"csv points {N}, waypoints {n0}, length {ds0.sum():.1f} m")
print("\nbaseline corridor through the hotspot (ub = the side the car drifts to):")
print("  wp   " + " ".join(f"{i:6d}" for i in range(120, 146, 2)))
print("  ub   " + " ".join(f"{ub0[i]:+6.2f}" for i in range(120, 146, 2)))
print("  lb   " + " ".join(f"{lb0[i]:+6.2f}" for i in range(120, 146, 2)))
print("  kap  " + " ".join(f"{k0[i]:+6.3f}" for i in range(120, 146, 2)))
print(f"\nover wp {WP_LO}-{WP_HI}: min ub {ub0[WP_LO:WP_HI+1].min():+.2f}, "
      f"max |kappa| {np.abs(k0[WP_LO:WP_HI+1]).max():.4f}")


def wp_to_csv(i):
    return int(round(i * N / n0))


c_lo, c_hi = wp_to_csv(WP_LO - 2), wp_to_csv(WP_HI + 2)
print(f"shift window: wp {WP_LO-2}-{WP_HI+2} -> csv {c_lo}-{c_hi}")

dx, dy = np.gradient(wx), np.gradient(wy)
nrm = np.hypot(dx, dy)
nx, ny = -dy / nrm, dx / nrm          # left normal, same convention as the bag analysis

print("\nshifted (raised-cosine bump, negative = away from the wall the car hits):")
for amp in (0.4, 0.6, 0.8, 1.0):
    for sign, label in ((-1.0, "toward lb"), (+1.0, "toward ub")):
        w = np.zeros(N)
        span = np.arange(c_lo, c_hi + 1)
        w[span] = 0.5 * (1 - np.cos(2 * np.pi * (span - c_lo) / max(len(span) - 1, 1)))
        sx, sy = wx + sign * amp * w * nx, wy + sign * amp * w * ny
        try:
            n1, k1, lb1, ub1, ds1 = build(sx, sy)
        except Exception as e:
            print(f"  {amp:.1f} m {label}: build failed ({type(e).__name__}: {e})")
            continue
        a, b = WP_LO * n1 // n0, (WP_HI + 1) * n1 // n0
        print(f"  {amp:.1f} m {label}: n={n1} min ub {ub1[a:b].min():+.2f} "
              f"(was {ub0[WP_LO:WP_HI+1].min():+.2f})  min width {(ub1-lb1)[a:b].min():.2f}  "
              f"max |kappa| {np.abs(k1[a:b]).max():.4f} "
              f"(was {np.abs(k0[WP_LO:WP_HI+1]).max():.4f})  len {ds1.sum():.1f} m")
