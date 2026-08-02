"""What is special about waypoints 284-286, where this car stalls?

Eight scored-condition races produced one in-race deadlock, at wp 284, and every run that
stalled at all stalled somewhere in 284-328 of 351 (tools/GOAL.md, 2026-08-02). If the
geometry there is ordinary the clustering needs another explanation; if it is not, the
band that stands out is the thing to fix.
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
rp = ReferencePath(m, wx, wy, cfg.reference_path.resolution,
                   cfg.reference_path.smoothing_distance,
                   cfg.reference_path.max_width, cfg.reference_path.circular)
W = rp.waypoints
n = len(W)
kappa = np.array([abs(getattr(w, "kappa", 0.0)) for w in W])
ub = np.array([w.ub for w in W]); lb = np.array([w.lb for w in W])
width = ub - lb
# static corridor as the MPC actually sees it
rp.update_simple_path_constraints(int(cfg.mpc.N), 0.0)
print(f"waypoints={n}  resolution={cfg.reference_path.resolution} m  max_width={cfg.reference_path.max_width}")
print(f"\n{'band':>12s} {'kappa med':>10s} {'kappa max':>10s} {'width med':>10s} {'width min':>10s}")
def row(lbl, idx):
    print(f"{lbl:>12s} {np.median(kappa[idx]):10.4f} {kappa[idx].max():10.4f} "
          f"{np.median(width[idx]):10.3f} {width[idx].min():10.3f}")
row("whole lap", np.arange(n))
row("284-286", np.arange(284, 287))
row("280-290", np.arange(280, 291))
row("284-328", np.arange(284, 329))
row("rest", np.r_[np.arange(0, 284), np.arange(329, n)])

print("\nthe ten narrowest waypoints on the lap:")
for i in np.argsort(width)[:10]:
    print(f"  wp {i:3d}  width={width[i]:.3f}  ub={ub[i]:+.3f} lb={lb[i]:+.3f}  kappa={kappa[i]:.4f}")
print("\nthe ten sharpest waypoints on the lap:")
for i in np.argsort(-kappa)[:10]:
    print(f"  wp {i:3d}  kappa={kappa[i]:.4f}  width={width[i]:.3f}")
print("\nprofile through the stall band:")
print(f"  {'wp':>4s} {'kappa':>8s} {'ub':>8s} {'lb':>8s} {'width':>8s}")
for i in range(278, 296):
    star = "  <-- stall" if 284 <= i <= 286 else ""
    print(f"  {i:4d} {kappa[i]:8.4f} {ub[i]:+8.3f} {lb[i]:+8.3f} {width[i]:8.3f}{star}")
