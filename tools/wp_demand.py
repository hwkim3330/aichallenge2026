"""Lateral demand per waypoint, to see whether the stall band is loaded at all."""
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
W = rp.waypoints; n = len(W)
kappa = np.array([abs(getattr(w, "kappa", 0.0)) for w in W])
width = np.array([w.ub - w.lb for w in W])

sec = yaml.safe_load(open(PKG + "config/ref_vel.yaml"))["ref_vel_configulator"]
pts = sorted(((v["wp_id"], v["ref_vel"]) for v in sec.values()))
v = np.zeros(n)
for i in range(n):
    kmh = pts[0][1]
    for wp0, sp in pts:
        if i >= wp0:
            kmh = sp
    v[i] = kmh / 3.6
ay = kappa * v ** 2
print(f"ay_max config = {cfg.mpc.ay_max}; the ref_vel notes record 18.60 as this vehicle's measured peak")
print(f"lateral demand over the lap: median={np.median(ay):.2f}  p90={np.percentile(ay,90):.2f}  max={ay.max():.2f} m/s^2")
print(f"\nstall band 284-286: ay={ay[284:287].round(2)}  v={v[285]:.2f} m/s  kappa={kappa[284:287].round(3)}")
print("\nthe ten most laterally loaded waypoints:")
for i in np.argsort(-ay)[:10]:
    print(f"  wp {i:3d}  ay={ay[i]:6.2f}  kappa={kappa[i]:.4f}  v={v[i]:5.2f}  width={width[i]:.3f}")
print(f"\nrank of the stall band by lateral demand (1 = most loaded, of {n}):")
order = np.argsort(-ay)
for i in (284, 285, 286):
    print(f"  wp {i}: rank {int(np.where(order == i)[0][0]) + 1}   ay={ay[i]:.2f}")
