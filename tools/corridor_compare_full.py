"""Compare all N horizon columns, not just column 0.

Column 0 matched almost exactly (width -0.012 m, centre +0.003 m), but the MPC uses
all 20. The two builders also differ in kind: update_simple_path_constraints takes no
pose, while update_path_constraints takes one and the provider feeds it a pose swept
synthetically along the path. Divergence would grow along the horizon, not at its head.
"""
import numpy as np, yaml
from ament_index_python.packages import get_package_share_directory
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.utils import load_ref_path
from multi_purpose_mpc_ros.common import convert_to_namedtuple

PKG = get_package_share_directory('multi_purpose_mpc_ros') + "/"
cfg = convert_to_namedtuple(yaml.safe_load(open(PKG + "config/config.yaml")))
m = Map(PKG + cfg.map.yaml_path)
wp_x, wp_y, _, _ = load_ref_path(PKG + cfg.reference_path.csv_path)
rp = ReferencePath(m, wp_x, wp_y, cfg.reference_path.resolution,
                   cfg.reference_path.smoothing_distance,
                   cfg.reference_path.max_width, cfg.reference_path.circular)
car = BicycleModel(rp, cfg.bicycle_model.length, cfg.bicycle_model.width,
                   1.0 / cfg.mpc.control_rate)
N, sm, nwp = cfg.mpc.N, car.safety_margin, rp.n_waypoints - 1

# Baseline: exactly what update_simple_path_constraints stores, row i column n.
simple_ub = np.zeros((nwp, N)); simple_lb = np.zeros((nwp, N))
for i in range(nwp):
    for n in range(N):
        wp = rp.get_waypoint(i + n)
        u, l = wp.ub - sm, wp.lb + sm
        if u < l: u = l = 0.0
        simple_ub[i, n], simple_lb[i, n] = u, l

# Provider: same sweep it publishes (post off-by-one fix, so wp_id not wp_id+1).
pose = None
prov_ub = np.zeros((nwp, N)); prov_lb = np.zeros((nwp, N))
for i in range(nwp):
    ub_hor, lb_hor, bc = rp.update_path_constraints(i, pose, N, car.length, car.width, sm)
    ub_pw, lb_pw = np.array(bc[1])
    pose = (np.array(ub_pw) + np.array(lb_pw)) / 2.
    prov_ub[i], prov_lb[i] = ub_hor, lb_hor

dw = (prov_ub - prov_lb) - (simple_ub - simple_lb)
dc = (prov_ub + prov_lb) / 2 - (simple_ub + simple_lb) / 2
print(f"rows={nwp} cols={N}")
print(f"{'col':>4} {'d_width mean':>13} {'d_centre mean':>14} {'|d_centre| p95':>15} {'|d_centre| max':>15}")
for n in list(range(0, N, 4)) + [N - 1]:
    print(f"{n:4d} {dw[:,n].mean():13.3f} {dc[:,n].mean():14.3f} "
          f"{np.percentile(np.abs(dc[:,n]),95):15.3f} {np.abs(dc[:,n]).max():15.3f}")
print()
print(f"whole horizon: |d_centre| mean={np.abs(dc).mean():.3f}  p95={np.percentile(np.abs(dc),95):.3f}  max={np.abs(dc).max():.3f}")
print(f"               |d_width|  mean={np.abs(dw).mean():.3f}  p95={np.percentile(np.abs(dw),95):.3f}  max={np.abs(dw).max():.3f}")
zero_p = int(((prov_ub - prov_lb) <= 0).sum()); zero_s = int(((simple_ub - simple_lb) <= 0).sum())
print(f"collapsed cells (ub<=lb): provider {zero_p}/{nwp*N}, simple {zero_s}/{nwp*N}")
