"""Do the provider's constraints shift the corridor sideways, not just resize it?

The MPC steers toward xr_e_y = (lb + ub) / 2, so a corridor of the same width but a
different centre changes the line the car is asked to hold.

wp.lb is a SIGNED coordinate, not a distance: width = (ub-sm)-(lb+sm) comes out ~3.34
with sm=1.061, so ub-lb ~ 5.46 and lb is negative. The centre is therefore (ub+lb)/2
and the safety margin cancels out.
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
print(f"sanity: wp0 ub={rp.get_waypoint(0).ub:+.3f} lb={rp.get_waypoint(0).lb:+.3f} sm={sm:.3f}")

simple_c = np.array([(rp.get_waypoint(i).ub + rp.get_waypoint(i).lb) / 2.0 for i in range(nwp)])
simple_w = np.array([max(0.0, (rp.get_waypoint(i).ub - sm) - (rp.get_waypoint(i).lb + sm))
                     for i in range(nwp)])

pose = None
map_c = np.zeros(nwp); map_w = np.zeros(nwp)
for wp_id in range(nwp):
    ub_hor, lb_hor, bc = rp.update_path_constraints(wp_id + 1, pose, N, car.length, car.width, sm)
    ub_pw, lb_pw = np.array(bc[1])
    pose = (np.array(ub_pw) + np.array(lb_pw)) / 2.
    map_c[wp_id] = (ub_hor[0] + lb_hor[0]) / 2.0
    map_w[wp_id] = ub_hor[0] - lb_hor[0]

dc = map_c - simple_c
print(f"width   simple mean={simple_w.mean():.3f}  provider mean={map_w.mean():.3f}")
print(f"centre offset (provider - simple): mean={dc.mean():+.3f} median={np.median(dc):+.3f} "
      f"p95|.|={np.percentile(np.abs(dc),95):.3f} max|.|={np.abs(dc).max():.3f}")
print(f"  |offset| > 0.25 m at {int((np.abs(dc)>0.25).sum())}/{nwp}, > 0.50 m at {int((np.abs(dc)>0.50).sum())}")
