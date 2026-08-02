"""How far are lidar returns from the nearest thing the static map knows about?

If the map describes the track, most returns are walls or cones and should sit near
zero. A distribution shifted well away from zero means either a frame mismatch or a map
that simply does not contain what the lidar sees -- and free-space elimination cannot
work in either case.
"""
import math
import numpy as np, yaml
from pathlib import Path
from scipy import ndimage
from ament_index_python.packages import get_package_share_directory
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.common import convert_to_namedtuple
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl

PKG = get_package_share_directory("multi_purpose_mpc_ros") + "/"
cfg = convert_to_namedtuple(yaml.safe_load(open(PKG + "config/config.yaml")))
m = Map(PKG + cfg.map.yaml_path)
occ = ~np.asarray(m.data, dtype=bool)
print(f"map: {m.width}x{m.height} px, res={m.resolution} m, origin={m.origin}")
print(f"     occupied cells: {int(occ.sum())} / {occ.size} ({100.0*occ.sum()/occ.size:.1f}%)")
dist = ndimage.distance_transform_edt(np.asarray(m.data, dtype=bool)) * m.resolution

ts = get_typestore(Stores.ROS2_HUMBLE); ex = {}
for i in sorted(Path('/aichallenge/ml_workspace/tiny_lidar_net/msgdefs').rglob('*.idl')):
    ex.update(get_types_from_idl(i.read_text()))
ts.register(ex)

pose = None; all_d = []; ego_d = []; n = 0
with AnyReader([Path('/out/npc-with-1/bag')], default_typestore=ts) as r:
    for con, t, raw in r.messages():
        msg = r.deserialize(raw, con.msgtype)
        if con.topic.endswith('kinematic_state'):
            q = msg.pose.pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        elif con.topic.endswith('lidar/scan') and pose is not None:
            n += 1
            if n % 20:      # every 20th scan is plenty
                continue
            x0, y0, yaw = pose
            gx = int((x0 - m.origin[0])/m.resolution + 0.5)
            gy = int((m.height-1) - (y0 - m.origin[1])/m.resolution + 0.5)
            if 0 <= gx < m.width and 0 <= gy < m.height:
                ego_d.append(dist[gy, gx])
            rng = np.asarray(msg.ranges, dtype=np.float64)
            ok = np.isfinite(rng) & (rng >= 0.5) & (rng <= 25.0)
            if not ok.any(): continue
            idx = np.nonzero(ok)[0]
            ang = msg.angle_min + idx*msg.angle_increment + yaw
            px = x0 + rng[idx]*np.cos(ang); py = y0 + rng[idx]*np.sin(ang)
            gxs = np.clip(((px - m.origin[0])/m.resolution + 0.5).astype(int), 0, m.width-1)
            gys = np.clip(((m.height-1) - (py - m.origin[1])/m.resolution + 0.5).astype(int), 0, m.height-1)
            all_d.append(dist[gys, gxs])

d = np.concatenate(all_d)
print(f"\nreturns sampled: {len(d)}")
print(f"  distance from return to nearest occupied cell:")
for p in (10, 25, 50, 75, 90, 99):
    print(f"    p{p:<3d} {np.percentile(d,p):6.2f} m")
print(f"    fraction within 0.6 m of a wall: {100.0*(d<=0.6).mean():.1f}%")
e = np.array(ego_d)
print(f"\n  ego position distance-to-wall: median={np.median(e):.2f} m (should be ~half the corridor, ~1.7 m)")
