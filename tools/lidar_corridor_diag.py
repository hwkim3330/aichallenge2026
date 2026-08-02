"""Second candidate filter: keep only returns that fall inside the racing corridor.

Free-space elimination failed because the map's drivable region is wider than the
physical track -- barriers sit inside map-free space, so their returns all read as
dynamic. But barriers are outside the *corridor*, and the NPC drives inside it, so
lateral distance to the reference path separates them without relying on the map.
"""
import math
import numpy as np, yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.utils import load_ref_path
from multi_purpose_mpc_ros.common import convert_to_namedtuple
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_idl

PKG = get_package_share_directory("multi_purpose_mpc_ros") + "/"
cfg = convert_to_namedtuple(yaml.safe_load(open(PKG + "config/config.yaml")))
m = Map(PKG + cfg.map.yaml_path)
wx, wy, _, _ = load_ref_path(PKG + cfg.reference_path.csv_path)
rp = ReferencePath(m, wx, wy, cfg.reference_path.resolution,
                   cfg.reference_path.smoothing_distance,
                   cfg.reference_path.max_width, cfg.reference_path.circular)
WP = np.array([[w.x, w.y] for w in rp.waypoints])
print(f"reference waypoints: {len(WP)}")

ts = get_typestore(Stores.ROS2_HUMBLE); ex = {}
for i in sorted(Path('/aichallenge/ml_workspace/tiny_lidar_net/msgdefs').rglob('*.idl')):
    ex.update(get_types_from_idl(i.read_text()))
ts.register(ex)

def dist_to_path(px, py):
    d = np.hypot(px[:, None] - WP[None, :, 0], py[:, None] - WP[None, :, 1])
    return d.min(axis=1)

for THR in (1.0, 1.5, 2.0):
    pose = None; n = 0; counts = []; sizes = []; rngs = []
    with AnyReader([Path('/out/npc-with-1/bag')], default_typestore=ts) as r:
        for con, t, raw in r.messages():
            msg = r.deserialize(raw, con.msgtype)
            if con.topic.endswith('kinematic_state'):
                q = msg.pose.pose.orientation
                yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
            elif con.topic.endswith('lidar/scan') and pose is not None:
                n += 1
                if n % 40: continue
                x0, y0, yaw = pose
                rr = np.asarray(msg.ranges, dtype=np.float64)
                ok = np.isfinite(rr) & (rr >= 1.5) & (rr <= 25.0)
                if not ok.any(): counts.append(0); continue
                idx = np.nonzero(ok)[0]
                ang = msg.angle_min + idx*msg.angle_increment + yaw
                px = x0 + rr[idx]*np.cos(ang); py = y0 + rr[idx]*np.sin(ang)
                inside = dist_to_path(px, py) < THR
                if not inside.any(): counts.append(0); continue
                ipx, ipy = px[inside], py[inside]
                out, start = [], 0
                for i in range(1, len(ipx)+1):
                    if i == len(ipx) or math.hypot(ipx[i]-ipx[i-1], ipy[i]-ipy[i-1]) > 0.8:
                        if i - start >= 3:
                            out.append((ipx[start:i].mean(), ipy[start:i].mean(), i-start))
                        start = i
                counts.append(len(out))
                for cx, cy, k in out:
                    sizes.append(k); rngs.append(math.hypot(cx-x0, cy-y0))
    c = np.array(counts)
    line = f"corridor<{THR} m: scans={len(c)} mean_det={c.mean():.2f} 0={int((c==0).sum())} 1={int((c==1).sum())} 2+={int((c>=2).sum())}"
    if sizes:
        s = np.array(sizes); rg = np.array(rngs)
        line += f" | cluster beams med={np.median(s):.0f} max={s.max()} | range med={np.median(rg):.1f} m"
    print(line)
