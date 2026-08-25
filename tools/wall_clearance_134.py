"""Where is the wall at the wp 126-142 hotspot, measured off the occupancy grid?

shift_probe_134.py reported the ReferencePath corridor (lb/ub), but those are clipped by
max_width=3.0, so a bound reading +3.00 only means "at least 3 m" and the sign convention
against the bag's left-normal offsets is not established. This reads the grid directly and
reports signed clearance in the SAME convention the bag analysis uses (+ = left normal of
the csv polyline), so the two can be compared without guessing.

Also reports the clearance at the pose where the car actually stopped, if one is given.
"""
import sys
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.utils import load_ref_path
from multi_purpose_mpc_ros.common import convert_to_namedtuple

PKG = get_package_share_directory("multi_purpose_mpc_ros") + "/"
cfg = convert_to_namedtuple(yaml.safe_load(open(PKG + "config/config.yaml")))
m = Map(PKG + cfg.map.yaml_path)
wx, wy, _, _ = load_ref_path(PKG + cfg.reference_path.csv_path)
wx, wy = np.array(wx, float), np.array(wy, float)

# Map.data is int8, 1 = drivable, and its row 0 is the TOP of the image rather than
# min-y. Both were established by probing all four conventions against the racing line:
# only row = H-1-(y-oy)/res with data==1 puts 350 of 350 line points on free cells; the
# other three give 0.49, 0.51 and 0.00.
free = np.asarray(m.data).astype(bool)
H, W = free.shape
res = float(m.resolution)
ox, oy = float(m.origin[0]), float(m.origin[1])


def is_free(x, y):
    j = int((x - ox) / res)
    i = H - 1 - int((y - oy) / res)
    if i < 0 or j < 0 or i >= H or j >= W:
        return False
    return bool(free[i, j])


on_line = sum(is_free(wx[i], wy[i]) for i in range(len(wx)))
print(f"map {free.shape} res {res} origin ({ox:.1f},{oy:.1f}); "
      f"{on_line}/{len(wx)} line points read as free")
assert on_line == len(wx), "map indexing convention is wrong; clearances would be noise"

dx, dy = np.gradient(wx), np.gradient(wy)
nrm = np.hypot(dx, dy)
nx, ny = -dy / nrm, dx / nrm            # left normal, identical to the bag analysis


def clearance(i, sign, limit=6.0, step=0.02):
    d = 0.0
    while d < limit:
        d += step
        if not is_free(wx[i] + sign * d * nx[i], wy[i] + sign * d * ny[i]):
            return d
    return limit


print("\nsigned clearance from the racing line, + = left normal (same sign as the bag table)")
print(" wp   left(+)  right(-)   |  car's habitual offset there")
habit = {128: "+0.3..+0.7", 131: "+0.8..+1.4", 133: "+1.0..+2.0",
         134: "+1.0..+2.2", 136: "+0.9..+2.7", 139: "+0.6..+2.2"}
for i in range(118, 146):
    L, R = clearance(i, +1.0), clearance(i, -1.0)
    mark = "   <-- " + habit[i] if i in habit else ""
    print(f" {i:3d}   {L:5.2f}    {R:5.2f}{mark}")

if len(sys.argv) > 2:
    px, py = float(sys.argv[1]), float(sys.argv[2])
    k = int(np.argmin(np.hypot(wx - px, wy - py)))
    off = (px - wx[k]) * nx[k] + (py - wy[k]) * ny[k]
    d = 0.0
    while d < 6.0:
        d += 0.02
        ang = np.linspace(0, 2 * np.pi, 72)
        if any(not is_free(px + d * np.cos(a), py + d * np.sin(a)) for a in ang):
            break
    print(f"\nimpact pose ({px:.2f},{py:.2f}): nearest wp {k}, signed offset {off:+.2f} m, "
          f"nearest obstacle {d:.2f} m in any direction")
