"""Relabel policy-visited states with a geometric expert, so the 237k LP frames become usable.

The problem this solves is the reason those frames were thrown away: they are states the NETWORK drove
into, labelled with the NETWORK's own steering, so training on them teaches the failure. DAgger needs
the EXPERT's action at those states, and the expert we have is the MPC -- which we cannot re-run offline
because it needs the whole planning stack.

But we do not need the MPC itself, only a function that returns a competent steering command for an
arbitrary pose. The bags contain /localization/kinematic_state, and the raceline the MPC follows is a
CSV of x, y, psi. So pure pursuit on that raceline is a candidate expert, defined everywhere including
the off-line states the policy visits, which is exactly the property expert demonstrations lack.

Whether it is a GOOD expert is an empirical question, and this script answers it before using it: it
relabels the EXPERT bags, where the MPC's real steering was recorded, and reports the agreement. A
geometric expert that cannot reproduce the MPC on-line has no business labelling off-line states, and
the lookahead law is calibrated on that agreement rather than guessed.

Usage:
  relabel.py calibrate <expert-bag> [<expert-bag> ...]     fit Ld = a + b*v, report R^2 vs the MPC
  relabel.py apply --a A --b B <bag> ...                   write pp_steers.npy beside each sequence
"""
from __future__ import annotations

import argparse
import bisect
import math
import pathlib
import sys

import numpy as np

RACELINE = pathlib.Path("/home/kim/aichallenge2026/aichallenge/workspace/src/aichallenge_submit/"
                        "multi_purpose_mpc_ros/env/final_ver3/traj_blend45_s110L_ramps.csv")
WHEELBASE = 1.087  # racing_kart_description/config/vehicle_info.param.yaml


def load_raceline():
    d = np.genfromtxt(RACELINE, delimiter=",", names=True)
    return np.column_stack([d["x_m"], d["y_m"]]), np.asarray(d["s_m"])


def read_bag(bag: pathlib.Path):
    """Return scan times, pose (x, y, yaw), speed and the recorded MPC steering, all on scan times."""
    # Reuse the extractor's own typestore builder rather than duplicating it: Autoware's messages come
    # from IDL files in msgdefs/, and registering them by hand got the type name wrong.
    sys.path.insert(0, "/home/kim/aichallenge2026/aichallenge/ml_workspace/tiny_lidar_net")
    from rosbags.highlevel import AnyReader
    from extract_data_from_bag import build_typestore

    scan_t, pose, spd_t, spd, cmd_t, cmd_s = [], [], [], [], [], []
    with AnyReader([bag], default_typestore=build_typestore()) as r:
        want = {"/sensing/lidar/scan", "/localization/kinematic_state",
                "/vehicle/status/velocity_status", "/control/command/control_cmd"}
        conns = [c for c in r.connections if c.topic in want]
        for conn, t, raw in r.messages(connections=conns):
            if conn.topic == "/sensing/lidar/scan":
                scan_t.append(t)
                continue
            m = r.deserialize(raw, conn.msgtype)
            if conn.topic == "/localization/kinematic_state":
                p, q = m.pose.pose.position, m.pose.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
                pose.append((t, p.x, p.y, yaw))
            elif conn.topic == "/vehicle/status/velocity_status":
                spd_t.append(t)
                spd.append(m.longitudinal_velocity)
            else:
                cmd_t.append(t)
                cmd_s.append(m.lateral.steering_tire_angle)
    pose.sort()
    return (np.asarray(scan_t), np.asarray(pose, dtype=float),
            np.asarray(spd_t), np.asarray(spd, dtype=float),
            np.asarray(cmd_t), np.asarray(cmd_s, dtype=float))


def at_times(src_t, src_v, tgt_t):
    """Nearest-earlier sample, which is what a controller would have seen."""
    out = np.empty((len(tgt_t),) + src_v.shape[1:], dtype=float)
    for i, t in enumerate(tgt_t):
        j = min(max(bisect.bisect_right(src_t, t) - 1, 0), len(src_t) - 1)
        out[i] = src_v[j]
    return out


def pure_pursuit(xy, yaw, v, line, a, b):
    """Steering to a lookahead point on the raceline. Ld = a + b*v, floored at a."""
    n = len(xy)
    steer = np.empty(n)
    # nearest raceline index per frame
    for i in range(n):
        d2 = (line[:, 0] - xy[i, 0]) ** 2 + (line[:, 1] - xy[i, 1]) ** 2
        j = int(np.argmin(d2))
        Ld = max(a + b * v[i], a)
        # walk forward along the line until the chord reaches Ld
        k = j
        m = len(line)
        for _ in range(m):
            k2 = (k + 1) % m
            if math.hypot(line[k2, 0] - xy[i, 0], line[k2, 1] - xy[i, 1]) >= Ld:
                k = k2
                break
            k = k2
        dx, dy = line[k, 0] - xy[i, 0], line[k, 1] - xy[i, 1]
        # into the vehicle frame
        c, s = math.cos(-yaw[i]), math.sin(-yaw[i])
        lx, ly = c * dx - s * dy, s * dx + c * dy
        L = math.hypot(lx, ly)
        steer[i] = math.atan2(2.0 * WHEELBASE * ly, max(L, 1e-3) ** 2)
    return steer


def frames(bag: pathlib.Path):
    scan_t, pose, spd_t, spd, cmd_t, cmd_s = read_bag(bag)
    if len(scan_t) == 0 or len(pose) == 0:
        return None
    P = at_times(pose[:, 0], pose[:, 1:], scan_t)
    v = at_times(spd_t, spd, scan_t) if len(spd_t) else np.full(len(scan_t), 5.0)
    mpc = at_times(cmd_t, cmd_s, scan_t) if len(cmd_t) else None
    return scan_t, P[:, :2], P[:, 2], v, mpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "apply"])
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--a", type=float, default=2.0)
    ap.add_argument("--b", type=float, default=0.3)
    ap.add_argument("--outname", default="pp_steers.npy")
    args = ap.parse_args()

    line, _ = load_raceline()
    print(f"raceline: {len(line)} points", flush=True)

    data = []
    for b in args.bags:
        f = frames(pathlib.Path(b))
        if f is None:
            print(f"  skip {b}: no scans or no pose", flush=True)
            continue
        data.append((pathlib.Path(b), f))
        print(f"  {pathlib.Path(b).parts[-3]}: {len(f[0])} frames", flush=True)

    if args.mode == "calibrate":
        print("\nfitting Ld = a + b*v against the MPC's recorded steering")
        best = None
        for a in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            for b_ in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                P, T = [], []
                for _, (st, xy, yaw, v, mpc) in data:
                    if mpc is None:
                        continue
                    P.append(pure_pursuit(xy, yaw, v, line, a, b_))
                    T.append(mpc)
                p, t = np.concatenate(P), np.concatenate(T)
                r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
                print(f"  a={a:.1f} b={b_:.1f}  R2 vs MPC {r2:+.3f}  "
                      f"std pp {p.std():.3f} / mpc {t.std():.3f}  MAE {np.abs(p - t).mean():.4f}",
                      flush=True)
                if best is None or r2 > best[0]:
                    best = (r2, a, b_)
        print(f"\nbest: a={best[1]} b={best[2]}  R2 {best[0]:+.3f}")
        return 0

    for bag, (st, xy, yaw, v, mpc) in data:
        pp = pure_pursuit(xy, yaw, v, line, args.a, args.b)
        out = bag.parent.parent  # <run>/d<slot>/rosbag2_autoware -> <run>
        np.save(bag / args.outname, pp)
        msg = f"{out.name}: wrote {len(pp)} labels, std {pp.std():.3f}"
        if mpc is not None:
            msg += f", R2 vs recorded cmd {1 - ((pp - mpc) ** 2).sum() / ((mpc - mpc.mean()) ** 2).sum():+.3f}"
        print("  " + msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
