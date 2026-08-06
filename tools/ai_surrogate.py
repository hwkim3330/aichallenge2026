"""Fit a geometric surrogate of the MPC's steering, to relabel policy-visited states.

Why not pure pursuit: calibrated against the MPC's recorded steering it reaches R^2 0.675 at best, and
it gets there by UNDER-steering -- std 0.112 against the MPC's 0.203. Labelling with it would reinject
the amplitude collapse that took days to remove, so it is rejected as the expert.

What replaces it: the MPC's steering is a function of the vehicle's pose relative to the raceline, and
that relationship can be fitted. The features are all computable at ANY pose, which is the property that
matters -- expert demonstrations exist only on the racing line, while the states we need to label are off
it. Features:

    e_y          signed cross-track error to the raceline
    e_psi        heading error against the raceline tangent
    v            measured speed
    kappa(s+d)   raceline curvature ahead at d = 1, 3, 5, 8, 12, 18 m, which is what a horizon sees
    pp(Ld)       pure-pursuit steering at Ld = 2, 4, 6 m, as physically-correct basis functions

Fitted with ridge on interactions that matter for a bicycle model (e_y*v, e_psi*v, kappa*v^2), then
checked on HELD-OUT expert bags, reporting amplitude alongside R^2. A surrogate is only accepted if it
tracks the MPC's amplitude, not merely its direction.

The honest limit, reported rather than hidden: the MPC keeps |e_y| small, so the fit is interpolating
near the line and extrapolating where the policy actually goes. The script measures the e_y overlap
between expert and policy frames and reports what fraction of policy frames fall outside the range the
fit ever saw.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/kim/aichallenge2026/tools")
from ai_relabel import WHEELBASE, frames, load_raceline  # noqa: E402

RACELINE_CSV = pathlib.Path("/home/kim/aichallenge2026/aichallenge/workspace/src/aichallenge_submit/"
                            "multi_purpose_mpc_ros/env/final_ver3/traj_blend45_s110L_ramps.csv")
LOOKAHEAD_D = (1.0, 3.0, 5.0, 8.0, 12.0, 18.0)
PP_LD = (2.0, 4.0, 6.0)


def raceline_full():
    d = np.genfromtxt(RACELINE_CSV, delimiter=",", names=True)
    return (np.column_stack([d["x_m"], d["y_m"]]), np.asarray(d["s_m"]),
            np.asarray(d["psi_rad"]), np.asarray(d["kappa_radpm"]))


def featurise(xy, yaw, v, line, s_arr, psi_arr, kap_arr):
    n = len(xy)
    total_s = float(s_arr[-1] + (s_arr[1] - s_arr[0]))
    F = np.empty((n, 3 + len(LOOKAHEAD_D) + len(PP_LD)))
    ey_out = np.empty(n)
    for i in range(n):
        dx = line[:, 0] - xy[i, 0]
        dy = line[:, 1] - xy[i, 1]
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        # signed cross-track: project the offset onto the raceline normal
        nx, ny = -np.sin(psi_arr[j]), np.cos(psi_arr[j])
        e_y = -(dx[j] * nx + dy[j] * ny)
        e_psi = np.arctan2(np.sin(yaw[i] - psi_arr[j]), np.cos(yaw[i] - psi_arr[j]))
        ey_out[i] = e_y
        row = [e_y, e_psi, v[i]]
        for d in LOOKAHEAD_D:
            k = int(np.searchsorted(s_arr, (s_arr[j] + d) % total_s)) % len(s_arr)
            row.append(kap_arr[k])
        for Ld in PP_LD:
            # nearest raceline point at chord >= Ld ahead
            k = j
            for _ in range(len(line)):
                k = (k + 1) % len(line)
                if np.hypot(line[k, 0] - xy[i, 0], line[k, 1] - xy[i, 1]) >= Ld:
                    break
            ddx, ddy = line[k, 0] - xy[i, 0], line[k, 1] - xy[i, 1]
            c, s = np.cos(-yaw[i]), np.sin(-yaw[i])
            lx, ly = c * ddx - s * ddy, s * ddx + c * ddy
            L = max(float(np.hypot(lx, ly)), 1e-3)
            row.append(float(np.arctan2(2.0 * WHEELBASE * ly, L * L)))
        F[i] = row
    return F, ey_out


def expand(F):
    """Add the bicycle-model interactions. Column 0 is e_y, 1 is e_psi, 2 is v."""
    e_y, e_psi, v = F[:, 0], F[:, 1], F[:, 2]
    kap = F[:, 3:3 + len(LOOKAHEAD_D)]
    extra = [e_y * v, e_psi * v, e_y / np.maximum(v, 1.0), e_psi / np.maximum(v, 1.0),
             e_y * e_y * np.sign(e_y), e_psi * e_psi * np.sign(e_psi)]
    extra += [kap[:, k] * v * v for k in range(kap.shape[1])]
    return np.column_stack([F, np.column_stack(extra), np.ones(len(F))])


def fit(X, y, lam=1e-3):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def r2(p, t):
    return 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="expert bags to fit on")
    ap.add_argument("--test", nargs="+", required=True, help="held-out expert bags")
    ap.add_argument("--policy", nargs="*", default=[], help="policy bags, for the overlap report")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    line, s_arr, psi_arr, kap_arr = raceline_full()
    print(f"raceline {len(line)} points, s 0..{s_arr[-1]:.1f} m", flush=True)

    def collect(paths, label):
        XS, YS, EY = [], [], []
        for b in paths:
            f = frames(pathlib.Path(b))
            if f is None:
                print(f"  skip {b}", flush=True)
                continue
            _, xy, yaw, v, mpc = f
            F, ey = featurise(xy, yaw, v, line, s_arr, psi_arr, kap_arr)
            XS.append(expand(F))
            YS.append(mpc if mpc is not None else np.zeros(len(F)))
            EY.append(ey)
            print(f"  {label} {pathlib.Path(b).parts[-3]}: {len(F)} frames, "
                  f"|e_y| mean {np.abs(ey).mean():.2f} p95 {np.percentile(np.abs(ey), 95):.2f}",
                  flush=True)
        if not XS:
            return None, None, None
        return np.vstack(XS), np.concatenate(YS), np.concatenate(EY)

    Xtr, ytr, eytr = collect(args.train, "train")
    Xte, yte, eyte = collect(args.test, "test ")
    w = fit(Xtr, ytr)

    ptr, pte = Xtr @ w, Xte @ w
    print(f"\nsurrogate MPC steering fit ({Xtr.shape[1]} features, {len(ytr)} train frames)")
    print(f"  train  R2 {r2(ptr, ytr):+.3f}  std pred {ptr.std():.3f} / mpc {ytr.std():.3f}")
    print(f"  HELD-OUT R2 {r2(pte, yte):+.3f}  std pred {pte.std():.3f} / mpc {yte.std():.3f}  "
          f"MAE {np.abs(pte - yte).mean():.4f}")
    for lo, hi, lab in ((0, .05, "near-straight"), (.05, .15, "gentle"),
                        (.15, .30, "corner"), (.30, 9, "hard")):
        m = (np.abs(yte) >= lo) & (np.abs(yte) < hi)
        if m.sum() > 20:
            print(f"    {lab:<14} n={int(m.sum()):>6} R2 {r2(pte[m], yte[m]):+.3f} "
                  f"std {pte[m].std():.3f}/{yte[m].std():.3f}")

    if args.policy:
        Xp, _, eyp = collect(args.policy, "policy")
        lo, hi = eytr.min(), eytr.max()
        out = float(np.mean((eyp < lo) | (eyp > hi)))
        print(f"\nextrapolation check")
        print(f"  expert e_y range {lo:+.2f}..{hi:+.2f} m, |e_y| p95 {np.percentile(np.abs(eytr),95):.2f}")
        print(f"  policy |e_y| mean {np.abs(eyp).mean():.2f} p95 {np.percentile(np.abs(eyp),95):.2f} "
              f"max {np.abs(eyp).max():.2f}")
        print(f"  policy frames OUTSIDE the fitted e_y range: {out:.1%}")

    if args.save:
        np.save(args.save, w)
        print(f"\nsaved coefficients to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
