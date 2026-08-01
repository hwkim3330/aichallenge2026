#!/usr/bin/env python3
"""Generate a minimum-time, braking-aware speed profile for a closed-loop raceline.

Reads traj_mincurv.csv (s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2), replaces
vx_mps/ax_mps2 with a profile computed by the standard forward/backward pass
algorithm under the MPC's active constraints, and writes traj_mintime.csv.

Algorithm (closed loop, circular wraparound handled by repeated passes):
  1. v_curv[i] = min(v_max, sqrt(ay_max / max(|kappa[i]|, eps)))
  2. forward pass  (accel limit a_max):   v[i+1] = min(v[i+1], sqrt(v[i]^2 + 2*a_max*ds[i]))
  3. backward pass (brake limit |a_min|): v[i]   = min(v[i],   sqrt(v[i+1]^2 + 2*|a_min|*ds[i]))
  4. ax[i] = (v[i+1]^2 - v[i]^2) / (2*ds[i])  (circular)

Default constraints follow the live MPC config (config/config.yaml):
  a_max = 0.99 m/s^2, a_min = -1.6 m/s^2, ay_max = 20.0 m/s^2, v_max = 50 km/h.
"""

import argparse
import numpy as np

EPS = 1e-12


def compute_profile(kappa: np.ndarray, ds: np.ndarray, a_max: float, a_min: float,
                    ay_max: float, v_max: float, n_loops: int = 4) -> np.ndarray:
    """ds[i] is the distance from point i to point i+1 (ds[-1] wraps to point 0)."""
    n = len(kappa)
    v = np.minimum(v_max, np.sqrt(ay_max / np.maximum(np.abs(kappa), EPS)))

    # Forward pass (acceleration-limited), repeated around the loop until converged
    for _ in range(n_loops):
        changed = False
        for i in range(n):
            j = (i + 1) % n
            cap = np.sqrt(v[i] ** 2 + 2.0 * a_max * ds[i])
            if cap < v[j]:
                v[j] = cap
                changed = True
        if not changed:
            break

    # Backward pass (braking-limited), repeated around the loop until converged
    for _ in range(n_loops):
        changed = False
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            cap = np.sqrt(v[j] ** 2 + 2.0 * abs(a_min) * ds[i])
            if cap < v[i]:
                v[i] = cap
                changed = True
        if not changed:
            break

    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="traj_mincurv.csv")
    ap.add_argument("--output", default="traj_mintime.csv")
    ap.add_argument("--a-max", type=float, default=0.99)
    ap.add_argument("--a-min", type=float, default=-1.6)
    ap.add_argument("--ay-max", type=float, default=20.0)
    ap.add_argument("--v-max-kmh", type=float, default=50.0)
    args = ap.parse_args()
    v_max = args.v_max_kmh / 3.6

    with open(args.input) as f:
        header = f.readline().strip()
    data = np.loadtxt(args.input, delimiter=",", skiprows=1)
    s, x, y, psi, kappa, vx_old, ax_old = data.T
    n = len(s)

    # Circular segment lengths from coordinates (ds[i] = dist point i -> i+1)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    ds_all = np.hypot(dx, dy)

    # The raceline closes the loop with a final point that duplicates the first
    # one (ds wrap ~ 0). Compute the profile on the unique points only, then
    # mirror the first point's values onto the duplicated closing point.
    closed_dup = ds_all[-1] < 1e-3
    m = n - 1 if closed_dup else n
    ds = ds_all[:m].copy()
    if closed_dup:
        # ds[m-1] is dist(point m-1 -> last) = dist(point m-1 -> first): correct wrap
        assert abs((np.hypot(x[m - 1] - x[0], y[m - 1] - y[0])) - ds[m - 1]) < 1e-3

    v = compute_profile(kappa[:m], ds, args.a_max, args.a_min, args.ay_max, v_max)

    # Signed longitudinal acceleration over each segment (circular)
    v_next = np.roll(v, -1)
    ax = (v_next ** 2 - v ** 2) / (2.0 * ds)

    if closed_dup:
        v = np.append(v, v[0])
        ax = np.append(ax, ax[0])
        ds = np.append(ds, ds_all[-1])

    rows = np.column_stack([s, x, y, psi, kappa, v, ax])
    with open(args.output, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(", ".join(f"{val:.7f}" for val in r) + "\n")

    # ---- diagnostics ----
    def lap_time(vp):
        v_seg = 0.5 * (vp + np.roll(vp, -1))
        return float(np.sum(ds / np.maximum(v_seg, 0.1)))

    print(f"points: {n}, track length: {ds.sum():.1f} m, ds wrap (last->first): {ds[-1]:.2f} m")
    print(f"old vx: min {vx_old.min():.2f}, max {vx_old.max():.2f} m/s; "
          f"theoretical lap {lap_time(vx_old):.2f} s")
    print(f"new vx: min {v.min():.2f}, max {v.max():.2f} m/s; "
          f"theoretical lap {lap_time(v):.2f} s")
    print(f"ax range: [{ax.min():.3f}, {ax.max():.3f}] m/s^2 "
          f"(caps: {args.a_min}, {args.a_max})")

    # sanity: accel/brake and curvature-speed limits respected
    assert ax.max() <= args.a_max + 1e-6, "a_max violated"
    assert ax.min() >= args.a_min - 1e-6, "a_min violated"
    v_curv = np.minimum(v_max, np.sqrt(args.ay_max / np.maximum(np.abs(kappa), EPS)))
    assert np.all(v <= v_curv + 1e-9), "curvature speed violated"
    print("sanity checks passed")

    print("\n  s_m     kappa    v_curv   v_new    v_old    ax_new   (s in [180, 260])")
    for i in range(n):
        if 180.0 <= s[i] <= 260.0:
            print(f"{s[i]:7.1f} {kappa[i]:9.4f} {v_curv[i]:8.2f} {v[i]:8.2f} "
                  f"{vx_old[i]:8.2f} {ax[i]:8.3f}")


if __name__ == "__main__":
    main()
