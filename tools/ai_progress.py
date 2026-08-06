"""Where does the car actually go? Arc-length progress along the raceline over time.

This exists because the covariate-shift story stopped fitting the measurements. The policy's cross-track
error is SMALLER than the MPC's (|e_y| p95 0.99 m against 1.05-2.52 m) and no policy frame sits outside
the range the expert data covers. A policy that stays on the line but completes no lap is not suffering
covariate shift, so "collect more expert data" cannot be the fix, and the actual failure has to be
located before anything else is trained.

At 3.7 m/s a 337.8 m lap takes 91 s, so 300 s should yield three laps. It yields zero. Either the car
does not traverse the circuit, or it does and the lap is not being counted. This distinguishes those:
it projects the recorded pose onto the raceline and prints the arc-length coordinate over time, so
running backwards, stalling at a fixed s, or oscillating between two values are all visible directly.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/kim/aichallenge2026/tools")
from ai_relabel import frames  # noqa: E402
from ai_surrogate import raceline_full  # noqa: E402

line, s_arr, psi_arr, kap_arr = raceline_full()
total = float(s_arr[-1] + (s_arr[1] - s_arr[0]))

for bag in sys.argv[1:]:
    f = frames(pathlib.Path(bag))
    if f is None:
        print(f"{bag}: no data")
        continue
    st, xy, yaw, v, mpc = f
    t = (st - st[0]) / 1e9
    idx = np.array([int(np.argmin((line[:, 0] - x) ** 2 + (line[:, 1] - y) ** 2)) for x, y in xy])
    s = s_arr[idx]
    # unwrap so a lap boundary reads as continuing progress rather than a jump back to zero
    ds = np.diff(s)
    ds[ds > total / 2] -= total
    ds[ds < -total / 2] += total
    cum = np.concatenate([[0.0], np.cumsum(ds)])

    name = pathlib.Path(bag).parts[-3]
    print(f"\n{name}: {len(t)} frames over {t[-1]:.1f} s")
    print(f"  distance travelled along the line: {cum[-1]:+.1f} m  "
          f"({cum[-1] / total:+.2f} laps)   circuit {total:.1f} m")
    print(f"  s span visited: {s.min():.1f}..{s.max():.1f} m of {total:.1f}")
    print(f"  speed: mean {v.mean():.2f} max {v.max():.2f}  frames with v<0.5: "
          f"{np.mean(v < 0.5):.1%}")
    fwd = float(np.sum(ds[ds > 0]))
    back = float(np.sum(ds[ds < 0]))
    print(f"  forward {fwd:+.1f} m, backward {back:+.1f} m  "
          f"(net/forward {cum[-1] / max(fwd, 1e-9):.2f})")
    # where does it lose time: progress per 20 s bucket
    print("  progress per 20 s:", " ".join(
        f"{cum[min(np.searchsorted(t, b + 20), len(t) - 1)] - cum[np.searchsorted(t, b)]:+.0f}"
        for b in range(0, int(t[-1]), 20)))
    # the single worst place: longest stretch with under 1 m of progress
    w = np.searchsorted(t, t + 10.0)
    stalled = [(i, cum[min(j, len(cum) - 1)] - cum[i]) for i, j in enumerate(w) if j < len(cum)]
    if stalled:
        i, d = min(stalled, key=lambda z: z[1])
        print(f"  worst 10 s window: {d:+.1f} m at t={t[i]:.0f}s, s={s[i]:.0f} m, "
              f"v there {v[i]:.2f}")
