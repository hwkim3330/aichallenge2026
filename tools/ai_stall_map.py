"""Where on the circuit does the AI stall? Cluster low-speed episodes by arc length.

The 18-slot official measurement said 14/18 complete six laps and that the failures begin with a stall:
median 15.1% of frames under 0.5 m/s, worst 73.1%. Recovering from the consequence (a car left facing
backwards) turned out to be the wrong thing to fix first -- the U-turn attempt deadlocked all three cars.
So the question is where the stalls START, because that decides what to change:

  clustered at s near 0        the grid start, i.e. three cars leaving together and colliding
  clustered at one s elsewhere one corner the policy cannot take at this pace
  spread uniformly            a pace problem rather than a place problem

An episode is a run of frames under 0.5 m/s lasting at least `min_s` seconds. The car's initial wait on
the grid is excluded by ignoring episodes that begin before the vehicle has ever exceeded 1 m/s, since
that is a countdown rather than a stall.
"""
import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/kim/aichallenge2026/tools")
from ai_relabel import frames  # noqa: E402
from ai_surrogate import raceline_full  # noqa: E402

MIN_EPISODE_S = 2.0
line, s_arr, _, _ = raceline_full()
total = float(s_arr[-1] + (s_arr[1] - s_arr[0]))

episodes = []
per_run = []
for bag in sys.argv[1:]:
    f = frames(pathlib.Path(bag))
    if f is None:
        continue
    st, xy, yaw, v, _ = f
    t = (st - st[0]) / 1e9
    idx = np.array([int(np.argmin((line[:, 0] - x) ** 2 + (line[:, 1] - y) ** 2)) for x, y in xy])
    s = s_arr[idx]
    moved = np.cumsum(v > 1.0) > 0          # has the car ever exceeded 1 m/s
    slow = (v < 0.5) & moved
    n = 0
    i = 0
    while i < len(slow):
        if not slow[i]:
            i += 1
            continue
        j = i
        while j < len(slow) and slow[j]:
            j += 1
        dur = t[j - 1] - t[i]
        if dur >= MIN_EPISODE_S:
            episodes.append((float(s[i]), float(dur)))
            n += 1
        i = j
    per_run.append((pathlib.Path(bag).parts[-3] + "/" + pathlib.Path(bag).parts[-2], n,
                    float(np.sum([d for _, d in episodes[-n:]])) if n else 0.0))

print(f"{len(per_run)} slots, {len(episodes)} stall episodes of >= {MIN_EPISODE_S:.0f} s\n")
print("per slot: episodes, total stalled seconds")
for name, n, sec in per_run:
    print(f"  {name:<34} {n:>3}  {sec:>7.1f} s")

if episodes:
    S = np.array([s for s, _ in episodes])
    D = np.array([d for _, d in episodes])
    print(f"\nstalled time total {D.sum():.0f} s; episode duration median {np.median(D):.1f} s "
          f"max {D.max():.1f} s")
    print(f"\nby arc-length bucket of {total/12:.0f} m (where the stall STARTS):")
    buckets = collections.Counter((S // (total / 12)).astype(int))
    for b in range(12):
        lo = b * total / 12
        secs = D[(S >= lo) & (S < lo + total / 12)].sum()
        bar = "#" * int(buckets[b])
        print(f"  s {lo:6.0f}-{lo + total/12:6.0f} m  {buckets[b]:>3} episodes  {secs:>7.1f} s  {bar}")
    hot = buckets.most_common(3)
    print("\nhottest buckets:", ", ".join(
        f"s {b * total/12:.0f}-{(b+1) * total/12:.0f} m ({c} episodes)" for b, c in hot))
