#!/usr/bin/env python3
"""Does running with the lidar on cost lap time?

The validated 6/6 submission (278.23 s, avg 46.37) was measured under eval.sh, which
passes --lidar off. The official evaluation does not: every
AWSIM_Data/StreamingAssets/RaceConfig/eval-{1,2,3,4}p.yaml sets `lidar: "on"`
(tools/GOAL.md, 2026-08-02 correction). Lidar simulation is CPU work on the same box as
the control loop, so the scored condition is strictly heavier than the one the
submission was validated under, and nothing has measured the difference.

solo6.sh and solo6lidar.sh differ in exactly one flag -- `--lidar off` against
`--lidar cpu` -- with laps, timeout, start mode, NPC count and handicap all held equal.
That makes them a clean pair. Runs alternate A/B/A/B so any thermal or background drift
over the session falls on both arms rather than on whichever went second.

Reuses evolve.py's compose plumbing, which already knows the failure modes: stale
containers sharing a ROS domain, and count-mode starts needing no /admin/awsim/start.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evolve import ROOT, compose, env_for, parse_laps, sweep  # noqa: E402

OUT = ROOT / "tools/lidar_load_ab.jsonl"
SLOT = 1
# solo6*.sh time out at 480 s; allow for AWSIM start-up and container teardown on top.
WALL = 620


def one_run(tag: str, sim_mode: str) -> dict:
    out = f"/output/{tag}"
    host_out = ROOT / "output" / tag
    sweep()
    env0 = dict(env_for(SLOT, "config.yaml", "ref_vel.yaml", out))
    env0.update(ROS_DOMAIN_ID="0", SIM_MODE=sim_mode, LOG_DIR=out)
    procs = [compose(["run", "--rm", "-T", "--name", f"lidarab-sim-{tag}",
                      "simulator"], env0, wait=False)]
    time.sleep(8)
    procs.append(compose(["run", "--rm", "-T", "--name", f"lidarab-{tag}",
                          "autoware"], env_for(SLOT, "config.yaml", "ref_vel.yaml", out),
                         wait=False))

    # Stop as soon as six laps are in the log; no reason to sit through the timeout.
    deadline = time.time() + WALL
    laps: list[float] = []
    log = host_out / f"d{SLOT}" / "autoware.log"
    while time.time() < deadline:
        if procs[0].poll() is not None:
            break
        laps, _ = parse_laps(log)
        if len(laps) >= 6:
            break
        time.sleep(5)

    for p in procs:
        if p.poll() is None:
            p.kill()
    sweep()
    time.sleep(5)

    laps, recov = parse_laps(log)
    return dict(tag=tag, sim_mode=sim_mode, laps=laps, recoveries=recov,
                finished=len(laps) >= 6, total6=sum(laps[:6]) if len(laps) >= 6 else None)


def summarise(rows: list[dict]) -> None:
    for mode in ("solo6", "solo6lidar"):
        got = [r for r in rows if r["sim_mode"] == mode]
        if not got:
            continue
        # Lap 1 carries the standing start and is not comparable to the rest.
        steady = [L for r in got for L in r["laps"][1:]]
        fin = sum(1 for r in got if r["finished"])
        line = f"{mode:11s} runs={len(got)} finished={fin}/{len(got)}"
        if steady:
            line += (f" steady laps n={len(steady)}"
                     f" median={statistics.median(steady):.2f}"
                     f" mean={statistics.mean(steady):.2f}")
            if len(steady) > 1:
                line += f" sd={statistics.stdev(steady):.2f}"
        tot = [r["total6"] for r in got if r["total6"]]
        if tot:
            line += f" | 6-lap total median={statistics.median(tot):.2f}"
        line += f" | recoveries={sum(r['recoveries'] for r in got)}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2,
                    help="one round is one solo6 run and one solo6lidar run")
    args = ap.parse_args()

    rows = [json.loads(l) for l in OUT.read_text().splitlines()] if OUT.exists() else []
    for r in range(args.rounds):
        for mode in ("solo6", "solo6lidar"):
            tag = f"lidarab-{mode}-{len(rows):02d}"
            print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
            row = one_run(tag, mode)
            rows.append(row)
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  laps={[round(x, 2) for x in row['laps']]} "
                  f"finished={row['finished']} recoveries={row['recoveries']}",
                  flush=True)
    summarise(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
