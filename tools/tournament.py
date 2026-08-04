#!/usr/bin/env python3
"""Race the local candidate configs against each other, repeatedly, and keep the winner.

The idea is the user's: there are about twenty config variants in the tree, several of them
known to drive well, so put them against each other under competition rules and let the
survivors define the champion instead of arguing over one parameter at a time.

Scenario is arena3 (three cars, collisions off, handicap ON, lidar ON, 480 s) -- screen3 with
the two flags the official RaceConfig sets, because screen3 turns the lidar off and that
silently removes lidar_guard, under which the eval-verified config takes 310.9 s for one lap.
Collisions off is not a shortcut -- evolve.py's docstring records why: with contact on, a bad challenger becomes a
parked obstacle and destroys the OTHER cars' measurements, three generations were lost that
way, and the official three-vehicle RaceConfig has collisions off too. So one run yields
three independent measurements.

Fitness is lexicographic, completion before pace, because 34 of 40 board battles are decided
on lap count. Completion uses race_outcome, not lap-message counting: the MPC's final "Lap N
completed" line races the orchestrator's shutdown and usually loses, which scored real six-lap
finishes as failures in eight of twenty-four historical measurements.

State lives in tools/tournament_log.jsonl so the loop resumes after any interruption.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evolve import (CFG_DIR, ROOT, compose, env_for, link_installed,  # noqa: E402
                    race_outcome, request_start, total_6, wait_grounded)

LOG = ROOT / "tools/tournament_log.jsonl"
SLOTS = [1, 2, 3]
WALL = 900

# The champion is the shipped config. Challengers are the variants worth a race: the ones
# whose names record a deliberate line or speed choice rather than a one-off probe. Today's
# throwaway screening configs are excluded by name.
CHAMPION = "config.yaml"
POOL = [
    "config_top36.yaml", "config_csvmt_plus.yaml", "config_csvmt_ay11.yaml",
    "config_csvmt_ay13.yaml", "config_csvmt_ramps.yaml", "config_expA.yaml",
    "config_expB.yaml", "config_expC.yaml", "config_horizon.yaml", "config_qtime.yaml",
    "config_officialline.yaml", "config_D_ey3x.yaml", "config_expV2.yaml",
]


def sweep() -> None:
    import subprocess
    for _ in range(12):
        subprocess.run(
            "docker ps -aq --filter name=tourney- | xargs -r docker rm -f; "
            "docker ps -aq --filter name=aichallenge2026-simulator | xargs -r docker rm -f",
            shell=True, capture_output=True)
        if not subprocess.run("docker ps -aq --filter name=tourney-", shell=True,
                              capture_output=True, text=True).stdout.strip():
            return
        time.sleep(5)
    raise RuntimeError("containers from a previous round would not die; refusing to measure")


def one_round(tag: str, entrants: list[str]) -> list[dict]:
    out = f"/output/{tag}"
    host = ROOT / "output" / tag
    sweep()
    env0 = dict(env_for(SLOTS[0], entrants[0], "ref_vel.yaml", out))
    env0.update(ROS_DOMAIN_ID="0", SIM_MODE="arena3", LOG_DIR=out)
    procs = [compose(["run", "--rm", "-T", "--name", f"tourney-sim-{tag}",
                      "simulator"], env0, wait=False)]
    time.sleep(10)
    for slot, cfg in zip(SLOTS, entrants):
        link_installed(cfg)
        procs.append(compose(["run", "--rm", "-T", "--name", f"tourney-{tag}-d{slot}",
                              "autoware"], env_for(slot, cfg, "ref_vel.yaml", out),
                             wait=False))
        time.sleep(8)

    # 420 s rather than the 240 s default: three Autoware instances plus AWSIM never all
    # reported ready inside 240, and with sync start an unready car begins uncontrolled.
    if wait_grounded(host, SLOTS, timeout=420) < len(SLOTS):
        print("  warning: not every car reported ready", flush=True)
    request_start()

    deadline = time.time() + WALL
    while time.time() < deadline:
        if procs[0].poll() is not None:
            break
        done = sum(1 for s in SLOTS
                   if (host / f"d{s}" / "autoware.log").exists()
                   and "latest link updated" in
                   (host / f"d{s}" / "autoware.log").read_text(errors="replace"))
        if done == len(SLOTS):
            time.sleep(5)
            break
        time.sleep(10)

    for p in procs:
        if p.poll() is None:
            p.kill()
    sweep()
    time.sleep(5)

    rows = []
    for slot, cfg in zip(SLOTS, entrants):
        log = host / f"d{slot}" / "autoware.log"
        o = (race_outcome(log, timeout_s=480.0) if log.exists()
             else dict(laps=[], completed=0, elapsed=None, finished=False))
        text = log.read_text(errors="replace") if log.exists() else ""
        rows.append(dict(tag=tag, slot=slot, config=cfg, laps=o["laps"],
                         completed=o["completed"], elapsed=o["elapsed"],
                         finished=o["finished"],
                         stalls=len(re.findall(r"STALL ANATOMY", text))))
    return rows


def fitness(row: dict):
    """Completion first, then the six-lap total. Missing measurement sorts last."""
    if not row["laps"]:
        return (2, float("inf"))
    total = row["elapsed"] if (row["finished"] and row["elapsed"]) else total_6(row["laps"])
    return (0 if row["finished"] else 1, total if total is not None else float("inf"))


def standings(rows: list[dict]) -> list[tuple[str, int, int, float]]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)
    out = []
    for cfg, rs in by.items():
        fin = sum(1 for r in rs if r["finished"])
        tot = [r["elapsed"] for r in rs if r["finished"] and r["elapsed"]]
        out.append((cfg, fin, len(rs), statistics.mean(tot) if tot else float("inf")))
    # most finishes first, then fastest mean among finishes
    out.sort(key=lambda t: (-(t[1] / t[2]) if t[2] else 0, t[3]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    rows = ([json.loads(l) for l in LOG.read_text().splitlines()]
            if LOG.exists() else [])
    seen = {r["config"] for r in rows}
    for i in range(args.rounds):
        # Champion always races, so every round carries a same-conditions control. Challengers
        # are whichever pool entries have the fewest measurements so far.
        order = sorted(POOL, key=lambda c: (sum(1 for r in rows if r["config"] == c),
                                            POOL.index(c)))
        entrants = [CHAMPION, order[0], order[1]]
        # Rotate which entrant sits in which grid slot; slot 1 is the worst and pinning the
        # champion there decided the result rather than measuring it.
        rnd = len(rows) // 3
        entrants = entrants[rnd % 3:] + entrants[:rnd % 3]
        tag = f"tourney-{rnd:02d}"
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: {entrants}", flush=True)
        got = one_round(tag, entrants)
        rows += got
        with LOG.open("a") as f:
            for r in got:
                f.write(json.dumps(r) + "\n")
        for r in sorted(got, key=fitness):
            el = f"{r['elapsed']:.1f}s" if r["elapsed"] else "-"
            print(f"   {r['config']:26s} slot{r['slot']} completed={r['completed']} "
                  f"{el} stalls={r['stalls']}", flush=True)

    print("\nstandings (finish rate, then mean total of finishes):")
    for cfg, fin, n, mean in standings(rows):
        m = f"{mean:.1f}s" if mean != float("inf") else "-"
        print(f"  {cfg:26s} {fin}/{n} finishes   mean {m}")
    _ = seen
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
