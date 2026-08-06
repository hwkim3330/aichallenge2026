#!/usr/bin/env python3
"""Search the AI controller's free parameters by racing three cars at once, repeatedly.

Why this exists separately from evolve.py: the AI path has no config.yaml to mutate. Everything tunable about
it is an environment variable read by tiny_lidar_net_controller_node.py, so the genes are env values and the
weights file is a separate axis chosen beforehand.

Screening runs with COLLISIONS OFF, for the reason evolve.py records with numbers: with collisions on, a bad
challenger becomes a parked obstacle and destroys the other cars' measurements. Its generations 0-2 measured
the same champion at 19, 37 and 17 recoveries against zero solo. Three cars with collisions off are three
independent solo runs, which is also what the official 3-vehicle RaceConfig does. Final validation of the
winner then runs official3_600.sh with collisions ON, because that is what is scored.

What is being scored here is completion first and pace second, the same order the board uses: a car that does
not finish scores nothing regardless of its best lap. So fitness is (laps, -total), not lap time.

The genes are the four levers that made the difference between a car that never moved and one that drives:

  TLN_LAUNCH_SPEED   the floor applied only while the vehicle is essentially stopped. Without it the closed
                     loop traps itself: v_meas 0.00 gives 0 + 0.6*0.05 = 0.03 m/s forever, measured.
  TLN_LEAD_MARGIN    how far the setpoint may lead the MEASURED speed. Pure open loop climbs to the cap while
                     the car is held against a wall; pure closed loop plateaus, measured at 2.5 m/s.
  TLN_MAX_SPEED      the cap. The rival 42.28 s lap in config/pathfinders_accel_profile.json runs 6.9-9.25
                     m/s, so caps below about 7 cannot reach that pace at all.
  TLN_FIXED_ACCEL    the constant acceleration. Upstream commit eefb9ef set this to 0.6 deliberately, raising
                     it from 0.3, so 0.6 is the incumbent rather than an arbitrary starting point.

GUARD_FRONT_LIMIT and GUARD_SIDE_LIMIT are pinned to 0, not searched. lidar_guard is a node upstream does not
have, and with its shipped limits it pinned the vehicle at 2.6 m/s while the setpoint asked for 4.1. Inert is
the only setting under which the car drives at all, so it is a constant here, not a gene.

    python3 tools/evolve_ai.py --generations 4 --races 2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATE = ROOT / "tools" / "evolve_ai_state.json"

# name, incumbent, [low, high]
GENES = [
    ("TLN_MAX_SPEED", 10.0, [7.5, 12.0]),
    ("TLN_LEAD_MARGIN", 1.5, [0.75, 3.0]),
    ("TLN_LAUNCH_SPEED", 2.0, [1.0, 3.5]),
    ("TLN_FIXED_ACCEL", 0.6, [0.45, 0.9]),
]

# Not genes. See the module docstring.
FIXED = {
    "GUARD_FRONT_LIMIT": "0.0",
    "GUARD_SIDE_LIMIT": "0.0",
    "TLN_CONTROL_MODE": "fixed",
}


def genome_env(genome: dict) -> list[str]:
    out = []
    for k, v in {**FIXED, **{k: f"{v}" for k, v in genome.items()}}.items():
        out += ["--env", f"{k}={v}"]
    return out


def race(tag: str, genomes: list[dict], scenario: str, races: int) -> list[dict]:
    """Three cars, one genome each, screened together with collisions off."""
    cmd = ["python3", str(ROOT / "tools" / "eval3p_race.py"),
           "--races", str(races), "--scenario", scenario, "--cars", str(len(genomes)),
           "--batch", tag, "--name", tag.lower(),
           "--configs", ",".join(["config.yaml"] * len(genomes)),
           "--methods", ",".join(["tiny_lidar_net"] * len(genomes))]
    # eval3p_race applies --env globally, so per-slot genomes go through --slot-env.
    for i, g in enumerate(genomes, start=1):
        for k, v in {**FIXED, **g}.items():
            cmd += ["--slot-env", f"{i}:{k}={v}"]
    subprocess.run(cmd, cwd=ROOT, timeout=1800 * races * len(genomes), check=False)
    return read_results(tag, len(genomes))


def read_results(tag: str, n: int) -> list[dict]:
    import re
    rows = []
    for d in sorted((ROOT / "output").glob(f"eval3p{tag}-*")):
        for slot in range(1, n + 1):
            log = d / f"d{slot}" / "autoware.log"
            if not log.exists():
                continue
            t = log.read_text(errors="ignore")
            laps = [float(x) for x in re.findall(r"Lap \d+ completed! Lap time: ([\d.]+)", t)]
            rows.append({"run": d.name, "slot": slot, "laps": len(laps),
                         "total": sum(laps) if laps else None,
                         "best": min(laps) if laps else None})
    return rows


def fitness(rows: list[dict]) -> tuple:
    """Completion first, then total time. A car that does not finish scores nothing."""
    laps = max((r["laps"] for r in rows), default=0)
    totals = [r["total"] for r in rows if r["total"] is not None]
    return (laps, -(min(totals) if totals else 1e9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--races", type=int, default=2)
    ap.add_argument("--scenario", default="arena3",
                    help="screening scenario; arena3 is 3 cars with collisions OFF, which is what makes "
                         "three simultaneous measurements independent")
    args = ap.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() else {"champion": {n: b for n, b, _ in GENES},
                                                                  "history": []}
    champ = state["champion"]
    for gen in range(args.generations):
        name, _, (lo, hi) = GENES[gen % len(GENES)]
        trials = [dict(champ), dict(champ), dict(champ)]
        trials[1][name] = lo
        trials[2][name] = hi
        tag = f"AIG{gen}{name[4:8]}"
        print(f"[gen {gen}] gene {name}: champion {champ[name]} vs {lo} vs {hi}", flush=True)
        rows = race(tag, trials, args.scenario, args.races)
        by_slot = {s: [r for r in rows if r["slot"] == s] for s in (1, 2, 3)}
        scored = {s: fitness(v) for s, v in by_slot.items()}
        for s, val in ((1, champ[name]), (2, lo), (3, hi)):
            print(f"   slot {s} ({name}={val}): laps {scored[s][0]}, best total "
                  f"{-scored[s][1] if scored[s][1] > -1e8 else None}", flush=True)
        best_slot = max(scored, key=lambda s: scored[s])
        if best_slot != 1:
            champ[name] = lo if best_slot == 2 else hi
            print(f"   -> champion updated: {name} = {champ[name]}", flush=True)
        else:
            print("   -> champion held", flush=True)
        state["history"].append({"gen": gen, "gene": name, "scored": {str(k): v for k, v in scored.items()},
                                 "champion": dict(champ)})
        STATE.write_text(json.dumps(state, indent=2))
    print("final champion:", json.dumps(champ), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
