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

Fitness gates on completing at least a lap and then ranks on the BEST FLYING LAP, not on whole-run mean
speed and not on total time. Measured reason: the same champion configuration returned mean speeds of
3.62, 4.27 and 5.89 m/s across three generations, because one wall contact drags a whole-run mean down and
that is luck rather than pace. The best flying lap over the same runs is stable to a quarter of a second.

The genes are the terms of the longitudinal loop, which is what actually sets pace here:

  TLN_STEER_SLOWDOWN how much speed to give up per radian of steering. This is the whole cornering policy:
                     v_des = max_speed - slowdown*|steer|, floored at corner_speed.
  TLN_MAX_SPEED      the straight-line cap.
  TLN_CORNER_SPEED   the floor that slowdown may not go below.
  TLN_LON_KP         the gain from speed error to commanded acceleration.

TLN_FIXED_ACCEL is deliberately NOT a gene: with a constant acceleration the vehicle simply equilibrates
against drag near 4.9 m/s regardless of the speed setpoint, so searching the constant cannot lift the
ceiling. TLN_LAUNCH_SPEED and TLN_LEAD_MARGIN are pinned for the same reason -- they matter only to the
old open-loop setpoint path, which the acceleration loop supersedes.

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
#
# TLN_FIXED_ACCEL is gone as a gene, and the reason is the whole point of this file's second revision:
# the vehicle tracks cmd.acceleration, not cmd.speed, so a CONSTANT acceleration equilibrates against
# drag at about 4.9 m/s no matter what the speed setpoint says. Measured on one track and clock, this
# node commanded speed 9.19 with acceleration pinned at 0.600 and reached 3.73 m/s, while the MPC
# commanded 7.94 with acceleration ranging -1.60..+1.35 and reached 6.97. Searching the constant only
# moves the equilibrium a little; closing the loop moves the car, so the genes are now the terms of that
# loop. With TLN_LON_KP=1.0 the same policy runs at 7.0-7.4 m/s.
GENES = [
    ("TLN_STEER_SLOWDOWN", 8.0, [4.0, 14.0]),
    ("TLN_MAX_SPEED", 9.5, [8.5, 11.0]),
    ("TLN_CORNER_SPEED", 4.5, [3.5, 6.0]),
    ("TLN_LON_KP", 1.0, [0.6, 2.0]),
]

# Not genes. See the module docstring.
FIXED = {
    "GUARD_FRONT_LIMIT": "0.0",
    "GUARD_SIDE_LIMIT": "0.0",
    "TLN_CONTROL_MODE": "fixed",
    "TLN_LAUNCH_SPEED": "2.0",
    "TLN_LEAD_MARGIN": "6.0",
    # REQUIRED. Fitness is read from the recorded pose, so without a bag every slot scores 0.00 and
    # every generation "holds the champion" for want of data -- which is exactly what the first run of
    # this file did, across four generations, while the cars were driving perfectly well.
    "ROSBAG": "1",
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
    """Score on MEASURED raceline progress, not on the log's lap lines.

    "Lap N completed!" comes from multi_purpose_mpc_ros/mpc_controller.py, i.e. from the MPC node, so
    it NEVER appears on the AI path. Reading it here reported 0 laps for a car that had driven 3.42 laps
    (1160 m in 304 s, forward only), which sent the whole AI effort after a steering problem that did not
    exist. Laps are counted officially by AWSIM, not by the controller, so the honest local proxy is the
    distance the recorded pose actually covered along the raceline.
    """
    import subprocess
    rows = []
    for d in sorted((ROOT / "output").glob(f"eval3p{tag}-*")):
        for slot in range(1, n + 1):
            bag = d / f"d{slot}" / "rosbag2_autoware"
            if not bag.exists():
                continue
            out = subprocess.run(["python3", str(ROOT / "tools" / "ai_progress.py"), str(bag)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=1200).stdout
            laps = speed = best = None
            for line in out.splitlines():
                if "laps)" in line:
                    laps = float(line.split("(")[1].split()[0])
                elif "speed: mean" in line:
                    speed = float(line.split("mean")[1].split()[0])
                elif "best flying" in line:
                    best = float(line.split("best flying")[1].split()[0])
            if laps is None:
                continue
            rows.append({"run": d.name, "slot": slot, "laps": laps, "speed": speed,
                         "best": best})
    return rows


def fitness(rows: list[dict]) -> tuple:
    """Laps as a gate, then BEST FLYING LAP. Mean speed is too noisy to rank on.

    Measured: the same champion configuration returned mean speeds of 3.62, 4.27 and 5.89 m/s across
    three generations, because a single wall contact or slow recovery drags the whole-run mean down and
    those events are luck. The best flying lap from the same runs is stable -- LON1's six laps came in at
    48.40 48.25 48.25 48.25 48.16, a 0.24 s spread -- because it measures pace where the car is actually
    running. So pace is judged on the flying lap, and the laps figure only gates out a car that failed to
    get round at all.
    """
    if not rows:
        return (0.0, -1e9)
    laps = max(r["laps"] for r in rows)
    bests = [r["best"] for r in rows if r.get("best")]
    return (1.0 if laps >= 1.5 else 0.0, -(min(bests) if bests else 1e9))


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
            got = -scored[s][1]
            print(f"   slot {s} ({name}={val}): "
                  f"{'ran' if scored[s][0] else 'DID NOT LAP'}, best flying "
                  f"{got:.2f} s" if got < 1e8 else
                  f"   slot {s} ({name}={val}): no flying lap", flush=True)
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
