#!/usr/bin/env python3
"""Screen one MPC parameter against the corner excursion at wp 275-300.

Why not screen against the stall itself: it happens in about one run in six, so telling a
real improvement from noise would take tens of hours per candidate. The excursion that
precedes it is measured every lap. Four stall dumps put the car 2.9-4.4 m off line through
this band against a corridor ceiling of 3.0 m, and the baseline lap that hit 4.126 m is
exactly the 62.95 s lap in solo6lidar-25 -- the quantity tracks the failure.

Baseline, 21 laps over four runs on 2026-08-02:

    median 1.048   mean 1.257   max 4.126   over 2 m: 2 of 21

The median is not the target. Excursions around 1 m cost nothing, and one lap reached
2.568 m with a perfectly normal 46.42 s time. What matters is the tail, so the verdict
here is the fraction past 2 m and the maximum, with the median reported to catch a
candidate that simply drives further off line everywhere.

Runs solo6lidar because the scored eval-*p.yaml configs all set lidar: "on", and twelve
runs per arm showed the lidar costs nothing in pace (identical 46.19 medians), so there is
no reason to screen under the cheaper condition.
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
from evolve import (BASE_CFG, BASE_REF, CFG_DIR, ROOT, compose, env_for,  # noqa: E402
                    link_installed, race_outcome)
from lidar_load_ab import WALL, sweep  # noqa: E402

OUT = ROOT / "tools/corner_screen.jsonl"
SLOT = 1
BASELINE = dict(median=1.048, mean=1.257, mx=4.126, over2=2, n=21)


def write_variant(name: str, edits: dict[str, str]) -> str:
    """Copy config.yaml with one or more scalar fields replaced.

    Textual, like evolve.write_config, and for the same reason: config.yaml's comments are
    the measured history of this project and a yaml round-trip would drop them.
    """
    text = BASE_CFG.read_text()
    for key, value in edits.items():
        pattern = rf"^(\s*{re.escape(key)}:\s*)\S+"
        text, n = re.subn(pattern, lambda m: f"{m.group(1)}{value}", text,
                          count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(f"{key}: matched {n} times in config.yaml, expected 1")
    (CFG_DIR / name).write_text(text)
    link_installed(name)
    return name


def write_ref_variant(name: str, section: str, value: str) -> str:
    """Change one section's ref_vel, leaving the rest of the profile alone.

    ref_vel.yaml's header rules out raising per-section speeds for pace -- that failed three
    times, coarse and fine, with and without braking lead-ins. This is the opposite trade:
    LOWERING one section to buy stability, paying pace for it. Completion dominates the
    scoring, so that direction is worth its own test.
    """
    text = BASE_REF.read_text()
    pattern = rf"(^  {re.escape(section)}:\n    ref_vel: )[\d.]+"
    text, n = re.subn(pattern, lambda m: f"{m.group(1)}{value}", text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"{section}: matched {n} times in ref_vel.yaml, expected 1")
    (CFG_DIR / name).write_text(text)
    link_installed(name)
    return name


def peaks_from(log: pathlib.Path) -> list[float]:
    if not log.exists():
        return []
    return [float(x) for x in re.findall(
        r"corner peak \|e_y\| wp275-300: ([\d.]+)", log.read_text(errors="replace"))]


def one_run(tag: str, cfg: str, ref: str = "ref_vel.yaml") -> dict:
    out = f"/output/{tag}"
    host_out = ROOT / "output" / tag
    sweep()
    env0 = dict(env_for(SLOT, cfg, ref, out))
    env0.update(ROS_DOMAIN_ID="0", SIM_MODE="solo6lidar", LOG_DIR=out)
    procs = [compose(["run", "--rm", "-T", "--name", f"lidarab-sim-{tag}",
                      "simulator"], env0, wait=False)]
    time.sleep(8)
    procs.append(compose(["run", "--rm", "-T", "--name", f"lidarab-{tag}",
                          "autoware"], env_for(SLOT, cfg, ref, out),
                         wait=False))
    log = host_out / f"d{SLOT}" / "autoware.log"
    deadline = time.time() + WALL
    while time.time() < deadline:
        if procs[0].poll() is not None:
            break
        if log.exists() and "latest link updated" in log.read_text(errors="replace"):
            time.sleep(5)
            break
        time.sleep(5)
    for p in procs:
        if p.poll() is None:
            p.kill()
    sweep()
    time.sleep(5)
    o = race_outcome(log, timeout_s=480.0)
    return dict(tag=tag, config=cfg, peaks=peaks_from(log), laps=o["laps"],
                completed=o["completed"], finished=o["finished"],
                stalls=len(re.findall(r"STALL ANATOMY",
                                      log.read_text(errors="replace"))) if log.exists() else 0)


def verdict(peaks: list[float], label: str, laps: list[float] | None = None,
            stalls: int = 0, runs: int = 0) -> None:
    """Lap times first, corner peaks second.

    wp_id_offset=3 is why this is not a one-number screen: it cut every corner peak
    (0.46-1.89 against a 1.048 median and 4.126 max) while running 87.9 s and 104.3 s laps
    with five stalls per run. The band got tidier and the car got much worse, because the
    failure moved somewhere the band does not watch. A candidate has to clear lap time
    before its excursion figures mean anything.
    """
    if laps:
        steady = sorted(laps[1:]) or laps
        med = statistics.median(steady)
        # Stall rate belongs in the verdict, not just lap median. s6=18.0 kept a 46.89 s
        # median -- passing any lap-time test -- while producing four stalls in five runs
        # against the control's zero in two. Completion is the scored quantity.
        reasons = []
        if med > 50.0:
            reasons.append("laps regressed")
        if runs and stalls / runs > 0.5:
            reasons.append(f"stall rate {stalls}/{runs} runs")
        flag = f"  REJECT ({'; '.join(reasons)})" if reasons else ""
        print(f"{label}: laps median={med:.2f} worst={max(laps):.2f} "
              f"stalls={stalls} runs={runs}{flag}")
    if not peaks:
        print(f"{label}: no laps")
        return
    over2 = sum(1 for p in peaks if p > 2.0)
    print(f"{label}: n={len(peaks)} median={statistics.median(peaks):.3f} "
          f"mean={statistics.mean(peaks):.3f} max={max(peaks):.3f} "
          f"over2m={over2}/{len(peaks)}")
    print(f"{'baseline':>{len(label)}}: n={BASELINE['n']} median={BASELINE['median']:.3f} "
          f"mean={BASELINE['mean']:.3f} max={BASELINE['mx']:.3f} "
          f"over2m={BASELINE['over2']}/{BASELINE['n']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=["steer_rate_max=0.35"],
                    metavar="KEY=VALUE",
                    help="config.yaml scalar to override, e.g. wp_id_offset=3")
    ap.add_argument("--name", required=True, help="short label for this candidate")
    ap.add_argument("--set-ref", metavar="SECTION=VALUE",
                    help="ref_vel.yaml section speed to override, e.g. s6=18.0")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--batch", default="a",
                    help="suffix so a rerun does not collide with a previous batch's output dirs")
    args = ap.parse_args()

    edits = dict(kv.split("=", 1) for kv in args.set)
    cfg = write_variant(f"config_{args.name}.yaml", edits)
    ref = "ref_vel.yaml"
    if args.set_ref:
        sec, val = args.set_ref.split("=", 1)
        ref = write_ref_variant(f"ref_vel_{args.name}.yaml", sec, val)
    print(f"{args.name}: {edits} ref[{args.set_ref or '-'}] -> {cfg} / {ref}", flush=True)

    peaks: list[float] = []
    all_laps: list[float] = []
    all_stalls = 0
    valid_runs = 0
    for i in range(args.runs):
        tag = f"corner-{args.name}{args.batch}-{i:02d}"
        print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
        row = one_run(tag, cfg, ref)
        peaks += row["peaks"]
        all_laps += row["laps"]
        all_stalls += row["stalls"]
        valid_runs += bool(row["laps"])   # empty runs are start-up failures, not samples
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  laps={[round(x, 2) for x in row['laps']]} completed={row['completed']} "
              f"stalls={row['stalls']} peaks={[round(p, 2) for p in row['peaks']]}",
              flush=True)
    verdict(peaks, args.name, laps=all_laps, stalls=all_stalls, runs=valid_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
