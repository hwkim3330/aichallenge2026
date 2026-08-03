#!/usr/bin/env python3
"""Run the official three-car evaluation condition and score every car.

Everything measured on 2026-08-02/03 was a single car. The board runs eval-3p.yaml: three
vehicles, sync start, collisions off, handicap ON, lidar on, six laps, 600 s. Handicap is
the part never tested -- screen3.sh turns it off on purpose so one bad car cannot spoil the
others' numbers, and eval.sh has it off too, so nothing in this project has ever run with
the field coupled the way the board couples it.

Collisions being off means the cars pass through each other, so this is not about contact.
It is about whether handicap changes the pace or the stall behaviour.

All three cars run the submitted configuration, which makes this three samples of the
shipped setup under the real condition rather than a comparison.

Reuses evolve.py's plumbing: sync-mode starts need exactly one /admin/awsim/start on
domain 0, and killing a compose-run client does not stop its container.
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
from evolve import (CFG_DIR, ROOT, compose, env_for, race_outcome,  # noqa: E402
                    request_start, wait_grounded)
from corner_screen import write_ref_variant, write_variant  # noqa: E402

OUT = ROOT / "tools/eval3p_log.jsonl"
SLOTS = [1, 2, 3]
WALL = 1000          # 600 s race plus AWSIM start-up for three Autoware instances


def sweep() -> None:
    import subprocess
    for _ in range(12):
        subprocess.run(
            "docker ps -aq --filter name=eval3p- | xargs -r docker rm -f; "
            "docker ps -aq --filter name=aichallenge2026-simulator | xargs -r docker rm -f",
            shell=True, capture_output=True)
        left = subprocess.run("docker ps -aq --filter name=eval3p-", shell=True,
                              capture_output=True, text=True).stdout.strip()
        if not left:
            return
        time.sleep(5)
    raise RuntimeError("containers from a previous run would not die; refusing to measure")


def one_race(tag: str, cfg: str = "config.yaml",
             ref: str = "ref_vel.yaml") -> list[dict]:
    out = f"/output/{tag}"
    host_out = ROOT / "output" / tag
    sweep()
    env0 = dict(env_for(SLOTS[0], cfg, ref, out))
    env0.update(ROS_DOMAIN_ID="0", SIM_MODE="eval3p", LOG_DIR=out)
    procs = [compose(["run", "--rm", "-T", "--name", f"eval3p-sim-{tag}",
                      "simulator"], env0, wait=False)]
    time.sleep(10)
    for slot in SLOTS:
        procs.append(compose(["run", "--rm", "-T", "--name", f"eval3p-{tag}-d{slot}",
                              "autoware"],
                             env_for(slot, cfg, ref, out), wait=False))
        time.sleep(4)

    ready = wait_grounded(host_out, SLOTS)
    if ready < len(SLOTS):
        print(f"  warning: {ready}/{len(SLOTS)} cars ready; sending start anyway", flush=True)
    if not request_start():
        print("  start signal failed", flush=True)

    deadline = time.time() + WALL
    while time.time() < deadline:
        if procs[0].poll() is not None:
            break
        done = sum(
            1 for s in SLOTS
            if (host_out / f"d{s}" / "autoware.log").exists()
            and "latest link updated" in
            (host_out / f"d{s}" / "autoware.log").read_text(errors="replace"))
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
    for slot in SLOTS:
        log = host_out / f"d{slot}" / "autoware.log"
        o = race_outcome(log, timeout_s=600.0) if log.exists() else dict(
            laps=[], recoveries=0, completed=0, elapsed=None, finished=False)
        text = log.read_text(errors="replace") if log.exists() else ""
        rows.append(dict(
            tag=tag, slot=slot, laps=o["laps"], completed=o["completed"],
            elapsed=o["elapsed"], finished=o["finished"],
            stalls=len(re.findall(r"STALL ANATOMY", text)),
            lap_peaks=[float(x) for x in re.findall(
                r"lap peak \|e_y\|: ([\d.]+) m at wp=\d+", text)],
            peak_wps=[int(x) for x in re.findall(
                r"lap peak \|e_y\|: [\d.]+ m at wp=(\d+)", text)]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", type=int, default=2)
    ap.add_argument("--batch", default="a")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--set-ref", metavar="SECTION=VALUE",
                    help="ref_vel section override, e.g. s1=20.0")
    ap.add_argument("--name", default="base")
    args = ap.parse_args()

    cfg, ref = "config.yaml", "ref_vel.yaml"
    if args.set:
        cfg = write_variant(f"config_3p_{args.name}.yaml", dict(kv.split("=", 1) for kv in args.set))
    if args.set_ref:
        sec, val = args.set_ref.split("=", 1)
        ref = write_ref_variant(f"ref_vel_3p_{args.name}.yaml", sec, val)
    print(f"config={cfg} ref={ref}", flush=True)

    all_rows = []
    for i in range(args.races):
        tag = f"eval3p{args.batch}-{args.name}-{i:02d}"
        print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
        rows = one_race(tag, cfg, ref)
        all_rows += rows
        with OUT.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        for r in rows:
            el = f"{r['elapsed']:.1f}s" if r["elapsed"] else "-"
            print(f"  d{r['slot']}: completed={r['completed']} elapsed={el} "
                  f"stalls={r['stalls']} laps={[round(x, 2) for x in r['laps']]}", flush=True)

    # Per-slot comparison against whatever baseline races are already in the log. All three
    # cars run the same config, so rotating grid assignment would change nothing; what does
    # remove the grid confound is comparing slot to slot. The baseline is already measured
    # at 363/355 in slot 1, 284/285 in slot 2 and 353/350 in slot 3.
    if args.name != "base" and OUT.exists():
        base = [json.loads(l) for l in OUT.read_text().splitlines()]
        base = [r for r in base if "-base-" in r["tag"] and r["laps"]]
        if base:
            print("\nper-slot, candidate against baseline:")
            for slot in SLOTS:
                b = [r["elapsed"] for r in base if r["slot"] == slot and r["elapsed"]]
                c = [r["elapsed"] for r in all_rows if r["slot"] == slot and r["elapsed"]]
                if b and c:
                    print(f"  slot {slot}: baseline {[round(x, 1) for x in b]} -> "
                          f"candidate {[round(x, 1) for x in c]}  "
                          f"delta {statistics.mean(c) - statistics.mean(b):+.1f}s")

    good = [r for r in all_rows if r["laps"]]
    if good:
        tot = [r["elapsed"] for r in good if r["elapsed"]]
        print(f"\ncars={len(good)} six-lap finishes={sum(r['finished'] for r in good)}/{len(good)}"
              + (f" race median={statistics.median(tot):.1f}s" if tot else "")
              + f" stalls={sum(r['stalls'] for r in good)}")
        peaks = [p for r in good for p in r["lap_peaks"]]
        if peaks:
            print(f"lap peak |e_y|: n={len(peaks)} median={statistics.median(peaks):.3f} "
                  f"max={max(peaks):.3f}")
            wps = [w for r in good for w in r["peak_wps"]]
            from collections import Counter
            print("where the peak lands: " + ", ".join(
                f"wp{w}x{c}" for w, c in Counter(wps).most_common(8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
