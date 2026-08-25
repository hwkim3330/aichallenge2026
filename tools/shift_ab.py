#!/usr/bin/env python3
"""A/B the local line shift at the wp 131-139 wall, scored on a per-lap metric.

The event being fixed is rare -- one wall contact per race -- so scoring on the event
itself would need dozens of races. It does not have to be scored that way. The contact is
the tail of a distribution that is measurable on every single lap: how far left of the
line the car gets through wp 128-142. The clean laps sit at +1.0 to +1.6 m, the contact lap
reached +2.7 m, and the wall is 3.7-3.9 m out. Six samples per race instead of one.

So the primary quantity is the per-lap peak offset, and lap time is the guard: a line that
buys margin by being slower is not worth having. Both are reported per arm, and the shift
amplitudes are run as a dose so a real effect has to show up as an ordering rather than a
single lucky arm.

Arms are interleaved rather than run in blocks; on this machine a run's neighbours matter
(thermal, disk) and blocking would confound arm with time.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evolve import ROOT, compose, env_for, race_outcome  # noqa: E402

OUT = ROOT / "tools/shift_ab.jsonl"
ENV = (ROOT / "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros"
       / "env/final_ver3")
SLOT = 1
WALL = 900
WP_LO, WP_HI = 128, 142


def sweep() -> None:
    import subprocess
    for _ in range(12):
        subprocess.run(
            "docker ps -aq --filter name=shiftab- | xargs -r docker rm -f; "
            "docker ps -aq --filter name=aichallenge2026-simulator | xargs -r docker rm -f",
            shell=True, capture_output=True)
        left = subprocess.run("docker ps -aq --filter name=shiftab-", shell=True,
                              capture_output=True, text=True).stdout.strip()
        if not left:
            return
        time.sleep(5)
    raise RuntimeError("containers from a previous run would not die; refusing to measure")


def lap_peaks(bag: pathlib.Path, csv: pathlib.Path) -> list[float]:
    """Peak signed offset per lap through the hotspot, + = left, the side that hits.

    Offsets are measured against the SHIPPED line for every arm. A shifted arm compared
    against its own line would report the tracking error, which is not the question; the
    question is where the car ends up relative to the wall, and the wall does not move.
    """
    from rosbags.highlevel import AnyReader
    traj = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=(1, 2))
    with AnyReader([bag]) as r:
        conns = [c for c in r.connections if c.topic == "/localization/kinematic_state"]
        P = []
        for con, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, con.msgtype)
            p = m.pose.pose.position
            P.append((p.x, p.y))
    if len(P) < 100:
        return []
    P = np.array(P)
    d = np.hypot(P[:, 0][:, None] - traj[None, :, 0], P[:, 1][:, None] - traj[None, :, 1])
    wp = d.argmin(1)
    tan = np.roll(traj, -1, 0) - traj
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    off = P - traj[wp]
    lat = d.min(1) * np.sign(tan[wp, 0] * off[:, 1] - tan[wp, 1] * off[:, 0])
    lap_id = np.cumsum(np.r_[0, (np.diff(wp.astype(int)) < -200).astype(int)])
    peaks = []
    for L in range(lap_id.max() + 1):
        sel = (lap_id == L) & (wp >= WP_LO) & (wp <= WP_HI)
        if sel.sum() >= 3:
            peaks.append(float(lat[sel].max()))
    return peaks


def one_run(tag: str, cfg: str) -> dict:
    out = f"/output/{tag}"
    host_out = ROOT / "output" / tag
    sweep()
    env0 = dict(env_for(SLOT, cfg, "ref_vel.yaml", out))
    env0.update(ROS_DOMAIN_ID="0", SIM_MODE="solo6", LOG_DIR=out, ROSBAG="1")
    procs = [compose(["run", "--rm", "-T", "--name", f"shiftab-sim-{tag}",
                      "simulator"], env0, wait=False)]
    time.sleep(8)
    env1 = dict(env_for(SLOT, cfg, "ref_vel.yaml", out))
    env1.update(ROSBAG="1")
    procs.append(compose(["run", "--rm", "-T", "--name", f"shiftab-{tag}",
                          "autoware"], env1, wait=False))
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
    text = log.read_text(errors="replace") if log.exists() else ""
    bag = host_out / f"d{SLOT}" / "rosbag2_autoware"
    peaks = lap_peaks(bag, ENV / "traj_top36_blend45.csv") if bag.exists() else []
    return dict(tag=tag, config=cfg, laps=o["laps"], completed=o["completed"],
                finished=o["finished"], peaks=peaks,
                stalls=len(re.findall(r"STALL ANATOMY", text)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch", default="a")
    ap.add_argument("--arms", default="config.yaml,config_shift_r06.yaml,config_shift_r09.yaml")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",")]

    rows = []
    for i in range(args.rounds):
        for cfg in arms:
            name = cfg.replace("config", "").replace(".yaml", "").strip("_") or "base"
            tag = f"shift{args.batch}-{name}-{i:02d}"
            print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
            r = one_run(tag, cfg)
            rows.append(r)
            with OUT.open("a") as f:
                f.write(json.dumps(r) + "\n")
            laps = r["laps"] or []
            print(f"    laps {len(laps)} total {sum(laps):7.2f}  "
                  f"peaks {' '.join('%.2f' % p for p in r['peaks'])}  stalls {r['stalls']}",
                  flush=True)

    print("\narm            runs  6-lap total   median lap   peak offset (median / max)  stalls")
    for cfg in arms:
        sel = [r for r in rows if r["config"] == cfg]
        tot = [sum(r["laps"]) for r in sel if r["completed"] >= 6]
        allp = [p for r in sel for p in r["peaks"]]
        med = [statistics.median(r["laps"][1:]) for r in sel if len(r["laps"]) > 2]
        print(f"{cfg:28s} {len(sel):2d}  "
              f"{statistics.mean(tot) if tot else float('nan'):9.2f}  "
              f"{statistics.mean(med) if med else float('nan'):10.2f}  "
              f"{statistics.median(allp) if allp else float('nan'):14.2f} / "
              f"{max(allp) if allp else float('nan'):.2f}  "
              f"{sum(r['stalls'] for r in sel):5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
