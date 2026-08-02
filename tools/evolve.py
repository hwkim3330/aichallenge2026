#!/usr/bin/env python3
"""Search the MPC parameter space by racing three cars against each other, repeatedly.

Shape of the search
-------------------
One gene per generation, low and high value, each screened SOLO. The champion is a fixed
measurement re-taken periodically rather than re-run alongside every trial.

All three run at once, and the flags in screen3.sh are what makes that legitimate:
collisions off, handicap off. With collisions ON -- which is how the first three
generations here were run -- a bad challenger becomes a parked obstacle and destroys the
OTHER cars' measurements:

    generation 0   challengers 49 and 28 recoveries -> champion measured 19
    generation 1   challengers 59 and 95 recoveries -> champion measured 37
    generation 2   challengers 33 and 120 recoveries -> champion measured 17

The same champion scores 0 recoveries solo, so nothing in those runs was attributable to
the gene. With collisions off the cars pass through each other and each is effectively a
solo run, which is what the official 3-vehicle config does as well
(AWSIM_Data/StreamingAssets/RaceConfig/eval-3p.yaml). Handicap is off for the same reason
-- it couples the field deliberately -- and because eval.sh, which produced the 46.53 s
reference, has it off too.

So one 8-minute run yields three clean measurements: the champion as a same-conditions
control plus both ends of one gene.

What is actually being scored
-----------------------------
Not lap time. The board runs challenger vs ranker vs a "Computer" NPC, 6 required laps,
480 s timeout, and the rating moves on position -- victory/defended. Real battle records
show ranked teams finishing 4 laps or 0, so FINISHING is the discriminator and pace is
the tiebreak between cars that both finish. Fitness here is therefore lexicographic:
completed 6 laps first, total time second. Chasing best-lap pace has misled this project
twice; a candidate with 44.3-44.6 s clean laps loses to one running 46.6 s because a
single 62 s lap costs more than five fast laps gain.

State lives in tools/evolve_log.jsonl, one record per generation, so the loop resumes
after any interruption. Nothing here writes to config.yaml -- generated configs are
config_gen*.yaml and are disposable.

Usage
-----
    python3 tools/evolve.py --generations 8
    python3 tools/evolve.py --status
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros"
CFG_DIR = PKG / "config"
BASE_CFG = CFG_DIR / "config.yaml"
BASE_REF = CFG_DIR / "ref_vel.yaml"
LOG = ROOT / "tools/evolve_log.jsonl"

RACE_WALL = 900      # 600 s sim + startup; kill past this
SOLO_WALL = 700

# The genes, in the order they get searched. Ranges exclude regions this project has
# already measured as bad -- Q[0] x3 gave 42 recoveries, corridor 2.0 gave 20, ref_vel
# scale above 1.10 wedges -- so the loop does not spend races re-finding them.
GENES = [
    # R[1] (the steering-input weight) is deliberately absent. It is not a tunable: the
    # input reference `ur` is zero for that channel, so any R[1] > 0 is a penalty on
    # turning at all. Measured at both ends before this was reasoned through -- 3000 and
    # 30000 each produced zero completed laps and 112/115 recoveries.
    #
    # Untested axes first. The three at the bottom were run in generations 0-2 under the
    # contaminated 3-car screen and are kept for a clean solo re-check, not dropped --
    # except that steer_rate_max=1.40 already HAS a clean solo verdict (48.9 s median,
    # 23 recoveries against the champion's zero), so only its low side is still open.
    ("mpc.Q_epsi", 1.0e8, [3.0e7, 3.0e8], "float"),
    ("mpc.QN_epsi", 1.0e6, [1.0e3, 1.0e8], "float"),
    ("mpc.Q_ey", 1.0e6, [4.0e5, 2.0e6], "float"),
    ("mpc.Q_t", 1.85e6, [1.0e6, 3.0e6], "float"),
    ("reference_path.max_width", 3.0, [2.8, 4.0], "float"),
    ("ref_vel_scale", 1.00, [1.04, 1.08], "float"),
    ("mpc.wp_id_offset", 2, [1, 3], "int"),
    ("mpc.N", 20, [18, 24], "int"),
    ("mpc.steer_rate_max", 0.35, [0.25, 0.50], "float"),
]

CHAMPION0 = {name: base for name, base, _, _ in GENES}
# The submission's 46.53 s was measured in eval mode. solo6.sh is close but not the same
# script, so the baseline is re-measured under solo6 in generation 0 rather than assumed.
CHAMPION0_SOLO = None

# colcon was run with --symlink-install, so the installed config directory is per-file
# symlinks back into source. A newly written config is invisible to the launch until a
# link exists, and creating it directly is what a rebuild would do anyway -- worth
# avoiding a two-minute build per generation.
INSTALL_CFG = (ROOT / "aichallenge/workspace/install/multi_purpose_mpc_ros"
                    / "share/multi_purpose_mpc_ros/config")
IN_CONTAINER_CFG = ("/aichallenge/workspace/src/aichallenge_submit"
                    "/multi_purpose_mpc_ros/config")


def link_installed(name: str) -> None:
    link = INSTALL_CFG / name
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(f"{IN_CONTAINER_CFG}/{name}")


# ---------------------------------------------------------------- config generation

def write_config(dst: pathlib.Path, genome: dict) -> None:
    """Apply a genome to config.yaml with line edits.

    Deliberately textual rather than a yaml round-trip: config.yaml carries the measured
    history of this project in its comments, and every one of those comments is a test
    someone does not have to repeat.
    """
    text = BASE_CFG.read_text()

    def sub(pattern, repl, why):
        nonlocal text
        new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(f"{why}: pattern matched {n} times, expected 1")
        text = new

    sub(r"^(\s*steer_rate_max:\s*)[\d.]+",
        lambda m: f"{m.group(1)}{genome['mpc.steer_rate_max']}", "steer_rate_max")
    sub(r"^(\s*N:\s*)\d+", lambda m: f"{m.group(1)}{genome['mpc.N']}", "N")
    sub(r"^(\s*wp_id_offset:\s*)\d+",
        lambda m: f"{m.group(1)}{genome['mpc.wp_id_offset']}", "wp_id_offset")
    # Screening only. The submission keeps these on; here they are the last channel
    # through which one car's failure reaches another. --collisions off stops the cars
    # touching, but v2x still publishes their positions and the MPC plans around them,
    # so two wedged cars were still costing the control two 62 s laps and 5 recoveries
    # against its own 0 solo.
    sub(r"^(\s*use_path_constraints_topic:\s*)\w+",
        lambda m: f"{m.group(1)}false", "use_path_constraints_topic")
    sub(r"^(\s*use_border_cells_topic:\s*)\w+",
        lambda m: f"{m.group(1)}false", "use_border_cells_topic")
    sub(r"^(\s*max_width:\s*)[\d.]+",
        lambda m: f"{m.group(1)}{genome['reference_path.max_width']}", "max_width")
    sub(r"^(\s*Q:\s*)\[[^\]]*\]",
        lambda m: (f"{m.group(1)}[{genome['mpc.Q_ey']}, {genome['mpc.Q_epsi']}, "
                   f"{genome['mpc.Q_t']}]"), "Q")
    sub(r"^(\s*QN:\s*)\[[^\]]*\]",
        lambda m: f"{m.group(1)}[1000000.0, {genome['mpc.QN_epsi']}, 10000.0]", "QN")

    dst.write_text(text)


def write_ref_vel(dst: pathlib.Path, scale: float) -> None:
    """Scale every section speed uniformly.

    Uniform because per-section ceilings have failed three separate times in this
    project -- the sections interact through braking distance, so lifting one in
    isolation just moves the stall downstream.
    """
    text = BASE_REF.read_text()
    if abs(scale - 1.0) < 1e-9:
        dst.write_text(text)
        return
    # `ref_vel:` lines only. wp_id values are integers on their own lines, so they are
    # untouched either way, but being explicit keeps a layout change from silently
    # rescaling waypoint indices.
    out, n = re.subn(r"(?m)^(\s*ref_vel:\s*)([\d.]+)\s*$",
                     lambda m: f"{m.group(1)}{float(m.group(2)) * scale:.2f}", text)
    if n == 0:
        raise RuntimeError("ref_vel.yaml layout changed; no ref_vel: lines matched")
    dst.write_text(out)


# ---------------------------------------------------------------- running

def env_for(slot: int, cfg: str, ref: str, log_dir: str) -> dict:
    e = dict(os.environ)
    e.update(HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()),
             ROS_DOMAIN_ID=str(slot), LOG_DIR=log_dir,
             MPC_CONFIG_FILE=cfg, MPC_REF_VEL_FILE=ref,
             CONTROL_METHOD="mpc", RUN_MODE="awsim",
             USE_OBSTACLE_AVOIDANCE="false")
    e.setdefault("DISPLAY", ":0")
    return e


def compose(args, env, wait=True, timeout=None):
    p = subprocess.Popen(["docker", "compose"] + args, cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait:
        return p
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
    return p


def request_start() -> bool:
    """Release a sync-mode start.

    online3.sh runs --start-mode sync, where AWSIM holds every car until a single
    /admin/awsim/start arrives on Domain 0. Normally awsim_state_manager_node owns that
    topic; the plain `simulator` compose service does not run it, so nothing publishes it
    and the cars sit in Grounded forever -- the MPC commands 8.33 m/s while the vehicle
    reports 0.0, which is what the first two attempts here looked like. Per the interface
    contract this must be published exactly once, and never by a per-vehicle orchestrator.
    solo6.sh does not need it: --start-mode count releases itself.
    """
    env = dict(os.environ)
    env.update(HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()),
               ROS_DOMAIN_ID="0",
               CMD="env ROS_DOMAIN_ID=0 ros2 topic pub -1 /admin/awsim/start "
                   "std_msgs/msg/Bool '{data: true}'")
    env.setdefault("DISPLAY", ":0")
    p = compose(["run", "--rm", "-T", "--no-deps", "autoware-command"], env,
                timeout=120)
    return p.returncode == 0


def wait_grounded(host_out: pathlib.Path, slots, timeout=240) -> int:
    """Wait until each car's orchestrator has seen Grounded and requested control mode."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        n = 0
        for slot in slots:
            log = host_out / f"d{slot}" / "autoware.log"
            if log.exists() and "control mode request: success=True" in \
                    log.read_text(errors="replace"):
                n += 1
        if n == len(slots):
            return n
        time.sleep(5)
    return n


def sweep():
    """Remove every container this harness may have started.

    Belt and braces on the name filter: a compose-run container that outlives its client
    keeps publishing, and a second AWSIM on the same ROS domains is indistinguishable
    from a broken configuration in the logs.
    """
    subprocess.run(
        "docker ps -aq --filter name=evolve- | xargs -r docker rm -f; "
        "docker ps -aq --filter name=aichallenge2026-simulator | xargs -r docker rm -f",
        shell=True, capture_output=True)


def parse_laps(log_path: pathlib.Path):
    if not log_path.exists():
        return [], 0
    text = log_path.read_text(errors="replace")
    laps = [float(x) for x in
            re.findall(r"Lap \d+ completed! Lap time: ([\d.]+) s", text)]
    recov = len(re.findall(r"stuck detected: velocity=", text))
    return laps, recov


def race_outcome(log_path: pathlib.Path, timeout_s: float = 480.0) -> dict:
    """Score completion from when the race ended, not from lap messages.

    The MPC's "Lap N completed!" line for the FINAL lap races the orchestrator's shutdown
    and usually loses. Measured over five solo6/solo6lidar runs on 2026-08-02: the gap
    from lap 5 to the orchestrator's "latest link updated" was 46.0 s every time -- one
    lap -- races ended at 277.5-280.0 s against a 480 s timeout, and yet only one of the
    five logged its sixth lap, in the same second as the finalize. All five had finished.

    That is a coin flip landing on fitness()'s FIRST key, where finishing outranks any
    amount of pace, so a genuine six-lap finish was being ranked below a slower one that
    happened to get its last message out. Every completion figure in evolve_log.jsonl
    predates this and should be read with it in mind.

    Ending well inside the timeout is what proves the laps were run, and the elapsed time
    is then the six-lap total exactly -- better than summing messages.
    """
    laps, recov = parse_laps(log_path)
    if not laps:
        return dict(laps=[], recoveries=recov, completed=0, elapsed=None, finished=False)
    text = log_path.read_text(errors="replace")
    fin = re.findall(r"\[(\d+)\.\d+\].*latest link updated", text)
    lap_ts = [float(m) for m in re.findall(r"\[(\d+)\.\d+\].*Lap \d+ completed", text)]
    completed, elapsed = len(laps), None
    if fin:
        elapsed = float(fin[0]) - (lap_ts[0] - laps[0])
        gap = float(fin[0]) - lap_ts[-1]
        typical = sorted(laps)[len(laps) // 2]
        if 0.6 * typical < gap < 1.4 * typical:
            completed += 1
    finished = completed >= 6 and elapsed is not None and elapsed < timeout_s - 20
    return dict(laps=laps, recoveries=recov, completed=completed, elapsed=elapsed,
                finished=finished)


def scored_total(outcome):
    """Six-lap total for fitness.

    Summed laps when all six were logged, elapsed only to fill in for a lap the log never
    reported. Elapsed carries the orchestrator's finalize latency -- a few tenths that
    vary run to run -- and preferring it everywhere put that noise straight into the
    comparison between near-identical candidates. In generation 1 it was enough to swap
    two candidates that sat 0.8 s apart, which is a coin flip dressed as a result.
    """
    laps = outcome["laps"]
    if len(laps) >= 6:
        return total_6(laps)
    if outcome["finished"] and outcome["elapsed"]:
        return outcome["elapsed"]
    return total_6(laps)


def total_6(laps):
    """6-lap total, the scored quantity. Missing laps are charged at the worst lap seen.

    Charging rather than discarding: a config that only manages 4 laps inside the window
    is worse than one that manages 6, and dropping the missing laps would rank it better.
    """
    if not laps:
        return None
    laps = laps[:6]
    if len(laps) < 6:
        return sum(laps) + (6 - len(laps)) * max(laps)
    return sum(laps)


def clean_median(laps, best_overall):
    if not laps:
        return None
    good = sorted(L for L in laps if L <= best_overall * 1.25)
    if not good:
        return None
    n = len(good)
    return good[n // 2] if n % 2 else (good[n // 2 - 1] + good[n // 2]) / 2


def run_race(tag, genomes, slots):
    """One 3-car race. genomes[i] goes to grid slot slots[i]."""
    out = f"/output/{tag}"
    host_out = ROOT / "output" / tag
    sweep()                 # nothing stale may share the ROS domains
    procs = []
    env0 = dict(os.environ)
    env0.update(HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()),
                SIM_MODE="online3", LOG_DIR=out)
    env0.setdefault("DISPLAY", ":0")
    procs.append(compose(["run", "--rm", "-T", "--name", f"evolve-sim-{tag}",
                          "simulator"], env0, wait=False))
    time.sleep(8)

    for g, slot in zip(genomes, slots):
        cfg, ref = g["_cfg"], g["_ref"]
        e = env_for(slot, cfg, ref, out)
        procs.append(compose(["run", "--rm", "-T", "--name",
                              f"evolve-{tag}-d{slot}", "autoware"], e, wait=False))
        time.sleep(3)

    ready = wait_grounded(host_out, slots)
    if ready < len(slots):
        print(f"  경고: {ready}/{len(slots)}대만 준비됨 — 그래도 스타트 신호 전송")
    if not request_start():
        print("  스타트 신호 전송 실패")

    deadline = time.time() + RACE_WALL
    while time.time() < deadline:
        if procs[0].poll() is not None:
            break
        time.sleep(10)

    for p in procs:
        if p.poll() is None:
            p.kill()
    sweep()
    time.sleep(5)

    results = []
    for g, slot in zip(genomes, slots):
        o = race_outcome(host_out / f"d{slot}" / "autoware.log", timeout_s=600.0)
        results.append(dict(label=g["_label"], slot=slot, control=g["_control"], **o))
    best = min((L for r in results for L in r["laps"]), default=None)
    for r in results:
        r["clean_median"] = clean_median(r["laps"], best) if best else None
    return results


def run_screen(tag, genomes, slots):
    """One screen3 run: three independent cars, one measurement each."""
    out = f"/output/{tag}"
    host_out = ROOT / "output" / tag
    sweep()
    procs = []
    env0 = dict(os.environ)
    env0.update(HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()),
                SIM_MODE="screen3", LOG_DIR=out)
    env0.setdefault("DISPLAY", ":0")
    procs.append(compose(["run", "--rm", "-T", "--name", f"evolve-sim-{tag}",
                          "simulator"], env0, wait=False))
    time.sleep(8)
    for g, slot in zip(genomes, slots):
        e = env_for(slot, g["_cfg"], g["_ref"], out)
        procs.append(compose(["run", "--rm", "-T", "--name",
                              f"evolve-{tag}-d{slot}", "autoware"], e, wait=False))
        time.sleep(3)

    ready = wait_grounded(host_out, slots)
    if ready < len(slots):
        print(f"  경고: {ready}/{len(slots)}대만 준비됨")
    if not request_start():
        print("  스타트 신호 전송 실패")

    deadline = time.time() + RACE_WALL
    while time.time() < deadline and procs[0].poll() is None:
        time.sleep(10)
    for p in procs:
        if p.poll() is None:
            p.kill()
    sweep()
    time.sleep(5)

    out_rows = []
    for g, slot in zip(genomes, slots):
        o = race_outcome(host_out / f"d{slot}" / "autoware.log", timeout_s=480.0)
        out_rows.append(dict(label=g["_label"], slot=slot, control=g["_control"],
                             total6=scored_total(o), **o))
    return out_rows


def fitness(row):
    """Lexicographic: finishing six laps beats any amount of pace."""
    if row["total6"] is None:
        return (2, float("inf"))
    return (0 if row["finished"] else 1, row["total6"])


def run_solo(tag, genome):
    out = f"/output/{tag}"
    host_out = ROOT / "output" / tag
    sweep()
    env0 = dict(os.environ)
    env0.update(HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()),
                SIM_MODE="solo6", LOG_DIR=out)
    env0.setdefault("DISPLAY", ":0")
    sim = compose(["run", "--rm", "-T", "--name", f"evolve-sim-{tag}",
                   "simulator"], env0, wait=False)
    time.sleep(8)
    e = env_for(1, genome["_cfg"], genome["_ref"], out)
    aw = compose(["run", "--rm", "-T", "--name", f"evolve-{tag}-solo", "autoware"],
                 e, wait=False)
    deadline = time.time() + SOLO_WALL
    while time.time() < deadline and sim.poll() is None:
        time.sleep(10)
    for p in (sim, aw):
        if p.poll() is None:
            p.kill()
    sweep()
    time.sleep(5)
    laps, recov = parse_laps(host_out / "d1" / "autoware.log")
    best = min(laps, default=None)
    return dict(laps=laps, recoveries=recov,
                clean_median=clean_median(laps, best) if best else None)


# ---------------------------------------------------------------- the loop

def load_state():
    """Replay the log. Race-mode records from the first three generations are skipped --
    their fitness was clean-lap median under traffic and is not comparable."""
    champion, total, gen, gene_i = dict(CHAMPION0), None, 0, 0
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("mode") != "screen":
                continue
            gen = rec["generation"] + 1
            gene_i = rec["gene_index"] + 1
            if rec.get("promoted"):
                champion = rec["promoted"]["genome"]
                total = rec["promoted"]["total6"]
    return champion, total, gen, gene_i


def append(rec):
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def materialise(genome, name, is_control=False):
    g = dict(genome)
    g["_control"] = is_control
    g["_cfg"] = f"config_{name}.yaml"
    g["_ref"] = f"ref_vel_{name}.yaml"
    g["_label"] = name
    write_config(CFG_DIR / g["_cfg"], genome)
    write_ref_vel(CFG_DIR / g["_ref"], genome["ref_vel_scale"])
    link_installed(g["_cfg"])
    link_installed(g["_ref"])
    return g


def fmt(row):
    tot = row["total6"]
    return (f"{len(row['laps'])}랩 "
            f"{'완주' if row['finished'] else '미완주'} "
            f"6랩환산 {'--' if tot is None else format(tot, '.1f')} "
            f"복구 {row['recoveries']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    champion, champ_total, gen, gene_i = load_state()

    if a.status:
        print(f"세대 {gen}, 다음 유전자 {GENES[gene_i % len(GENES)][0]}")
        print(f"챔피언 6랩 {champ_total if champ_total else '미측정'}"
              f"   (참고: 학생부 1위 KSK 282.155)")
        for k, v in champion.items():
            mark = "" if v == CHAMPION0[k] else "   <-- 개선됨"
            print(f"  {k:32s} {v}{mark}")
        return 0

    for _ in range(a.generations):
        name, _base, (lo, hi), kind = GENES[gene_i % len(GENES)]
        cast = int if kind == "int" else float
        tag = f"evolve-g{gen:02d}"

        variants = [("champ", champion, True)]
        for suffix, val in (("lo", cast(lo)), ("hi", cast(hi))):
            if cast(champion[name]) == cast(val):
                continue
            child = dict(champion)
            child[name] = cast(val)
            variants.append((f"{name.split('.')[-1]}_{suffix}", child, False))
        while len(variants) < 3:
            variants.append((f"pad{len(variants)}", champion, True))

        genomes = [materialise(g, f"{tag}-{lbl}", ctl) for lbl, g, ctl in variants]
        slots = [(gen + i) % 3 + 1 for i in range(3)]
        print(f"\n=== 세대 {gen}: {name}  (챔피언 {champion[name]})  슬롯 {slots}")

        rows = run_screen(tag, genomes, slots)
        for r in rows:
            print(f"  {r['label'].split('-')[-1]:20s} 슬롯{r['slot']} {fmt(r)}"
                  f"{'  <- 대조군' if r['control'] else ''}")

        rec = dict(generation=gen, gene=name, gene_index=gene_i % len(GENES),
                   mode="screen", slots=slots, results=rows,
                   champion=dict(champion))

        ctrl = next((r for r in rows if r["control"] and r["total6"] is not None), None)
        if ctrl is None:
            print("  대조군 측정 실패 — 이 세대는 판정 불가")
            append(rec)
            gen += 1
            gene_i += 1
            continue

        # The control's own number, not the historical champion figure, is the reference:
        # it absorbs whatever this particular run's conditions were.
        ref = fitness(ctrl)
        print(f"  대조군 기준 {ref}")

        best, best_g = None, None
        for r, g in zip(rows, genomes):
            if r["control"]:
                continue
            f = fitness(r)
            if f < ref and (best is None or f < best):
                best, best_g = f, g

        if best is not None and (ref[1] - best[1] > 1.0 or best[0] < ref[0]):
            val = best_g[name]
            print(f"  -> {name} = {val} 이 앞섬 {best} vs {ref}, 재현 확인")
            conf = run_screen(f"{tag}-confirm",
                             [materialise(dict(champion), f"{tag}-confirm-champ", True),
                              materialise({k: v for k, v in best_g.items()
                                           if not k.startswith("_")},
                                          f"{tag}-confirm-cand", False),
                              materialise(dict(champion), f"{tag}-confirm-pad", True)],
                             [slots[1], slots[2], slots[0]])
            for r in conf:
                print(f"     {r['label'].split('-')[-1]:14s} {fmt(r)}")
            rec["confirm"] = conf
            c_ctrl = [r for r in conf if r["control"] and r["total6"] is not None]
            c_cand = next((r for r in conf if not r["control"]), None)
            if c_ctrl and c_cand and c_cand["total6"] is not None:
                cref = min(fitness(r) for r in c_ctrl)
                ccand = fitness(c_cand)
                if ccand < cref and (cref[1] - ccand[1] > 1.0 or ccand[0] < cref[0]):
                    champion = {k: v for k, v in best_g.items()
                                if not k.startswith("_")}
                    champ_total = ccand[1]
                    rec["promoted"] = dict(genome=dict(champion), total6=champ_total)
                    print(f"     채택: {name} = {val}, 6랩 {champ_total:.1f}초")
                else:
                    print(f"     재현 안 됨 ({ccand} vs 대조군 {cref}) — 기각")
            else:
                print("     확인 실행 측정 실패 — 기각")
        else:
            print("  -> 대조군을 1초 이상 넘은 후보 없음")

        append(rec)
        gen += 1
        gene_i += 1

    print(f"\n최종 챔피언" + (f" 6랩 {champ_total:.1f}초" if champ_total else ""))
    for k, v in champion.items():
        if v != CHAMPION0[k]:
            print(f"  {k} : {CHAMPION0[k]} -> {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
