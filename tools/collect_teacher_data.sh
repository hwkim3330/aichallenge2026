#!/bin/bash
# Record one clean solo run as Track 2 teacher data, bounded.
#
# Why solo and not a race: the npc-with-* race bags are 68% frames below 0.5 m/s
# (the car wedged), and behavioural cloning on those teaches the net to stand
# still. solo6lidar has no NPC to wedge against, so every frame is a good label.
# See tools/GOAL.md, "레이스 bag 은 학습 데이터로 거의 못 쓴다".
#
# solo6lidar.sh is the right mode: --lidar cpu (the model's input exists),
# --timeout 480 (what the scored battles allow), wall-recovery off, 6 laps.
#
# Everything is bounded. An unbounded simulator run has burned whole sessions
# here before, and this box's NVMe dies under sustained load.
#
# Usage: tools/collect_teacher_data.sh [run_label] [wait_seconds]
set -uo pipefail

ROOT=/home/kim/aichallenge2026
cd "$ROOT"

LABEL=${1:-teacher_$(date +%Y%m%d-%H%M%S)}
# Container path the recorder writes to, and the same place on the host.
# These must stay in step: the stall check below used to watch a hardcoded
# rawdata/ while REC_DIR pointed elsewhere, so it saw no growth and aborted
# every run after two minutes.
REC_DIR=${REC_DIR:-/aichallenge/ml_workspace/rawdata}
REC_DIR_HOST=$ROOT${REC_DIR#/aichallenge}
REC_DIR_HOST=${REC_DIR_HOST/#$ROOT\/ml_workspace/$ROOT\/aichallenge\/ml_workspace}
WAIT_S=${2:-620}          # 480 s race + ~30 s AWSIM boot + margin
DOMAIN=1                  # AWSIM bridges vehicle 1's topics onto domain 1
LOG_DIR=/output/$LABEL
HOST_LOG_DIR=$ROOT/output/$LABEL

log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
    log "tearing down"
    docker compose exec -T autoware bash -lc 'pkill -INT -f "ros2 bag record"' >/dev/null 2>&1 || true
    sleep 3
    make down >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

mkdir -p "$HOST_LOG_DIR"

# Baseline for the race-end check below; the file survives between sessions.
SUMMARY=$ROOT/aichallenge/result-summary.json
BASE_MTIME=$(stat -c %Y "$SUMMARY" 2>/dev/null || echo 0)

# SIM_MODE selects the AWSIM launch script. solo6lidar (default) is 1 vehicle +
# LiDAR, the right shape for teacher data. npc1 adds the NPC that every scored
# battle includes, which is the condition where lap progress collapses.
MODE=${SIM_MODE:-solo6lidar}
log "starting AWSIM (mode=$MODE)"
LOG_DIR=$LOG_DIR SIM_MODE=$MODE ROS_DOMAIN_ID=0 \
    docker compose up -d simulator >/dev/null 2>&1 || { log "simulator up failed"; exit 1; }

log "starting Autoware on domain $DOMAIN (control_method=${CONTROL_METHOD:-default})"
LOG_DIR=$LOG_DIR RUN_MODE=awsim CONTROL_METHOD=${CONTROL_METHOD:-} \
    docker compose up -d autoware >/dev/null 2>&1 \
    || { log "autoware up failed"; exit 1; }

# Wait for AWSIM to reach Start, read from Unity's own Player.log.
#
# Two earlier probes failed here. `ros2 topic list | grep scan` matched 5 s in
# because that command lists a topic with only a subscriber, and Autoware had
# already subscribed while AWSIM was still booting -- 8 KB bag. Replacing it with
# `ros2 topic echo --once` then timed out for 240 s even though AWSIM was
# perfectly healthy: the shell `docker compose exec` gives us does not carry the
# RMW/cyclonedds environment the real nodes run with, and stderr was being sent
# to /dev/null so the reason was invisible.
#
# Player.log needs no ROS environment at all -- AWSIM writes "state -> Start"
# there once the vehicle is Ready and the countdown has elapsed. Do not silence
# stderr on the probe.
PLAYER_LOG=/tmp/.config/unity3d/TIERIV/AWSIM/Player.log
log "waiting for AWSIM to reach Start (max 240 s)"
sim_up=0
for _ in $(seq 1 80); do
    if docker compose exec -T simulator bash -lc \
        "grep -q 'state → Start' $PLAYER_LOG 2>/dev/null"; then
        log "AWSIM reached Start"
        sim_up=1
        break
    fi
    sleep 3
done
if (( sim_up == 0 )); then
    log "ERROR: AWSIM did not reach Start in 240 s -- refusing to record an empty bag"
    log "last 15 lines of Player.log:"
    docker compose exec -T simulator bash -lc "tail -15 $PLAYER_LOG" 2>&1 | sed 's/^/    /'
    exit 1
fi

if [[ ${RECORD:-1} == 0 ]]; then
    log "recording disabled (RECORD=0) -- no telemetry will exist for this run"
else
    # An AI-driven run must not land in ml_workspace/rawdata/: its bag is
    # structurally identical to a teacher bag, so a later
    # `extract --bags-dir rawdata` would silently train the model on its own
    # commands. But recording nothing costs the diagnostics -- the 2026-08-02 BC
    # test scored 0 laps and there was no telemetry to say whether the car never
    # moved or drove off the line. So send non-MPC runs to a separate directory
    # instead of switching recording off.
    log "starting recorder -> $REC_DIR (log: $HOST_LOG_DIR/recorder.log)"
    # Keep the recorder's own output. The first attempts sent it to /dev/null, which
    # is how an empty bag got mistaken for a successful capture.
    docker compose exec -T -d autoware bash -lc \
        "ROS_DOMAIN_ID=$DOMAIN TLN_REC_DIR=$REC_DIR /aichallenge/ml_workspace/record_data.bash > $LOG_DIR/recorder.log 2>&1" \
        || log "WARNING: recorder launch returned non-zero"
fi

log "running for up to ${WAIT_S}s"
deadline=$((SECONDS + WAIT_S))
last_size=0
stall_checks=0
while (( SECONDS < deadline )); do
    # Stop when the race ends, only if asked to.
    #
    # The summary lands at $ROOT/aichallenge/result-summary.json (the repo dir is
    # /aichallenge inside the container), NOT under LOG_DIR. A previous version
    # tested it with no baseline and matched a leftover from an earlier session in
    # the same second the run started; the version after that tested LOG_DIR and so
    # never matched anything. Compare against the mtime captured before launch.
    #
    # Default is off: AWSIM keeps the car driving after the 6 laps are scored, and
    # for teacher data that overrun is free clean labels. The 2026-08-01 run got
    # 14,093 usable samples in 705 s where stopping at the flag would have given
    # roughly a third of that. Turn it on for Track 1 screening runs, where the
    # laps are the measurement and the overrun is waste.
    if [[ ${STOP_AT_RACE_END:-0} != 0 ]]; then
        now_mtime=$(stat -c %Y "$SUMMARY" 2>/dev/null || echo 0)
        if [[ -s $SUMMARY && $now_mtime -gt $BASE_MTIME ]]; then
            log "race finished (result-summary.json updated)"
            break
        fi
    fi
    # Guard against the other silent failure: recorder alive but capturing nothing.
    size=$(du -sk "$REC_DIR_HOST" 2>/dev/null | cut -f1)
    size=${size:-0}
    # Only meaningful while recording; with RECORD=0 the bag never grows and this
    # would abort every run after two minutes.
    if [[ ${RECORD:-1} != 0 ]]; then
        if (( size <= last_size )); then
            stall_checks=$((stall_checks + 1))
            if (( stall_checks >= 8 )); then
                log "ERROR: bag has not grown in ~2 min (${size} KB) -- aborting"
                break
            fi
        else
            stall_checks=0
        fi
        last_size=$size
    fi
    sleep 15
done

log "stopping recorder cleanly (SIGINT, so the mcap is closed and readable)"
docker compose exec -T autoware bash -lc 'pkill -INT -f "ros2 bag record"' >/dev/null 2>&1 || true
sleep 8

# Preserve this run's score. AWSIM writes it to aichallenge/result-summary.json and
# the next run overwrites it, which is why the 2026-07-31 npc-with/npc-without races
# left no scores behind and had to be reconstructed from their bags.
if [[ -s $SUMMARY ]]; then
    now_mtime=$(stat -c %Y "$SUMMARY" 2>/dev/null || echo 0)
    if (( now_mtime > BASE_MTIME )); then
        cp -f "$SUMMARY" "$HOST_LOG_DIR/result-summary.json"
        log "saved result-summary.json to $HOST_LOG_DIR"
    else
        log "no new result-summary.json (race did not finish)"
    fi
fi

BAGS=$REC_DIR_HOST
log "bags now in $BAGS:"
ls -1t "$BAGS" 2>/dev/null | head -5
du -sh "$BAGS" 2>/dev/null

log "done — extract with:"
echo "  cd $ROOT/aichallenge/ml_workspace/tiny_lidar_net && \\"
echo "    python3 extract_data_from_bag.py --bags-dir $BAGS \\"
echo "      --outdir $ROOT/aichallenge/ml_workspace/dataset/$LABEL --min-speed 2.0"
