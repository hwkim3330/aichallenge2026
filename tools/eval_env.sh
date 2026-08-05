#!/bin/bash
# Run the official evaluation N times under a given environment, and report the penalty breakdown.
#
# Usage: eval_env.sh <label> <N> [VAR=VAL ...]
#
# No image rebuild is needed for the recovery flags. stuck_recovery_controller reads them with
# std::getenv and docker-compose injects them into the container, so they take effect on the
# evaluation path even though run_evaluation.bash never mentions them -- confirmed by finding
# RECOVERY_DIRECTED in the strings of the installed node inside aichallenge-2025-eval.
#
# Config changes are NOT switchable this way: run_evaluation.bash passes no mpc_config_file, so a
# different config means rebuilding the tarball and the image.
#
# The container keeps driving after the six laps are scored, so waiting for it to exit wasted about
# twelve minutes a run. Here the loop waits for d1-result-details.json instead and then removes it.
set -uo pipefail
cd /home/kim/aichallenge2026
LABEL=${1:?label}
N=${2:?count}
shift 2
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "$LABEL: $N runs with env: ${*:-<none>}"

for i in $(seq 1 "$N"); do
    make down >/dev/null 2>&1
    docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
    sleep 5
    env "$@" make eval > /dev/null 2>&1
    sleep 25
    D=$(ls -dt output/*/ | head -1)
    for t in $(seq 1 60); do
        [ -f "$D/d1/d1-result-details.json" ] && break
        docker ps --format '{{.Names}}' | grep -q simulator-evaluation || break
        sleep 15
    done
    sleep 3
    docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
    python3 - "$LABEL" "$i" "$D" <<'PY'
import json, pathlib, sys
label, i, d = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
f = next(iter(d.glob("d1/*result-details.json")), None)
if f is None:
    print(f"  {label} run {i}: NO RESULT -- did not finish six laps"); raise SystemExit
r = json.loads(f.read_text())
pk = r.get("penalty_by_kind", {})
n = lambda k: pk.get(k, {}).get("count", 0)
print(f"  {label} run {i}: laps={r.get('lap_count')} finished={r.get('finished')} "
      f"total={r.get('total_lap_time', 0):.2f}s min={r.get('min_lap_time', 0):.2f}s "
      f"penalty={r.get('penalty_total_seconds', 0):.2f}s "
      f"(crash {n('crash')}, wall {n('wall')}, over {n('over')})  {d.name}")
print("    laps: " + " ".join(f"{x:.2f}" for x in r.get("laps", [])))
for e in r.get("penalty_events", []):
    print(f"      {e['kind']} lap {e['lap']} at {e['race_time']:.1f}s for {e['duration']:.2f}s")
PY
done
log "$LABEL done"
