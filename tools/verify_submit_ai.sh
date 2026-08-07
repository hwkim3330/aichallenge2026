#!/bin/bash
# Verify the AI-track tarball the way the organisers will: bake it into the eval image, then run it
# with NO environment overrides at all, so the launch file's own defaults are what drives.
#
# This is a separate script from verify_submit.sh, which hardcodes submit/aichallenge_submit.tar.gz.
# That path holds the verified SW entry (md5 e40bd2b704ba10727b7c5de2e2ccab9f) and must not be
# overwritten while its upload is still pending.
#
# NOTE, and it matters: `docker_build.sh eval` overwrites the single aichallenge-2025-eval image, which
# currently contains the SW submission. After this runs, that image holds the AI entry. Rebuild it from
# the SW tarball before verifying SW again.
#
# What this actually proves that a dev-tree run cannot: the tarball compiles from a clean tree (the
# packaging step tars source that may have been edited but never built), and control_method defaults to
# tiny_lidar_net inside the artifact rather than relying on CONTROL_METHOD being set by our harness.
set -uo pipefail
cd /home/kim/aichallenge2026
TAR=${1:-submit/aichallenge_submit_ai.tar.gz}
S=/tmp/claude-1000/-home-kim/d4fa5e0d-a691-4eb7-a3e2-380727849d2c/scratchpad
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "0. stop whatever is running"
make down >/dev/null 2>&1
docker rm -f $(docker ps -aq) >/dev/null 2>&1
sleep 4

log "1. tarball under test"
md5sum "$TAR"
tar zxfO "$TAR" aichallenge_submit/aichallenge_submit_launch/launch/aichallenge_submit.launch.xml \
    | grep 'arg name="control_method"' | head -1

log "2. build the eval image from it (full workspace compile, slow)"
if ! ./docker_build.sh eval --submit "$TAR" > "$S/evalbuild_ai.log" 2>&1; then
    log "EVAL IMAGE BUILD FAILED -- this submission would not compile for the organisers"
    grep -iE "error|failed" "$S/evalbuild_ai.log" | tail -25
    exit 1
fi
log "built OK"

log "3. run it with no overrides"
make eval > "$S/makeeval_ai.log" 2>&1
log "waiting for the eval container to exit"
until [ -z "$(docker ps -q)" ]; do sleep 20; done
D=$(ls -dt output/*/ | head -1)
log "run dir $D"
find "$D" -name "*result*.json" | head -3
for f in $(find "$D" -name "*result*.json" | head -2); do echo "--- $f"; head -c 700 "$f"; echo; done
log "4. measured progress from the recorded pose (if a bag exists)"
B=$(find "$D" -type d -name rosbag2_autoware | head -1)
[ -n "$B" ] && timeout 900 python3 tools/ai_progress.py "$B" 2>&1 | tail -10 || log "no bag (ROSBAG unset in eval, expected)"
log "done"
