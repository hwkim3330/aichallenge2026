#!/bin/bash
# Verify the submission the way the organisers will: build the eval image FROM the tarball, then run.
#
# The previous attempt verified nothing. `make eval` only runs `docker compose up -d
# autoware-simulator-evaluation`, and that service uses the image aichallenge-2025-eval without
# mounting ./aichallenge -- so the code comes from inside the image, which was three days old and
# built from the previous tarball. The giveaway was in the lap times: 47.58 / 45.74 / 46.31 / 46.09 /
# 46.30, which is the champion's 45.9-46.3 signature, not the promoted profile's 44.2-44.8.
#
# So the tarball has to be baked in first, via docker_build.sh eval --submit. That is also the step
# that proves the tarball compiles from a clean tree, which matters because create_submit_file.bash
# tars everything including source edited but never built.
set -uo pipefail
cd /home/kim/aichallenge2026
S=/tmp/claude-1000/-home-kim/d4fa5e0d-a691-4eb7-a3e2-380727849d2c/scratchpad
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "0. stop whatever is running"
make down >/dev/null 2>&1
docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
sleep 4

log "1. md5 of the tarball being verified"
md5sum submit/aichallenge_submit.tar.gz

log "2. build the eval image from that tarball (slow: full workspace compile)"
if ! ./docker_build.sh eval --submit submit/aichallenge_submit.tar.gz > "$S/evalbuild.log" 2>&1; then
    log "EVAL IMAGE BUILD FAILED -- the submission would not compile for the organisers"
    grep -iE "error|failed" "$S/evalbuild.log" | tail -25
    exit 1
fi
log "built OK"
docker images --format '{{.Repository}}:{{.Tag}} {{.CreatedSince}}' | grep aichallenge-2025-eval

log "3. run it"
RUN_BEFORE=$(ls -d output/*/ 2>/dev/null | wc -l)
make eval > "$S/makeeval2.log" 2>&1
sleep 20
D=$(ls -dt output/*/ | head -1)
log "run dir $D"

log "4. wait for the run to end (480 s cap plus startup, hard stop at 20 min)"
for i in $(seq 1 80); do
    docker ps --format '{{.Names}}' | grep -q simulator-evaluation || break
    sleep 15
done
if docker ps --format '{{.Names}}' | grep -q simulator-evaluation; then
    log "still running at the hard stop; capturing what exists and stopping"
    docker logs "$(docker ps --format '{{.Names}}' | grep simulator-evaluation | head -1)" \
        > "$S/evalrun.log" 2>&1
    make down >/dev/null 2>&1
else
    cp "$D"/*/autoware.log "$S/evalrun.log" 2>/dev/null || true
fi

log "5. result"
grep -hoE "Lap [0-9]+ completed! Lap time: [0-9.]+" "$S/evalrun.log" 2>/dev/null || true
grep -hiE "result|finish|total|elapsed|score" "$D"/*/autoware.log "$D"/*.json 2>/dev/null | tail -10 || true
ls -R "$D" 2>/dev/null | head -20
log done
