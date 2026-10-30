#!/bin/bash
# Human (G923 on the Autoware PC) vs our racing AI in AWSIM, on this PC; the screen streamed to the Autoware PC.
#   tools/race.sh up      AWSIM 2 karts (P1 AI on domain 1, P2 the human on domain 2), the AI's Autoware, the wheel
#                         node, the scoreboard (http://127.0.0.1:8097) and the stream to 10.42.0.90 (a window there)
#   tools/race.sh down    all of it
#   tools/race.sh watch   (started by up) the scoreboard's 새 레이스: restart the AI, then reset AWSIM
# LOCAL=1 (on the Autoware PC itself, where the wheel and its screen are; the default there): the wheel bridge on
# 127.0.0.1, the windows on its own screen (:0, 2560x1440), no stream
set -u
cd "$(dirname "$0")/.."
AW=${AW:-10.42.0.90}
[ "$(hostname -I 2>/dev/null | grep -c 10.42.0.90)" = 1 ] && LOCAL=${LOCAL:-1}
LOCAL=${LOCAL:-0}
if [ "$LOCAL" = 1 ]; then
  export DISPLAY=:0
  XA=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
  # the containers reach X through the socket (xhost local), not the host's auth file (not mounted in them: with
  # its path passed in, AWSIM opened no window here)
  XAUTHORITY=$XA xhost +local: >/dev/null 2>&1
  export XAUTHORITY=
  WHEEL=ws://127.0.0.1:8771
  DC="sg docker -c"
else
  export DISPLAY=:1
  WHEEL=ws://$AW:8771
  DC="bash -c"
fi
SSH="sshpass -e ssh -o ConnectTimeout=5 keti@$AW"   # the password in SSHPASS (not stored here)
run_cmd() { $DC "CMD='source /opt/ros/humble/setup.bash; source /autoware/install/setup.bash 2>/dev/null; $2' ROS_DOMAIN_ID=${3:-1} docker compose run -d --rm --no-deps --name $1 -v $PWD/tools:/tools autoware-command" >/dev/null; }
case "${1:-up}" in
up)
  # race2: two karts held on the grid until /admin/awsim/start (sync start), so 새 레이스 can start them together
  $DC "make simulator SIM_MODE=race2" >/dev/null
  sleep 5
  # SIM_MODE=race2 in the AI's container: its lidar_guard keeps clear of the human's kart (race mode)
  $DC "SIM_MODE=race2 ROS_DOMAIN_ID=1 docker compose -p 1 up -d autoware" >/dev/null
  run_cmd aic-human "WHEEL_URL=$WHEEL python3 -u /tools/human_driver.py" 2
  run_cmd aic-board "python3 -u /tools/race_board.py"
  sleep 15
  if [ "$LOCAL" = 1 ]; then
    touch tools/.new_race          # the first race: the watcher restarts AWSIM and the AI
    XAUTHORITY=$XA setsid -f google-chrome --no-first-run --no-default-browser-check --password-store=basic --disable-features=Translate --lang=ko --new-window --app=http://127.0.0.1:8097/ \
      --user-data-dir=/tmp/aic-board-chrome --window-size=950,540 --window-position=1610,0 >/dev/null 2>&1
    (sleep 40; XAUTHORITY=$XA "$0" arrange) &
    setsid -f "$0" watch > /tmp/race_watch.log 2>&1 < /dev/null
    echo "race up here: AWSIM + scoreboard http://127.0.0.1:8097 on this screen"
    exit 0
  fi
  wmctrl -r AWSIM -e 0,0,60,2620,2000
  setsid -f google-chrome --no-first-run --no-default-browser-check --password-store=basic --disable-features=Translate --lang=ko --new-window --app=http://127.0.0.1:8097/ \
    --user-data-dir=/tmp/aic-board-chrome --window-size=1180,1500 --window-position=2640,60 >/dev/null 2>&1
  # the screen to the Autoware PC (the wheel is there): MJPEG frames over TCP (no H.264 decoder there; RTP/UDP lost
  # fragments of the big frames and the picture broke up), 1080p30, ffmpeg listening and the viewer connecting
  setsid -f ffmpeg -loglevel warning -f x11grab -framerate 30 -video_size 3840x2160 -i :1 -vf scale=1920:1080 \
    -c:v mjpeg -q:v 5 -f mpjpeg "tcp://0.0.0.0:5601?listen=1&tcp_nodelay=1" > /tmp/race_stream.log 2>&1 < /dev/null
  sleep 2
  $SSH 'DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority setsid -f gst-launch-1.0 -q tcpclientsrc host=10.42.0.1 port=5601 ! multipartdemux ! jpegdec ! videoconvert ! xvimagesink sync=false force-aspect-ratio=true > /tmp/race_view.log 2>&1 < /dev/null'
  setsid -f "$0" watch > /tmp/race_watch.log 2>&1 < /dev/null
  echo "race up: scoreboard http://127.0.0.1:8097, the stream on $AW"
  ;;
watch)
  while sleep 0.5; do
    [ -f tools/.new_race ] || continue
    rm -f tools/.new_race
    echo "$(date +%T) new race: restarting the AI"
    # a fresh AWSIM per race (its count-down start runs once, at boot). A kart joins that count-down only if it asked
    # for its control mode while still Grounded (asked at Ready, the AI was left out and held at 5 km/h): so AWSIM
    # comes up last, with the AI already waiting for it and the wheel node already asking
    # the scoreboard's settings (tools/race_config.json, option values from fixed lists; quoted all the same)
    eval "$(python3 - <<'PY'
import json, shlex
c = {'ai_style': 'safe', 'ai_ctrl': 'mpc', 'ai_speed': '85', 'collisions': 'on', 'boosts': '2', 'laps': 'unlimited', 'view': 'p2', 'lock': '150'}
try:
    c.update(json.load(open('tools/race_config.json')))
except (OSError, ValueError):
    pass
for k, v in c.items():
    print(f'C_{k}={shlex.quote(str(v))}')
PY
)"
    printf 'RACE_COLLISIONS=%q RACE_BOOSTS=%q RACE_LAPS=%q\n' "$C_collisions" "$C_boosts" "$C_laps" > aichallenge/simulator_scripts/race2.env
    SIMV=race2; [ "$C_ai_style" = race ] && SIMV=race2noavoid
    REF=""; [ "$C_ai_speed" != 100 ] && REF="ref_vel_race$C_ai_speed.yaml"
    echo "$(date +%T) settings: AI $C_ai_style/$C_ai_ctrl/$C_ai_speed% collisions $C_collisions boosts $C_boosts laps $C_laps view $C_view lock $C_lock"
    $DC "docker stop aichallenge2026-simulator-1" >/dev/null
    $DC "docker rm -f aic-human" >/dev/null 2>&1
    touch /tmp/race_ai_restart
    # the AI recreated with this race's brain, character and speed (a restart keeps the old environment)
    $DC "SIM_MODE=$SIMV CONTROL_METHOD=$C_ai_ctrl MPC_REF_VEL_FILE=$REF ROS_DOMAIN_ID=1 docker compose -p 1 up -d --force-recreate autoware" >/dev/null 2>&1
    for _ in $(seq 1 60); do
      L=$(ls -td output/2026*/d1 2>/dev/null | head -1)
      [ -n "$L" ] && [ "$L/autoware.log" -nt /tmp/race_ai_restart ] && grep -q "wait start" "$L/autoware.log" && break
      sleep 1
    done
    run_cmd aic-human "LOCK_DEG=$C_lock WHEEL_URL=$WHEEL python3 -u /tools/human_driver.py" 2
    sleep 3
    $DC "docker start aichallenge2026-simulator-1" >/dev/null
    curl -s -X POST "http://127.0.0.1:8097/clear" >/dev/null
    # the human's own view: AWSIM's observer camera (C) goes split -> P1 -> P2; the AI's side is RViz beside it
    [ "$LOCAL" = 1 ] && (sleep 25; XAUTHORITY=$XA "$0" arrange; XAUTHORITY=$XA "$0" p2view "$C_view") &
    echo "$(date +%T) AWSIM and the AI restarted: the count-down follows"
  done
  ;;
arrange)
  # the Autoware PC's screen (2560x1440): AWSIM large on the left; on the right the board above the AI's RViz (its
  # race line, pose and trajectory); the orchestrator's debug window out of the way
  if [ "$LOCAL" = 1 ]; then export XAUTHORITY=$XA; fi
  for id in $(xdotool search --name "Autostart Orchestrator Debug" 2>/dev/null); do xdotool windowminimize "$id" 2>/dev/null; done
  wmctrl -r AWSIM -b remove,maximized_vert,maximized_horz; wmctrl -r AWSIM -e 0,0,0,1600,1420
  for t in "사람 vs" "127.0.0.1"; do wmctrl -r "$t" -e 0,1610,0,950,540; done
  wmctrl -r RViz -b remove,hidden; wmctrl -r RViz -e 0,1610,560,950,860
  for t in RViz "사람 vs" "127.0.0.1" AWSIM; do wmctrl -a "$t" 2>/dev/null; done
  ;;
p2view)
  if [ "$LOCAL" = 1 ]; then export XAUTHORITY=$XA; fi
  W=$(xdotool search --name "^AWSIM$" | head -1)
  # observer camera (C): split -> P1 -> P2
  case "${2:-p2}" in p1) N=1;; split) N=0;; *) N=2;; esac
  for _ in $(seq 1 $N); do xdotool windowactivate --sync "$W" key c 2>/dev/null; sleep 0.7; done
  "$0" arrange
  ;;
down)
  for p in $(ps -eo pid=,args= | awk '/race.sh watch/ && !/awk/ {print $1}'); do kill "$p"; done
  for p in $(ps -eo pid=,args= | awk '/x11grab/ && /5601/ && !/awk/ {print $1}'); do kill "$p"; done
  [ "$LOCAL" = 1 ] || $SSH 'for p in $(ps -eo pid=,args= | awk "/tcpclientsrc/ && !/awk/ {print \$1}"); do kill $p; done'
  $DC "docker rm -f aic-human aic-board" >/dev/null 2>&1
  $DC "make down" >/dev/null 2>&1
  echo "race down"
  ;;
esac
