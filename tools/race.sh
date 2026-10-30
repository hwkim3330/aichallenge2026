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
  export DISPLAY=:0 XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
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
  $DC "ROS_DOMAIN_ID=1 docker compose -p 1 up -d autoware" >/dev/null
  run_cmd aic-human "WHEEL_URL=$WHEEL python3 -u /tools/human_driver.py" 2
  run_cmd aic-board "python3 -u /tools/race_board.py"
  sleep 15
  if [ "$LOCAL" = 1 ]; then
    command -v wmctrl >/dev/null && wmctrl -r AWSIM -e 0,0,0,1900,1400
    touch tools/.new_race          # the first race: the watcher brings the AI up to the grid, then counts down
    setsid -f firefox --new-window http://127.0.0.1:8097/ >/dev/null 2>&1
    setsid -f "$0" watch > /tmp/race_watch.log 2>&1 < /dev/null
    echo "race up here: AWSIM + scoreboard http://127.0.0.1:8097 on this screen"
    exit 0
  fi
  wmctrl -r AWSIM -e 0,0,60,2620,2000
  setsid -f google-chrome --no-first-run --no-default-browser-check --new-window --app=http://127.0.0.1:8097/ \
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
    # the AI's start sequence (initial pose, control mode) runs once per Autoware start: restart it, put the karts
    # back on the grid (held there: sync start), give the AI time to settle on it, then 3-2-1 and go
    $DC "docker restart 1-autoware-1" >/dev/null
    sleep 3
    curl -s -X POST "http://127.0.0.1:8097/reset?now" >/dev/null
    sleep 14
    curl -s -X POST "http://127.0.0.1:8097/start?in=3" >/dev/null
    echo "$(date +%T) reset, counting down"
  done
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
