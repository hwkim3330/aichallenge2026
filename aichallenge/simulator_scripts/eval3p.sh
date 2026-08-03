#!/bin/bash
#
# The official three-car evaluation condition, reproduced from
# AWSIM_Data/StreamingAssets/RaceConfig/eval-3p.yaml.
#
# Neither existing three-car script matches it:
#
#              collisions  handicap  timeout  lidar
#   eval-3p.yaml   off        on       600      on     <- what the board runs
#   online3.sh     on         on       600      off
#   screen3.sh     off        off      480      off
#
# screen3.sh turns handicap off deliberately, to decouple the cars so one bad challenger
# cannot spoil the others' measurements -- correct for parameter screening, wrong for
# asking "what happens in the actual battle". This script is for the latter, so handicap
# stays on and the lidar stays on.
#
# Collisions off means the cars pass through each other, which is the board's own setting.

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

exec "$AWSIM_DIRECTORY/AWSIM.x86_64" \
    --venue citycircuit \
    --start-mode sync \
    --start-count-seconds 5 \
    --vehicles 3 \
    --npcs 0 \
    --boosts 2 \
    --laps 6 \
    --timeout 600 \
    --steer-source ackermann \
    --sound off \
    --collisions off \
    --handicap on \
    --wall-recovery off \
    --ranking on \
    --camera off \
    --lidar cpu \
    -screen-fullscreen 0 \
    -screen-width 1280 \
    -screen-height 720 \
    -screen-quality low \
    -window-mode windowed
