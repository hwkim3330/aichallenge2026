#!/bin/bash
# Two submission vehicles plus one NPC, 600 s -- the rules-text reading of the online battle.
#
# Which is official is genuinely unsettled from the files. StreamingAssets/Race/official.yaml says
# "Official AIChallenge2026 race start positions (3 vehicles, grid layout)" with three grid coordinates and no
# mention of NPCs, while the rules text describes the challenger plus a similarly-rated opponent plus an NPC.
# Both put three entities on track; they disagree on whether the third is a submission or an NPC. So this runs
# the second reading, and official3_600.sh runs the first.
#
# Derived from eval.sh, changing only vehicles, npcs and lidar -- writing one from scratch previously omitted
# --start-mode and the race never began.

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

exec "$AWSIM_DIRECTORY/AWSIM.x86_64" \
    --venue citycircuit \
    --start-mode sync \
    --start-count-seconds 5 \
    --vehicles 2 \
    --npcs 1 \
    --boosts 2 \
    --laps 6 \
    --timeout 600 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --ranking off \
    --camera off \
    --lidar cpu

# Cameraを使う場合 : --camera cpu or gpu
# LiDARを使う場合 : --lidar cpu or gpu
# GPUがない場合 -headlessを末尾に追加
