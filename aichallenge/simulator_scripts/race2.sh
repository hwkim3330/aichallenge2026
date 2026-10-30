#!/bin/bash
# human vs AI race (tools/race.sh): 2 karts, the count-down start (AWSIM restarts for every race: sync starts after a reset never released here, 2026-10-30)

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

# 車両数: 第1引数（既定 1）
vehicles="${1:-2}"

# the scoreboard's settings (tools/race.sh writes them before each race)
RACE_COLLISIONS=on RACE_BOOSTS=2 RACE_LAPS=unlimited
[ -f /aichallenge/simulator_scripts/race2.env ] && . /aichallenge/simulator_scripts/race2.env

exec $AWSIM_DIRECTORY/AWSIM.x86_64 \
    --venue citycircuit \
    --start-mode count \
    --start-count-seconds 5 \
    --vehicles "${vehicles}" \
    --npcs 0 \
    --boosts "${RACE_BOOSTS}" \
    --laps "${RACE_LAPS}" \
    --timeout 10000000.0 \
    --steer-source ackermann \
    --sound off \
    --collisions "${RACE_COLLISIONS}" \
    --handicap off \
    --wall-recovery on \
    --ranking off \
    --observer-camera on \
    --camera gpu \
    --lidar cpu

# Cameraを使う場合 : --camera cpu or gpu
# LiDARを使う場合 : --lidar cpu or gpu
# GPUがない場合 -headlessを末尾に追加
