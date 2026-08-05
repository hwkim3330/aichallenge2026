#!/bin/bash
# The official race scenario, with the knobs the scenario file does not pin.
#
# Authoritative parts, from files byte-identical to upstream/main:
#   StreamingAssets/Race/official.yaml   name: official_race
#                                        "Official AIChallenge2026 race start positions
#                                         (3 vehicles, grid layout)"
#                                        three fixed grid coordinates, all yaw -121.84
#   sample-scenario.sh                   --scenario official.yaml --collisions on
#
# What those files do NOT specify: laps or the time limit. So the 300 s figure does not come from the
# official race scenario -- it comes from e2e.sh, which the official simulator_scripts/README.md does not
# list among its modes. The only laps/timeout pair the official docs do state is eval.sh's 6 laps / 600 s,
# and that is the single-car submission check.
#
# 300 s is therefore chosen, not derived: passing at 300 s implies passing at 600 s, and it halves the cost
# of every experiment. Six laps at 300 s means 50 s a lap against our clean 44.4 s, so 34 s of slack for the
# whole race while one wedge costs 70-125 s -- under this clock a wedge is a DNF rather than a slow race.
#
# LiDAR is on, unlike sample-scenario.sh: the AI division's only sensors are camera, lidar, steering and
# wheel odometry, and lidar_guard needs it on the SW side too. Camera stays off -- nothing reads it.
AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0
SCENARIO=/aichallenge/simulator/AWSIM/AWSIM_Data/StreamingAssets/Race/official.yaml

exec "$AWSIM_DIRECTORY/AWSIM.x86_64" \
    --scenario "${SCENARIO}" \
    --collisions on \
    --laps 6 \
    --timeout 300.0 \
    --steer-source ackermann \
    --sound off \
    --camera off \
    --lidar cpu \
    --imu off \
    -screen-fullscreen 0 \
    -screen-width 1280 \
    -screen-height 720 \
    "$@"
