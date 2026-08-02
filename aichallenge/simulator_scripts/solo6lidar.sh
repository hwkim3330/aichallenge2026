#!/bin/bash
# solo6 with the LiDAR on, so lidar_guard actually has a scan to act on.
#
# The A/B is done by swapping this script for solo6.sh rather than by rewiring: the guard
# sits in the MPC's command chain either way and is an exact passthrough without a scan.
# Note the LiDAR costs CPU, which on this 16-core box is not free -- compare guard runs to
# each other, not to a no-LiDAR baseline, when judging pace.
#
# This mirrors eval.sh -- the script that produced the 46.53 s reference -- with one
# correction: the timeout is 480 s, which is what the scored battles allow
# (required_laps 6, timeout 480.0, read off the live board's payload). eval.sh's 600 s
# hides a configuration that needs 520 s for six laps.
#
# dev.sh is unusable for scoring a candidate: it runs wall-recovery on and laps
# unlimited, so its times are not comparable to anything the event measures.

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

exec "$AWSIM_DIRECTORY/AWSIM.x86_64" \
    --camera off \
    --lidar cpu \
    --start-mode count \
    --start-count-seconds 5 \
    --vehicles 1 \
    --npcs 0 \
    --boosts 2 \
    --laps 6 \
    --timeout 480.0 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --ranking on \
    -screen-fullscreen 0 \
    -screen-width 1280 \
    -screen-height 720 \
    -screen-quality low \
    -window-mode windowed
