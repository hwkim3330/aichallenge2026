#!/bin/bash
# One Autoware car, one NPC, LiDAR on, 480 s. For measuring NPC response and collecting
# AI-track training data at the same time.
#
# Not two cars plus an NPC, which is the battle's real shape, because that measured nothing:
# with --lidar cpu the 750-beam 20 Hz raycast pushed this 16-core box past its limit and
# stalls per car went from ~12 to 68, completion to 0/4 for both arms. Our two cars also
# share one racing line, so they tangle far more than real opponents on different lines do.
#
# One car keeps the control rate intact with LiDAR on, and one NPC is enough to answer the
# question that matters: the NPC never appears in /v2x/vehicle_positions -- measured, both
# with 1 car + NPC and in KSK's own battle bag, which lists exactly 2 vehicles and never the
# NPC -- so it is a slow obstacle only LiDAR can see, and lidar_guard has never been measured
# against one.

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

exec "$AWSIM_DIRECTORY/AWSIM.x86_64" \
    --venue citycircuit \
    --start-mode count \
    --start-count-seconds 5 \
    --vehicles 1 \
    --npcs 1 \
    --boosts 2 \
    --laps 6 \
    --timeout 480.0 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --ranking on \
    --camera off \
    --lidar cpu \
    -screen-fullscreen 0 \
    -screen-width 1280 \
    -screen-height 720 \
    -screen-quality low \
    -window-mode windowed
