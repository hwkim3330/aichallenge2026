#!/bin/bash
# The scored battle's actual shape: two Autoware cars plus one AWSIM NPC, 6 laps, 480 s.
#
# online3.sh differs from what the board runs in two ways that matter, both read off the
# live board's own payload and battle records:
#   timeout 480, not 600. Several runs recorded as 6-lap finishes at 490-510 s would be
#   DNFs where it counts.
#   an NPC is always present ("Computer", lapping in ~149 s). It is invisible to our MPC:
#   /v2x/vehicle_positions carried zero messages with one Autoware car and one NPC on
#   track, and KSK's battle bag shows v2x listing exactly 2 vehicles, never the NPC. So
#   the NPC is a slow obstacle only LiDAR can see, and lidar_guard's value can only be
#   measured with it present.

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
    --timeout 480.0 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
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
