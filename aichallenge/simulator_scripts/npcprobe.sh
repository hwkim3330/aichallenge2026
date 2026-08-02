#!/bin/bash
# One car, one NPC. Only question: does the NPC appear in /v2x/vehicle_positions?
#
# The scored battles run challenger + ranker + a "Computer" NPC. Our stack sees rivals
# ONLY through v2x (reference_path.use_path_constraints_topic). If AWSIM's native v2x
# publisher covers Autoware-driven vehicles but not NPCs, then the NPC -- which the board's
# own battle records show lapping in ~149 s -- is invisible to us and parking itself on our
# line. That would make LiDAR the only sensor that can see it, and the official 3-vehicle
# config (RaceConfig/eval-3p.yaml) does run lidar on.
#
# Long timeout and unlimited laps: this is a probe, the harness kills it.

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

exec "$AWSIM_DIRECTORY/AWSIM.x86_64" \
    --venue citycircuit \
    --start-mode count \
    --start-count-seconds 5 \
    --vehicles 1 \
    --npcs 1 \
    --boosts 2 \
    --laps unlimited \
    --timeout 600.0 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --ranking off \
    --camera off \
    --lidar cpu \
    -screen-fullscreen 0 \
    -screen-width 1280 \
    -screen-height 720 \
    -screen-quality low \
    -window-mode windowed
