#!/bin/bash
#
# screen3 with the two flags the official RaceConfig actually sets: lidar on and handicap on.
#
# screen3 runs --lidar off, which silently removes lidar_guard from the chain. The guard caps
# speed at 5.5 m/s when a wall is inside 1.25 m; without it the car meets walls at full pace.
# Measured 2026-08-04: under screen3 the shipped, eval-verified config took 310.9 s for ONE lap
# after leaving the line at wp 40 with e_y 3.53 m, where the same config does 288 s for six laps
# under prelimnc, which runs lidar cpu. Every eval-*p.yaml sets lidar: "on", so screen3 was
# screening a car that does not exist -- and every historical screening run used it.
#
# Three submissions, no NPC, collisions off so one bad entrant cannot spoil the others.
# Three cars, independent. A screening rig, not a race.
#
# Three configurations are measured in one 8-minute run instead of three sequential solo
# runs, and the point of every flag below is to stop the cars from influencing each other.
#
#   --collisions off   The official 3-vehicle config ships this way too
#                      (AWSIM_Data/StreamingAssets/RaceConfig/eval-3p.yaml). With
#                      collisions ON, a bad configuration becomes a parked obstacle and
#                      destroys the OTHER cars' measurements -- generations 0-2 of the
#                      search here recorded the champion at 19, 37 and 17 recoveries in
#                      races whose challengers scored 49, 95 and 120, while the same
#                      champion scores 0 solo. Nothing in those runs was attributable.
#
#   --handicap on     Handicap couples the field by design. eval.sh, which produced the
#                      46.53 s reference, also runs with it off.
#
#   --timeout 480      What the scored battles actually allow (required_laps 6,
#                      timeout 480.0, read off the live board's payload), not the 600 in
#                      eval.sh. A configuration that needs 520 s for six laps looks fine
#                      at 600 and does not finish where it counts.

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
    --timeout 480.0 \
    --steer-source ackermann \
    --sound off \
    --collisions off \
    --handicap on \
    --wall-recovery off \
    --ranking off \
    --camera off \
    --lidar cpu \
    -screen-fullscreen 0 \
    -screen-width 1280 \
    -screen-height 720 \
    -screen-quality low \
    -window-mode windowed
