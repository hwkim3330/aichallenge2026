#!/bin/bash
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
#   --handicap off     Handicap couples the field by design. eval.sh, which produced the
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
    --handicap off \
    --wall-recovery off \
    --ranking off \
    --camera off \
    --lidar off \
    -screen-fullscreen 0 \
    -screen-width 1280 \
    -screen-height 720 \
    -screen-quality low \
    -window-mode windowed
