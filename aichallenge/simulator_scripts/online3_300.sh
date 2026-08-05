#!/bin/bash
# Three cars on the OFFICIAL clock.
#
# The official multi-car scenario is e2e.sh -- four vehicles, no NPCs, six laps, 300 s, collisions on,
# handicap off, start-random on -- and our copy of it is byte-identical to upstream/main's. 300 s for six
# laps means an average of 50 s a lap against our clean 44.4 s, so there is 34 s of slack in the whole race
# and a single wedge costs 70-125 s. Under that clock a wedge is not a slow race, it is a DNF.
#
# online3 was on 600 s, which both hid that and doubled how long every experiment took. This is online3 at
# 300 s: three cars, collisions on, for continuous multi-car running.

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

exec "$AWSIM_DIRECTORY/AWSIM.x86_64" \
    --camera off \
    --lidar off \
    --start-mode sync \
    --start-count-seconds 5 \
    --vehicles 3 \
    --npcs 0 \
    --boosts 2 \
    --laps 6 \
    --timeout 300.0 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap on \
    --wall-recovery off \
    --ranking on \
    -screen-fullscreen 0 \
    -screen-width 1280 \
    -screen-height 720 \
    -screen-quality low \
    -window-mode windowed
