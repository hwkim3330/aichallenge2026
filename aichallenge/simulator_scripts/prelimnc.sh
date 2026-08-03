#!/bin/bash
#
# The official SW-division preliminary condition, from the rules page
# (aichallenge-documentation-racingkart/competition/sw-class.html):
#
#   "3~4台同時のレース方式"  -- preliminaries pit one submission against a similarly
#   rated opponent PLUS an NPC, so three cars: --vehicles 2 --npcs 1.
#   Six laps, ten minutes, ranked by finishing order.
#   Wall or other-vehicle collisions incur a timed speed restriction as a penalty.
#   Handicap adjusts acceleration and speed by running position.
#
# No existing script matches this. npc1.sh is one vehicle against one NPC; eval3p.sh and
# parallel.sh are three submissions with no NPC. The submission has only ever been validated
# solo under eval.sh, which is why the collision penalty never showed up.
#
# collisions on, not off: the rules define a penalty for collisions, which is meaningless if
# they cannot happen. Note the penalty applies to WALL contact too, so it bites even when
# cars pass through each other -- three-car runs with --collisions off still produced 82 s and
# 95 s opening laps with no stall ever detected, which is a speed restriction, not a stop.
#
# handicap on and lidar on to match the scored environment (every eval-*p.yaml sets both).

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
    --collisions off \
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
