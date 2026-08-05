#!/bin/bash
# The official race shape on the only clock the official docs actually state.
#
# official3.sh uses 300 s, which I CHOSE on the reasoning that passing 300 implies passing 600. That choice
# turned out to dominate every three-car result today. Of the twelve slots that failed to reach six laps across
# two speed profiles, every one was at five laps when the clock stopped it at 293-295 s, and one more clean lap
# would have landed at 338-340 s. So the 61 percent completion figure is a property of my chosen clock, not of
# the car.
#
# The only laps/timeout pair the official repo documents is eval.sh's 6 laps / 600 s, for the single-car
# submission check. StreamingAssets/Race/official.yaml and sample-scenario.sh pin the three-car grid and
# collisions on, and pin neither laps nor timeout. So this variant runs the official three-car shape on the
# documented 600 s.
# Official three-car race: eval.sh's flag set with three vehicles and the chosen 300 s clock.
#
# Derived from eval.sh deliberately. My first attempt at this script was written from scratch around
# --scenario official.yaml and omitted --start-mode, --venue, --vehicles and --handicap. Without a start mode
# the race never begins: eighteen races produced zero laps and no elapsed time for both versions under test,
# and the 68-lap build reported stalls=0 because its recovery never armed -- the car had never moved. That
# comparison measured this script, not the cars.
#
# --scenario is not used. official.yaml's own comment says its coordinates come from RacingKarts.prefab, i.e.
# they are the positions AWSIM already places three vehicles at, so --vehicles 3 reproduces the official grid
# without combining two flags whose interaction is untested.
#
# The shape is official: three vehicles, collisions on, per StreamingAssets/Race/official.yaml
# ("Official AIChallenge2026 race start positions (3 vehicles, grid layout)") and sample-scenario.sh, both
# byte-identical to upstream/main. Laps and timeout are NOT pinned by those files: 6 laps at 300 s is chosen,
# because passing 300 s implies 600 s and it halves the cost of every experiment.
#
# LiDAR on: the AI division's sensors are camera, lidar, steering and wheel odometry, and lidar_guard needs it
# on the SW side too.

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
    --timeout 600 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --ranking off \
    --camera off \
    --lidar cpu

# Cameraを使う場合 : --camera cpu or gpu
# LiDARを使う場合 : --lidar cpu or gpu
# GPUがない場合 -headlessを末尾に追加
