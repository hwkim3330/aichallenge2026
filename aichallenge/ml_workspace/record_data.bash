#!/bin/bash

# Function to handle cleanup on exit
cleanup_rosbag() {
    echo "Rosbag recording cleanup..."
    # Stop any running ros2 bag record processes
    pkill -f "ros2 bag record" 2>/dev/null || true
    sleep 1
}

# Trap signals to ensure cleanup
trap cleanup_rosbag EXIT SIGINT SIGTERM

# shellcheck disable=SC1091
source "/aichallenge/workspace/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

# Topics with data (excluding 0-message topics from original bag)
#
# velocity_status is here because the model cannot learn an acceleration target
# without knowing its current speed -- that is the leading suspect for the
# "acceleration did not train" note in the README, and it was missing from every
# bag recorded before 2026-08-01.
TOPICS=(
    "/admin/awsim/state"
    "/control/command/actuation_cmd"
    "/control/command/control_cmd"
    "/sensing/camera/image_raw"
    "/sensing/lidar/scan"
    "/vehicle/status/velocity_status"
)

# TLN_REC_DIR lets a non-MPC run record somewhere other than rawdata/. Bags from an
# AI-driven run look exactly like teacher bags, so leaving them in rawdata/ risks a
# later `extract --bags-dir rawdata` training the model on its own commands.
REC_DIR="${TLN_REC_DIR:-/aichallenge/ml_workspace/rawdata}"
mkdir -p "$REC_DIR"
cd "$REC_DIR" || exit
ros2 bag record "${TOPICS[@]}" -o "$(date +%Y%m%d-%H%M%S)" -s mcap --compression-format zstd --compression-mode file
