#!/bin/bash

mode="${1}"
id="${2:-${ROS_DOMAIN_ID:-0}}"
out_dir="${3:+${3}/d${id}}"
out_dir="${out_dir:-/output/$(date +%Y%m%d-%H%M%S)/d${id}}"

case "${mode}" in
"awsim")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=true")
    [[ -n "${CAPTURE:-}" ]] && opts+=("capture:=${CAPTURE}")
    ;;
"awsim-no-viz")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=false")
    ;;
"vehicle")
    opts=("simulation:=false" "use_sim_time:=false" "run_rviz:=false")
    ;;
"rosbag")
    opts=("simulation:=false" "use_sim_time:=true" "run_rviz:=true")
    ;;
*)
    echo "invalid argument (use 'awsim' or 'vehicle' or 'rosbag')"
    exit 1
    ;;
esac

[[ -n "${CONTROL_METHOD:-}" ]] && opts+=("control_method:=${CONTROL_METHOD}")
[[ -n "${INPUT_SOURCE:-}" ]] && opts+=("input_source:=${INPUT_SOURCE}")
[[ -n "${MPC_CONFIG_FILE:-}" ]] && opts+=("mpc_config_file:=${MPC_CONFIG_FILE}")
[[ -n "${MPC_REF_VEL_FILE:-}" ]] && opts+=("mpc_ref_vel_file:=${MPC_REF_VEL_FILE}")
[[ -n "${USE_OBSTACLE_AVOIDANCE:-}" ]] && opts+=("use_obstacle_avoidance:=${USE_OBSTACLE_AVOIDANCE}")
# lidar_guard thresholds. Needed to separate the guard's effect from the CPU cost of
# CPU-rendered LiDAR: the launch file says to A/B the guard by turning the simulator's
# LiDAR on or off, but that changes both at once, and solo6lidar.sh separately warns
# that LiDAR is not free on this 16-core box. Setting the limits to 0 leaves LiDAR on
# and makes the guard inert, which isolates it.
[[ -n "${GUARD_FRONT_LIMIT:-}" ]] && opts+=("guard_front_limit:=${GUARD_FRONT_LIMIT}")
[[ -n "${GUARD_SIDE_LIMIT:-}" ]] && opts+=("guard_side_limit:=${GUARD_SIDE_LIMIT}")
[[ -n "${GUARD_SPEED:-}" ]] && opts+=("guard_speed:=${GUARD_SPEED}")
[[ -n "${LEAD_LIMIT:-}" ]] && opts+=("lead_limit:=${LEAD_LIMIT}")
# Recording is off by default and has to be asked for. It is needed to harvest expert
# demonstrations: a clean MPC solo run is exactly the behavioural-cloning data the AI track's
# network lacks, and every eval3p run so far logged "rosbag: false" and threw it away.
[[ -n "${ROSBAG:-}" ]] && opts+=("rosbag:=${ROSBAG}")
# 2026-08-02 escape experiments, read straight from the environment by
# stuck_recovery_controller (not launch args). Default off; see GOAL.md.
export RECOVERY_STRAIGHT_ESCAPE="${RECOVERY_STRAIGHT_ESCAPE:-}"
export RECOVERY_CREEP_ESCALATION="${RECOVERY_CREEP_ESCALATION:-}"
# 0 or unset keeps the blind alternation the verified submission uses; 1 and 2 aim the first
# escape burst from the latched nominal steering with opposite sign conventions.
export RECOVERY_DIRECTED="${RECOVERY_DIRECTED:-}"
# Escape geometry and the forced trigger. Both are read with getenv by
# stuck_recovery_controller, but a variable absent from docker-compose.yml never reaches the
# container at all -- setting it would have measured nothing and looked like the fix not working.
export RECOVERY_STRAIGHT4="${RECOVERY_STRAIGHT4:-}"
export RECOVERY_FORCE="${RECOVERY_FORCE:-}"
export RECOVERY_OFFICIAL="${RECOVERY_OFFICIAL:-}"
# Car-to-car avoidance from V2X opponent positions. Read by mpc_controller.py but never exported
# here or listed in docker-compose.yml, so every attempt to enable it so far was a no-op.
export V2X_AVOIDANCE="${V2X_AVOIDANCE:-}"
# Read by tiny_lidar_net_controller_node.py. With speed-trained weights this MUST be 1: the head's
# slot 0 is a normalised speed, and the default convention integrates it as an acceleration instead.
export TLN_DIRECT_SPEED="${TLN_DIRECT_SPEED:-}"
# The zero-centred speed mapping. These MUST equal what training used; the node prints them in its
# 1 Hz report so a mismatch is visible instead of just making the car drive at the wrong speed.
export TLN_SPEED_MEAN="${TLN_SPEED_MEAN:-}"
export TLN_SPEED_STD="${TLN_SPEED_STD:-}"
export TLN_SPEED_K="${TLN_SPEED_K:-}"
export TLN_CONTROL_MODE="${TLN_CONTROL_MODE:-}"
export TLN_FIXED_ACCEL="${TLN_FIXED_ACCEL:-}"
export TLN_VMAX="${TLN_VMAX:-}"
export TLN_MAX_SPEED="${TLN_MAX_SPEED:-}"
export TLN_LAUNCH_SPEED="${TLN_LAUNCH_SPEED:-}"
export TLN_LEAD_MARGIN="${TLN_LEAD_MARGIN:-}"
export TLN_REPLAY_PROFILE="${TLN_REPLAY_PROFILE:-}"
export TLN_REPLAY_V0="${TLN_REPLAY_V0:-}"
export TLN_LON_KP="${TLN_LON_KP:-}"
export TLN_ACCEL_MAX="${TLN_ACCEL_MAX:-}"
export TLN_ACCEL_MIN="${TLN_ACCEL_MIN:-}"
export TLN_STEER_SLOWDOWN="${TLN_STEER_SLOWDOWN:-}"
export TLN_CORNER_SPEED="${TLN_CORNER_SPEED:-}"
export TLN_WRONGWAY="${TLN_WRONGWAY:-}"
export TLN_WRONGWAY_WINDOW="${TLN_WRONGWAY_WINDOW:-}"
export TLN_WRONGWAY_YAW_DEG="${TLN_WRONGWAY_YAW_DEG:-}"
export TLN_UTURN_SPEED="${TLN_UTURN_SPEED:-}"
export TLN_UTURN_STEER="${TLN_UTURN_STEER:-}"
export TLN_UTURN_YAW_DEG="${TLN_UTURN_YAW_DEG:-}"
export TLN_UTURN_TIMEOUT="${TLN_UTURN_TIMEOUT:-}"

export ROS_DOMAIN_ID=$id

mkdir -p "${out_dir}"
exec >"${out_dir}/autoware.log" 2>&1

cd "${out_dir}" || exit
# Persist ROS node logs under the run output directory (so autostart_orchestrator logs are collectible).
export ROS_HOME="${out_dir}/ros"
export ROS_LOG_DIR="${ROS_HOME}/log"
mkdir -p "${ROS_LOG_DIR}"

# set -m keeps bash from setting SIGINT to SIG_IGN on the backgrounded child (then the forwarded INT would be a no-op).
set -m
ros2 launch aichallenge_system_launch aichallenge_system.launch.xml "${opts[@]}" "domain_id:=$id" &
trap 'kill -INT $! 2>/dev/null' TERM INT
while kill -0 $! 2>/dev/null; do wait; done
