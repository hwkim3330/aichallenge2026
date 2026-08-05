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
