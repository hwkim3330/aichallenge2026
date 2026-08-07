#!/usr/bin/env python3
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport

from tiny_lidar_net_controller_core import TinyLidarNetCore


class TinyLidarNetNode(Node):
    """ROS 2 Node for TinyLidarNet autonomous driving control.

    This node subscribes to LaserScan messages, processes them using the
    TinyLidarNetCore logic, and publishes AckermannControlCommand messages.
    """

    def __init__(self):
        super().__init__('tiny_lidar_net_node')

        # --- Parameter Declaration ---
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.output_dim', 2)
        self.declare_parameter('model.architecture', 'large')
        self.declare_parameter('model.ckpt_path', '')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('acceleration', 0.1)
        self.declare_parameter('control_mode', 'ai')
        self.declare_parameter('debug', False)
        self.declare_parameter('max_speed_mps', 10.0)
        self.declare_parameter('min_speed_mps', 0.0)

        # --- Initialization ---
        input_dim = self.get_parameter('model.input_dim').value
        output_dim = self.get_parameter('model.output_dim').value
        architecture = self.get_parameter('model.architecture').value
        ckpt_path = self.get_parameter('model.ckpt_path').value
        max_range = self.get_parameter('max_range').value
        acceleration = self.get_parameter('acceleration').value
        control_mode = self.get_parameter('control_mode').value
        # Override so the July design and any alternative can be compared without file edits.
        control_mode = os.environ.get('TLN_CONTROL_MODE', '') or control_mode
        
        self.debug = self.get_parameter('debug').value
        self.log_interval = self.get_parameter('log_interval_sec').value
        self.launch_speed_mps = float(os.environ.get('TLN_LAUNCH_SPEED', '') or '2.0')
        self.lead_margin_mps = float(os.environ.get('TLN_LEAD_MARGIN', '') or '1.5')
        # Longitudinal P controller on acceleration. Now the default, on this measurement over the same
        # track and 300 s clock (tools/ai_progress.py, laps read from the pose rather than the log):
        #   kp = 0.0 (constant 0.6 m/s^2)  best lap 79.46 s, 3.47 laps, mean 3.73 m/s
        #   kp = 1.0                       best lap 48.16 s, 6.05 laps, mean 6.78 m/s
        # and the MPC on the same run manages 6.05 laps with a 44.35 s best but a 47.49 s mean, because
        # one lap cost 59.75 s. The AI's mean flying lap of 48.26 s is the more consistent of the two,
        # and it spends 1.7% of frames under 0.5 m/s against the MPC's 5.2%.
        # kp = 0.0 still reproduces the old constant-acceleration behaviour exactly.
        # Limits are the MPC's measured envelope (-1.60..+1.35 m/s^2), not a guess. `or default` because
        # docker-compose injects declared-but-unset variables as EMPTY STRINGS, and float('') raises,
        # which killed the node once already.
        # 1.8 rather than 1.0 from batch PACE1, three cars collisions-off in one race so the comparison
        # shares the conditions: kp 1.8 with max_speed 11.0 gave best flying 44.09 s / mean flying
        # 44.40 s, against kp 1.0 with max_speed 9.5 at 48.38 / 48.58. A third slot with
        # steer_slowdown 5.0 set the fastest single lap of the session, 41.67 s, but is NOT adopted: it
        # spent 35.3% of frames under 0.5 m/s and ran 30.5 m backwards, for a 55.46 s mean flying lap.
        # That is the pace ceiling of this control law and also the evidence that corner braking is what
        # buys the consistency an Elo battle scores.
        self.lon_kp = float(os.environ.get('TLN_LON_KP', '') or '1.8')
        self.accel_max = float(os.environ.get('TLN_ACCEL_MAX', '') or '1.35')
        self.accel_min = float(os.environ.get('TLN_ACCEL_MIN', '') or '-1.60')
        self.steer_slowdown = float(os.environ.get('TLN_STEER_SLOWDOWN', '') or '8.0')
        self.corner_speed_mps = float(os.environ.get('TLN_CORNER_SPEED', '') or '4.5')
        # Take the target speed from the NETWORK's slot-0 head instead of the |steer| law.
        #
        # The |steer| law is reactive: it slows only once the wheel is already turned, and the AI's one
        # remaining failure is the tightest corner on the circuit (4.3 m radius at s 56-85 m) reached
        # straight off the longest straight at 10.01 m/s. Front clearance and steering rate were both
        # tested as anticipatory substitutes and both read the WIDER corner as more dangerous, so neither
        # can serve. The scan does contain the corner's shape, so the fix is a target that encodes it:
        # the MPC's commanded speed, recorded beside every scan.
        #
        # Measured on 59226 held-out frames, this checkpoint trained on both heads:
        #   speed    R2 +0.968, std 0.5444 against target 0.5495, MAE 0.056 m/s, range 6.03..9.33
        #   steering R2 +0.918, against 0.922 for the steering-only checkpoint
        # so the speed head is real and steering paid 0.004 for it. The earlier attempt at this head
        # emitted a CONSTANT (std 0.0009) because the old min-max mapping put the target mean at +0.775
        # where tanh' is about 0.016; the zero-centred mapping below is what fixed it.
        #
        # These three constants MUST equal the ones training used, or the car drives at the wrong speed
        # with no error anywhere, so they are reported in the 1 Hz line.
        self.vdes_from_net = (os.environ.get('TLN_VDES_FROM_NET', '') or '0') not in ('0', 'false')
        self._v_mean = float(os.environ.get('TLN_SPEED_MEAN', '') or '7.86')
        self._v_std = float(os.environ.get('TLN_SPEED_STD', '') or '1.83')
        self._v_k = float(os.environ.get('TLN_SPEED_K', '') or '1.0')
        self._v_des = 0.0

        # --- Wrong-way detection, from integrated yaw only ---------------------------------------
        # Why this exists: in 18 slots of the official three-car race, 4 failed to complete six laps,
        # and the cause was not pace. A car that gets turned around DRIVES THE WRONG WAY AT RACING
        # SPEED and never recovers -- measured at -144 and -76 m at 7.55 m/s, and -118/-95/-157/-103/-116 m
        # at up to 8.41 m/s. TinyLidarNet decides from a single LaserScan and the circuit looks much the
        # same in both directions, so nothing in the policy can represent "this is the wrong way".
        #
        # Yaw is enough to tell, because traversing a closed circuit in reverse flips the sign of the
        # accumulated heading. Measured over 40 s windows across five runs: forward driving gives net yaw
        # -246 deg on average (median -274, p95 -14; this circuit runs clockwise), while wrong-way gives
        # +142 (median +124, p5 +22). A threshold of +50 deg catches 85% of wrong-way windows with ZERO
        # false positives on forward driving, and +0 deg catches 96% with 3% false positives. +50 is used:
        # a false positive would throw away a good run, and the failure it guards against is slow anyway.
        #
        # Gyro only, so this is legal whichever way the division's sensor restrictions are read. It uses
        # /sensing/imu/imu_raw, which participant-interface.md requires every submission to subscribe.
        self.wrongway_window_s = float(os.environ.get('TLN_WRONGWAY_WINDOW', '') or '40.0')
        self.wrongway_yaw_deg = float(os.environ.get('TLN_WRONGWAY_YAW_DEG', '') or '50.0')
        # DEFAULT OFF. The first live test of this deadlocked ALL THREE cars: each entered the U-turn and
        # froze with v_meas 0.00 and the turned angle stuck at 47, 148 and 102 of 150 deg. The exit
        # condition was "yaw actually turned", and a car that cannot move accumulates no yaw, so the state
        # was absorbing -- strictly worse than having no detector at all. The timeout and the no-progress
        # abort below fix that, but the feature stays off by default until it is measured to help.
        self.wrongway_enable = (os.environ.get('TLN_WRONGWAY', '') or '0') not in ('0', 'false')
        self.uturn_timeout_s = float(os.environ.get('TLN_UTURN_TIMEOUT', '') or '6.0')
        self.uturn_speed_mps = float(os.environ.get('TLN_UTURN_SPEED', '') or '1.5')
        self.uturn_steer = float(os.environ.get('TLN_UTURN_STEER', '') or '0.40')
        self.uturn_yaw_target_deg = float(os.environ.get('TLN_UTURN_YAW_DEG', '') or '150.0')
        self._yaw_hist = []          # (t_sec, yaw_rate) inside the detection window
        self._uturn_until_yaw = None  # set while a recovery U-turn is in progress
        self._uturn_yaw_acc = 0.0
        self._uturn_count = 0
        self._uturn_start_s = 0.0
        self._last_imu_t = None
        self.create_subscription(Imu, '/sensing/imu/imu_raw', self._imu_cb,
                                 QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                                            history=HistoryPolicy.KEEP_LAST))
        # Time-indexed acceleration replay. See the module this was patched by: the trace is a flying lap,
        # so it needs an entry speed, and it is indexed by time because the AI division bans position.
        self._replay_t = None
        self._replay_a = None
        self._replay_v0 = float(os.environ.get('TLN_REPLAY_V0', '') or '6.2')
        _rp = os.environ.get('TLN_REPLAY_PROFILE', '')
        if _rp:
            import json
            _d = json.load(open(_rp))
            self._replay_t = np.asarray(_d['t_sec'], dtype=float)
            self._replay_a = np.asarray(_d['accel_mps2'], dtype=float)
            _dt = np.diff(self._replay_t)
            _dv = np.concatenate([[0.0], np.cumsum(_dt * (self._replay_a[:-1] + self._replay_a[1:]) / 2)])
            self._replay_v = self._replay_v0 + _dv
            self._replay_start = None
            self.get_logger().info(
                f'replay profile {_rp}: {len(self._replay_t)} points over '
                f'{self._replay_t[-1]:.2f} s, speed {self._replay_v.min():.2f}..'
                f'{self._replay_v.max():.2f} m/s from v0={self._replay_v0:.2f}')
        self.max_speed_mps = float(os.environ.get('TLN_MAX_SPEED', '')
                                   or self.get_parameter('max_speed_mps').value)
        self.min_speed_mps = self.get_parameter('min_speed_mps').value

        # AckermannControlCommand.longitudinal.speed is the setpoint AWSIM's
        # vehicle bridge actually tracks; .acceleration alone never moves the
        # vehicle. No planned trajectory exists here to read a target speed
        # from like MPC does, so integrate one from the model's own accel
        # output against the last known real velocity.
        self._last_velocity_mps = 0.0
        self._last_callback_time = None

        try:
            self.core = TinyLidarNetCore(
                input_dim=input_dim,
                output_dim=output_dim,
                architecture=architecture,
                ckpt_path=ckpt_path,
                acceleration=acceleration,
                control_mode=control_mode,
                max_range=max_range
            )
            self.get_logger().info(
                f"Core initialized. Arch: {architecture}, MaxRange: {max_range}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize core logic: {e}")
            raise e

        # --- Communication Setup ---
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos
        )
        self.sub_velocity = self.create_subscription(
            VelocityReport, "/vehicle/status/velocity_status",
            self._velocity_callback, qos
        )
        self.pub_control = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1
        )

        self.get_logger().info("TinyLidarNetNode is ready.")

    def _velocity_callback(self, msg: VelocityReport):
        """Tracks the vehicle's real current speed for speed-setpoint integration."""
        self._last_velocity_mps = msg.longitudinal_velocity

    def _imu_cb(self, msg: Imu):
        """Accumulates yaw rate over a sliding window, for the wrong-way test."""
        t = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        if t <= 0.0:
            return
        wz = float(msg.angular_velocity.z)
        if self._last_imu_t is not None:
            dt = t - self._last_imu_t
            if 0.0 < dt < 0.5:
                self._yaw_hist.append((t, wz * dt))
                if self._uturn_until_yaw is not None:
                    self._uturn_yaw_acc += wz * dt
        self._last_imu_t = t
        cut = t - self.wrongway_window_s
        while self._yaw_hist and self._yaw_hist[0][0] < cut:
            self._yaw_hist.pop(0)

    def _net_yaw_deg(self) -> float:
        return float(np.degrees(sum(d for _, d in self._yaw_hist)))

    def _uturn_command(self, cmd, now):
        """Return a U-turn command while recovering from a wrong-way heading, else None.

        Entry needs a FULL window of yaw history, so the test cannot fire during the first
        `wrongway_window_s` of the race -- the grid start accumulates very little yaw and would otherwise
        read as ambiguous. Exit is on yaw turned during the manoeuvre, not on a timer, because how long a
        given turn takes depends on where the car is wedged.
        """
        now_s = now.nanoseconds * 1e-9
        if self._uturn_until_yaw is None:
            if len(self._yaw_hist) < 20:
                return None
            span = self._yaw_hist[-1][0] - self._yaw_hist[0][0]
            if span < self.wrongway_window_s * 0.8:
                return None
            if self._net_yaw_deg() <= self.wrongway_yaw_deg:
                return None
            self._uturn_until_yaw = self.uturn_yaw_target_deg
            self._uturn_yaw_acc = 0.0
            self._uturn_count += 1
            self._uturn_start_s = now_s
            self.get_logger().warn(
                f'[tln] WRONG WAY: net yaw {self._net_yaw_deg():+.0f} deg over '
                f'{span:.0f} s exceeds +{self.wrongway_yaw_deg:.0f}; U-turn #{self._uturn_count}')

        turned = abs(np.degrees(self._uturn_yaw_acc))
        elapsed = now_s - self._uturn_start_s
        # A HARD timeout as well as the yaw target. Without it this state is absorbing: the exit test is
        # "yaw actually turned", a wedged car turns no yaw, and all three cars in the first live test froze
        # here at 47, 148 and 102 of 150 deg with v_meas 0.00 for the rest of the race. Whatever the
        # manoeuvre achieved, hand control back and let the normal policy and the recovery node try.
        if turned >= self._uturn_until_yaw or elapsed >= self.uturn_timeout_s:
            why = 'turned enough' if turned >= self._uturn_until_yaw else 'TIMEOUT'
            self.get_logger().warn(
                f'[tln] U-turn end ({why}): turned {turned:.0f} deg in {elapsed:.1f} s; resuming')
            self._uturn_until_yaw = None
            self._yaw_hist.clear()   # the manoeuvre's own yaw must not re-trigger the test
            self._last_imu_t = None
            return None

        # Reverse under steering lock. Reversing rather than driving forward because the car reaches this
        # state by being stopped or wedged against something it just hit, and reverse is the direction with
        # room in it -- the same reasoning stuck_recovery_controller's directed escape already uses.
        out = AckermannControlCommand()
        out.stamp = self.get_clock().now().to_msg()
        out.longitudinal.speed = -abs(self.uturn_speed_mps)
        out.longitudinal.acceleration = -0.8 if self._last_velocity_mps > -abs(self.uturn_speed_mps) else 0.0
        out.lateral.steering_tire_angle = float(self.uturn_steer)
        if getattr(self, '_last_uturn_report', None) is None or now_s - self._last_uturn_report >= 1.0:
            self._last_uturn_report = now_s
            self.get_logger().info(
                f'[tln] UTURN turned={turned:.0f}/{self._uturn_until_yaw:.0f} deg '
                f'v_meas={self._last_velocity_mps:.2f}')
        return out

    def scan_callback(self, msg: LaserScan):
        """Callback for LaserScan subscription.

        Processes the scan data via the core logic and publishes a control command.

        Args:
            msg (LaserScan): The incoming ROS 2 LaserScan message.
        """
        start_time = time.monotonic()

        # 1. Convert ROS message to Numpy
        # We pass the raw array; the core logic handles NaN/Inf and normalization.
        ranges = np.array(msg.ranges, dtype=np.float32)

        # 2. Process via Core Logic
        accel, steer = self.core.process(ranges)

        # 2b. Integrate a speed setpoint from the model's own acceleration
        # (see note above __init__ on why .acceleration alone can't move
        # the vehicle).
        now = self.get_clock().now()
        if self._last_callback_time is not None:
            dt = (now - self._last_callback_time).nanoseconds / 1e9
        else:
            dt = 0.1  # first callback: assume the nominal ~10Hz scan rate
        self._last_callback_time = now
        dt = max(0.0, min(dt, 0.5))  # guard against a stale/huge gap

        # July's closed loop, plus a floor. Integrating from the MEASURED velocity is what makes the
        # setpoint track reality instead of climbing to the cap while the car is stuck against a wall.
        # But at a standstill it traps itself: v_meas 0.00 gives 0 + 0.6*0.05 = 0.03 m/s forever, which is
        # exactly what the restored node did -- the car never left the grid. The floor only applies while
        # the vehicle is essentially stopped, so it restores launch without weakening the tracking.
        # Both pure forms fail, in opposite directions, and this is measured rather than argued:
        #   open loop  (setpoint += accel*dt)         climbs to the cap while the car sits against a wall,
        #                                            so the command bears no relation to reality
        #   closed loop (setpoint = v_meas + accel*dt) is July's, and it plateaus: the vehicle tracks the
        #                                            setpoint with a steady-state error, so the two hold
        #                                            each other at an equilibrium -- measured 2.5 m/s here
        # So ratchet the setpoint up like the open loop, but never let it lead the measured speed by more
        # than lead_margin. That accelerates properly and still collapses back to reality when the car is
        # held up, which is what the closed loop was protecting.

        self._setpoint = getattr(self, "_setpoint", 0.0) + float(accel) * dt
        self._setpoint = min(self._setpoint, self._last_velocity_mps + self.lead_margin_mps)
        if self._last_velocity_mps < self.launch_speed_mps:
            self._setpoint = max(self._setpoint, self.launch_speed_mps)
        target_speed = self._setpoint
        target_speed = float(np.clip(target_speed, self.min_speed_mps, self.max_speed_mps))

        # 3. Publish Command
        cmd = AckermannControlCommand()
        if self._replay_t is not None:
            # Wall-clock replay, started on the first frame the vehicle is actually moving so the grid
            # countdown does not consume the trace.
            now_s = now.nanoseconds * 1e-9
            if self._replay_start is None:
                if self._last_velocity_mps > 0.5:
                    self._replay_start = now_s
                self._setpoint = max(self._replay_v0, self.launch_speed_mps)
            else:
                tt = (now_s - self._replay_start) % float(self._replay_t[-1])
                self._setpoint = float(np.interp(tt, self._replay_t, self._replay_v))
            target_speed = float(np.clip(self._setpoint, 0.0, self.max_speed_mps))
            # Stamp and acceleration matter: this branch sits before the normal stanza that fills them,
            # and an unstamped command with acceleration 0 is not the same message the vehicle expects.
            cmd.stamp = self.get_clock().now().to_msg()
            cmd.longitudinal.speed = target_speed
            cmd.longitudinal.acceleration = float(accel)
            cmd.lateral.steering_tire_angle = float(steer)
            self.pub_control.publish(cmd)
            if getattr(self, '_last_report', None) is None or now_s - self._last_report >= 1.0:
                self._last_report = now_s
                self.get_logger().info(
                    f'[tln] REPLAY steer={float(steer):+.3f} v_meas={self._last_velocity_mps:.2f} '
                    f'setpoint={target_speed:.2f} cap={self.max_speed_mps:.2f}')
            return
        # Wrong-way recovery takes priority over everything else: a car pointing the wrong way does not
        # need a better speed profile, it needs to be pointing the other way. See the notes in __init__
        # for the measurements that motivate this and for the threshold's false-positive rate.
        if self.wrongway_enable:
            uturn = self._uturn_command(cmd, now)
            if uturn is not None:
                self.pub_control.publish(uturn)
                return

        # Longitudinal control on ACCELERATION, which is the channel the vehicle actually tracks.
        #
        # Measured on the same track, same 300 s, from the recorded /control/command/control_cmd:
        #   this node   cmd.speed mean 9.19  cmd.acceleration CONSTANT 0.600  ->  measured v mean 3.73
        #   the MPC     cmd.speed mean 7.94  cmd.acceleration mean 1.064, range -1.60..+1.35
        #                                                                 ->  measured v mean 6.97
        # So the vehicle does not chase cmd.speed; a constant 0.6 m/s^2 simply equilibrates against drag
        # at about 4.9 m/s, which is the "speed ceiling" that resisted every change to the setpoint. It
        # was never a steering-scrub consequence: raising the setpoint to 9.5 cannot help when the
        # acceleration channel is pinned. Upstream's own 0.3 -> 0.6 bump (eefb9ef) is the same lever.
        #
        # The MPC's envelope is the evidence for the limits used here: -1.60..+1.35 m/s^2 is proven
        # feasible on this vehicle, so the P controller is clipped to it rather than to a guess.
        #
        # Target speed comes from the steering the network just produced: the harder it is turning, the
        # less speed the corner allows. That keeps the AI division's sensor rules -- it needs no position,
        # only the lidar the network already saw.
        if self.lon_kp > 0.0:
            if self.vdes_from_net:
                # Inverse of the dataset's zero-centred mapping. Needs control_mode "ai", because "fixed"
                # substitutes a constant for slot 0 before it ever reaches here.
                v_des = self._v_mean + float(accel) * self._v_k * self._v_std
            else:
                v_des = self.max_speed_mps - self.steer_slowdown * abs(float(steer))
            v_des = float(np.clip(v_des, self.corner_speed_mps, self.max_speed_mps))
            if self._last_velocity_mps < self.launch_speed_mps:
                v_des = max(v_des, self.launch_speed_mps)
            accel = float(np.clip(self.lon_kp * (v_des - self._last_velocity_mps),
                                  self.accel_min, self.accel_max))
            target_speed = v_des
            self._v_des = v_des

        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.speed = target_speed
        cmd.longitudinal.acceleration = float(accel)
        cmd.lateral.steering_tire_angle = float(steer)
        self.pub_control.publish(cmd)
        # Report the head, the measured speed it was integrated from, and the setpoint, once a
        # second. Without this the node runs silently and a constant output looks like driving.
        if getattr(self, '_last_report', None) is None or \
                (now.nanoseconds * 1e-9) - self._last_report >= 1.0:
            self._last_report = now.nanoseconds * 1e-9
            self.get_logger().info(
                f'[tln] head=({float(accel):+.3f}, {float(steer):+.3f}) '
                f'v_meas={self._last_velocity_mps:.2f} setpoint={target_speed:.2f} '
                f'lead={self.lead_margin_mps:.1f} '
                f'cap={self.max_speed_mps:.2f} mode={self.core.control_mode} '
                f'lon_kp={self.lon_kp:.2f} v_des={self._v_des:.2f} '
                f'a_cmd={float(accel):+.3f} '
                # imu_n is the gate's input: if it stays 0 the IMU topic is not arriving and the
                # wrong-way test silently cannot fire, which is exactly the failure to look for.
                f'yaw={self._net_yaw_deg():+.0f}deg imu_n={len(self._yaw_hist)} '
                f'uturns={self._uturn_count} '
                # vnet shows whether the speed head is actually driving v_des, and the three constants
                # must match training -- a mismatch is otherwise silent.
                f'vnet={int(self.vdes_from_net)} '
                f'map=({self._v_mean:.2f},{self._v_std:.2f},k{self._v_k:.1f})')

        # 4. Debug Logging
        if self.debug:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            self.inference_times.append(duration_ms)
            self._log_performance_metrics()

    def _log_performance_metrics(self):
        """Logs internal performance metrics at fixed intervals."""
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_log_time).nanoseconds / 1e9

        if elapsed_sec > self.log_interval:
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                max_time = np.max(self.inference_times)
                fps = 1000.0 / avg_time if avg_time > 0 else 0.0

                self.get_logger().info(
                    f"DEBUG: Avg Inference: {avg_time:.2f}ms ({fps:.2f}Hz) | "
                    f"Max: {max_time:.2f}ms"
                )
                self.inference_times.clear()
            
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = TinyLidarNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
