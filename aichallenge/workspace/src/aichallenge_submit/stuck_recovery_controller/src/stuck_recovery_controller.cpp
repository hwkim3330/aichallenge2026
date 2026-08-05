#include "stuck_recovery_controller/stuck_recovery_controller.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>

namespace stuck_recovery_controller
{
namespace
{

constexpr float kStuckSpeedThreshold = 0.1;
constexpr double kStuckDurationSec = 0.4;  // faster detection (was 0.6s)
constexpr float kCommandSpeedThreshold = 1.0;
constexpr float kCommandAccelerationThreshold = 0.3;
constexpr float kMovingSpeedThreshold = 0.5;
constexpr float kRecoveryResetSpeedThreshold = 2.0;
constexpr double kRecoveryResetDurationSec = 1.0;
// A second, easier reset. Measured in a two-car prelim race: the attempt counter reached 28 on one car and
// 23 on the other, because the reset above needs 2.0 m/s sustained for 1.0 s and a car trading blows with
// another never gets there. Past attempt 3 deep_escape_mode_ latches on permanently, so every escape becomes
// full lock -- the worst choice in the 1.36 m gap where these wedges happen, and something the upstream node
// does not do at all (it has no deep escape). Modest progress is enough to prove the last escape worked.
constexpr float kAttemptResetSpeedThreshold = 0.6F;
constexpr double kAttemptResetDurationSec = 2.0;
constexpr double kReverseDurationSec = 1.8;
constexpr double kDeepReverseDurationSec = 2.6;
constexpr double kRecoveryCooldownSec = 1.2;
constexpr double kDriveSettleDurationSec = 0.2;  // was 0.3s
// Mutual-yield deadlock: two vehicles both choosing to stop for each other
// (e.g. under V2X obstacle avoidance) never trips the "commanded to move but
// can't" check above, since the nominal command itself is ~0. Detect a plain
// prolonged standstill regardless of what was commanded, and creep forward to
// break the tie instead of reversing (nothing is physically blocking us).
constexpr double kYieldTimeoutSec = 0.8;  // faster detection (was 1.5s)
constexpr double kCreepDurationSec = 0.6;  // shorter creep burst (was 1.0s)
constexpr float kCreepSpeed = 2.0;
constexpr float kCreepAcceleration = 0.5;
// Creeps that failed to restore motion before we stop believing this is a
// mutual yield and start reversing instead.
constexpr int kYieldCreepFailuresBeforeReverse = 2;

// Both 2026-08-02 escape changes are OFF by default. Measured together in the
// npc1 condition they did what they were designed to do -- longest stall fell
// from 42.8 s to 15.8 s and total stopped time from 60.1% to 55.4% -- but lap
// progress regressed, 3.46 -> 2.85 laps@480s with stalls rising 23 -> 37. The car
// escapes fast and re-wedges, which is the thrashing this repo already recorded
// for aggressive escapes. They were also changed together, so neither is
// attributable on its own; enable one at a time to settle it.
bool envFlag(const char * name)
{
  const char * raw = std::getenv(name);
  if (raw == nullptr) { return false; }
  const std::string value(raw);
  return !(value.empty() || value == "0" || value == "false");
}

// 0 keeps the verified blind alternation; 1 and 2 seed the first escape from the latched nominal
// steering with opposite sign conventions. Reversing with the wheels turned swings the tail one way
// and the nose the other, so which sign frees a jammed car is a question for measurement rather
// than for assertion -- hence two directed modes instead of one.
// Reproduce the upstream implementation exactly (upstream/agent/recovery-supervisor): detect for 1.0 s,
// reverse STRAIGHT for 4.0 s at -1.0, settle 0.5 s, and nothing else -- no cooldown, no creep, no yield
// handling, no deep escape, no steering alternation. Ours diverged from it on every one of those points,
// and the package README still describes the official behaviour, "直進で後退".
// Official escape GEOMETRY with our multi-car handling kept. Separate from RECOVERY_OFFICIAL because
// the official node has no yield handling, and in the scored three-car format two cars both commanding
// about zero to avoid each other would never resume.
// Recover a stationary car without waiting for the nominal command to ask for motion. Measured need:
// during a 73 s standstill the published command averaged 0.48 m/s, under the 1.0 gate, so recovery fired
// once every 9.1 s against a 3.2 s cycle. The upstream supervisor has the same gate, so this is a flaw
// both share rather than a divergence.
bool forceMode()
{
  const char * v = std::getenv("RECOVERY_FORCE");
  return v != nullptr && std::strcmp(v, "0") != 0 && *v != '\0';
}

constexpr double kForceStuckDurationSec = 1.2;
constexpr double kForceCooldownSec = 0.4;

bool straight4Mode()
{
  const char * v = std::getenv("RECOVERY_STRAIGHT4");
  return v != nullptr && std::strcmp(v, "0") != 0 && *v != '\0';
}

bool officialMode()
{
  const char * v = std::getenv("RECOVERY_OFFICIAL");
  return v != nullptr && std::strcmp(v, "0") != 0 && *v != '\0';
}

constexpr double kOfficialStuckDurationSec = 1.0;
constexpr double kOfficialReverseDurationSec = 4.0;
constexpr double kOfficialDriveSettleDurationSec = 0.5;
constexpr float kOfficialReverseSpeed = -1.0F;

int directedEscapeMode()
{
  // Default 2, measured on the official evaluation path with only this convention varied:
  //   0 (blind alternation)  n=4  mean 324.05 s  12 wall penalties
  //   1                      n=3  mean 354.99 s  12, and one run at 499.04 s, past the 480 s cap
  //   2                      n=6  mean 287.08 s   7, three runs clean at 268.5 s
  // So the escape steering takes the OPPOSITE sign to the latched nominal steering, which is also what
  // the kinematics say: turning the wheels while reversing sends the rear of the car the other way.
  // Convention 1 is worse than doing nothing, so guessing the sign would have been a coin flip on a
  // change that can make things worse rather than merely fail to help.
  //
  // This has to be the compiled default rather than an environment variable: the organisers' evaluation
  // sets none, so an env-gated improvement ships nothing. RECOVERY_DIRECTED=0 still forces the old blind
  // alternation for A/B work.
  const char * v = std::getenv("RECOVERY_DIRECTED");
  if (v == nullptr) {
    return 2;
  }
  if (std::strcmp(v, "0") == 0) {
    return 0;
  }
  if (std::strcmp(v, "1") == 0) {
    return 1;
  }
  if (std::strcmp(v, "2") == 0) {
    return 2;
  }
  return 2;
}

}  // namespace

StuckRecoveryController::StuckRecoveryController() : Node("stuck_recovery_controller")
{
  // The online race launches one identical submission per vehicle.  Use the
  // vehicle's ROS domain as a deterministic role selector: the first car
  // stays conservative, while cars starting behind get a stronger lateral
  // escape during recovery instead of pushing straight into the bumper ahead.
  if (const char * domain = std::getenv("ROS_DOMAIN_ID")) {
    const int domain_id = std::atoi(domain);
    if (domain_id == 2) {
      recovery_steering_ = 0.45F;
    } else if (domain_id >= 3) {
      recovery_steering_ = 0.60F;
    } else {
      recovery_steering_ = 0.15F;
    }
  }

  control_pub_ = create_publisher<AckermannControlCommand>("/control/command/control_cmd", 1);
  gear_pub_ = create_publisher<GearCommand>("/control/command/gear_cmd", 1);

  nominal_sub_ = create_subscription<AckermannControlCommand>(
    "/control/command/nominal_control_cmd", 1,
    std::bind(&StuckRecoveryController::onNominalCommand, this, std::placeholders::_1));
  velocity_sub_ = create_subscription<VelocityReport>(
    "/vehicle/status/velocity_status", 1,
    [this](const VelocityReport::ConstSharedPtr msg) {
      latest_velocity_ = msg->longitudinal_velocity;
    });
}

void StuckRecoveryController::onNominalCommand(
  const AckermannControlCommand::ConstSharedPtr msg)
{
  const auto now = this->now();
  if (runRecovery(now)) {
    return;
  }
  control_pub_->publish(*msg);
  // Latch the steering direction the MPC was asking for while it still had one, for the escape
  // to aim by. Only while not in recovery, so the recovery's own commands never feed this back.
  constexpr float kMeaningfulSteering = 0.05F;
  if (std::abs(msg->lateral.steering_tire_angle) > kMeaningfulSteering) {
    last_meaningful_steering_ = msg->lateral.steering_tire_angle;
  }
  updateStuckDetection(*msg, now);
}

void StuckRecoveryController::updateStuckDetection(
  const AckermannControlCommand & command, const rclcpp::Time & now)
{
  const float velocity = latest_velocity_;
  // Require movement once to avoid detecting the initial stationary state as stuck.
  if (velocity >= kMovingSpeedThreshold) {
    moving_observed_ = true;
    yield_recovery_attempted_ = false;
    // A brief wheel-speed spike while still pressed against a wall must not
    // reset the recovery episode.  Reset only after sustained forward motion.
    if (velocity >= kRecoveryResetSpeedThreshold) {
      if (!forward_progress_start_time_.has_value()) {
        forward_progress_start_time_ = now;
      } else if (
        (now - forward_progress_start_time_.value()).seconds() >=
        kRecoveryResetDurationSec)
      {
        recovery_attempts_ = 0;
        deep_escape_mode_ = false;
        yield_creep_failures_ = 0;
        recovery_cooldown_until_.reset();
        forward_progress_start_time_.reset();
      }
    } else {
      forward_progress_start_time_.reset();
    }
    // Easier reset: sustained modest motion clears the escalation, so a car that is making progress
    // again does not keep escaping at full lock on the strength of attempts it already recovered from.
    static const bool kAttemptReset = envFlag("RECOVERY_ATTEMPT_RESET");
    if (kAttemptReset && velocity >= kAttemptResetSpeedThreshold) {
      if (!attempt_reset_start_time_.has_value()) {
        attempt_reset_start_time_ = now;
      } else if ((now - attempt_reset_start_time_.value()).seconds() >= kAttemptResetDurationSec) {
        if (recovery_attempts_ > 0) {
          RCLCPP_INFO(
            get_logger(), "progress for %.1f s at >= %.1f m/s: clearing %d attempts",
            kAttemptResetDurationSec, kAttemptResetSpeedThreshold, recovery_attempts_);
        }
        recovery_attempts_ = 0;
        deep_escape_mode_ = false;
        attempt_reset_start_time_.reset();
      }
    } else {
      attempt_reset_start_time_.reset();
    }
  } else {
    forward_progress_start_time_.reset();
    attempt_reset_start_time_.reset();
  }

  static const bool kForce = forceMode();
  if (kForce && moving_observed_ && !recovery_start_time_.has_value() &&
    std::abs(velocity) <= kStuckSpeedThreshold)
  {
    if (!force_stuck_start_time_.has_value()) {
      force_stuck_start_time_ = now;
    } else if ((now - force_stuck_start_time_.value()).seconds() >= kForceStuckDurationSec) {
      // The cooldown exists to stop a MOVING car thrashing between nominal and recovery, which cannot
      // happen to one sitting still, so only a short remainder of it is honoured here.
      const bool cooling = recovery_cooldown_until_.has_value() &&
        now < recovery_cooldown_until_.value() &&
        (recovery_cooldown_until_.value() - now).seconds() > kForceCooldownSec;
      if (!cooling) {
        force_stuck_start_time_.reset();
        stuck_start_time_.reset();
        yield_start_time_.reset();
        creep_mode_ = false;
        recovery_start_time_ = now;
        ++recovery_attempts_;
        deep_escape_mode_ = recovery_attempts_ >= 3;
        recovery_steering_ = -recovery_steering_;
        RCLCPP_INFO(
          get_logger(),
          "stuck detected (forced, nominal gate bypassed): velocity=%.3f attempt=%d",
          velocity, recovery_attempts_);
        return;
      }
    }
  } else if (std::abs(velocity) > kStuckSpeedThreshold) {
    force_stuck_start_time_.reset();
  }

  static const bool kOfficial = officialMode();
  // The official node has no yield handling, no cooldown and no escalation. In official mode the
  // detector is only the upstream one: nominal asking for motion, velocity under threshold for
  // kOfficialStuckDurationSec, then reverse.
  if (kOfficial) {
    if (!moving_observed_ || command.longitudinal.speed < kCommandSpeedThreshold ||
      command.longitudinal.acceleration < kCommandAccelerationThreshold)
    {
      stuck_start_time_.reset();
      return;
    }
    if (std::abs(velocity) <= kStuckSpeedThreshold) {
      if (!stuck_start_time_.has_value()) {
        stuck_start_time_ = now;
      } else if ((now - stuck_start_time_.value()).seconds() >= kOfficialStuckDurationSec) {
        stuck_start_time_.reset();
        recovery_start_time_ = now;
        creep_mode_ = false;
        RCLCPP_INFO(get_logger(), "stuck detected (official): velocity=%.3f", velocity);
      }
    } else {
      stuck_start_time_.reset();
    }
    return;
  }

  // Mutual-yield deadlock check: runs regardless of what was commanded, as
  // long as we've moved at least once before. Two vehicles both commanding
  // ~0 to avoid each other otherwise never resumes on its own.
  if (moving_observed_ && !yield_recovery_attempted_ &&
    std::abs(velocity) <= kStuckSpeedThreshold)
  {
    if (!yield_start_time_.has_value()) {
      yield_start_time_ = now;
    } else if ((now - yield_start_time_.value()).seconds() >= kYieldTimeoutSec) {
      yield_start_time_.reset();
      recovery_start_time_ = now;
      creep_mode_ = true;
      yield_recovery_attempted_ = true;
      RCLCPP_INFO(get_logger(), "yield deadlock detected: velocity=%.3f", velocity);
      return;
    }
  } else {
    yield_start_time_.reset();
  }

  if (
    !moving_observed_ || command.longitudinal.speed < kCommandSpeedThreshold ||
    command.longitudinal.acceleration < kCommandAccelerationThreshold)
  {
    stuck_start_time_.reset();
    return;
  }

  if (std::abs(velocity) <= kStuckSpeedThreshold) {
    if (!stuck_start_time_.has_value()) {
      stuck_start_time_ = now;
    } else if ((now - stuck_start_time_.value()).seconds() >= kStuckDurationSec) {
      if (recovery_cooldown_until_.has_value() &&
        now < recovery_cooldown_until_.value())
      {
        // Give the vehicle a short window to regain traction after an
        // escape.  Without this guard, a wall contact is detected again on
        // the very next command and the controller loops forever at the same
        // corner.
        stuck_start_time_.reset();
        return;
      }
      stuck_start_time_.reset();
      recovery_start_time_ = now;
      creep_mode_ = false;
      ++recovery_attempts_;
      deep_escape_mode_ = recovery_attempts_ >= 3;
      static const int kDirected = directedEscapeMode();
      const float mag = std::abs(recovery_steering_);
      if (kDirected != 0 && recovery_attempts_ == 1 &&
        std::abs(last_meaningful_steering_) > 0.0F)
      {
        // First attempt of this stall: aim it rather than guess. Later attempts still alternate,
        // so a seed with the wrong sign costs one burst instead of repeating forever.
        const float sign = last_meaningful_steering_ >= 0.0F ? 1.0F : -1.0F;
        recovery_steering_ = (kDirected == 1 ? sign : -sign) * mag;
      } else {
        recovery_steering_ = -recovery_steering_;
      }
      RCLCPP_INFO(
        get_logger(),
        "stuck detected: velocity=%.3f attempt=%d deep_escape=%s directed=%d "
        "nominal_steer=%.3f escape_steer=%.3f",
        velocity, recovery_attempts_, deep_escape_mode_ ? "true" : "false", kDirected,
        last_meaningful_steering_, recovery_steering_);
    }
  } else {
    stuck_start_time_.reset();
  }
}

bool StuckRecoveryController::runRecovery(const rclcpp::Time & now)
{
  if (!recovery_start_time_.has_value()) {
    return false;
  }

  const double elapsed = (now - recovery_start_time_.value()).seconds();

  static const bool kOfficial = officialMode();
  if (kOfficial) {
    // STEP1 reverse straight, STEP2 settle in DRIVE, STEP3 resume. Steering is left at 0 by
    // publishCommand's default, which is what the upstream node hardcodes.
    if (elapsed < kOfficialReverseDurationSec) {
      publishGear(GearCommand::REVERSE);
      publishCommand(kOfficialReverseSpeed, 1.0);
      return true;
    }
    if (elapsed < kOfficialReverseDurationSec + kOfficialDriveSettleDurationSec) {
      publishGear(GearCommand::DRIVE);
      publishCommand(0.0, 0.0);
      return true;
    }
    publishGear(GearCommand::DRIVE);
    recovery_start_time_.reset();
    return false;
  }

  if (creep_mode_) {
    // Nothing is physically blocking us in the yield-deadlock case, so just
    // creep forward instead of reversing.
    if (elapsed < kCreepDurationSec) {
      publishGear(GearCommand::DRIVE);
      publishCommand(kCreepSpeed, kCreepAcceleration);
      return true;
    }
    creep_mode_ = false;
    // Did the creep actually free us? If the car is still stationary it was not a
    // mutual yield at all -- it is jammed, and creeping forward into whatever is
    // holding it will never help.
    //
    // This matters because the reverse manoeuvre, the only thing that can extract a
    // jammed car, lives behind the stuck detector, which requires the nominal
    // command to be asking for motion (speed >= 1.0 and accel >= 0.3). When the QP
    // goes infeasible the MPC commands ~0, so that gate stays shut. Measured
    // 2026-08-02: during a 144 s standstill the nominal command met the stuck gate
    // in only 16.6% of samples while the car was below 0.1 m/s for 71.5% of it, and
    // recovery never reversed once. Escalate here instead of waiting for a gate that
    // is not going to open.
    static const bool kCreepEscalationEnabled = envFlag("RECOVERY_CREEP_ESCALATION");
    if (kCreepEscalationEnabled && std::abs(latest_velocity_) <= kStuckSpeedThreshold) {
      ++yield_creep_failures_;
      if (yield_creep_failures_ >= kYieldCreepFailuresBeforeReverse) {
        recovery_start_time_ = now;
        ++recovery_attempts_;
        deep_escape_mode_ = recovery_attempts_ >= 3;
        recovery_steering_ = -recovery_steering_;
        RCLCPP_INFO(
          get_logger(),
          "yield creep failed %d times, escalating to reverse escape: attempt=%d deep=%s",
          yield_creep_failures_, recovery_attempts_, deep_escape_mode_ ? "true" : "false");
        return true;
      }
      // Allow the yield detector to fire again rather than latching after one try.
      yield_recovery_attempted_ = false;
    } else {
      yield_creep_failures_ = 0;
    }
    recovery_start_time_.reset();
    return false;
  }

  static const bool kStraight4 = straight4Mode();
  const double reverse_duration = kStraight4
    ? kOfficialReverseDurationSec
    : (deep_escape_mode_ ? kDeepReverseDurationSec : kReverseDurationSec);
  // Deep escape used to alternate only between +0.55 and -0.55 -- full lock both
  // ways. Measured 2026-08-02 in the npc1 condition: three stalls of 21-43 s where
  // every lidar sector read 0.05 m and the node was commanding full-lock reverse the
  // whole time without the car moving. Reversing at full lock swings the tail
  // sideways, and when the car is jammed there is no lateral room to swing into.
  // Straight reverse needs the least clearance and was never being tried, so cycle
  // through it every third attempt.
  static const bool kStraightEscapeEnabled = envFlag("RECOVERY_STRAIGHT_ESCAPE");
  const bool straight_escape =
    kStraightEscapeEnabled && deep_escape_mode_ && (recovery_attempts_ % 3 == 0);
  const float escape_steering = !deep_escape_mode_
    ? recovery_steering_
    : (straight_escape ? 0.0F : (recovery_steering_ >= 0.0F ? 0.55F : -0.55F));

  // STEP1. Reverse.  After repeated failed attempts, use a longer escape
  // burst so the vehicle can clear the wall instead of settling back into it.
  if (elapsed < reverse_duration) {
    publishGear(GearCommand::REVERSE);
    // AWSIM expects positive acceleration with reverse gear and negative target speed.
    if (kStraight4) {
      // Straight and slower, the upstream values: least clearance needed, and no tail swing into a
      // wall that is 1.36 m away at the place this fires.
      publishCommand(kOfficialReverseSpeed, 1.0, 0.0F);
    } else {
      publishCommand(deep_escape_mode_ ? -2.0F : -1.5F, 1.0, escape_steering);
    }
    return true;
  }

  // STEP2. Shift to DRIVE and stop for kDriveSettleDurationSec.
  const double settle = kStraight4 ? kOfficialDriveSettleDurationSec : kDriveSettleDurationSec;
  if (elapsed < reverse_duration + settle) {
    publishGear(GearCommand::DRIVE);
    publishCommand(0.0, 0.0);
    return true;
  }

  // STEP3. Finish recovery and resume nominal commands.
  // If STEP2 was skipped due to a callback dropout, gear would still be REVERSE, so publish DRIVE here too.
  publishGear(GearCommand::DRIVE);
  recovery_start_time_.reset();
  recovery_cooldown_until_ = now + rclcpp::Duration::from_seconds(kRecoveryCooldownSec);
  return false;
}

void StuckRecoveryController::publishCommand(float speed, float acceleration, float steering)
{
  const auto stamp = this->now();
  AckermannControlCommand msg;
  msg.stamp = stamp;
  msg.lateral.stamp = stamp;
  msg.lateral.steering_tire_angle = steering;
  msg.lateral.steering_tire_rotation_rate = 2.0;
  msg.longitudinal.stamp = stamp;
  msg.longitudinal.speed = speed;
  msg.longitudinal.acceleration = acceleration;
  control_pub_->publish(msg);
}

void StuckRecoveryController::publishGear(std::uint8_t command)
{
  GearCommand msg;
  msg.stamp = this->now();
  msg.command = command;
  gear_pub_->publish(msg);
}

}  // namespace stuck_recovery_controller

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<stuck_recovery_controller::StuckRecoveryController>());
  rclcpp::shutdown();
  return 0;
}
