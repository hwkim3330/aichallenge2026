#include "stuck_recovery_controller/stuck_recovery_controller.hpp"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>

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
        recovery_cooldown_until_.reset();
        forward_progress_start_time_.reset();
      }
    } else {
      forward_progress_start_time_.reset();
    }
  } else {
    forward_progress_start_time_.reset();
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
      recovery_steering_ = -recovery_steering_;
      RCLCPP_INFO(
        get_logger(), "stuck detected: velocity=%.3f attempt=%d deep_escape=%s",
        velocity, recovery_attempts_, deep_escape_mode_ ? "true" : "false");
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

  if (creep_mode_) {
    // Nothing is physically blocking us in the yield-deadlock case, so just
    // creep forward instead of reversing.
    if (elapsed < kCreepDurationSec) {
      publishGear(GearCommand::DRIVE);
      publishCommand(kCreepSpeed, kCreepAcceleration);
      return true;
    }
    recovery_start_time_.reset();
    creep_mode_ = false;
    return false;
  }

  const double reverse_duration =
    deep_escape_mode_ ? kDeepReverseDurationSec : kReverseDurationSec;
  const float escape_steering = deep_escape_mode_
    ? (recovery_steering_ >= 0.0F ? 0.55F : -0.55F)
    : recovery_steering_;

  // STEP1. Reverse.  After repeated failed attempts, use a longer escape
  // burst so the vehicle can clear the wall instead of settling back into it.
  if (elapsed < reverse_duration) {
    publishGear(GearCommand::REVERSE);
    // AWSIM expects positive acceleration with reverse gear and negative target speed.
    publishCommand(deep_escape_mode_ ? -2.0F : -1.5F, 1.0, escape_steering);
    return true;
  }

  // STEP2. Shift to DRIVE and stop for kDriveSettleDurationSec.
  if (elapsed < reverse_duration + kDriveSettleDurationSec) {
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
