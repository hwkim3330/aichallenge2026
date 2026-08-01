#ifndef STUCK_RECOVERY_CONTROLLER_HPP_
#define STUCK_RECOVERY_CONTROLLER_HPP_

#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_vehicle_msgs/msg/gear_command.hpp>
#include <autoware_auto_vehicle_msgs/msg/velocity_report.hpp>

#include <cstdint>
#include <optional>

namespace stuck_recovery_controller
{

using autoware_auto_control_msgs::msg::AckermannControlCommand;
using autoware_auto_vehicle_msgs::msg::GearCommand;
using autoware_auto_vehicle_msgs::msg::VelocityReport;

class StuckRecoveryController : public rclcpp::Node
{
public:
  StuckRecoveryController();

private:
  void onNominalCommand(const AckermannControlCommand::ConstSharedPtr msg);
  void updateStuckDetection(
    const AckermannControlCommand & command, const rclcpp::Time & now);
  bool runRecovery(const rclcpp::Time & now);
  void publishCommand(float speed, float acceleration, float steering = 0.0F);
  void publishGear(std::uint8_t command);

  rclcpp::Publisher<AckermannControlCommand>::SharedPtr control_pub_;
  rclcpp::Publisher<GearCommand>::SharedPtr gear_pub_;
  rclcpp::Subscription<AckermannControlCommand>::SharedPtr nominal_sub_;
  rclcpp::Subscription<VelocityReport>::SharedPtr velocity_sub_;

  float latest_velocity_{0.0};
  bool moving_observed_{false};
  std::optional<rclcpp::Time> stuck_start_time_;
  std::optional<rclcpp::Time> recovery_start_time_;
  std::optional<rclcpp::Time> recovery_cooldown_until_;
  std::optional<rclcpp::Time> forward_progress_start_time_;
  std::optional<rclcpp::Time> yield_start_time_;
  bool creep_mode_{false};
  // Do not repeatedly re-trigger creep while the vehicle remains stationary.
  // A new yield recovery is allowed only after real forward movement resumes.
  bool yield_recovery_attempted_{false};
  // Alternate the escape side on consecutive physical recoveries.  This is
  // useful when the vehicle is boxed in by the car immediately ahead.
  float recovery_steering_{0.45F};
  int recovery_attempts_{0};
  bool deep_escape_mode_{false};
  // Consecutive yield creeps that did not restore motion. A jammed car looks
  // like a mutual-yield standstill to the detector, but a 0.6 s forward creep
  // cannot free it; after this many failures escalate to the reverse escape.
  int yield_creep_failures_{0};
};

}  // namespace stuck_recovery_controller

#endif  // STUCK_RECOVERY_CONTROLLER_HPP_
