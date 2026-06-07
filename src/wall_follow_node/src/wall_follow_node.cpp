#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <cmath>
#include <limits>

class WallFollowNode : public rclcpp::Node {
public:
  WallFollowNode()
  : Node("wall_follow_node"),
    // Parameters as in the F1Tenth Lab3 template
    desired_distance_(1.0), // Desired distance from the right wall [meters]
    k_(0.8),               // Proportional gain for steering control
    L_(1.0),               // Lookahead distance [meters]
    speed_(20.0),           // Constant forward speed [m/s]
    phi_(M_PI/4)           // Angle offset (45° in radians) for the first measurement
  {
    // Subscribe to the laser scan topic (usually "/scan")
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10, std::bind(&WallFollowNode::scan_callback, this, std::placeholders::_1)
    );

    // Publisher for drive commands (usually "/drive")
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      "/drive", 10
    );

    RCLCPP_INFO(this->get_logger(), "Wall Follow Node Initialized");
  }

private:
  // Parameters
  double desired_distance_;
  double k_;
  double L_;
  double speed_;
  double phi_;  // Offset angle (radians) for the first laser measurement

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  // Callback processing the laser scan message for wall following
  void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // Define the angles (in radians) for the two measurements:
    // b: directly to the right (–90°)
    // a: offset by φ from the right (i.e. –90° + φ)
    double angle_b = -M_PI / 2.0;
    double angle_a = angle_b + phi_;

    // Verify that these angles are within the laser scan range.
    if (angle_a < scan_msg->angle_min || angle_b < scan_msg->angle_min ||
        angle_a > scan_msg->angle_max || angle_b > scan_msg->angle_max) {
      RCLCPP_WARN(this->get_logger(), "Desired laser angles (%.2f, %.2f) out of scan range (%.2f, %.2f).",
                  angle_a, angle_b, scan_msg->angle_min, scan_msg->angle_max);
      return;
    }

    // Calculate the corresponding indices in the scan array.
    int index_b = static_cast<int>((angle_b - scan_msg->angle_min) / scan_msg->angle_increment);
    int index_a = static_cast<int>((angle_a - scan_msg->angle_min) / scan_msg->angle_increment);

    // Check index validity.
    if (index_a < 0 || index_a >= static_cast<int>(scan_msg->ranges.size()) ||
        index_b < 0 || index_b >= static_cast<int>(scan_msg->ranges.size())) {
      RCLCPP_WARN(this->get_logger(), "Calculated laser scan indices out of range.");
      return;
    }

    double b = scan_msg->ranges[index_b];
    double a = scan_msg->ranges[index_a];

    // Check that both measurements are valid.
    if (!std::isfinite(a) || !std::isfinite(b)) {
      RCLCPP_WARN(this->get_logger(), "Invalid laser scan measurements: a = %.2f, b = %.2f", a, b);
      return;
    }

    // Compute the wall’s angle (alpha) relative to the vehicle.
    // Formula: alpha = arctan((a*cos(phi) - b) / (a*sin(phi)))
    double alpha = std::atan((a * std::cos(phi_) - b) / (a * std::sin(phi_)));

    // Current distance from the wall (projection using measurement b)
    double D_t = b * std::cos(alpha);
    // Predicted (projected) distance from the wall after moving ahead by lookahead distance L_
    double D_t1 = D_t + L_ * std::sin(alpha);

    // Compute the error between the desired distance and the projected distance.
    double error = desired_distance_ - D_t1;

    // Compute the steering angle command using a proportional controller.
    double steering_angle = k_ * error;

    // Build and publish the drive command.
    ackermann_msgs::msg::AckermannDriveStamped drive_msg;
    drive_msg.header.stamp = this->now();
    drive_msg.drive.speed = speed_;
    drive_msg.drive.steering_angle = steering_angle;
    drive_pub_->publish(drive_msg);

    // Log the values for debugging.
    RCLCPP_INFO(this->get_logger(), 
                "Laser measurements: a = %.2f, b = %.2f, alpha = %.2f, D_t = %.2f, D_t1 = %.2f, error = %.2f, steer = %.2f",
                a, b, alpha, D_t, D_t1, error, steering_angle);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<WallFollowNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

