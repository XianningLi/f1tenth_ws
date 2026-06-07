#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

#include <limits>
#include <cmath>

using std::placeholders::_1;

class SafetyNode : public rclcpp::Node {
public:
  SafetyNode() 
  : Node("safety_node"), ttc_threshold_(1.5), current_speed_(0.0)
  {
    // Subscribe to LIDAR scan data (usually published on "/scan")
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10, std::bind(&SafetyNode::scan_callback, this, _1)
    );

    // Subscribe to odometry data to get the vehicle's current speed.
    // Note: Adjust the topic name if your simulator publishes odom data under a different namespace.
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/ego_racecar/odom", 10, std::bind(&SafetyNode::odom_callback, this, _1)
    );

    // Publisher for drive commands.
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      "/drive", 10
    );

    RCLCPP_INFO(this->get_logger(), "Safety Node Initialized (TTC threshold: %.2f s)", ttc_threshold_);
  }

private:
  // Parameters
  double ttc_threshold_;    // Time-to-collision threshold [seconds]
  double current_speed_;    // Current vehicle speed [m/s]

  // ROS Subscribers and Publisher
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  // Callback to update the current speed from odometry.
  void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    current_speed_ = msg->twist.twist.linear.x;
  }

  // Callback to process LIDAR scan data and compute the minimum TTC.
  void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg)
  {
    // Only consider TTC calculations if the car is moving forward.
    if (current_speed_ <= 0.0) {
      return;
    }

    double min_ttc = std::numeric_limits<double>::max();

    // Iterate over each laser beam in the scan.
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      double range = scan_msg->ranges[i];
      double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
      
      // Compute the component of speed in the direction of the laser beam.
      double closing_speed = current_speed_ * std::cos(angle);
      
      // Only consider beams that are pointing forward.
      if (closing_speed > 0.0) {
        double ttc = range / closing_speed;
        if (ttc < min_ttc) {
          min_ttc = ttc;
        }
      }
    }

    // If the minimum TTC is below the threshold, publish an emergency brake command.
    if (min_ttc < ttc_threshold_) {
      RCLCPP_WARN(this->get_logger(), "Emergency braking activated! TTC: %.2f s", min_ttc);

      ackermann_msgs::msg::AckermannDriveStamped brake_msg;
      brake_msg.header.stamp = this->now();
      brake_msg.drive.speed = 0.0;          // Force a complete stop.
      brake_msg.drive.steering_angle = 0.0;   // Maintain current steering (or set to 0 if desired).

      drive_pub_->publish(brake_msg);
    }
    // Otherwise, do nothing and allow normal driving commands to pass.
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SafetyNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

