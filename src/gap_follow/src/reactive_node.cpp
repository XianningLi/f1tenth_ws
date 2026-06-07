#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

class ReactiveGapFollower : public rclcpp::Node {
public:
  ReactiveGapFollower() : Node("reactive_gap_follower")
  {
    // Set topics (make sure these match your simulation setup)
    lidarscan_topic_ = "/scan";
    drive_topic_ = "/drive";

    // Create the subscriber for LiDAR scans
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      lidarscan_topic_, 1000,
      std::bind(&ReactiveGapFollower::callback, this, std::placeholders::_1)
    );

    // Create the publisher for drive commands
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      drive_topic_, 1000
    );

    RCLCPP_INFO(this->get_logger(), "Reactive Gap Follower node initialized.");
  }

private:
  // Topics
  std::string lidarscan_topic_;
  std::string drive_topic_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  // Variables for processing
  double angle_;                     // Final chosen steering angle
  std::vector<double> ranges_;       // Processed LiDAR ranges

  // Constants / Parameters
  static constexpr double KP = 1.5;
  static constexpr double KD = 0.05;
  static constexpr double KI = 0.25;
  static constexpr double SERVO_OFFSET = 0.00;
  static constexpr int ANGLE_RANGE = 270; // (not directly used in this code)
  static constexpr double DESIRED_DISTANCE_RIGHT = 0.9;
  static constexpr double DESIRED_DISTANCE_LEFT = 1.20;
  static constexpr double CAR_LENGTH = 0.50;
  static constexpr double PI = 3.1415927;

  // Callback: process LiDAR data and compute drive command
  void callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr lidar_info) {
    // Copy raw LiDAR data into a local vector (convert from float to double)
    ranges_ = std::vector<double>(lidar_info->ranges.begin(), lidar_info->ranges.end());
    size_t n = ranges_.size();

    // Define the indices for angles between -70 and 70 degrees
    double min_angle = -70.0 / 180.0 * PI;
    double max_angle = 70.0 / 180.0 * PI;
    unsigned int min_indx = static_cast<unsigned int>(std::floor((min_angle - lidar_info->angle_min) / lidar_info->angle_increment));
    unsigned int max_indx = static_cast<unsigned int>(std::ceil((max_angle - lidar_info->angle_min) / lidar_info->angle_increment));
    if (max_indx >= n)
      max_indx = n - 1;

    // Preprocess: For indices between min_indx and max_indx, filter out invalid values
    for (unsigned int i = min_indx; i <= max_indx && i < n; i++) {
      if (std::isinf(lidar_info->ranges[i]) || std::isnan(lidar_info->ranges[i])) {
        ranges_[i] = 0.0;
      } else if (lidar_info->ranges[i] > lidar_info->range_max) {
        ranges_[i] = lidar_info->range_max;
      }
    }

    // 1. Find the closest point in the selected region (using a window of 5 points)
    unsigned int closest_indx = min_indx;
    double closest_distance = lidar_info->range_max * 5.0; // large initial value
    // To avoid boundary issues, start at min_indx+2 and end at max_indx-2
    for (unsigned int i = min_indx + 2; i + 2 <= max_indx && i < n; i++) {
      double sum = ranges_[i - 2] + ranges_[i - 1] + ranges_[i] + ranges_[i + 1] + ranges_[i + 2];
      if (sum < closest_distance) {
        closest_distance = sum;
        closest_indx = i;
      }
    }

    // 2. Eliminate points inside a "bubble" around the closest point
    unsigned int radius = 150;
    unsigned int start_bubble = (closest_indx > radius) ? closest_indx - radius : 0;
    unsigned int end_bubble = std::min(static_cast<unsigned int>(n - 1), closest_indx + radius);
    for (unsigned int i = start_bubble; i <= end_bubble; i++) {
      ranges_[i] = 0.0;
    }

    // 3. Find the maximum gap in the free-space region within [min_indx, max_indx]
    unsigned int best_start = min_indx;
    unsigned int best_end = min_indx;
    unsigned int current_start = min_indx;
    unsigned int longest_gap = 0;
    bool in_gap = false;

    for (unsigned int i = min_indx; i <= max_indx && i < n; i++) {
      if (ranges_[i] > 0.0) {
        if (!in_gap) {
          in_gap = true;
          current_start = i;
        }
      } else {
        if (in_gap) {
          unsigned int gap_length = i - current_start;
          if (gap_length > longest_gap) {
            longest_gap = gap_length;
            best_start = current_start;
            best_end = i - 1;
          }
          in_gap = false;
        }
      }
    }
    // Check for gap reaching to max_indx
    if (in_gap) {
      unsigned int gap_length = max_indx - current_start + 1;
      if (gap_length > longest_gap) {
        best_start = current_start;
        best_end = max_indx;
      }
    }

    // 4. Within the maximum gap, find the best point (naively: the point with maximum range)
    double current_max = 0.0;
    angle_ = 0.0;
    for (unsigned int i = best_start; i <= best_end && i < n; i++) {
      if (ranges_[i] > current_max) {
        current_max = ranges_[i];
        angle_ = lidar_info->angle_min + i * lidar_info->angle_increment;
      } else if (ranges_[i] == current_max) {
        double candidate_angle = lidar_info->angle_min + i * lidar_info->angle_increment;
        if (std::abs(candidate_angle) < std::abs(angle_)) {
          angle_ = candidate_angle;
        }
      }
    }

    RCLCPP_INFO(this->get_logger(), "Best gap: [%u, %u], Chosen angle: %.2f rad", best_start, best_end, angle_);

    // 5. Publish drive command based on the chosen angle
    reactive_control();
  }

  // Publish drive commands based on the computed steering angle.
  void reactive_control() {
    auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
    drive_msg.header.stamp = this->now();
    drive_msg.drive.steering_angle = angle_;

    // Choose speed based on the magnitude of the steering angle (in radians)
    double abs_angle_deg = std::abs(angle_) * 180.0 / PI;
    if (abs_angle_deg > 20.0) {
      drive_msg.drive.speed = 0.5;
    } else if (abs_angle_deg > 10.0) {
      drive_msg.drive.speed = 1.0;
    } else {
      drive_msg.drive.speed = 1.5;
    }

    drive_pub_->publish(drive_msg);
    RCLCPP_INFO(this->get_logger(), "Published steering angle: %.2f rad", angle_);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ReactiveGapFollower>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

