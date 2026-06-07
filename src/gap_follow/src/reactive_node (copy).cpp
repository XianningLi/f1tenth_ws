#include "rclcpp/rclcpp.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

class ReactiveFollowGap : public rclcpp::Node {
public:
  ReactiveFollowGap() : Node("reactive_follow_gap_node")
  {
    // Topics (ensure these match your simulator configuration)
    lidarscan_topic_ = "/scan";
    drive_topic_ = "/drive";

    // Create subscriber for LiDAR data
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      lidarscan_topic_, 10,
      std::bind(&ReactiveFollowGap::lidar_callback, this, std::placeholders::_1)
    );

    // Create publisher for drive commands
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      drive_topic_, 10
    );

    RCLCPP_INFO(this->get_logger(), "Reactive Follow Gap Node Initialized.");
  }

private:
  // Topics
  std::string lidarscan_topic_;
  std::string drive_topic_;

  // Subscribers and publishers
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  // Constants / Parameters
  const float MAX_RANGE = 3.0;    // Cap maximum range to 3.0 meters
  const int SMOOTH_WINDOW = 5;    // Window size for moving average
  const int BUBBLE_RADIUS = 10;   // Number of indices to remove around the closest obstacle
  const float VELOCITY = 1.5;     // Constant speed command
  // (Additional tuning parameters can be added here.)

  // Preprocess the LiDAR scan: smooth and threshold ranges.
  void preprocess_lidar(std::vector<float>& ranges)
  {
    int n = ranges.size();
    std::vector<float> smoothed = ranges;
    for (int i = 0; i < n; i++) {
      float sum = 0.0;
      int count = 0;
      int start = std::max(0, i - SMOOTH_WINDOW / 2);
      int end = std::min(n - 1, i + SMOOTH_WINDOW / 2);
      for (int j = start; j <= end; j++) {
        sum += ranges[j];
        count++;
      }
      smoothed[i] = sum / count;
      if (smoothed[i] > MAX_RANGE)
        smoothed[i] = MAX_RANGE;
    }
    ranges = smoothed;
  }

  // Find the maximum contiguous gap (free space) in the preprocessed ranges.
  void find_max_gap(const std::vector<float>& ranges, int &start_idx, int &end_idx)
  {
    int best_start = 0, best_end = 0, best_length = 0;
    int current_start = -1;
    int n = ranges.size();
    for (int i = 0; i < n; i++) {
      if (ranges[i] > 0.0) { // Consider nonzero values as free
        if (current_start == -1)
          current_start = i;
      } else {
        if (current_start != -1) {
          int length = i - current_start;
          if (length > best_length) {
            best_length = length;
            best_start = current_start;
            best_end = i - 1;
          }
          current_start = -1;
        }
      }
    }
    // Check if the free gap extends to the end of the scan
    if (current_start != -1) {
      int length = n - current_start;
      if (length > best_length) {
        best_start = current_start;
        best_end = n - 1;
      }
    }
    start_idx = best_start;
    end_idx = best_end;
  }

  // Within the identified gap, find the "best" point.
  // Naively, we choose the index with the maximum range.
  int find_best_point(const std::vector<float>& ranges, int start_idx, int end_idx)
  {
    int best_index = start_idx;
    float best_range = 0.0;
    for (int i = start_idx; i <= end_idx; i++) {
      if (ranges[i] > best_range) {
        best_range = ranges[i];
        best_index = i;
      }
    }
    return best_index;
  }

  // LiDAR callback: process scan and publish an Ackermann drive command.
  void lidar_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg)
  {
    // Copy the scan data to a local vector for processing.
    std::vector<float> proc_ranges = scan_msg->ranges;
    int n = proc_ranges.size();

    // Preprocess LiDAR data (smoothing and capping maximum values)
    preprocess_lidar(proc_ranges);

    // Find the closest point in the scan (to create a safety bubble)
    float min_range = std::numeric_limits<float>::max();
    int min_index = 0;
    for (int i = 0; i < n; i++) {
      if (proc_ranges[i] < min_range) {
        min_range = proc_ranges[i];
        min_index = i;
      }
    }

    // Eliminate points in a "bubble" around the closest point.
    int start_bubble = std::max(0, min_index - BUBBLE_RADIUS);
    int end_bubble = std::min(n - 1, min_index + BUBBLE_RADIUS);
    for (int i = start_bubble; i <= end_bubble; i++) {
      proc_ranges[i] = 0.0;
    }

    // Find the maximum gap in the remaining free space.
    int gap_start = 0, gap_end = 0;
    find_max_gap(proc_ranges, gap_start, gap_end);

    // Find the best point within the maximum gap.
    int best_index = find_best_point(proc_ranges, gap_start, gap_end);

    // Calculate the angle corresponding to the best point.
    float best_angle = scan_msg->angle_min + best_index * scan_msg->angle_increment;

    // Create and publish the drive command.
    ackermann_msgs::msg::AckermannDriveStamped drive_msg;
    drive_msg.header.stamp = this->now();
    drive_msg.drive.speed = VELOCITY;
    drive_msg.drive.steering_angle = best_angle;  // Adjust sign if needed based on coordinate system

    drive_pub_->publish(drive_msg);

    // For debugging, you may log useful info:
    RCLCPP_INFO(this->get_logger(), "Best gap: [%d, %d] Best index: %d, Best angle: %.2f",
                gap_start, gap_end, best_index, best_angle);
  }
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ReactiveFollowGap>());
  rclcpp::shutdown();
  return 0;
}

