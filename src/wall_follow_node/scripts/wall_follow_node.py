#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

# PID control parameters
kp = 1.5
kd = 0.05
ki = 0.25
servo_offset = 0.0
prev_error = 0.0 
integral = 0.0
prev_time = 0.0

# Wall follow parameters
ANGLE_RANGE = 270              # Hokuyo 10LX has 270° scan
DESIRED_DISTANCE_RIGHT = 0.9   # meters
DESIRED_DISTANCE_LEFT = 0.85   # meters
VELOCITY = 5.0                 # m/s
CAR_LENGTH = 1.0               # meters

class WallFollow(Node):
    def __init__(self):
        super().__init__('wall_follow_node')
        global prev_time
        # Initialize time using the node's clock
        prev_time = self.get_clock().now().nanoseconds / 1e9

        # Subscribe to the LIDAR scan topic
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # Publisher for drive commands (publish on /nav as per your configuration)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        self.get_logger().info("WallFollow node initialized.")

    def get_range(self, data, angle):
        """
        Get distance reading for a given angle from LIDAR data.
        Angle is in degrees (from -45 to 225, where 0° is directly to the right).
        """
        if angle < -45 or angle > 225:
            return float('nan')
        # Map the angle to an index in the data array.
        # Assuming the LIDAR provides a 360° reading (or a subset mapped accordingly).
        index = int(len(data) * (angle + 90) / 360)
        if index < 0 or index >= len(data):
            return float('nan')
        if not np.isnan(data[index]) and not np.isinf(data[index]):
            return data[index]
        return float('nan')

    def pid_control(self, error, velocity):
        global integral, prev_error, kp, ki, kd, prev_time
        current_time = self.get_clock().now().nanoseconds / 1e9  # seconds
        dt = current_time - prev_time if (current_time - prev_time) > 0 else 1e-3
        integral += error * dt
        derivative = (error - prev_error) / dt
        angle = kp * error + ki * integral + kd * derivative
        prev_error = error
        prev_time = current_time

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = -angle + servo_offset

        abs_angle_deg = abs(math.degrees(angle))
        if abs_angle_deg <= 10:
            drive_msg.drive.speed = velocity
        elif abs_angle_deg <= 20:
            drive_msg.drive.speed = 1.0
        else:
            drive_msg.drive.speed = 0.5

        self.drive_pub.publish(drive_msg)

    def follow_left(self, data, left_dist):
        """
        Calculate error between desired and actual distance from left wall.
        Uses two LIDAR measurements.
        """
        front_scan_angle = 125  # degrees
        back_scan_angle = 180   # degrees
        teta = math.radians(abs(front_scan_angle - back_scan_angle))
        front_scan_dist = self.get_range(data, front_scan_angle)
        back_scan_dist = self.get_range(data, back_scan_angle)
        if np.isnan(front_scan_dist) or np.isnan(back_scan_dist):
            return 0.0
        alpha = math.atan2(front_scan_dist * math.cos(teta) - back_scan_dist,
                             front_scan_dist * math.sin(teta))
        wall_dist = back_scan_dist * math.cos(alpha)
        ahead_wall_dist = wall_dist + CAR_LENGTH * math.sin(alpha)
        return left_dist - ahead_wall_dist

    def lidar_callback(self, msg):
        error_val = self.follow_left(msg.ranges, DESIRED_DISTANCE_LEFT)
        self.pid_control(error_val, VELOCITY)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

