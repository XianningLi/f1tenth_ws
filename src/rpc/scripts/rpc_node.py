#!/usr/bin/env python3
"""ROS 2 RPC node using the trained neural feedback controller."""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from rpc.dynamics import clip_control, make_tracking_features, rollout_policy
from rpc.neural_controller import NeuralController
from rpc.run_logger import RunCsvLogger, default_log_directory
from rpc.trajectory import load_waypoints, waypoint_reference_trajectory


class RPC(Node):
    """Drop-in neural counterpart to the existing MPC tracker node."""

    def __init__(self) -> None:
        super().__init__("rpc_node")
        default_model_path = str(
            get_package_share_directory("rpc") + "/models/hfrpc.pt"
        )

        self.declare_parameter("model_path", default_model_path)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("virtual_road_mode", True)
        self.declare_parameter(
            "waypoint_path", "/home/nvidia/Downloads/levine_centerline.csv"
        )
        self.declare_parameter("waypoint_index_step", 2)
        self.declare_parameter("virtual_lane_width", 0.3)
        self.declare_parameter("virtual_road_length", 5.0)
        self.declare_parameter("virtual_reference_speed", 0.6)
        self.declare_parameter("virtual_lane_change_delay", 2.0)
        self.declare_parameter("control_period", 0.1)
        self.declare_parameter("odom_topic", "/pf/pose/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("visualization_frame", "map")
        self.declare_parameter("save_run_data", True)
        self.declare_parameter("log_directory", default_log_directory())

        model_path = self.get_parameter("model_path").value
        device = self.get_parameter("device").value
        self.virtual_road_mode = bool(self.get_parameter("virtual_road_mode").value)
        self.waypoint_path = self.get_parameter("waypoint_path").value
        self.waypoint_index_step = int(
            self.get_parameter("waypoint_index_step").value
        )
        self.virtual_lane_width = float(
            self.get_parameter("virtual_lane_width").value
        )
        self.virtual_road_length = float(
            self.get_parameter("virtual_road_length").value
        )
        self.virtual_reference_speed = float(
            self.get_parameter("virtual_reference_speed").value
        )
        self.virtual_lane_change_delay = float(
            self.get_parameter("virtual_lane_change_delay").value
        )
        self.control_period = float(self.get_parameter("control_period").value)
        self.visualization_frame = self.get_parameter("visualization_frame").value
        if self.control_period <= 0.0:
            raise ValueError("control_period must be positive")

        self.controller = NeuralController(model_path, device=device)
        self.get_logger().info(
            "Loaded %s checkpoint (horizon=%d, model_dt=%.3f s, device=%s)"
            % (
                self.controller.method,
                self.controller.horizon,
                self.controller.model_dt,
                self.controller.device,
            )
        )

        self.waypoints = None
        if not self.virtual_road_mode:
            self.waypoints, _ = load_waypoints(self.waypoint_path)

        self.run_logger = None
        if bool(self.get_parameter("save_run_data").value):
            self.run_logger = RunCsvLogger(
                self.get_parameter("log_directory").value,
                "RPC",
                self.controller.horizon,
            )
            self.get_logger().info(f"Run data: {self.run_logger.path}")

        drive_topic = self.get_parameter("drive_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10
        )
        self.pose_subscription = self.create_subscription(
            Odometry, odom_topic, self.pose_callback, 1
        )
        self.reference_points_publisher = self.create_publisher(
            MarkerArray, "ref_goal_points", 1
        )
        self.reference_trajectory_publisher = self.create_publisher(
            Marker, "ref_trajectory", 1
        )
        self.predicted_trajectory_publisher = self.create_publisher(
            Marker, "opt_trajectory", 1
        )
        self.virtual_road_publisher = self.create_publisher(
            MarkerArray, "virtual_road", 1
        )
        self.reference_target_publisher = self.create_publisher(
            Marker, "ref_target_point", 1
        )

        self.latest_odometry = None
        self.virtual_origin = None
        self.virtual_yaw = None
        self.virtual_forward = None
        self.virtual_left = None
        self.virtual_start_time = None
        self.reference_points = (
            None if self.virtual_road_mode else self.make_reference_point_markers()
        )
        self.control_timer = self.create_timer(
            self.control_period, self.control_callback
        )

    def pose_callback(self, message: Odometry) -> None:
        self.latest_odometry = message

    def control_callback(self) -> None:
        if self.latest_odometry is None:
            return
        callback_start_ns = time.perf_counter_ns()
        state = self.state_from_odometry(self.latest_odometry)

        if self.virtual_road_mode:
            if self.virtual_origin is None:
                self.initialize_virtual_road(state)
            reference = self.virtual_reference_trajectory(state)
            self.publish_virtual_road()
            self.publish_reference_target(reference)
        else:
            reference = waypoint_reference_trajectory(
                state,
                self.waypoints,
                self.controller.horizon,
                self.waypoint_index_step,
            )

        controller_start_ns = time.perf_counter_ns()
        control_inference_ms = None
        prediction_rollout_ms = None
        predicted_states = None
        predicted_controls = None
        actual_control = None
        try:
            inference_start_ns = time.perf_counter_ns()
            features = make_tracking_features(state, reference[:, 0])
            actual_control = clip_control(
                self.controller.act(features), self.controller.vehicle
            )
            control_inference_ms = (
                time.perf_counter_ns() - inference_start_ns
            ) / 1.0e6

            prediction_start_ns = time.perf_counter_ns()
            predicted_states, predicted_controls = rollout_policy(
                self.controller,
                state,
                reference,
                self.controller.model_dt,
                self.controller.vehicle,
                first_control=actual_control,
            )
            prediction_rollout_ms = (
                time.perf_counter_ns() - prediction_start_ns
            ) / 1.0e6
        except Exception as exc:  # Keep the car command unchanged after inference failure.
            controller_compute_ms = (
                time.perf_counter_ns() - controller_start_ns
            ) / 1.0e6
            self.write_run_log(
                status="inference_failed",
                timings={
                    "controller_compute_ms": controller_compute_ms,
                    "solver_internal_ms": None,
                    "control_inference_ms": control_inference_ms,
                    "prediction_rollout_ms": prediction_rollout_ms,
                    "callback_total_ms": (
                        time.perf_counter_ns() - callback_start_ns
                    )
                    / 1.0e6,
                },
                state=state,
                reference=reference,
                predicted_states=predicted_states,
                predicted_controls=predicted_controls,
                command_acceleration=None,
                command_steering=None,
                command_speed=None,
            )
            self.get_logger().error(f"RPC inference failed; skipping command: {exc}")
            return

        controller_compute_ms = (
            time.perf_counter_ns() - controller_start_ns
        ) / 1.0e6

        acceleration = float(actual_control[0])
        steering = float(actual_control[1])
        speed_command = float(
            np.clip(
                state[2] + acceleration * self.controller.model_dt,
                self.controller.vehicle.min_speed,
                self.controller.vehicle.max_speed,
            )
        )

        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.drive.speed = speed_command
        command.drive.steering_angle = steering
        self.drive_publisher.publish(command)

        self.publish_trajectory(reference, self.reference_trajectory_publisher, False)
        self.publish_trajectory(
            predicted_states, self.predicted_trajectory_publisher, True
        )
        if self.reference_points is not None:
            self.reference_points_publisher.publish(self.reference_points)
        callback_total_ms = (
            time.perf_counter_ns() - callback_start_ns
        ) / 1.0e6
        self.write_run_log(
            status="ok",
            timings={
                "controller_compute_ms": controller_compute_ms,
                "solver_internal_ms": None,
                "control_inference_ms": control_inference_ms,
                "prediction_rollout_ms": prediction_rollout_ms,
                "callback_total_ms": callback_total_ms,
            },
            state=state,
            reference=reference,
            predicted_states=predicted_states,
            predicted_controls=predicted_controls,
            command_acceleration=acceleration,
            command_steering=steering,
            command_speed=speed_command,
        )
        self.get_logger().debug(
            "RPC control %.3f ms, prediction %.3f ms, total %.3f ms: "
            "a=%.3f m/s^2, steer=%.3f rad, speed=%.3f m/s"
            % (
                control_inference_ms,
                prediction_rollout_ms,
                callback_total_ms,
                acceleration,
                steering,
                speed_command,
            )
        )

    def write_run_log(self, **sample) -> None:
        if self.run_logger is None:
            return
        try:
            sample["ros_time_sec"] = self.get_clock().now().nanoseconds / 1.0e9
            self.run_logger.log(**sample)
        except Exception as exc:
            self.get_logger().error(f"Disabling run data logging after error: {exc}")
            self.run_logger.close()
            self.run_logger = None

    @staticmethod
    def state_from_odometry(message: Odometry) -> np.ndarray:
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0
            * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0
            - 2.0
            * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        return np.array(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                max(0.0, message.twist.twist.linear.x),
                yaw,
            ],
            dtype=float,
        )

    def initialize_virtual_road(self, state: np.ndarray) -> None:
        self.virtual_origin = state[:2].copy()
        self.virtual_yaw = float(state[3])
        self.virtual_forward = np.array(
            [math.cos(self.virtual_yaw), math.sin(self.virtual_yaw)], dtype=float
        )
        self.virtual_left = np.array(
            [-math.sin(self.virtual_yaw), math.cos(self.virtual_yaw)], dtype=float
        )
        self.virtual_start_time = self.get_clock().now().nanoseconds * 1.0e-9
        self.get_logger().info(
            "Virtual road initialized at x=%.2f, y=%.2f, yaw=%.2f"
            % (state[0], state[1], state[3])
        )

    def virtual_point(self, distance: float, lateral_offset: float) -> tuple[float, float]:
        point = (
            self.virtual_origin
            + distance * self.virtual_forward
            + lateral_offset * self.virtual_left
        )
        return float(point[0]), float(point[1])

    def virtual_reference_trajectory(self, state: np.ndarray) -> np.ndarray:
        reference = np.zeros((4, self.controller.horizon + 1), dtype=float)
        current_distance = float(
            np.dot(state[:2] - self.virtual_origin, self.virtual_forward)
        )
        current_distance = max(0.0, current_distance)
        current_time = self.get_clock().now().nanoseconds * 1.0e-9
        elapsed = current_time - self.virtual_start_time
        target_offset = (
            self.virtual_lane_width
            if elapsed >= self.virtual_lane_change_delay
            else 0.0
        )
        step_distance = (
            max(abs(state[2]), self.virtual_reference_speed, 0.1)
            * self.controller.model_dt
        )
        for step in range(self.controller.horizon + 1):
            x_position, y_position = self.virtual_point(
                current_distance + step * step_distance, target_offset
            )
            reference[:, step] = [
                x_position,
                y_position,
                self.virtual_reference_speed,
                self.virtual_yaw,
            ]
        return reference

    def publish_trajectory(
        self,
        trajectory: np.ndarray,
        publisher,
        predicted: bool,
    ) -> None:
        marker = Marker(type=Marker.LINE_STRIP)
        marker.header.frame_id = self.visualization_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rpc_prediction" if predicted else "rpc_reference"
        marker.id = 1
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
        marker.color.a = 1.0
        if predicted:
            marker.color.r = 1.0
            marker.color.b = 1.0
        else:
            marker.color.b = 1.0
        for step in range(trajectory.shape[1]):
            marker.points.append(
                Point(
                    x=float(trajectory[0, step]),
                    y=float(trajectory[1, step]),
                    z=0.0,
                )
            )
        publisher.publish(marker)

    def publish_reference_target(self, reference: np.ndarray) -> None:
        target = Marker(type=Marker.SPHERE)
        target.header.frame_id = self.visualization_frame
        target.header.stamp = self.get_clock().now().to_msg()
        target.ns = "ref_target"
        target.id = 0
        target.action = Marker.ADD
        target.pose.orientation.w = 1.0
        target.pose.position = Point(
            x=float(reference[0, -1]), y=float(reference[1, -1]), z=0.0
        )
        target.scale = Vector3(x=0.14, y=0.14, z=0.14)
        target.color.a = 1.0
        target.color.r = 1.0
        target.color.g = 0.55
        self.reference_target_publisher.publish(target)

    def publish_virtual_road(self) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        edge_specs = [
            (-0.5 * self.virtual_lane_width, 0.9, 0.9, 0.9),
            (0.5 * self.virtual_lane_width, 1.0, 1.0, 1.0),
            (1.5 * self.virtual_lane_width, 0.9, 0.9, 0.9),
        ]
        for marker_id, (offset, red, green, blue) in enumerate(edge_specs):
            line = Marker(type=Marker.LINE_STRIP)
            line.header.frame_id = self.visualization_frame
            line.header.stamp = stamp
            line.ns = "virtual_road_edges"
            line.id = marker_id
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale = Vector3(x=0.025, y=0.025, z=0.025)
            line.color.a = 1.0
            line.color.r = red
            line.color.g = green
            line.color.b = blue
            for distance in np.linspace(-0.5, self.virtual_road_length, 40):
                x_position, y_position = self.virtual_point(distance, offset)
                line.points.append(Point(x=x_position, y=y_position, z=0.0))
            markers.markers.append(line)

        dash_length = 0.18
        gap_length = 0.12
        center_specs = [
            (0.0, 0.0, 0.9, 0.1),
            (self.virtual_lane_width, 0.0, 0.8, 1.0),
        ]
        for marker_id, (offset, red, green, blue) in enumerate(
            center_specs, start=10
        ):
            center = Marker(type=Marker.LINE_LIST)
            center.header.frame_id = self.visualization_frame
            center.header.stamp = stamp
            center.ns = "virtual_lane_centers"
            center.id = marker_id
            center.action = Marker.ADD
            center.pose.orientation.w = 1.0
            center.scale = Vector3(x=0.018, y=0.018, z=0.018)
            center.color.a = 1.0
            center.color.r = red
            center.color.g = green
            center.color.b = blue
            distance = -0.5
            while distance < self.virtual_road_length:
                start = self.virtual_point(distance, offset)
                end = self.virtual_point(
                    min(distance + dash_length, self.virtual_road_length), offset
                )
                center.points.append(Point(x=start[0], y=start[1], z=0.02))
                center.points.append(Point(x=end[0], y=end[1], z=0.02))
                distance += dash_length + gap_length
            markers.markers.append(center)
        self.virtual_road_publisher.publish(markers)

    def make_reference_point_markers(self) -> MarkerArray:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for marker_id, waypoint in enumerate(self.waypoints):
            marker = Marker(type=Marker.SPHERE)
            marker.header.frame_id = self.visualization_frame
            marker.header.stamp = stamp
            marker.ns = "rpc_reference_points"
            marker.id = marker_id
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.pose.position = Point(
                x=float(waypoint[1]), y=float(waypoint[2]), z=0.0
            )
            marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
            marker.color.a = 1.0
            marker.color.r = 1.0
            markers.markers.append(marker)
        return markers

    def destroy_node(self):
        if self.run_logger is not None:
            self.run_logger.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = RPC()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
