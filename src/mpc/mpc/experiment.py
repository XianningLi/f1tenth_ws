"""RB-gated startup, speed settling, and experiment lifecycle.

This module is kept identical in the MPC and RPC packages so either package
can be built independently. No ROS imports are required for the state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import time

import numpy as np

from .run_logger import RunCsvLogger


def make_warmup_inputs(horizon, model_dt, speed):
    """Return synthetic states and references without initializing a real scene."""
    state = np.array([0.0, 0.05, speed, 0.03], dtype=float)
    reference = np.zeros((4, horizon + 1), dtype=float)
    reference[0] = np.arange(horizon + 1) * model_dt * speed
    reference[2] = speed
    return state, reference


@dataclass
class StartupGate:
    """Reach target speed, then hold motion; only near-zero speed resets the timer."""

    target_speed: float
    acceleration: float = 0.5
    zero_speed_threshold: float = 0.1
    stable_duration: float = 1.0
    startup_timeout: float = 0.0
    max_step_time: float = 0.2
    boost_speed: float = 1.0
    boost_duration: float = 0.0

    def __post_init__(self):
        values = (
            self.target_speed, self.acceleration,
            self.stable_duration, self.max_step_time,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Startup parameters must be finite and positive")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (self.startup_timeout, self.boost_duration)
        ):
            raise ValueError("Startup duration limits must be finite and nonnegative; 0 disables")
        if not math.isfinite(self.boost_speed) or self.boost_speed < 0.0:
            raise ValueError("startup_boost_speed must be finite and nonnegative")
        if (
            self.boost_speed > 0.0 and self.boost_duration > 0.0
            and self.startup_timeout > 0.0
            and self.boost_duration >= self.startup_timeout
        ):
            raise ValueError("startup_boost_duration must be shorter than startup_timeout")
        if (
            not math.isfinite(self.zero_speed_threshold)
            or not 0.0 <= self.zero_speed_threshold < self.target_speed
        ):
            raise ValueError("startup_zero_speed_threshold must be nonnegative and below startup_speed")
        if 0.0 < self.startup_timeout <= self.stable_duration:
            raise ValueError("startup_timeout must exceed startup_stable_duration")
        self.reset()

    def reset(self):
        self.phase = "WAITING"
        self.command_speed = 0.0
        self.started_at = None
        self.stable_since = None
        self.last_tick = None
        self.abort_reason = None
        self.boost_started_at = None

    def abort(self, reason):
        self.phase = "ABORTED"
        self.command_speed = 0.0
        self.stable_since = None
        self.abort_reason = reason

    def update(self, now, speed, enabled, feedback_ok):
        if not enabled:
            self.reset()
            return 0.0
        if self.phase == "ABORTED":
            return 0.0
        if not feedback_ok or not math.isfinite(speed):
            if self.phase != "WAITING":
                self.abort("feedback_timeout")
            return 0.0
        if self.phase in {"PREPARING", "RUNNING"}:
            return None
        if self.phase == "WAITING":
            self.phase = "STARTUP"
            self.started_at = now
            self.last_tick = now
            self.command_speed = min(self.target_speed, max(0.0, speed))
            if (
                self.boost_speed > self.target_speed
                and speed < self.target_speed
            ):
                self.phase = "BOOSTING"
                self.boost_started_at = now
        if self.startup_timeout > 0.0 and now - self.started_at >= self.startup_timeout:
            self.abort("startup_timeout")
            return 0.0

        if self.phase == "BOOSTING":
            if speed >= self.target_speed:
                # Hold the experiment speed only after measured speed reaches it.
                self.command_speed = self.target_speed
                self.last_tick = now
                self.phase = "STARTUP"
            elif self.boost_duration > 0.0 and now - self.boost_started_at >= self.boost_duration:
                self.abort("startup_boost_timeout")
                return 0.0
            else:
                self.command_speed = self.boost_speed
                self.stable_since = None
                return self.command_speed

        dt = min(max(now - self.last_tick, 0.0), self.max_step_time)
        self.last_tick = now
        change = self.acceleration * dt
        difference = self.target_speed - self.command_speed
        self.command_speed += max(-change, min(change, difference))

        at_command_target = abs(self.command_speed - self.target_speed) <= 1.0e-9
        if speed <= self.zero_speed_threshold:
            self.stable_since = None
        elif self.stable_since is None and at_command_target and speed >= self.target_speed:
            self.stable_since = now
        if self.stable_since is not None:
            self.phase = "STABILIZING"
            if now - self.stable_since >= self.stable_duration:
                self.phase = "PREPARING"
                return None
        else:
            self.phase = "STARTUP"
        return self.command_speed


class ExperimentNodeMixin:
    """Use the same startup and logging lifecycle in both ROS controllers."""

    def setup_experiment(self, prefix, horizon, max_speed, max_accel):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Joy

        defaults = {
            "joy_topic": "/joy",
            "autonomous_button": 5,
            "manual_button": 4,
            "startup_speed": -1.0,
            "startup_acceleration": 0.5,
            "startup_zero_speed_threshold": 0.1,
            "startup_stable_duration": 1.0,
            "startup_timeout": 0.0,
            "startup_boost_speed": 1.0,
            "startup_boost_duration": 0.0,
            "joy_timeout": 0.3,
            "odometry_timeout": 0.3,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {name: self.get_parameter(name).value for name in defaults}
        target = float(values["startup_speed"])
        if target == -1.0:
            target = float(self.virtual_reference_speed)
        acceleration = float(values["startup_acceleration"])
        boost_speed = float(values["startup_boost_speed"])
        if not 0.0 < target <= max_speed or not 0.0 < acceleration <= max_accel:
            raise ValueError("Startup speed or acceleration exceeds vehicle limits")
        if not math.isfinite(boost_speed) or not 0.0 <= boost_speed <= max_speed:
            raise ValueError("startup_boost_speed must be between 0 and the vehicle speed limit")
        self.startup_gate = StartupGate(
            target_speed=target,
            acceleration=acceleration,
            zero_speed_threshold=float(values["startup_zero_speed_threshold"]),
            stable_duration=float(values["startup_stable_duration"]),
            startup_timeout=float(values["startup_timeout"]),
            max_step_time=2.0 * self.control_period,
            boost_speed=boost_speed,
            boost_duration=float(values["startup_boost_duration"]),
        )
        self._autonomous_button = int(values["autonomous_button"])
        self._manual_button = int(values["manual_button"])
        if min(self._autonomous_button, self._manual_button) < 0:
            raise ValueError("Joystick button indices must be nonnegative")
        if self._autonomous_button == self._manual_button:
            raise ValueError("Manual and autonomous buttons must differ")
        self._joy_timeout = float(values["joy_timeout"])
        self._odometry_timeout = float(values["odometry_timeout"])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.control_period, self._joy_timeout, self._odometry_timeout)
        ):
            raise ValueError("Control period and feedback timeouts must be positive")
        self._experiment_prefix = prefix
        self._experiment_horizon = horizon
        self._rb_pressed = False
        self._lb_pressed = False
        self._joy_received_at = None
        self._odometry_received_at = None
        self.experiment_start_time = None
        self.experiment_sample_time = 0.0
        self.experiment_reference_time = 0.0
        self._scene_started_at = None
        self.scene_preparation_ms = None
        self.first_command_latency_ms = None
        self._last_control_publication = None
        self.run_logger = None
        self.joy_subscription = self.create_subscription(
            Joy, values["joy_topic"], self.experiment_joy_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "WAITING: hold RB to reach %.3f m/s, then keep moving for %.1f s. "
            "Only speed <= %.3f m/s resets the timer; the real scene is created afterward. Recording starts "
            "with the first published controller command."
            % (target, self.startup_gate.stable_duration, self.startup_gate.zero_speed_threshold)
        )
        boost_limit = (
            "%.2f s" % self.startup_gate.boost_duration
            if self.startup_gate.boost_duration > 0.0 else "disabled"
        )
        startup_limit = (
            "%.2f s" % self.startup_gate.startup_timeout
            if self.startup_gate.startup_timeout > 0.0 else "disabled"
        )
        self.get_logger().info(
            "Startup boost: %.3f m/s until measured speed reaches target %.3f m/s. "
            "Boost duration limit: %s; total startup limit: %s. Hold RB throughout."
            % (self.startup_gate.boost_speed, target, boost_limit, startup_limit)
        )

    def experiment_joy_callback(self, message):
        self._joy_received_at = time.monotonic()
        buttons = message.buttons
        self._rb_pressed = (
            len(buttons) > self._autonomous_button
            and bool(buttons[self._autonomous_button])
        )
        self._lb_pressed = (
            len(buttons) > self._manual_button
            and bool(buttons[self._manual_button])
        )
        if not self._rb_pressed or self._lb_pressed:
            self.startup_gate.reset()
            self.finish_experiment("manual_override" if self._lb_pressed else "RB_released")
            self.publish_experiment_command(0.0)

    def note_experiment_odometry(self):
        self._odometry_received_at = time.monotonic()

    def prepare_experiment(self, state):
        """Return True only while the controller owns a qualified experiment."""
        now = time.monotonic()
        enabled = self._rb_pressed and not self._lb_pressed
        fresh_joy = (
            self._joy_received_at is not None
            and now - self._joy_received_at <= self._joy_timeout
        )
        fresh_odometry = (
            self._odometry_received_at is not None
            and now - self._odometry_received_at <= self._odometry_timeout
        )
        valid_state = state is not None and bool(np.isfinite(state).all())
        previous_phase = self.startup_gate.phase
        command = self.startup_gate.update(
            now, float(state[2]) if valid_state else 0.0,
            enabled, fresh_joy and fresh_odometry and valid_state,
        )
        phase = self.startup_gate.phase
        if phase != previous_phase:
            self.get_logger().info(
                "Experiment phase: %s; measured speed: %s m/s; target: %.3f m/s."
                % (phase, "%.3f" % state[2] if valid_state else "unavailable",
                   self.startup_gate.target_speed)
            )
        if phase not in {"PREPARING", "RUNNING"}:
            self.finish_experiment(self.startup_gate.abort_reason or phase)
            self.publish_experiment_command(command or 0.0)
            if phase == "ABORTED" and previous_phase != phase:
                self.get_logger().error(
                    "Experiment aborted (%s). Release RB before retrying."
                    % self.startup_gate.abort_reason
                )
            return False
        if self._scene_started_at is None:
            self._scene_started_at = now
            try:
                self.start_experiment_scenario(state)
            except Exception as exc:
                self.abort_experiment("experiment_initialization_failed")
                self.get_logger().error("Cannot start experiment: " + str(exc))
                return False
            self.get_logger().info(
                "PREPARING: real scene generated from the current pose; "
                "waiting for the first controller command."
            )
        self.experiment_reference_time = (
            0.0 if self.experiment_start_time is None
            else max(0.0, now - self.experiment_start_time)
        )
        return True

    def complete_scene_preparation(self):
        """Measure real-scene and first-reference preparation, excluding solving."""
        if self.scene_preparation_ms is None and self._scene_started_at is not None:
            self.scene_preparation_ms = (
                time.monotonic() - self._scene_started_at
            ) * 1000.0

    def publish_controller_command(self, message):
        """Start experiment time only after the first successful ROS publication."""
        now = time.monotonic()
        fresh_inputs = (
            self._rb_pressed and not self._lb_pressed
            and self._joy_received_at is not None
            and now - self._joy_received_at <= self._joy_timeout
            and self._odometry_received_at is not None
            and now - self._odometry_received_at <= self._odometry_timeout
        )
        if self.startup_gate.phase not in {"PREPARING", "RUNNING"} or not fresh_inputs:
            self.abort_experiment("feedback_timeout_before_publication")
            return False
        if not all(math.isfinite(value) for value in (
            message.drive.speed, message.drive.steering_angle,
        )):
            self.abort_experiment("invalid_controller_command")
            return False
        try:
            message.header.stamp = self.get_clock().now().to_msg()
            self._experiment_drive_publisher.publish(message)
        except Exception as exc:
            self.abort_experiment("controller_publication_failed")
            self.get_logger().error("Controller publication failed: " + str(exc))
            return False
        published_at = time.monotonic()
        ros_time_sec = self.get_clock().now().nanoseconds / 1.0e9
        wall_time = datetime.now().astimezone()
        if self.experiment_start_time is None:
            self.experiment_start_time = published_at
            self.virtual_start_time = published_at
            self.first_command_latency_ms = (
                published_at - self._scene_started_at
            ) * 1000.0
            self.startup_gate.phase = "RUNNING"
            self.get_logger().info(
                "RUNNING: first controller command published; experiment time is 0.0 s."
            )
        self.experiment_sample_time = max(0.0, published_at - self.experiment_start_time)
        self._last_control_publication = (ros_time_sec, wall_time)
        return True

    def abort_experiment(self, reason):
        self.startup_gate.abort(reason)
        self.finish_experiment(reason)
        self.get_logger().error("Experiment aborted: " + reason + "; release RB to retry.")
        try:
            self.publish_experiment_command(0.0)
        except Exception:
            # A failed transport can also prevent the final stop publication.
            pass

    def finish_experiment(self, reason):
        if self.run_logger is not None:
            self.run_logger.close()
            self.run_logger = None
        if self._scene_started_at is not None:
            self.get_logger().info("Experiment ended: " + reason)
            try:
                self.clear_experiment_markers()
            except Exception:
                # Marker publishers may already be invalid during ROS shutdown.
                pass
        self.experiment_start_time = None
        self._scene_started_at = None
        self.scene_preparation_ms = None
        self.first_command_latency_ms = None
        self._last_control_publication = None
        self.experiment_sample_time = 0.0
        self.experiment_reference_time = 0.0

    def publish_experiment_command(self, speed):
        from ackermann_msgs.msg import AckermannDriveStamped

        message = AckermannDriveStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.drive.speed = float(speed)
        message.drive.steering_angle = 0.0
        self._experiment_drive_publisher.publish(message)

    def clear_experiment_markers(self):
        from visualization_msgs.msg import Marker, MarkerArray

        marker = Marker()
        marker.action = Marker.DELETEALL
        for publisher in self._experiment_marker_publishers:
            publisher.publish(marker)
        for publisher in self._experiment_array_publishers:
            message = MarkerArray()
            message.markers.append(marker)
            publisher.publish(message)

    def write_run_log(self, **sample):
        if (
            self.experiment_start_time is None
            or self.startup_gate.phase != "RUNNING"
            or not bool(self.get_parameter("save_run_data").value)
        ):
            return
        try:
            if self.run_logger is None:
                self.run_logger = RunCsvLogger(
                    self.get_parameter("log_directory").value,
                    self._experiment_prefix, self._experiment_horizon,
                    started_at=self._last_control_publication[1],
                )
                self.get_logger().info("Run data: " + str(self.run_logger.path))
            if sample["status"] == "ok":
                ros_time_sec, wall_time = self._last_control_publication
                sample["experiment_time_sec"] = self.experiment_sample_time
            else:
                ros_time_sec = self.get_clock().now().nanoseconds / 1.0e9
                wall_time = datetime.now().astimezone()
                sample["experiment_time_sec"] = max(
                    0.0, time.monotonic() - self.experiment_start_time
                )
            sample["ros_time_sec"] = ros_time_sec
            sample["wall_time_iso"] = wall_time.isoformat(timespec="milliseconds")
            sample["phase"] = "RUNNING"
            sample["reference_time_sec"] = self.experiment_reference_time
            sample["timings"] = dict(
                sample["timings"],
                scene_preparation_ms=self.scene_preparation_ms,
                first_command_latency_ms=self.first_command_latency_ms,
            )
            self.run_logger.log(**sample)
        except Exception as exc:
            self.get_logger().error("Experiment logging failed: " + str(exc))
            self.abort_experiment("logging_failed")

    def close_experiment(self):
        self.startup_gate.reset()
        self.finish_experiment("node_shutdown")
        try:
            self.publish_experiment_command(0.0)
        except Exception:
            # A ROS shutdown can invalidate the publisher before node destruction.
            pass

