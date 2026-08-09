"""Model rollout and feature construction for the RPC node."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


STATE_SIZE = 4
CONTROL_SIZE = 2
FEATURE_SIZE = 6


@dataclass(frozen=True)
class VehicleLimits:
    """Vehicle parameters stored in the trained checkpoint."""

    wheelbase: float
    min_speed: float
    max_speed: float
    max_accel: float
    max_steer: float

    def __post_init__(self) -> None:
        if self.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if self.min_speed < 0.0 or self.min_speed >= self.max_speed:
            raise ValueError("invalid speed limits")
        if self.max_accel <= 0.0 or not 0.0 < self.max_steer < math.pi / 2.0:
            raise ValueError("invalid control limits")


class FeaturePolicy(Protocol):
    """Minimal interface needed by the model rollout."""

    def act(self, features: np.ndarray) -> np.ndarray:
        """Return [acceleration, steering_angle] for one feature vector."""


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def make_tracking_features(
    state: np.ndarray,
    reference_state: np.ndarray,
) -> np.ndarray:
    """Express a map-frame state in the reference pose's local frame.

    The trained policy expects
    [x_relative, y_error, speed, yaw_error, y_reference_relative, speed_ref].
    Its training data used a straight road with zero relative longitudinal and
    lateral references. Projecting onto the local reference frame preserves
    that convention for the MPC waypoint interface.
    """

    state_array = np.asarray(state, dtype=float)
    reference_array = np.asarray(reference_state, dtype=float)
    if state_array.shape != (STATE_SIZE,) or reference_array.shape != (STATE_SIZE,):
        raise ValueError("state and reference_state must have shape (4,)")

    dx = state_array[0] - reference_array[0]
    dy = state_array[1] - reference_array[1]
    reference_yaw = reference_array[3]
    lateral_error = -math.sin(reference_yaw) * dx + math.cos(reference_yaw) * dy
    heading_error = wrap_angle(state_array[3] - reference_yaw)

    return np.array(
        [
            0.0,
            lateral_error,
            state_array[2],
            heading_error,
            0.0,
            reference_array[2],
        ],
        dtype=np.float32,
    )


def clip_control(control: np.ndarray, vehicle: VehicleLimits) -> np.ndarray:
    """Apply the same acceleration and steering limits used for training."""

    clipped = np.asarray(control, dtype=float).copy()
    if clipped.shape != (CONTROL_SIZE,):
        raise ValueError("control must have shape (2,)")
    clipped[0] = np.clip(clipped[0], -vehicle.max_accel, vehicle.max_accel)
    clipped[1] = np.clip(clipped[1], -vehicle.max_steer, vehicle.max_steer)
    return clipped


def kinematic_step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    vehicle: VehicleLimits,
) -> np.ndarray:
    """Advance the nonlinear rear-axle kinematic bicycle model one step."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    current = np.asarray(state, dtype=float)
    if current.shape != (STATE_SIZE,):
        raise ValueError("state must have shape (4,)")
    acceleration, steering = clip_control(control, vehicle)
    x_position, y_position, speed, yaw = current

    next_state = np.array(
        [
            x_position + dt * speed * math.cos(yaw),
            y_position + dt * speed * math.sin(yaw),
            np.clip(
                speed + dt * acceleration,
                vehicle.min_speed,
                vehicle.max_speed,
            ),
            wrap_angle(yaw + dt * speed / vehicle.wheelbase * math.tan(steering)),
        ],
        dtype=float,
    )
    return next_state


def rollout_policy(
    policy: FeaturePolicy,
    initial_state: np.ndarray,
    reference_trajectory: np.ndarray,
    dt: float,
    vehicle: VehicleLimits,
    first_control: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recursively infer controls and propagate the vehicle model.

    Returns a state array shaped [4, horizon + 1] and a control array shaped
    [2, horizon]. The first control is the command applied by the ROS node.
    """

    references = np.asarray(reference_trajectory, dtype=float)
    if references.ndim != 2 or references.shape[0] != STATE_SIZE:
        raise ValueError("reference_trajectory must have shape (4, horizon + 1)")
    if references.shape[1] < 2:
        raise ValueError("reference_trajectory must contain at least two states")

    horizon = references.shape[1] - 1
    states = np.empty((STATE_SIZE, horizon + 1), dtype=float)
    controls = np.empty((CONTROL_SIZE, horizon), dtype=float)
    states[:, 0] = np.asarray(initial_state, dtype=float)

    for step in range(horizon):
        if step == 0 and first_control is not None:
            controls[:, step] = clip_control(first_control, vehicle)
        else:
            features = make_tracking_features(states[:, step], references[:, step])
            controls[:, step] = clip_control(policy.act(features), vehicle)
        states[:, step + 1] = kinematic_step(
            states[:, step], controls[:, step], dt, vehicle
        )

    return states, controls
