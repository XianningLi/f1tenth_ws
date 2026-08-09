"""Reference trajectory helpers matching the existing MPC node."""

from __future__ import annotations

from pathlib import Path

import numpy as np

def nearest_point(point: np.ndarray, trajectory: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Return the closest projection on a piecewise-linear trajectory."""

    query = np.asarray(point, dtype=float)
    path = np.asarray(trajectory, dtype=float)
    if query.shape != (2,) or path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("invalid point or trajectory shape")
    differences = path[1:] - path[:-1]
    squared_lengths = np.sum(differences * differences, axis=1)
    if np.any(squared_lengths <= 0.0):
        raise ValueError("trajectory contains duplicate consecutive points")
    interpolation = np.sum((query - path[:-1]) * differences, axis=1) / squared_lengths
    interpolation = np.clip(interpolation, 0.0, 1.0)
    projections = path[:-1] + interpolation[:, None] * differences
    distances = np.linalg.norm(query - projections, axis=1)
    index = int(np.argmin(distances))
    return projections[index], float(distances[index]), float(interpolation[index]), index


def load_waypoints(path: str | Path) -> tuple[np.ndarray, float]:
    """Load the same semicolon-separated waypoint format as mpc_node.py."""

    points = np.loadtxt(path, delimiter=";", skiprows=3, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 6:
        raise ValueError("waypoint file must contain at least two rows and six columns")
    points = points.copy()
    points[:, 5] *= 0.5
    distance_step = float(points[1, 0] - points[0, 0])
    if distance_step <= 0.0:
        raise ValueError("waypoint distance column must be strictly increasing")
    return points, distance_step


def waypoint_reference_trajectory(
    state: np.ndarray,
    waypoints: np.ndarray,
    horizon: int,
    index_step: int = 2,
) -> np.ndarray:
    """Select the MPC node's next horizon of reference points."""

    if horizon < 1 or index_step < 1:
        raise ValueError("horizon and index_step must be positive")
    vehicle_state = np.asarray(state, dtype=float)
    points = np.asarray(waypoints, dtype=float)
    _, _, _, nearest_index = nearest_point(vehicle_state[:2], points[:, 1:3])
    indices = (nearest_index + np.arange(horizon + 1) * index_step) % len(points)

    reference = np.zeros((4, horizon + 1), dtype=float)
    reference[0] = points[indices, 1]
    reference[1] = points[indices, 2]
    reference[2] = points[indices, 5]
    selected_yaw = points[indices, 3].copy()
    yaw_difference = selected_yaw - vehicle_state[3]
    selected_yaw[yaw_difference > 4.5] -= 2.0 * np.pi
    selected_yaw[yaw_difference < -4.5] += 2.0 * np.pi
    reference[3] = selected_yaw
    return reference
