"""Per-experiment CSV logging for controller timing and trajectories."""

from __future__ import annotations

import csv
from datetime import datetime
import os
from pathlib import Path
from typing import Mapping

import numpy as np


STATE_NAMES = ("x", "y", "v", "yaw")
CONTROL_NAMES = ("acceleration", "steering")
TIMING_NAMES = (
    "controller_compute_ms",
    "solver_internal_ms",
    "control_inference_ms",
    "prediction_rollout_ms",
    "callback_total_ms",
    "scene_preparation_ms",
    "first_command_latency_ms",
)


def default_log_directory() -> str:
    """Return a writable ROS-oriented default log directory."""

    ros_home = Path(os.environ.get("ROS_HOME", Path.home() / ".ros"))
    return str(ros_home / "f1tenth_controller_logs")


class RunCsvLogger:
    """Write one flushed CSV row per controller update."""

    def __init__(
        self, directory: str | Path, prefix: str, horizon: int,
        *, started_at: datetime | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        output_directory = Path(directory).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = (started_at or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S_%f")
        self.path = output_directory / f"{prefix}_{timestamp}.csv"
        self.horizon = horizon
        self.sample_index = 0
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, lineterminator="\n")
        self._writer.writerow(self._header())
        self._file.flush()

    def _header(self) -> list[str]:
        header = [
            "sample", "wall_time_iso", "ros_time_sec", "status",
            "phase", "experiment_time_sec", "reference_time_sec",
        ]
        header.extend(TIMING_NAMES)
        header.extend(f"state_{name}" for name in STATE_NAMES)
        header.extend(
            ["command_acceleration", "command_steering", "command_speed"]
        )
        for step in range(self.horizon + 1):
            header.extend(f"reference_{step}_{name}" for name in STATE_NAMES)
        for step in range(self.horizon + 1):
            header.extend(f"predicted_{step}_{name}" for name in STATE_NAMES)
        for step in range(self.horizon):
            header.extend(f"predicted_control_{step}_{name}" for name in CONTROL_NAMES)
        return header

    @staticmethod
    def _state_values(state: np.ndarray | None) -> list[float | str]:
        if state is None:
            return [""] * 4
        array = np.asarray(state, dtype=float)
        if array.shape != (4,):
            raise ValueError("state must have shape (4,)")
        return array.tolist()

    def _trajectory_values(
        self, trajectory: np.ndarray | None
    ) -> list[float | str]:
        if trajectory is None:
            return [""] * (4 * (self.horizon + 1))
        array = np.asarray(trajectory, dtype=float)
        if array.shape != (4, self.horizon + 1):
            raise ValueError(
                f"trajectory must have shape (4, {self.horizon + 1})"
            )
        return array.T.reshape(-1).tolist()

    def _control_values(self, controls: np.ndarray | None) -> list[float | str]:
        if controls is None:
            return [""] * (2 * self.horizon)
        array = np.asarray(controls, dtype=float)
        if array.shape != (2, self.horizon):
            raise ValueError(f"controls must have shape (2, {self.horizon})")
        return array.T.reshape(-1).tolist()

    def log(
        self,
        *,
        ros_time_sec: float,
        status: str,
        timings: Mapping[str, float | None],
        state: np.ndarray,
        reference: np.ndarray,
        predicted_states: np.ndarray | None,
        predicted_controls: np.ndarray | None,
        command_acceleration: float | None,
        command_steering: float | None,
        command_speed: float | None,
        phase: str = "RUNNING",
        experiment_time_sec: float | None = None,
        reference_time_sec: float | None = None,
        wall_time_iso: str | None = None,
    ) -> None:
        row: list[float | int | str] = [
            self.sample_index,
            wall_time_iso or datetime.now().astimezone().isoformat(timespec="milliseconds"),
            ros_time_sec,
            status,
            phase,
            "" if experiment_time_sec is None else float(experiment_time_sec),
            "" if reference_time_sec is None else float(reference_time_sec),
        ]
        row.extend(
            "" if timings.get(name) is None else float(timings[name])
            for name in TIMING_NAMES
        )
        row.extend(self._state_values(state))
        row.extend(
            "" if value is None else float(value)
            for value in (
                command_acceleration,
                command_steering,
                command_speed,
            )
        )
        row.extend(self._trajectory_values(reference))
        row.extend(self._trajectory_values(predicted_states))
        row.extend(self._control_values(predicted_controls))
        self._writer.writerow(row)
        self._file.flush()
        self.sample_index += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

