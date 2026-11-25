"""Test startup and experiment boundaries without ROS hardware."""

import csv
import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(params=["rpc", "mpc"])
def experiment(request):
    return pytest.importorskip(request.param + ".experiment")


def test_startup_ramp_accumulates_even_when_wheels_are_stopped(experiment):
    gate = experiment.StartupGate(target_speed=1.0)
    assert gate.update(0.0, 0.0, True, True) == 0.0
    assert gate.update(0.1, 0.0, True, True) == pytest.approx(0.05)
    assert gate.update(0.2, 0.0, True, True) == pytest.approx(0.1)
    for index in range(3, 40):
        gate.update(index * 0.1, 0.0, True, True)
    assert gate.command_speed == pytest.approx(1.0)
    assert gate.phase == "STARTUP"


def test_requires_two_full_seconds_of_measured_speed_stability(experiment):
    gate = experiment.StartupGate(target_speed=0.6)
    gate.update(0.0, 0.6, True, True)
    assert gate.phase == "STABILIZING"
    gate.update(1.99, 0.6, True, True)
    assert gate.phase == "STABILIZING"
    assert gate.update(2.0, 0.6, True, True) is None
    assert gate.phase == "PREPARING"


def test_startup_never_commands_above_target_when_already_moving(experiment):
    gate = experiment.StartupGate(target_speed=0.6)
    assert gate.update(0.0, 1.0, True, True) == 0.6
    assert gate.phase == "STARTUP"


def test_speed_outside_tolerance_restarts_stability_timer(experiment):
    gate = experiment.StartupGate(target_speed=0.6)
    gate.update(0.0, 0.6, True, True)
    gate.update(1.0, 0.7, True, True)
    assert gate.stable_since is None
    gate.update(1.1, 0.6, True, True)
    gate.update(3.09, 0.6, True, True)
    assert gate.phase == "STABILIZING"
    gate.update(3.11, 0.6, True, True)
    assert gate.phase == "PREPARING"


def test_lost_feedback_aborts_running_experiment_until_rb_released(experiment):
    gate = experiment.StartupGate(target_speed=0.6)
    gate.update(0.0, 0.6, True, True)
    gate.update(2.0, 0.6, True, True)
    gate.phase = "RUNNING"  # A first successful command publication commits takeover.
    assert gate.update(2.1, 0.6, True, False) == 0.0
    assert gate.phase == "ABORTED"
    gate.update(2.2, 0.6, True, True)
    assert gate.phase == "ABORTED"
    gate.update(2.3, 0.6, False, True)
    gate.update(2.4, 0.6, True, True)
    assert gate.phase == "STABILIZING"
    assert gate.stable_since == 2.4


def test_unreachable_target_times_out_instead_of_starting_scenario(experiment):
    gate = experiment.StartupGate(target_speed=0.6, startup_timeout=3.0)
    gate.update(0.0, 0.0, True, True)
    assert gate.update(3.0, 0.0, True, True) == 0.0
    assert gate.phase == "ABORTED"
    assert gate.abort_reason == "startup_timeout"


@pytest.fixture
def node(experiment, tmp_path, monkeypatch):
    # ROS imports are confined to setup; the harness supplies its Node API.
    qos_module = ModuleType("rclpy.qos")
    qos_module.qos_profile_sensor_data = object()
    sensor_module = ModuleType("sensor_msgs.msg")
    sensor_module.Joy = object
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_module)
    clock = [0.0]
    monkeypatch.setattr(experiment.time, "monotonic", lambda: clock[0])

    class Harness(experiment.ExperimentNodeMixin):
        def __init__(self):
            self.parameters = {
                "save_run_data": True,
                "log_directory": str(tmp_path),
            }
            self.virtual_reference_speed = 0.6
            self.control_period = 0.1
            self.commands = []
            self.publications = []
            self.publication_delay = 0.0
            self.scenarios = []
            self.messages = []
            self.marker_clears = 0
            self.clock = clock
            self.setup_experiment("RPC", 2, 6.0, 3.0)
            self._experiment_drive_publisher = SimpleNamespace(publish=self.record_publication)

        def record_publication(self, message):
            self.clock[0] += self.publication_delay
            self.publications.append((
                self.clock[0], message.drive.speed, message.drive.steering_angle,
            ))

        def publish_valid_command(self):
            return self.publish_controller_command(SimpleNamespace(
                header=SimpleNamespace(stamp=None),
                drive=SimpleNamespace(speed=0.6, steering_angle=0.0),
            ))

        def declare_parameter(self, name, value):
            self.parameters.setdefault(name, value)

        def get_parameter(self, name):
            return SimpleNamespace(value=self.parameters[name])

        def create_subscription(self, *args):
            return args

        def get_logger(self):
            return SimpleNamespace(
                info=self.messages.append, error=self.messages.append,
                debug=self.messages.append, warn=self.messages.append,
            )

        def get_clock(self):
            stamp = SimpleNamespace(
                nanoseconds=int(self.clock[0] * 1.0e9), to_msg=lambda: None,
            )
            return SimpleNamespace(now=lambda: stamp)

        def publish_experiment_command(self, speed):
            self.commands.append((speed, 0.0))

        def clear_experiment_markers(self):
            self.marker_clears += 1

        def start_experiment_scenario(self, state):
            self.scenarios.append(state.copy())

        def tick(self, now, speed=0.6, rb=True, lb=False, joy=True, odometry=True):
            self.clock[0] = now
            if joy:
                self.experiment_joy_callback(
                    SimpleNamespace(buttons=[0, 0, 0, 0, int(lb), int(rb)])
                )
            if odometry:
                self.note_experiment_odometry()
            return self.prepare_experiment(np.array([now, 2.0, speed, 0.4]))

        def write_sample(self):
            self.write_run_log(
                status="ok", timings={}, state=np.zeros(4),
                reference=np.zeros((4, 3)), predicted_states=np.zeros((4, 3)),
                predicted_controls=np.zeros((2, 2)),
                command_acceleration=0.0, command_steering=0.0, command_speed=0.6,
            )

    return Harness()


def test_scenario_and_csv_start_only_after_settling_and_restart_per_hold(node, tmp_path):
    assert not node.tick(0.0, rb=False)
    assert not node.tick(0.1)
    assert not node.tick(2.09)
    assert not node.scenarios
    assert not list(tmp_path.glob("*.csv"))
    assert node.tick(2.11)
    np.testing.assert_allclose(node.scenarios[0], [2.11, 2.0, 0.6, 0.4])
    assert node.experiment_start_time is None
    assert node.run_logger is None
    assert not list(tmp_path.glob("*.csv"))
    node.complete_scene_preparation()
    assert node.publish_valid_command()
    assert node.experiment_sample_time == 0.0
    node.write_sample()
    first_logger = node.run_logger
    assert node.tick(2.21)
    assert node.publish_valid_command()
    node.write_sample()
    node.tick(2.3, rb=False)
    assert first_logger._file.closed
    with first_logger.path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["phase"] == "RUNNING"
    assert float(rows[0]["experiment_time_sec"]) == 0.0
    assert float(rows[1]["experiment_time_sec"]) == pytest.approx(0.1)
    assert not node.tick(2.4)
    assert node.tick(4.41)
    assert len(node.scenarios) == 2
    assert node.publish_valid_command()
    node.write_sample()
    assert node.run_logger.path != first_logger.path
    assert node.run_logger.sample_index == 1
    node.close_experiment()


def test_lb_override_ends_logging_and_resets_settling(node):
    node.tick(0.0)
    assert node.tick(2.01)
    node.publish_valid_command()
    node.write_sample()
    logger = node.run_logger
    assert not node.tick(2.1, lb=True)
    assert logger._file.closed
    assert node.commands[-1] == (0.0, 0.0)
    assert node.startup_gate.phase == "WAITING"
    assert not node.tick(2.2)
    assert len(node.scenarios) == 1
    node.close_experiment()


@pytest.mark.parametrize("lost_stream", ["joy", "odometry"])
def test_stale_input_prevents_qualification(node, tmp_path, lost_stream):
    node.tick(0.0)
    assert not node.tick(0.31, **{lost_stream: False})
    assert node.startup_gate.phase == "ABORTED"
    assert not node.tick(2.0)
    assert not node.scenarios
    assert not list(tmp_path.glob("*.csv"))


def test_log_open_failure_aborts_immediately_after_first_publication(node, experiment, monkeypatch):
    def fail_open(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(experiment, "RunCsvLogger", fail_open)
    node.tick(0.0)
    assert node.tick(2.01)
    assert node.publish_valid_command()
    node.write_sample()
    assert node.startup_gate.phase == "ABORTED"
    assert len(node.scenarios) == 1
    assert node.run_logger is None
    assert node.commands[-1] == (0.0, 0.0)


def test_recording_can_be_disabled_without_skipping_settling(node, tmp_path):
    node.parameters["save_run_data"] = False
    assert not node.tick(0.0)
    assert node.tick(2.01)
    assert node.publish_valid_command()
    node.write_sample()
    assert len(node.scenarios) == 1
    assert not list(tmp_path.glob("*.csv"))
    node.close_experiment()


def test_time_zero_is_publication_not_scene_generation_or_csv_creation(
    node, experiment, tmp_path, monkeypatch,
):
    node.tick(0.0)
    assert node.tick(2.01)
    assert node.startup_gate.phase == "PREPARING"
    assert node.experiment_start_time is None
    assert not list(tmp_path.glob("*.csv"))
    node.clock[0] += 0.02
    node.complete_scene_preparation()
    node.clock[0] += 0.04
    node.publication_delay = 0.003
    assert node.publish_valid_command()
    published_at = node.publications[-1][0]
    assert node.experiment_start_time == published_at
    assert node.experiment_sample_time == 0.0
    assert node.first_command_latency_ms == pytest.approx(63.0)
    assert node.scene_preparation_ms == pytest.approx(20.0)
    assert not list(tmp_path.glob("*.csv"))

    original_logger = experiment.RunCsvLogger

    class SlowLogger(original_logger):
        def __init__(self, *args, **kwargs):
            node.clock[0] += 0.7  # Simulate slow filesystem initialization.
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(experiment, "RunCsvLogger", SlowLogger)
    node.write_sample()
    assert node.experiment_start_time == published_at
    with node.run_logger.path.open(newline="") as stream:
        first = next(csv.DictReader(stream))
    assert float(first["experiment_time_sec"]) == 0.0
    assert float(first["ros_time_sec"]) == pytest.approx(published_at)
    assert float(first["reference_time_sec"]) == 0.0
    assert float(first["scene_preparation_ms"]) == pytest.approx(20.0)
    assert float(first["first_command_latency_ms"]) == pytest.approx(63.0)

    node.tick(3.0)
    node.clock[0] += 0.02
    assert node.publish_valid_command()
    node.write_sample()
    with node.run_logger.path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[1]["reference_time_sec"]) == pytest.approx(3.0 - published_at)
    assert float(rows[1]["experiment_time_sec"]) == pytest.approx(3.023 - published_at)
    node.close_experiment()


def test_first_publication_failure_creates_no_experiment_file(node, tmp_path):
    node.tick(0.0)
    node.tick(2.01)

    def fail_publish(message):
        raise RuntimeError("transport unavailable")

    node._experiment_drive_publisher.publish = fail_publish
    assert not node.publish_valid_command()
    node.write_sample()
    assert node.experiment_start_time is None
    assert node.startup_gate.phase == "ABORTED"
    assert not list(tmp_path.glob("*.csv"))


def test_preparation_timeout_does_not_publish_a_stale_first_command(node, tmp_path):
    node.tick(0.0)
    node.tick(2.01)
    node.clock[0] += 0.31
    assert not node.publish_valid_command()
    assert not node.publications
    assert not list(tmp_path.glob("*.csv"))
    assert node.experiment_start_time is None
    assert node.startup_gate.phase == "ABORTED"


def test_experiment_modules_stay_identical():
    rpc_module = importlib.import_module("rpc.experiment")
    mpc_module = pytest.importorskip("mpc.experiment")
    from pathlib import Path

    assert Path(rpc_module.__file__).read_bytes() == Path(mpc_module.__file__).read_bytes()


def test_real_controller_callbacks_gate_scene_solve_and_csv(node, experiment, tmp_path):
    """Execute the real callbacks with fake transport and control backends."""
    import ast
    import math
    from pathlib import Path
    import time
    from types import MethodType

    from rpc.dynamics import (
        VehicleLimits, clip_control, make_tracking_features, rollout_policy,
    )

    package = experiment.__name__.split(".")[0]
    source = Path(experiment.__file__).parents[1] / "scripts" / (package + "_node.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    controller_class = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == package.upper()
    )
    methods = {
        "pose_callback", "control_callback", "start_experiment_scenario",
        "init_virtual_road", "initialize_virtual_road", "virtual_point",
        "virtual_reference_trajectory", "calc_virtual_ref_trajectory",
        "state_from_odometry",
        "warmup_controller",
    }
    controller_class.bases = []
    controller_class.body = [
        item for item in controller_class.body
        if isinstance(item, ast.FunctionDef) and item.name in methods
    ]
    fake_message = lambda: SimpleNamespace(
        header=SimpleNamespace(stamp=None),
        drive=SimpleNamespace(speed=0.0, steering_angle=0.0),
    )
    scope = {
        "np": np, "math": math, "time": time, "State": SimpleNamespace,
        "AckermannDriveStamped": fake_message,
        "clip_control": clip_control, "make_tracking_features": make_tracking_features,
        "rollout_policy": rollout_policy,
        "make_warmup_inputs": experiment.make_warmup_inputs,
    }
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0],
                              controller_class], type_ignores=[])
    exec(compile(module, str(source), "exec"), scope)
    extracted = scope[package.upper()]
    for name in methods:
        if hasattr(extracted, name):
            function = getattr(extracted, name)
            setattr(node, name, function if name == "state_from_odometry"
                    else MethodType(function, node))

    node._experiment_prefix = package.upper()
    node.virtual_road_mode = True
    node.virtual_origin = None
    node.virtual_road_initialized = False
    node.virtual_lane_width = 0.3
    node.virtual_lane_change_delay = 2.0
    node.latest_odometry = None
    node.latest_odom_msg = None
    node.oa = None
    node.odelta_v = None
    node.last_solver_time_ms = 0.2
    node.ref_goal_points_data = None
    node.reference_points = None
    node.reference_trajectory_publisher = object()
    node.predicted_trajectory_publisher = object()
    node.fail_backend = False
    backend_calls = []

    def act(features):
        if node.fail_backend:
            raise RuntimeError("simulated inference failure")
        backend_calls.append("inference")
        return np.zeros(2)

    def solve(reference, state, *args):
        if node.fail_backend:
            raise RuntimeError("simulated solver failure")
        backend_calls.append("solve")
        return (np.zeros(2), np.zeros(2), reference[0], reference[1],
                reference[3], reference[2], reference)

    node.controller = SimpleNamespace(
        horizon=2, model_dt=0.2,
        vehicle=VehicleLimits(0.33, 0.0, 6.0, 3.0, 0.4189), act=act,
    )
    node.config = SimpleNamespace(NXK=4, TK=2, DTK=0.2, MIN_SPEED=0.0, MAX_SPEED=6.0)
    node.linear_mpc_control = solve
    publisher = node._experiment_drive_publisher
    node.drive_publisher = node.drive_pub_ = publisher
    node.drive_msg_ = fake_message()
    for name in ("publish_virtual_road", "publish_reference_target", "publish_trajectory",
                 "viz_virtual_road", "viz_ref_target", "viz_rej_traj", "viz_opt_traj"):
        setattr(node, name, lambda *args, **kwargs: None)

    node.parameters["warmup_iterations"] = 3
    node.warmup_controller()
    assert len(backend_calls) == (6 if package == "rpc" else 3)
    assert node.virtual_origin is None
    assert node.experiment_start_time is None
    assert not node.publications
    assert not list(tmp_path.glob("*.csv"))
    backend_calls.clear()

    def cycle(now, x_position, rb=True):
        node.clock[0] = now
        node.experiment_joy_callback(SimpleNamespace(buttons=[0, 0, 0, 0, 0, int(rb)]))
        message = SimpleNamespace(
            pose=SimpleNamespace(pose=SimpleNamespace(
                position=SimpleNamespace(x=x_position, y=20.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(0.2), w=math.cos(0.2)),
            )),
            twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=0.6))),
        )
        node.pose_callback(message)
        node.control_callback()

    cycle(0.0, 0.0, rb=False)
    cycle(0.1, 1.0)
    cycle(2.09, 9.0)
    assert node.virtual_origin is None
    assert not backend_calls
    assert not list(tmp_path.glob("*.csv"))

    cycle(2.11, 10.0)
    np.testing.assert_allclose(node.virtual_origin, [10.0, 20.0])
    assert node.virtual_yaw == pytest.approx(0.4)
    assert backend_calls
    first_path = node.run_logger.path
    with first_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["experiment_time_sec"] == "0.0"
    assert rows[0]["reference_0_y"] == "20.0"
    assert rows[0]["state_x"] == "10.0"

    cycle(2.2, 10.1, rb=False)
    assert node.run_logger is None
    cycle(3.0, 30.0)
    cycle(5.01, 50.0)
    np.testing.assert_allclose(node.virtual_origin, [50.0, 20.0])
    assert node.run_logger.path != first_path
    assert node.run_logger.sample_index == 1
    node.close_experiment()

    published_count = len(node.publications)
    node.fail_backend = True
    cycle(6.0, 60.0)
    cycle(8.01, 80.0)
    assert node.startup_gate.phase == "ABORTED"
    assert node.experiment_start_time is None
    assert node.run_logger is None
    assert len(node.publications) == published_count
    assert len(list(tmp_path.glob("*.csv"))) == 2
