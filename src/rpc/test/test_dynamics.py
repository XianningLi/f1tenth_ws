import math

import numpy as np

from rpc.dynamics import (
    VehicleLimits,
    make_tracking_features,
    rollout_policy,
)


VEHICLE = VehicleLimits(
    wheelbase=0.33,
    min_speed=0.0,
    max_speed=6.0,
    max_accel=3.0,
    max_steer=0.4189,
)


class ConstantPolicy:
    def __init__(self, acceleration, steering):
        self.control = np.array([acceleration, steering], dtype=float)
        self.call_count = 0

    def act(self, features):
        assert features.shape == (6,)
        self.call_count += 1
        return self.control


def test_tracking_features_use_reference_local_frame():
    state = np.array([1.0, 2.0, 3.0, math.pi / 2.0 + 0.1])
    reference = np.array([1.5, 1.0, 4.0, math.pi / 2.0])

    features = make_tracking_features(state, reference)

    np.testing.assert_allclose(features, [0.0, 0.5, 3.0, 0.1, 0.0, 4.0])


def test_rollout_recursively_propagates_horizon():
    initial_state = np.array([0.0, 0.0, 1.0, 0.0])
    reference = np.zeros((4, 4))
    reference[0] = [0.0, 0.2, 0.4, 0.6]
    reference[2] = 2.0

    states, controls = rollout_policy(
        ConstantPolicy(acceleration=1.0, steering=0.0),
        initial_state,
        reference,
        dt=0.2,
        vehicle=VEHICLE,
    )

    assert states.shape == (4, 4)
    assert controls.shape == (2, 3)
    np.testing.assert_allclose(states[2], [1.0, 1.2, 1.4, 1.6])
    np.testing.assert_allclose(states[1], 0.0)
    np.testing.assert_allclose(controls[0], 1.0)
    np.testing.assert_allclose(controls[1], 0.0)


def test_rollout_clips_policy_output_to_checkpoint_limits():
    reference = np.zeros((4, 2))
    states, controls = rollout_policy(
        ConstantPolicy(acceleration=10.0, steering=2.0),
        np.array([0.0, 0.0, 5.8, 0.0]),
        reference,
        dt=0.2,
        vehicle=VEHICLE,
    )

    np.testing.assert_allclose(controls[:, 0], [3.0, 0.4189])
    assert states[2, 1] == VEHICLE.max_speed


def test_rollout_reuses_actual_first_control_without_second_inference():
    policy = ConstantPolicy(acceleration=0.5, steering=0.1)
    reference = np.zeros((4, 4))
    _, controls = rollout_policy(
        policy,
        np.array([0.0, 0.0, 1.0, 0.0]),
        reference,
        dt=0.2,
        vehicle=VEHICLE,
        first_control=np.array([1.0, -0.2]),
    )

    np.testing.assert_allclose(controls[:, 0], [1.0, -0.2])
    assert policy.call_count == 2

