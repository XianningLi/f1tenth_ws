import numpy as np

from rpc.trajectory import waypoint_reference_trajectory


def test_waypoint_reference_matches_mpc_two_index_lookahead():
    waypoints = np.zeros((6, 6))
    waypoints[:, 1] = np.arange(6, dtype=float)
    waypoints[:, 2] = 1.0
    waypoints[:, 3] = 0.0
    waypoints[:, 5] = 2.0

    reference = waypoint_reference_trajectory(
        np.array([0.1, 1.0, 1.0, 0.0]),
        waypoints,
        horizon=3,
        index_step=2,
    )

    np.testing.assert_allclose(reference[0], [0.0, 2.0, 4.0, 0.0])
    np.testing.assert_allclose(reference[1], 1.0)
    np.testing.assert_allclose(reference[2], 2.0)

