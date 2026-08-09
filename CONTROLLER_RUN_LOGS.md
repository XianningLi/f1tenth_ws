# MPC and RPC Run Logs

Both controller nodes create one CSV file per process start. Logging is enabled
by default.

Default directory:

```text
~/.ros/f1tenth_controller_logs
```

File names use the local process start time:

```text
MPC_YYYYMMDD_HHMMSS_microseconds.csv
RPC_YYYYMMDD_HHMMSS_microseconds.csv
```

Examples:

```text
MPC_20260809_153012_123456.csv
RPC_20260809_154455_654321.csv
```

Every successful control update is flushed to disk immediately. Failed MPC
solves and failed RPC inferences are also recorded with an English `status`
value and empty unavailable output fields.

## Timing columns

- `controller_compute_ms`: Complete algorithm time. For MPC this includes
  motion prediction, CVXPY parameter updates, and the OSQP call. For RPC this
  includes the actual one-step control inference and the future trajectory
  rollout.
- `solver_internal_ms`: OSQP's internal solver time reported by CVXPY. This is
  populated only for MPC.
- `control_inference_ms`: One feature construction and one neural network
  inference used for the actual RPC command. This is the fair RPC value to
  compare against the time needed by MPC to obtain its actual command.
- `prediction_rollout_ms`: RPC-only future trajectory generation. The already
  computed first command is reused, so the rollout does not infer the first
  command a second time.
- `callback_total_ms`: Full callback work before the CSV write, including
  reference generation, command publication, and visualization publication.

All timing values use `time.perf_counter_ns()` and are stored in milliseconds.
CSV disk-write time is intentionally excluded from the controller timing.

## State and control columns

Each row contains:

- Current state: `state_x`, `state_y`, `state_v`, `state_yaw`.
- Applied command: acceleration, steering angle, and speed command.
- Full reference sequence: `reference_0_*` through `reference_8_*`.
- Full predicted state sequence: `predicted_0_*` through `predicted_8_*`.
- Full predicted control sequence: `predicted_control_0_*` through
  `predicted_control_7_*`.

The exact number of sequence columns follows the configured/checkpoint horizon.
State order is `[x, y, v, yaw]`; control order is
`[acceleration, steering_angle]`. All values use SI units.

## ROS parameters

Use a custom output directory:

```bash
ros2 launch mpc mpc_launch.py \
  log_directory:=/home/nvidia/controller_logs

ros2 launch rpc rpc_launch.py \
  log_directory:=/home/nvidia/controller_logs
```

Disable logging when needed:

```bash
ros2 launch rpc rpc_launch.py save_run_data:=false
```

MPC and RPC do not need to run simultaneously. Their timestamped files can be
loaded afterward with pandas, MATLAB, Excel, or another analysis tool for
comparison.
