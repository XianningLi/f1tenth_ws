# MPC and RPC Run Logs

The Python MPC node and RPC node use the same RB-gated experiment workflow.
Logging is enabled by default. Each file begins with the first successfully
published controller command, after computational warmup and vehicle startup.

## Experiment workflow

1. Start the controller node and the usual joystick, localization, and vehicle
   bringup nodes. The controller runs `warmup_iterations` synthetic calculations
   before creating its drive publisher or accepting joystick input. MPC compiles
   and solves dummy optimization problems; RPC runs dummy inference and model
   rollouts. No real scene, drive commands, or experiment CSVs are produced by
   warmup. The node then waits in `WAITING`.
2. Hold RB (button index 5). In `STARTUP`, the speed command ramps toward
   `startup_speed` at `startup_acceleration`. Steering is held at zero. The ramp
   accumulates the commanded speed even when the measured wheel speed is zero.
3. Once the command reaches the target and measured speed stays within
   `startup_speed_tolerance` of it, enter `STABILIZING`. Any excursion outside
   the tolerance resets the entire settling timer.
4. After 2.0 consecutive seconds by default, enter `PREPARING`. Capture the
   vehicle's current position and heading as the real virtual-road origin,
   construct its first reference trajectory, and calculate the first command.
   Experiment time has not started, and no CSV exists yet.
5. Successfully publish the first controller command to enter `RUNNING`. The
   monotonic timestamp immediately after the ROS publish call returns is the
   experiment zero. The first CSV row has `experiment_time_sec=0.0` and contains
   that command, its measured input state, reference, and prediction sequences.
   The CSV is opened after publication, outside the measured callback work;
   its filename uses the captured first-publication wall-clock time.
6. Release RB or press LB (button index 4) to end the experiment, close the CSV,
   clear its visualization, and reset the startup process. Another qualified
   RB hold creates another CSV even if the node has not been restarted.

The initial speed (`startup_speed`) and tracking target
(`virtual_reference_speed`) may differ. A startup speed of `-1.0` means to use
`virtual_reference_speed`; both default to an effective 0.6 m/s. In waypoint
mode, the waypoint file remains fixed and is not relocated at experiment start.

The existing `virtual_lane_change_delay` is measured from the first published
controller command, not from node startup, acceleration, or scene preparation.
The first reference uses experiment time zero. Subsequent references use the
time at the start of their calculation, stored as `reference_time_sec`.
The default delay is 2.0 seconds; use `virtual_lane_change_delay:=0.0` to include
the lane-change target in the first command.

This timing definition describes software command publication, not physical
motor response or acknowledgement by the mux. The scene origin is the measured
pose used to compute the first command; it is not relocated after the solve.

The speed ramp is not a motor breakaway boost: it never commands a speed above
the selected startup target to overcome a stalled motor. An unreachable target
aborts startup after `startup_timeout` (15.0 s by default). The vehicle needs a
straight run-up area for acceleration and the speed-settling interval.

Fresh joystick and odometry messages are required. Losing either stream for
more than 0.3 s aborts an active startup or experiment and sends a zero command.
Release RB before retrying after an abort. The existing mux keeps LB manual
commands at higher priority than autonomous commands.

Default directory:

```text
~/.ros/f1tenth_controller_logs
```

File names use the local experiment start time:

```text
MPC_YYYYMMDD_HHMMSS_microseconds.csv
RPC_YYYYMMDD_HHMMSS_microseconds.csv
```

Examples:

```text
MPC_20260809_153012_123456.csv
RPC_20260809_154455_654321.csv
```

Only `RUNNING` controller updates are written. Synthetic warmup, waiting,
acceleration, settling, and preparation samples are excluded. Every row is
flushed immediately. If the first solve, inference, or publication fails, the
experiment aborts without opening a CSV. Failed MPC
solves and failed RPC inferences during the experiment are also recorded with
an English `status` value and empty unavailable output fields. Files close
on RB release, LB takeover, feedback loss, or node shutdown. If file creation
or writing fails with logging enabled, the experiment aborts.

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
- `scene_preparation_ms`: Time from speed qualification to completion of the
  first real-scene/reference construction, before the controller calculation.
- `first_command_latency_ms`: Time from speed qualification to the first
  successful controller publication. This includes scene/reference preparation
  and the first controller calculation. It precedes experiment zero and is
  stored separately rather than added to `experiment_time_sec`.

The two preparation/first-command metrics are constant metadata repeated in
each row of a run. First-solve/inference time remains in the first row's normal
controller timing columns. Startup computational warmup duration is printed in
the terminal and excluded from experiment measurements.

Controller compute values use `time.perf_counter_ns()`; lifecycle and
publication intervals use `time.monotonic()`. All `*_ms` values are milliseconds.
CSV creation and row-write time are excluded from controller/callback timings.

## State and control columns

Each row contains:

- Current state: `state_x`, `state_y`, `state_v`, `state_yaw`.
- `phase`: always `RUNNING` for recorded rows.
- `experiment_time_sec`: elapsed monotonic time at each successful command
  publication, relative to the first publication. The first row is exactly zero.
- `reference_time_sec`: experiment time at the beginning of reference generation.
  It is zero for the first command. The measured state and reference belong to
  the input of that calculation, before command publication.
- `ros_time_sec` and `wall_time_iso`: timestamps captured at command publication,
  not at the later CSV write. Failed calculations after takeover use their
  failure-time timestamps and have no newly published command.
- Controller command: requested acceleration, steering angle, and speed command.
- Full reference sequence: `reference_0_*` through `reference_8_*`.
- Full predicted state sequence: `predicted_0_*` through `predicted_8_*`.
- Full predicted control sequence: `predicted_control_0_*` through
  `predicted_control_7_*`.

The exact number of sequence columns follows the configured/checkpoint horizon.
State order is `[x, y, v, yaw]`; control order is
`[acceleration, steering_angle]`. All values use SI units.

## ROS parameters

Both launch files expose the same startup parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `warmup_iterations` | `3` | Synthetic controller calculations before waiting for RB |
| `startup_speed` | `-1.0` | Initial speed in m/s; -1 uses the tracking reference |
| `startup_acceleration` | `0.5` | Speed-command ramp rate in m/s^2 |
| `startup_speed_tolerance` | `0.05` | Allowed measured-speed error in m/s |
| `startup_stable_duration` | `2.0` | Required uninterrupted settling time in s |
| `startup_timeout` | `15.0` | Maximum startup and settling duration in s |
| `virtual_reference_speed` | `0.6` | Scenario speed target in m/s |
| `virtual_lane_change_delay` | `2.0` | Delay from the first controller publication in s |
| `joy_topic` | `/joy` | Joystick input topic |
| `joy_timeout` | `0.3` | Maximum joystick receive age in s |
| `odometry_timeout` | `0.3` | Maximum odometry receive age in s |

Button indices can also be configured with the ROS node parameters
`autonomous_button` and `manual_button` when using `ros2 run`. They must match
the existing `joy_teleop` configuration. Its default 20 Hz autorepeat supplies
fresh joystick messages while a button is held.

For a 1.0 m/s startup and a 1.0 m/s scenario target:

```bash
colcon build --packages-select mpc rpc
source install/setup.bash
ros2 launch rpc rpc_launch.py \
  virtual_reference_speed:=1.0 \
  startup_stable_duration:=2.0
```

For a speed-change experiment starting at 1.0 m/s and tracking 2.0 m/s:

```bash
ros2 launch rpc rpc_launch.py \
  startup_speed:=1.0 virtual_reference_speed:=2.0
```

Use `ros2 launch mpc mpc_launch.py` with the same arguments to run MPC instead.

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
