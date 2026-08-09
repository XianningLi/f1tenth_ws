# RPC 神经网络控制节点

`rpc_node` 保持现有 MPC 节点的 ROS 接口和参考轨迹生成方式，但把在线
CVXPY/OSQP 求解替换为训练好的 RPC/HFRPC 网络。网络每次输出一步控制量
`[acceleration, steering_angle]`，节点再用非线性运动学自行车模型递推整个预测
时域，并把结果发布到 `opt_trajectory`。

默认模型 `models/hfrpc.pt` 来自用户提供的 handoff，SHA-256 为：

```text
57A7FAE35CEE6B0EED6F732E05603B794EC21DC0270CB548437104E32FDACA3C
```

该 checkpoint 的方法名是 `hfrpc`；节点同时兼容相同 checkpoint 格式的
`rpc` 模型。默认参数为 8 步预测、模型步长 0.2 s、控制周期 0.1 s、轴距
0.33 m，并从 checkpoint 读取这些模型参数和执行器限制。

## 构建与启动

运行环境除 ROS 2 依赖外还需要 NumPy 和 PyTorch。PyTorch 的版本应与目标
机器的 CPU/CUDA 环境匹配。

```bash
colcon build --packages-select rpc
source install/setup.bash
ros2 launch rpc rpc_launch.py
```

默认接口：

- 订阅：`/pf/pose/odom` (`nav_msgs/Odometry`)
- 控制发布：`/drive` (`ackermann_msgs/AckermannDriveStamped`)
- 参考轨迹：`ref_trajectory`
- 模型递推轨迹：`opt_trajectory`
- 虚拟道路和目标：`virtual_road`、`ref_target_point`

主要参数：

```bash
ros2 run rpc rpc_node.py --ros-args \
  -p model_path:=/absolute/path/to/model.pt \
  -p device:=cpu \
  -p virtual_road_mode:=true \
  -p odom_topic:=/pf/pose/odom \
  -p drive_topic:=/drive
```

非虚拟道路模式使用与 MPC 相同的分号分隔 waypoint 文件：

```bash
ros2 run rpc rpc_node.py --ros-args \
  -p virtual_road_mode:=false \
  -p waypoint_path:=/absolute/path/to/raceline.csv
```

## 模型边界

当前 HFRPC 是用直线路面上的车道中心与速度跟踪数据训练的。节点在 waypoint
模式下会把车辆误差投影到每个参考点的局部坐标系，因此接口和递推可以用于曲线
参考轨迹；但模型没有曲率预瞄输入，也没有障碍物、赛道边界或车辆外形约束。
在真实曲线赛道使用前，应使用包含目标赛道曲率/轨迹分布的数据重新训练并验证。
