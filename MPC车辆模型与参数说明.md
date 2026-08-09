# MPC 控制器车辆模型与参数说明

> 审查对象：当前工作区 `src/mpc` 下的 Python MPC 实现  
> 主要文件：`src/mpc/scripts/mpc_node.py`  
> 审查日期：2026-07-26

## 1. 结论

本项目的 MPC 使用**平面运动学单轨模型**（Kinematic Single-Track Model，也常称 Kinematic Bicycle Model），不是包含轮胎侧偏、横摆惯量和侧向力的动力学车辆模型。

控制器内部的非线性模型为：

$$
\begin{aligned}
\dot{x} &= v\cos\psi \\
\dot{y} &= v\sin\psi \\
\dot{v} &= a \\
\dot{\psi} &= \frac{v}{L}\tan\delta
\end{aligned}
$$

其中：

- 状态向量：$\mathbf{z}=[x,\ y,\ v,\ \psi]^T$
- 控制向量：$\mathbf{u}=[a,\ \delta]^T$
- $x,y$：车辆在全局坐标系中的位置
- $v$：纵向速度
- $\psi$：航向角
- $a$：纵向加速度
- $\delta$：等效前轮转角
- $L$：轴距，代码中为 `WB = 0.33 m`

从方程形式看，$x,y$ 对应的是标准的后轴中心参考点形式；但代码直接使用里程计的位置作为 $x,y$，没有对里程计参考点与后轴中心之间的偏移做补偿。

MPC 优化器没有直接使用上述非线性方程，而是对它进行前向欧拉离散化和逐步线性化，得到离散仿射模型：

$$
\mathbf{z}_{k+1}=A_k\mathbf{z}_k+B_k\mathbf{u}_k+C_k
$$

因此，更准确地说，这是一个基于运动学单轨模型的**线性时变 MPC（LTV-MPC）**。

## 2. 离散非线性预测模型

代码中的 `update_state()` 使用时间步长 $\Delta t=0.2\ \mathrm{s}$：

$$
\begin{aligned}
x_{k+1} &= x_k+v_k\cos\psi_k\,\Delta t \\
y_{k+1} &= y_k+v_k\sin\psi_k\,\Delta t \\
\psi_{k+1} &= \psi_k+\frac{v_k}{L}\tan\delta_k\,\Delta t \\
v_{k+1} &= v_k+a_k\,\Delta t
\end{aligned}
$$

这个非线性模型用于根据上一轮控制序列生成工作轨迹 `path_predict`，随后优化器围绕该轨迹构造线性模型。

代码位置：`src/mpc/scripts/mpc_node.py:478-513`。

## 3. 优化器实际采用的线性化模型

`get_model_matrix(v, phi, delta)` 返回：

$$
A=
\begin{bmatrix}
1 & 0 & \Delta t\cos\bar{\psi} & -\Delta t\bar{v}\sin\bar{\psi} \\
0 & 1 & \Delta t\sin\bar{\psi} & \Delta t\bar{v}\cos\bar{\psi} \\
0 & 0 & 1 & 0 \\
0 & 0 & \dfrac{\Delta t\tan\bar{\delta}}{L} & 1
\end{bmatrix}
$$

$$
B=
\begin{bmatrix}
0 & 0 \\
0 & 0 \\
\Delta t & 0 \\
0 & \dfrac{\Delta t\bar{v}}{L\cos^2\bar{\delta}}
\end{bmatrix}
$$

$$
C=
\begin{bmatrix}
\Delta t\bar{v}\sin\bar{\psi}\,\bar{\psi} \\
-\Delta t\bar{v}\cos\bar{\psi}\,\bar{\psi} \\
0 \\
-\dfrac{\Delta t\bar{v}\bar{\delta}}{L\cos^2\bar{\delta}}
\end{bmatrix}
$$

这里的 $\bar{v},\bar{\psi},\bar{\delta}$ 是线性化工作点。

当前求解代码把每一步的工作点转角固定为：

```python
get_model_matrix(path_predict[2, t], path_predict[3, t], 0.0)
```

即 $\bar{\delta}=0$。因此实际求解时：

$$
A_{4,3}=0,\qquad
B_{4,2}=\frac{\Delta t\bar{v}}{L},\qquad
C_4=0
$$

这种处理简化了模型，但在大转角时会降低线性化精度。

代码位置：

- 矩阵定义：`src/mpc/scripts/mpc_node.py:515-548`
- 求解时的工作点：`src/mpc/scripts/mpc_node.py:550-570`
- 动力学等式约束：`src/mpc/scripts/mpc_node.py:336-389`

## 4. 车辆几何参数

| 参数 | 代码变量 | 当前值 | 是否进入车辆动力学 | 说明 |
|---|---:|---:|---|---|
| 车辆长度 | `LENGTH` | `0.58 m` | 否 | 只声明，当前 MPC 中没有用于碰撞包络或动力学 |
| 车辆宽度 | `WIDTH` | `0.31 m` | 否 | 只声明，当前 MPC 中没有用于碰撞包络或动力学 |
| 轴距 | `WB` | `0.33 m` | **是** | 用于 $\dot{\psi}=v\tan\delta/L$ |

代码位置：`src/mpc/scripts/mpc_node.py:53-55`。

需要特别注意：这个运动学模型不需要质量、横摆惯量、前后轴侧偏刚度、质心到前后轴距离等动力学参数。当前模型也没有描述轮胎侧偏、车身侧滑、载荷转移或执行器迟滞。

## 5. 状态、控制量与预测参数

| 类别 | 代码变量 | 当前值 | 含义 |
|---|---:|---:|---|
| 状态维数 | `NXK` | `4` | $[x,y,v,\psi]$ |
| 控制维数 | `NU` | `2` | $[a,\delta]$ |
| 预测步数 | `TK` | `8` | 8 个控制区间、9 个状态节点 |
| 模型时间步长 | `DTK` | `0.2 s` | 模型离散化步长 |
| 预测时域 | `TK × DTK` | `1.6 s` | 8 × 0.2 s |
| 控制定时器周期 | `control_period` | `0.1 s` | ROS 节点默认每 0.1 s 重新求解一次 |
| 默认路径距离步长 | `dlk` | `0.03 m` | 用于参考点索引，不是车辆物理参数 |
| 搜索点数 | `N_IND_SEARCH` | `20` | 已声明，但当前主节点中未实际使用 |

`dlk` 在非虚拟道路模式加载路径时会被改写为路径文件前两个采样点第一列之差，见 `src/mpc/scripts/mpc_node.py:147-154`。

模型步长为 `0.2 s`，而控制器默认每 `0.1 s` 重规划一次。这意味着“预测离散步长”和“闭环重规划周期”并不相同；这种多速率设置可以使用，但调参和分析时不能把两者混为一谈。

## 6. 状态与输入约束

| 约束 | 代码变量 | 当前值 | 换算/说明 |
|---|---:|---:|---|
| 最小转角 | `MIN_STEER` | `-0.4189 rad` | 约 `-24.0°`；该变量只声明，约束实际通过 `abs(delta)` 实现 |
| 最大转角 | `MAX_STEER` | `0.4189 rad` | 约 `24.0°` |
| 最大转角变化率 | `MAX_DSTEER` | `π rad/s` | `180°/s` |
| 每个模型步最大转角变化 | `MAX_DSTEER × DTK` | `0.6283 rad` | `36°/step` |
| 最低速度 | `MIN_SPEED` | `0.0 m/s` | 不允许倒车 |
| 最高速度 | `MAX_SPEED` | `6.0 m/s` | 状态约束 |
| 最大加速度绝对值 | `MAX_ACCEL` | `3.0 m/s²` | 即 $-3\le a\le3$ |

代码位置：

- 参数声明：`src/mpc/scripts/mpc_node.py:56-61`
- 约束实现：`src/mpc/scripts/mpc_node.py:391-408`

虽然代码注释有时把第二个控制量称为 `steering speed`，但从模型矩阵、转角限幅以及最终发布到 `drive.steering_angle` 的行为可以确认，它实际是**转向角 $\delta$**，不是转向角速度。

## 7. MPC 代价权重

以下参数影响控制器行为，但不是车辆物理参数。

### 7.1 控制输入权重

控制顺序为 $[a,\delta]$：

$$
R=
\operatorname{diag}(0.01,\ 100.0)
$$

转角的代价远大于加速度的代价。

### 7.2 控制增量权重

$$
R_d=
\operatorname{diag}(0.01,\ 100.0)
$$

它惩罚相邻预测步之间的加速度变化和转角变化。

### 7.3 状态跟踪权重

状态顺序为 $[x,y,v,\psi]$：

$$
Q=
\operatorname{diag}(100.0,\ 100.0,\ 5.5,\ 50.0)
$$

### 7.4 终端状态权重

$$
Q_f=
\operatorname{diag}(13.5,\ 13.5,\ 5.5,\ 13.0)
$$

代码位置：`src/mpc/scripts/mpc_node.py:33-47`。

完整目标函数为：

$$
J=
\sum_{k=0}^{T-1}
\left(
\mathbf{u}_k^TR\mathbf{u}_k+
(\mathbf{z}_k-\mathbf{z}_{k,\mathrm{ref}})^TQ
(\mathbf{z}_k-\mathbf{z}_{k,\mathrm{ref}})
\right)
+
\sum_{k=0}^{T-2}
\Delta\mathbf{u}_k^TR_d\Delta\mathbf{u}_k
+
(\mathbf{z}_T-\mathbf{z}_{T,\mathrm{ref}})^TQ_f
(\mathbf{z}_T-\mathbf{z}_{T,\mathrm{ref}})
$$

优化问题由 CVXPY 构造，并使用 OSQP 求解。

## 8. 状态来源与控制输出

车辆当前状态来自里程计：

- 实车模式在代码中被固定为 `sim_real = True`
- 订阅话题：`/pf/pose/odom`
- $x,y$：`pose.pose.position`
- $v$：`max(0, twist.twist.linear.x)`
- $\psi$：由四元数计算的 yaw

MPC 发布：

- 话题：`/drive`
- 速度命令：$v_{\mathrm{cmd}}=\operatorname{clip}(v+a_0\Delta t,\ 0,\ 6)$
- 转向命令：$\delta_{\mathrm{cmd}}=\delta_0$

代码位置：`src/mpc/scripts/mpc_node.py:118-144`、`202-278`。

## 9. 参数一致性检查

### 9.1 轴距存在不一致

当前仓库内至少存在三个轴距值：

| 使用位置 | 轴距 |
|---|---:|
| MPC 模型 `src/mpc/scripts/mpc_node.py` | `0.33 m` |
| 当前 bringup 默认加载的 `src/f1tenth_system/f1tenth_stack/config/vesc.yaml` | `0.25 m` |
| 旧式 `src/f1tenth_system/vesc/vesc_ackermann/launch/vesc_to_odom_node.launch.xml` | `0.20 m` |

`bringup_launch.py` 默认加载 `f1tenth_stack/config/vesc.yaml`，因此按当前默认 bringup 路径，底盘里程计使用的是 `0.25 m`，而 MPC 预测使用的是 `0.33 m`。

横摆角速度满足：

$$
\dot{\psi}=\frac{v}{L}\tan\delta
$$

在相同速度和转角下，`0.33 m` 模型预测的横摆角速度约为 `0.25 m` 模型的：

$$
\frac{0.25}{0.33}\approx0.758
$$

即约低 `24.2%`。建议实测轴距或通过转向实验辨识等效轴距，然后让 MPC 与底盘里程计配置使用同一数值。

### 9.2 长度和宽度没有参与约束

`LENGTH = 0.58 m` 和 `WIDTH = 0.31 m` 没有被引用。因此当前控制器在优化中把车辆当作一个点，没有利用车辆外形做赛道边界或障碍物碰撞约束。

### 9.3 转角线性化点固定为零

QP 每一步均以 $\bar{\delta}=0$ 构造 $A,B,C$，没有使用预测转角作为线性化工作点。小转角时通常较合理，但在接近 `±24°` 的工况下，模型误差会变大。

### 9.4 `mpc_cbf_node.py` 使用相同模型

当前工作区中的 `src/mpc/scripts/mpc_cbf_node.py` 与 `mpc_node.py` 的数值模型、参数、目标函数和约束相同，只修改了类名、节点名和启动入口。就当前代码而言，它没有增加 CBF 安全约束。

## 10. 建议优先核实的参数

1. **统一轴距**：重点确认 `0.33 m` 和 `0.25 m` 哪个与实车等效轴距一致。
2. **校准最大等效前轮转角**：确认 Ackermann 转角命令经过舵机映射后是否确实对应 `±0.4189 rad`。
3. **核实模型步长**：结合实际求解频率和执行周期，决定 `DTK = 0.2 s` 是否合适。
4. **大转角时改进线性化**：可以用上一轮预测转角作为 $\bar{\delta}$，并进行顺序线性化迭代。
5. **需要高速极限控制时升级模型**：若侧滑不可忽略，应考虑动力学单轨模型，并辨识质量、横摆惯量、质心位置和轮胎侧偏刚度。

## 11. 代码索引

| 内容 | 文件与行号 |
|---|---|
| 模型维数、权重、车辆参数与约束参数 | `src/mpc/scripts/mpc_node.py:27-61` |
| 状态定义 | `src/mpc/scripts/mpc_node.py:64-72` |
| ROS 参数与节点初始化 | `src/mpc/scripts/mpc_node.py:74-144` |
| 当前状态读取与控制发布 | `src/mpc/scripts/mpc_node.py:202-278` |
| 目标函数和约束 | `src/mpc/scripts/mpc_node.py:283-416` |
| 非线性状态更新 | `src/mpc/scripts/mpc_node.py:478-513` |
| 离散线性模型矩阵 | `src/mpc/scripts/mpc_node.py:515-548` |
| OSQP 求解 | `src/mpc/scripts/mpc_node.py:550-594` |
| 默认 MPC 启动参数 | `src/mpc/launch/mpc_launch.py:1-21` |
| 底盘默认轴距 | `src/f1tenth_system/f1tenth_stack/config/vesc.yaml:30-38` |
| bringup 加载 VESC 配置 | `src/f1tenth_system/f1tenth_stack/launch/bringup_launch.py:39-42,59-62,86-102` |

