#!/usr/bin/env python3
import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from utils import nearest_point



# TODO CHECK: include needed ROS msg type headers and libraries
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import time

@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        # default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
        default_factory=lambda: np.diag([100.0, 100.0, 5.5, 50])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.2  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        # TODO: create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        # TODO: get waypoints here
        # print("here")
        self.config = mpc_config()
        self.declare_parameter('virtual_road_mode', True)
        self.declare_parameter('waypoint_path', '/home/nvidia/Downloads/levine_centerline.csv')
        self.declare_parameter('virtual_lane_width', 0.3)
        self.declare_parameter('virtual_road_length', 5.0)
        self.declare_parameter('virtual_reference_speed', 0.6)
        self.declare_parameter('virtual_lane_change_delay', 2.0)
        self.declare_parameter('control_period', 0.1)

        self.virtual_road_mode = self.get_parameter('virtual_road_mode').value
        self.waypoint_path = self.get_parameter('waypoint_path').value
        self.virtual_lane_width = self.get_parameter('virtual_lane_width').value
        self.virtual_road_length = self.get_parameter('virtual_road_length').value
        self.virtual_reference_speed = self.get_parameter('virtual_reference_speed').value
        self.virtual_lane_change_delay = self.get_parameter('virtual_lane_change_delay').value
        self.control_period = self.get_parameter('control_period').value
        self.virtual_road_initialized = False
        self.virtual_origin = None
        self.virtual_yaw = None
        self.virtual_forward = None
        self.virtual_left = None
        self.virtual_start_time = None
        self.latest_odom_msg = None

        
        # self.waypoint_path = "/home/bosky2001/Downloads/traj_race_cl.csv" #mincurv centerline 8ish seconds at 80% velo and no crash
        # self.waypoint_path = "/home/bosky2001/f1tenth_gym/sim_ws/src/lab-8-model-predictive-controlc-team-3/mpc/reflines/traj_race_cl_mincurv.csv"
        self.waypoints = None
        if not self.virtual_road_mode:
            self.waypoints = self.load_waypoints(self.waypoint_path)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.drive_msg_ = AckermannDriveStamped()

        # False-sim, True -on car
        self.sim_real = True
        if(not self.sim_real):
            self.pose_sub_ = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 1)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 1)

        self.ref_goal_points_ = self.create_publisher(MarkerArray, 'ref_goal_points', 1)
        self.ref_trajectory_ = self.create_publisher(Marker,'ref_trajectory', 1)
        self.opt_trajectory_ = self.create_publisher(Marker,'opt_trajectory', 1)
        self.virtual_road_ = self.create_publisher(MarkerArray, 'virtual_road', 1)
        self.ref_target_point_ = self.create_publisher(Marker, 'ref_target_point', 1)
        
        
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # initialize MPC problem
        self.mpc_prob_init()

        # visualize goal points
        self.ref_goal_points_data = None
        if not self.virtual_road_mode:
            self.ref_goal_points_data = self.viz_ref_points()
        self.control_timer = self.create_timer(self.control_period, self.control_callback)

    
    def load_waypoints(self, path):
        points = np.loadtxt(path, delimiter=';',skiprows=3, dtype=np.float64)
        #  points = np.loadtxt(path, delimiter=';', skiprows=3)
        # points[:, 3] += 0.5*math.pi
        points[:, 5] *= 0.5
        # CONFIGURE DLK
        self.config.dlk = points[1, 0] - points[0, 0]
        return points

    def init_virtual_road(self, state):
        self.virtual_origin = np.array([state.x, state.y], dtype=np.float64)
        self.virtual_yaw = state.yaw
        self.virtual_forward = np.array(
            [math.cos(self.virtual_yaw), math.sin(self.virtual_yaw)],
            dtype=np.float64,
        )
        self.virtual_left = np.array(
            [-math.sin(self.virtual_yaw), math.cos(self.virtual_yaw)],
            dtype=np.float64,
        )
        self.virtual_start_time = time.time()
        self.virtual_road_initialized = True
        self.get_logger().info(
            f'Virtual road initialized at x={state.x:.2f}, y={state.y:.2f}, yaw={state.yaw:.2f}'
        )

    def virtual_point(self, s, lateral_offset):
        point = (
            self.virtual_origin
            + s * self.virtual_forward
            + lateral_offset * self.virtual_left
        )
        return float(point[0]), float(point[1])

    def calc_virtual_ref_trajectory(self, state):
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        current_position = np.array([state.x, state.y], dtype=np.float64)
        current_s = float(np.dot(current_position - self.virtual_origin, self.virtual_forward))
        current_s = max(0.0, current_s)
        elapsed = time.time() - self.virtual_start_time
        target_offset = 0.0
        if elapsed >= self.virtual_lane_change_delay:
            target_offset = self.virtual_lane_width

        step_distance = max(abs(state.v), self.virtual_reference_speed, 0.1) * self.config.DTK
        for i in range(self.config.TK + 1):
            x, y = self.virtual_point(current_s + i * step_distance, target_offset)
            ref_traj[0, i] = x
            ref_traj[1, i] = y
            ref_traj[2, i] = self.virtual_reference_speed
            ref_traj[3, i] = self.virtual_yaw

        return ref_traj


    def pose_callback(self, pose_msg):
        self.latest_odom_msg = pose_msg

    def control_callback(self):
        if self.latest_odom_msg is None:
            return

        pose_msg = self.latest_odom_msg
        # print("hi")
        start = time.time()
        
        # TODO: extract pose from ROS msg
        #state values are correct

        x_state = pose_msg.pose.pose.position.x
        y_state = pose_msg.pose.pose.position.y
        curr_orien = pose_msg.pose.pose.orientation
        # print(x_state, y_state)
        vel_state = max(0.0, pose_msg.twist.twist.linear.x)

        
        q = [curr_orien.x, curr_orien.y, curr_orien.z, curr_orien.w]
        yaw_state = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        # print("current yaw", yawp)
        vehicle_state = State(x = x_state, y = y_state, v = vel_state, yaw = yaw_state)

        if self.virtual_road_mode:
            if not self.virtual_road_initialized:
                self.init_virtual_road(vehicle_state)
            ref_path = self.calc_virtual_ref_trajectory(vehicle_state)
            self.viz_virtual_road()
            self.viz_ref_target(ref_path)
        else:
            ref_x = self.waypoints[:, 1]
            ref_y = self.waypoints[:, 2]
            ref_yaw = self.waypoints[:, 3]
            ref_v = self.waypoints[:, 5]
            # TODO: Calculate the next reference trajectory for the next T steps
            #       with current vehicle pose.
            #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
            ref_path = self.calc_ref_trajectory( vehicle_state, ref_x, ref_y, ref_yaw, ref_v)

        self.viz_rej_traj(ref_traj=ref_path)
        #print(ref_path.shape)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        if self.oa is None or self.odelta_v is None:
            self.get_logger().warn('MPC solve failed; skipping drive command.')
            return
        
        # print("reference yaw", ref_path[2, :])
        # print("predicted yaw", oyaw)
        # print("predicted yaw2", state_predict[2, :])
        # print(state_predict.shape)
        opt_traj = np.vstack((ox, oy, ov, oyaw))
        self.viz_opt_traj(opt_traj)
        print("Time to solve",  time.time() - start)
        print("steer output is",self.odelta_v[0] )
        # TODO: publish drive message.
        steer_output = float(self.odelta_v[0])
        speed_output = float(np.clip(vehicle_state.v + self.oa[0] * self.config.DTK, self.config.MIN_SPEED, self.config.MAX_SPEED))

        
        self.drive_msg_.drive.speed = speed_output
        self.drive_msg_.drive.steering_angle = steer_output
        self.drive_pub_.publish(self.drive_msg_)
        if self.ref_goal_points_data is not None:
            self.ref_goal_points_.publish(self.ref_goal_points_data)


    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere

        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective +=  cvxpy.quad_form(cvxpy.vec(self.uk), R_block) #need to flatten to multiply with R block since its all stacked up
        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)
        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        # print(cvxpy.diff(self.uk, axis = 1).shape) 2x7
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis = 1)), Rd_block)
        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        constraints += [cvxpy.vec(self.xk[:, 1:]) == self.Ak_@cvxpy.vec(self.xk[:, :-1]) + self.Bk_@cvxpy.vec(self.uk) + (self.Ck_)]

        # # TODO: Constraint part 2:
        # #       Add constraints on steering, change in steering angle
        # #       cannot exceed steering angle speed limit. Should be based on:
        # #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        constraints += [cvxpy.abs(cvxpy.diff(self.uk[1,:])) <= self.config.MAX_DSTEER*self.config.DTK]
        # # TODO: Constraint part 3:
        # #       Add constraints on upper and lower bounds of states and inputs
        # #       and initial state constraint, should be based on:
        # #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        # #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER

        # # input limits
        constraints +=[ cvxpy.abs(self.uk[0, :]) <= self.config.MAX_ACCEL ]
        constraints += [ cvxpy.abs(self.uk[1, :]) <= self.config.MAX_STEER ]
        
        # # speed limits + setting initial state
        constraints += [self.xk[2, :] <= self.config.MAX_SPEED, self.config.MIN_SPEED <=  self.xk[2, :]]
        constraints += [self.xk[:, 0] == self.x0k]
        # -------------------------------------------------------------



        
        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        dind = 2
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]

        angle_thres = 4.5
        # https://edstem.org/us/courses/34340/discussion/2817574

        for i in range(len(cyaw)):
            if cyaw[i] - state.yaw > angle_thres:
                cyaw[i] -= 2*np.pi
                # print(cyaw[i] - state.yaw)
            if state.yaw - cyaw[i] > angle_thres:
                cyaw[i] += 2*np.pi
                # print(cyaw[i] - state.yaw)

        # cyaw[cyaw - state.yaw > angle_thres] = np.abs(
        #     cyaw[cyaw - state.yaw > angle_thres] - (2 * np.pi)
        # )
        # cyaw[cyaw - state.yaw < -angle_thres] = np.abs(
        #     cyaw[cyaw - state.yaw < -angle_thres] + (2 * np.pi)
        # )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
        print("MPC solve time:", self.MPC_prob.solver_stats.solve_time)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    ## Visualization MPC utils
    def viz_virtual_road(self):
        road = MarkerArray()
        now = self.get_clock().now().to_msg()
        edge_specs = [
            (-0.5 * self.virtual_lane_width, 0.9, 0.9, 0.9),
            (0.5 * self.virtual_lane_width, 1.0, 1.0, 1.0),
            (1.5 * self.virtual_lane_width, 0.9, 0.9, 0.9),
        ]

        for marker_id, (offset, red, green, blue) in enumerate(edge_specs):
            line = Marker(type=Marker.LINE_STRIP, scale=Vector3(x=0.025, y=0.025, z=0.025))
            line.header.frame_id = 'map'
            line.header.stamp = now
            line.ns = 'virtual_road_edges'
            line.id = marker_id
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.color.a = 1.0
            line.color.r = red
            line.color.g = green
            line.color.b = blue

            for s in np.linspace(-0.5, self.virtual_road_length, 40):
                x, y = self.virtual_point(s, offset)
                line.points.append(Point(x=x, y=y, z=0.0))

            road.markers.append(line)

        center_specs = [
            (0.0, 0.0, 0.9, 0.1),
            (self.virtual_lane_width, 0.0, 0.8, 1.0),
        ]
        dash_length = 0.18
        gap_length = 0.12
        s_start = -0.5
        s_end = self.virtual_road_length

        for marker_id, (offset, red, green, blue) in enumerate(center_specs, start=10):
            center = Marker(type=Marker.LINE_LIST, scale=Vector3(x=0.018, y=0.018, z=0.018))
            center.header.frame_id = 'map'
            center.header.stamp = now
            center.ns = 'virtual_lane_centers'
            center.id = marker_id
            center.action = Marker.ADD
            center.pose.orientation.w = 1.0
            center.color.a = 1.0
            center.color.r = red
            center.color.g = green
            center.color.b = blue

            s = s_start
            while s < s_end:
                x0, y0 = self.virtual_point(s, offset)
                x1, y1 = self.virtual_point(min(s + dash_length, s_end), offset)
                center.points.append(Point(x=x0, y=y0, z=0.02))
                center.points.append(Point(x=x1, y=y1, z=0.02))
                s += dash_length + gap_length

            road.markers.append(center)

        self.virtual_road_.publish(road)

    def viz_ref_target(self, ref_traj):
        target = Marker(type=Marker.SPHERE)
        target.header.frame_id = 'map'
        target.header.stamp = self.get_clock().now().to_msg()
        target.ns = 'ref_target'
        target.id = 0
        target.action = Marker.ADD
        target.pose.orientation.w = 1.0
        target.pose.position.x = float(ref_traj[0, -1])
        target.pose.position.y = float(ref_traj[1, -1])
        target.pose.position.z = 0.0
        target.scale.x = 0.14
        target.scale.y = 0.14
        target.scale.z = 0.14
        target.color.a = 1.0
        target.color.r = 1.0
        target.color.g = 0.55
        target.color.b = 0.0
        self.ref_target_point_.publish(target)

    def viz_ref_points(self):
        ref_points = MarkerArray()

        for i in range(self.waypoints.shape[0]):
            message = Marker()
            message.header.frame_id="map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.type= Marker.SPHERE
            message.action = Marker.ADD
            message.id=i
            message.pose.orientation.x=0.0
            message.pose.orientation.y=0.0
            message.pose.orientation.z=0.0
            message.pose.orientation.w=1.0
            message.scale.x=0.2
            message.scale.y=0.2
            message.scale.z=0.2
            message.color.a=1.0
            message.color.r=1.0
            message.color.b=0.0
            message.color.g=0.0
            message.pose.position.x=float(self.waypoints[i,1])
            message.pose.position.y=float(self.waypoints[i,2])
            message.pose.position.z=0.0
            ref_points.markers.append(message)
        return ref_points
    
    def viz_rej_traj(self, ref_traj):

        traj = Marker(type=Marker.LINE_STRIP,
                        scale=Vector3(x=0.1, y=0.1, z=0.1))
        traj.header.frame_id = 'map'
        traj.color.r = 0.0
        traj.color.g = 0.0
        traj.color.b = 1.0
        traj.color.a = 1.0
        traj.id = 1
        for i in range(ref_traj.shape[1]):
            x, y, _, _ = ref_traj[:, i]
            # print(f'Publishing ref traj x={x}, y={y}')
            traj.points.append(Point(x=x, y=y, z=0.0))
        self.ref_trajectory_.publish(traj)
    
    def viz_opt_traj(self, opt_traj):

        traj = Marker(type=Marker.LINE_STRIP,
                        scale=Vector3(x=0.1, y=0.1, z=0.1))
        traj.header.frame_id = 'map'
        traj.color.r = 1.0
        traj.color.g = 0.0
        traj.color.b = 1.0
        traj.color.a = 1.0
        traj.id = 1
        for i in range(opt_traj.shape[1]):
            x, y, _, _ = opt_traj[:, i]
            # print(f'Publishing ref traj x={x}, y={y}')
            traj.points.append(Point(x=x, y=y, z=0.0))
        self.opt_trajectory_.publish(traj)



def main(args=None):

    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
