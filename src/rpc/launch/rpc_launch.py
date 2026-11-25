from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    log_directory = LaunchConfiguration("log_directory")
    save_run_data = LaunchConfiguration("save_run_data")
    experiment_defaults = {
        "virtual_reference_speed": "0.6",
        "virtual_lane_width": "0.3",
        "virtual_road_length": "5.0",
        "virtual_lane_change_delay": "2.0",
        "startup_speed": "-1.0",
        "startup_acceleration": "0.5",
        "startup_zero_speed_threshold": "0.1",
        "startup_stable_duration": "1.0",
        "startup_timeout": "0.0",
        "startup_boost_speed": "1.0",
        "startup_boost_duration": "0.0",
        "joy_timeout": "0.3",
        "odometry_timeout": "0.3",
    }
    experiment_parameters = {
        name: ParameterValue(LaunchConfiguration(name), value_type=float)
        for name in experiment_defaults
    }
    rpc_node = Node(
        package="rpc",
        executable="rpc_node.py",
        name="rpc_node",
        output="screen",
        parameters=[
            {
                "device": "cpu",
                "virtual_road_mode": True,
                **experiment_parameters,
                "control_period": 0.1,
                "warmup_iterations": ParameterValue(
                    LaunchConfiguration("warmup_iterations"), value_type=int,
                ),
                "odom_topic": "/pf/pose/odom",
                "drive_topic": "/drive",
                "joy_topic": LaunchConfiguration("joy_topic"),
                "save_run_data": ParameterValue(save_run_data, value_type=bool),
                "log_directory": log_directory,
            }
        ],
    )

    return LaunchDescription(
        [
            *[
                DeclareLaunchArgument(name, default_value=value)
                for name, value in experiment_defaults.items()
            ],
            DeclareLaunchArgument("joy_topic", default_value="/joy"),
            DeclareLaunchArgument("warmup_iterations", default_value="3"),
            DeclareLaunchArgument(
                "log_directory",
                default_value="~/.ros/f1tenth_controller_logs",
            ),
            DeclareLaunchArgument("save_run_data", default_value="true"),
            rpc_node,
        ]
    )
