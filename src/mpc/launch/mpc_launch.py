from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    log_directory = LaunchConfiguration('log_directory')
    save_run_data = LaunchConfiguration('save_run_data')
    mpc_node = Node(
        package='mpc',
        executable='mpc_node.py',
        name='mpc_node',
        output='screen',
        parameters=[{
            'virtual_road_mode': True,
            'virtual_lane_width': 0.3,
            'virtual_road_length': 5.0,
            'virtual_reference_speed': 0.6,
            'virtual_lane_change_delay': 2.0,
            'control_period': 0.1,
            'save_run_data': ParameterValue(save_run_data, value_type=bool),
            'log_directory': log_directory,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'log_directory',
            default_value='~/.ros/f1tenth_controller_logs',
        ),
        DeclareLaunchArgument('save_run_data', default_value='true'),
        mpc_node,
    ])
