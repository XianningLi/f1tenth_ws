from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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
        }],
    )

    return LaunchDescription([mpc_node])
