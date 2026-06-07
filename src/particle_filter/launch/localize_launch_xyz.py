from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    # Load configuration parameters from localize.yaml
    localize_config = os.path.join(
        get_package_share_directory('particle_filter'),
        'config',
        'localize.yaml'
    )
    localize_config_dict = yaml.safe_load(open(localize_config, 'r'))
    map_name = localize_config_dict['map_server']['ros__parameters']['map']
    
    # Declare launch argument for the localization config if needed
    localize_la = DeclareLaunchArgument(
        'localize_config',
        default_value=localize_config,
        description='Localization configuration file'
    )
    
    ld = LaunchDescription([localize_la])

    # Launch the map_server node directly, bypassing lifecycle management
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[
            {
                #'yaml_filename': os.path.join(get_package_share_directory('particle_filter'),'maps',map_name + '.yaml')
                'yaml_filename':'/home/can-02/f1tenth_ws/src/particle_filter/maps/8th_floor.yaml'
            },
            {'topic': 'map'},
            {'frame_id': 'map'},
            {'use_sim_time': True},
            {'output': 'screen'}
        ]
    )

    # Launch the particle_filter node
    pf_node = Node(
        package='particle_filter',
        executable='particle_filter',
        name='particle_filter',
        parameters=[LaunchConfiguration('localize_config')]
    )

    # Add the nodes to the launch description (lifecycle manager omitted)
    ld.add_action(map_server_node)
    ld.add_action(pf_node)

    return ld

