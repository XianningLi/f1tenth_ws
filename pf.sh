#!/bin/bash

# Navigate to your F1TENTH ROS 2 workspace
cd ~/f1tenth_ws

# Source the workspace environment to overlay ROS 2 packages
source install/setup.bash

ros2 launch particle_filter localize_launch.py
