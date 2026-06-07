#!/bin/bash

# Navigate to your F1TENTH ROS 2 workspace
cd ~/f1tenth_ws

# Source the workspace environment to overlay ROS 2 packages
source install/setup.bash

# Run the pure_pursuit node
ros2 run pure_pursuit pure_pursuit_node.py

