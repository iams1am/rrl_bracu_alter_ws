#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('serial_motor_bridge')
    
    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM0',
        description='Serial port for Arduino connection'
    )
    
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Baud rate for serial communication'
    )
    
    # Serial Motor Bridge Node
    serial_motor_node = Node(
        package='serial_motor_bridge',
        executable='serial_motor_node.py',
        name='serial_motor_bridge',
        parameters=[
            os.path.join(pkg_share, 'config', 'serial_motor_params.yaml'),
            {
                'serial_port': LaunchConfiguration('serial_port'),
                'baud_rate': LaunchConfiguration('baud_rate'),
            }
        ],
        output='screen'
    )
    
    return LaunchDescription([
        serial_port_arg,
        baud_rate_arg,
        serial_motor_node,
    ])
