#!/usr/bin/env python3
"""
Launch file for joystick teleop with serial motor bridge
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('serial_motor_bridge')
    
    # Declare launch arguments
    joy_dev_arg = DeclareLaunchArgument(
        'joy_dev',
        default_value='/dev/input/js0',
        description='Joystick device'
    )
    
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM0',
        description='Serial port for Arduino'
    )
    
    # Joy node - reads joystick input
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'device_id': 0,
            'deadzone': 0.1,
            'autorepeat_rate': 20.0,
        }],
        remappings=[
            ('/joy', '/joy'),
        ],
        output='screen'
    )
    
    # Teleop twist joy node - converts joystick to cmd_vel
    teleop_twist_joy_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        parameters=[
            os.path.join(pkg_share, 'config', 'teleop_joy_params.yaml')
        ],
        remappings=[
            ('/cmd_vel', '/cmd_vel'),
        ],
        output='screen'
    )
    
    # Serial Motor Bridge Node
    serial_motor_node = Node(
        package='serial_motor_bridge',
        executable='serial_motor_node.py',
        name='serial_motor_bridge',
        parameters=[
            os.path.join(pkg_share, 'config', 'serial_motor_params.yaml'),
            {'serial_port': LaunchConfiguration('serial_port')},
        ],
        output='screen'
    )
    
    return LaunchDescription([
        joy_dev_arg,
        serial_port_arg,
        joy_node,
        teleop_twist_joy_node,
        serial_motor_node,
    ])
