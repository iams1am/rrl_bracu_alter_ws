#!/usr/bin/env python3
"""
Complete launch file for differential drive robot teleop control
Launches: Joy node + Teleop twist joy + Serial motor bridge
"""

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
    
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Baud rate for serial communication'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug_serial',
        default_value='false',
        description='Enable serial debug output'
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
        output='screen'
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
                'debug_serial': LaunchConfiguration('debug_serial'),
            }
        ],
        output='screen'
    )
    
    return LaunchDescription([
        joy_dev_arg,
        serial_port_arg,
        baud_rate_arg,
        debug_arg,
        joy_node,
        teleop_twist_joy_node,
        serial_motor_node,
    ])
