# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch Cartographer 2D SLAM with RPLidar A3 and CMP10A IMU."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config_directory = os.path.join(
        get_package_share_directory('cartographer_slam'), 'config'
    )
    
    configuration_basename = 'cartographer_2d.lua'

    use_sim_time = LaunchConfiguration('use_sim_time')
    lidar_serial_port = LaunchConfiguration('lidar_serial_port')
    imu_serial_port = LaunchConfiguration('imu_serial_port')
    launch_rviz = LaunchConfiguration('launch_rviz')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time',
        ),
        DeclareLaunchArgument(
            'lidar_serial_port',
            default_value='/dev/ttyUSB0',
            description='Serial port for RPLidar A3 at 256000 baud',
        ),
        DeclareLaunchArgument(
            'imu_serial_port',
            default_value='/dev/ttyUSB1',
            description='Serial port for Yahboom CMP10A IMU at 115200 baud',
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Start RViz2 with Cartographer',
        ),

        # Static transform: base_link -> laser
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser'],
        ),
        
        # Static transform: base_link -> imu_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'],
        ),

        # Static transform: base_link -> camera_link
        # Adjust x/y/z to match the physical mounting position of the RealSense D435i
        # on your robot (x=forward, y=left, z=up from base_link origin, in metres).
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera',
            arguments=['0.1', '0', '0.2', '0', '0', '0', 'base_link', 'camera_link'],
        ),

        # RealSense ROS prefixes its internal base frame as camera_camera_link
        # when the node is launched as /camera/camera. Bridge that frame back
        # to the robot-mounted camera_link so depth detections can transform to map.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_link_to_realsense',
            arguments=['0', '0', '0', '0', '0', '0', 'camera_link', 'camera_camera_link'],
        ),
        
        # RPLidar A3 node
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            parameters=[{
                'channel_type': 'serial',
                'serial_port': lidar_serial_port,
                'serial_baudrate': 256000,
                'frame_id': 'laser',
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': 'Sensitivity',
            }],
            output='screen',
        ),
        
        # IMU node
        Node(
            package='wit_ros2_imu',
            executable='wit_ros2_imu',
            name='imu',
            parameters=[{'port': imu_serial_port, 'baud': 115200}],
            output='screen',
        ),
        
        # Cartographer node
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '-configuration_directory', config_directory,
                '-configuration_basename', configuration_basename,
            ],
            remappings=[
                ('scan', '/scan'),
                # wit_ros2_imu publishes on imu/data; cartographer_ros expects imu
                ('imu', 'imu/data'),
            ],
            output='screen',
        ),
        
        # Occupancy grid node
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='occupancy_grid_node',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'resolution': 0.05},
            ],
            output='screen',
        ),

        # Robot path tracker for final RRL GeoTIFF export
        Node(
            package='cartographer_slam',
            executable='path_tracker.py',
            name='path_tracker',
            parameters=[{
                'map_frame': 'map',
                'base_frame': 'base_link',
                'publish_topic': '/robot_path',
            }],
            output='screen',
        ),
        
        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            condition=IfCondition(launch_rviz),
            output='screen',
        ),
    ])
