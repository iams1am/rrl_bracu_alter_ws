"""Master bringup for BRACU Alter RRL robot.

Starts the full pipeline:
  1. Cartographer 2D SLAM (RPLidar A3 + IMU + TF tree)
  2. Intel RealSense D435i (colour + aligned-depth)
  3. YOLOv8 + ArUco detector
  4. Object localizer (depth + TF → map coords)
  5. CSV logger (writes competition pois.csv)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # ------------------------------------------------------------------ #
    # Launch arguments                                                     #
    # ------------------------------------------------------------------ #
    team_name_arg = DeclareLaunchArgument(
        'team_name', default_value='BracU ALTER',
        description='Team name written in the CSV header')

    file_team_name_arg = DeclareLaunchArgument(
        'file_team_name', default_value='BracUAlter',
        description='Team token used in exported CSV/TIFF filenames')

    country_arg = DeclareLaunchArgument(
        'country', default_value='Bangladesh',
        description='Country used in exported CSV header')

    mission_arg = DeclareLaunchArgument(
        'mission', default_value='Prelim1',
        description='Mission name used in exported CSV/TIFF filenames')

    robot_arg = DeclareLaunchArgument(
        'robot', default_value='Alter',
        description='Robot name recorded in the CSV')

    mode_arg = DeclareLaunchArgument(
        'mode', default_value='T',
        description='Operation mode: A (autonomous) or T (teleoperated)')

    output_dir_arg = DeclareLaunchArgument(
        'output_dir', default_value='/home/sbuntu/rrl_bracu_alter_ws/exports',
        description='Directory for pois.csv output')

    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='/home/sbuntu/rrl_bracu_alter_ws/models/best.pt',
        description='Absolute path to the YOLOv8 .pt weights file')

    lidar_serial_port_arg = DeclareLaunchArgument(
        'lidar_serial_port', default_value='/dev/ttyUSB0',
        description='Serial port for RPLidar A3')

    imu_serial_port_arg = DeclareLaunchArgument(
        'imu_serial_port', default_value='/dev/ttyUSB1',
        description='Serial port for CMP10A IMU')

    # ------------------------------------------------------------------ #
    # 1. Cartographer SLAM (includes RPLidar, IMU, TF, robot path)        #
    # ------------------------------------------------------------------ #
    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('cartographer_slam'),
                'launch', 'cartographer_2d.launch.py'
            )
        ),
        launch_arguments={
            'lidar_serial_port': LaunchConfiguration('lidar_serial_port'),
            'imu_serial_port': LaunchConfiguration('imu_serial_port'),
            'launch_rviz': 'false',
        }.items(),
    )

    # ------------------------------------------------------------------ #
    # 2. Intel RealSense D435i                                            #
    # Publishes:                                                           #
    #   /camera/camera/color/image_raw                                   #
    #   /camera/camera/aligned_depth_to_color/image_raw                  #
    #   /camera/camera/color/camera_info                                 #
    # ------------------------------------------------------------------ #
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        parameters=[{
            'align_depth.enable': True,
            'base_frame_id': 'camera_link',
            'enable_color': True,
            'enable_depth': True,
            'enable_accel': False,
            'enable_gyro': False,
            'enable_motion': False,
            'pointcloud.enable': False,
            'publish_tf': True,
        }],
        output='screen',
    )

    # ------------------------------------------------------------------ #
    # 3. YOLOv8 + ArUco detector                                          #
    # Publishes: /yolo/detections (ObjectDetectionArray)                  #
    # ------------------------------------------------------------------ #
    yolo_node = Node(
        package='yolov8_detector',
        executable='yolo_node',
        name='yolov8_detector',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'image_topic': '/camera/camera/color/image_raw',
            'confidence': 0.35,
            'device': 'cpu',
            'publish_annotated': False,
            'enable_aruco': True,
        }],
        output='screen',
    )

    # ------------------------------------------------------------------ #
    # 4. Object localizer                                                  #
    # Subscribes: /yolo/detections, /camera/aligned_depth_to_color/image_raw
    # Publishes:  /localization/objects (ObjectLocalization)              #
    # ------------------------------------------------------------------ #
    localizer_node = Node(
        package='object_localizer',
        executable='object_localizer',
        name='object_localizer',
        parameters=[{
            'detections_topic': '/yolo/detections',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'map_frame': 'map',
            'camera_frame': '',   # use frame_id from depth image header
            'depth_window': 5,
            'depth_scale': 0.001,
            'min_depth': 0.2,
            'max_depth': 10.0,
        }],
        output='screen',
    )

    # ------------------------------------------------------------------ #
    # 5. CSV logger                                                        #
    # Subscribes: /localization/objects                                   #
    # Writes: exports/RoboCup<Year>-<team>-<mission>-<time>-pois.csv     #
    # ------------------------------------------------------------------ #
    csv_logger_node = Node(
        package='csv_logger',
        executable='csv_logger',
        name='csv_logger',
        parameters=[{
            'localizations_topic': '/localization/objects',
            'team_name': LaunchConfiguration('team_name'),
            'file_team_name': LaunchConfiguration('file_team_name'),
            'country': LaunchConfiguration('country'),
            'mission': LaunchConfiguration('mission'),
            'robot': LaunchConfiguration('robot'),
            'mode': LaunchConfiguration('mode'),
            'output_dir': LaunchConfiguration('output_dir'),
            'min_distance': 0.3,
        }],
        output='screen',
    )

    return LaunchDescription([
        team_name_arg,
        file_team_name_arg,
        country_arg,
        mission_arg,
        robot_arg,
        mode_arg,
        output_dir_arg,
        model_path_arg,
        lidar_serial_port_arg,
        imu_serial_port_arg,
        cartographer_launch,
        realsense_node,
        yolo_node,
        localizer_node,
        csv_logger_node,
    ])
