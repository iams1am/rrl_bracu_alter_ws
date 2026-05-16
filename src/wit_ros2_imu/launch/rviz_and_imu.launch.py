from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Static transform from world to imu_link for RViz visualization
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_to_imu',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'imu_link'],
        output='screen'
    )

    # IMU driver node
    rviz_and_imu_node = Node(
        package='wit_ros2_imu',
        executable='wit_ros2_imu',
        name='imu',
        remappings=[('/wit/imu', '/imu/data')],
        parameters=[{'port': '/dev/imu_usb'},
                    {"baud": 115200}],
        output="screen"
    )

    # RViz display node
    rviz_display_node = Node(
        package='rviz2',
        executable="rviz2",
        name='rviz2',
        output="screen"
    )

    return LaunchDescription(
        [
            static_tf_node,
            rviz_and_imu_node,
            rviz_display_node
        ]
    )