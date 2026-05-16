#!/usr/bin/env python3
"""Publish the robot path in map coordinates for final RRL map export."""

import math

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf2_ros


class PathTracker(Node):
    def __init__(self):
        super().__init__('path_tracker')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_topic', '/robot_path')
        self.declare_parameter('sample_period', 0.5)
        self.declare_parameter('min_distance', 0.03)

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        sample_period = self.get_parameter('sample_period').get_parameter_value().double_value
        self.min_distance = self.get_parameter('min_distance').get_parameter_value().double_value

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.path_pub = self.create_publisher(Path, publish_topic, qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.path = Path()
        self.path.header.frame_id = self.map_frame

        self.timer = self.create_timer(sample_period, self.sample_pose)
        self.get_logger().info(f'Publishing robot path on {publish_topic}')

    def sample_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        if self.path.poses:
            last = self.path.poses[-1].pose.position
            if math.hypot(x - last.x, y - last.y) < self.min_distance:
                return

        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)


def main():
    rclpy.init()
    node = PathTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
