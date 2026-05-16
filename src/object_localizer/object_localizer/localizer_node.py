from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs

from rrl_interfaces.msg import ObjectDetectionArray, ObjectLocalization


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


class ObjectLocalizer(Node):
    def __init__(self) -> None:
        super().__init__("object_localizer")

        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("depth_window", 5)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth", 0.2)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("marker_topic", "/localization/object_markers")
        self.declare_parameter("marker_min_distance", 0.3)

        detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.depth_window = self.get_parameter("depth_window").get_parameter_value().integer_value
        self.depth_scale = self.get_parameter("depth_scale").get_parameter_value().double_value
        self.min_depth = self.get_parameter("min_depth").get_parameter_value().double_value
        self.max_depth = self.get_parameter("max_depth").get_parameter_value().double_value
        marker_topic = self.get_parameter("marker_topic").get_parameter_value().string_value
        self.marker_min_distance = self.get_parameter("marker_min_distance").get_parameter_value().double_value

        self.bridge = CvBridge()
        self.last_depth: Optional[Image] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.marker_records = []
        self.markers = []
        self.next_marker_id = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, camera_info_topic, self.on_camera_info, 10)
        self.create_subscription(ObjectDetectionArray, detections_topic, self.on_detections, 10)

        self.pub = self.create_publisher(ObjectLocalization, "localization/objects", 10)
        marker_qos = QoSProfile(depth=1)
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, marker_qos)

        self.get_logger().info(f"Listening for detections on: {detections_topic}")
        self.get_logger().info(f"Depth topic: {depth_topic}")
        self.get_logger().info(f"Camera info topic: {camera_info_topic}")
        self.get_logger().info(f"Publishing object markers on: {marker_topic}")

    def on_depth(self, msg: Image) -> None:
        self.last_depth = msg

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.intrinsics = CameraIntrinsics(
            fx=msg.k[0],
            fy=msg.k[4],
            cx=msg.k[2],
            cy=msg.k[5],
        )

    def on_detections(self, msg: ObjectDetectionArray) -> None:
        if self.last_depth is None or self.intrinsics is None:
            self.get_logger().warn("Waiting for depth image and camera info.")
            return

        depth_img = self.bridge.imgmsg_to_cv2(self.last_depth, desired_encoding="passthrough")
        intr = self.intrinsics

        for det in msg.detections:
            cx = int((det.xmin + det.xmax) / 2)
            cy = int((det.ymin + det.ymax) / 2)
            depth = self._depth_at(depth_img, cx, cy, self.last_depth.encoding)
            if depth is None:
                continue

            x = (cx - intr.cx) * depth / intr.fx
            y = (cy - intr.cy) * depth / intr.fy
            z = depth

            point = PointStamped()
            point.header = self.last_depth.header
            if self.camera_frame:
                point.header.frame_id = self.camera_frame
            point.point.x = float(x)
            point.point.y = float(y)
            point.point.z = float(z)

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    point.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
                map_point = tf2_geometry_msgs.do_transform_point(point, transform)
            except Exception as exc:
                self.get_logger().warn(f"TF transform failed: {exc}")
                continue

            loc = ObjectLocalization()
            loc.header = map_point.header
            loc.obj_type = det.obj_type
            loc.name = det.name
            loc.confidence = det.confidence
            loc.x = map_point.point.x
            loc.y = map_point.point.y
            loc.z = map_point.point.z
            self.pub.publish(loc)
            self._publish_marker_if_new(loc)

    def _depth_at(self, depth_img, cx: int, cy: int, encoding: str) -> Optional[float]:
        if depth_img is None:
            return None

        h, w = depth_img.shape[:2]
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return None

        window = max(1, int(self.depth_window))
        half = window // 2
        x0 = max(cx - half, 0)
        x1 = min(cx + half + 1, w)
        y0 = max(cy - half, 0)
        y1 = min(cy + half + 1, h)
        roi = depth_img[y0:y1, x0:x1].astype(np.float32)

        if encoding in ("16UC1", "mono16"):
            roi = roi * float(self.depth_scale)

        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size == 0:
            return None

        depth = float(np.median(valid))
        if depth < self.min_depth or depth > self.max_depth:
            return None
        return depth

    def _publish_marker_if_new(self, loc: ObjectLocalization) -> None:
        if not self._is_new_marker(loc):
            return

        marker_id = self.next_marker_id
        self.next_marker_id += 1
        self.marker_records.append((loc.obj_type, loc.name, loc.x, loc.y, loc.z))

        object_marker = Marker()
        object_marker.header = loc.header
        object_marker.ns = "rrl_objects"
        object_marker.id = marker_id * 2
        object_marker.action = Marker.ADD
        object_marker.pose.position.x = float(loc.x)
        object_marker.pose.position.y = float(loc.y)
        object_marker.pose.position.z = float(loc.z)
        object_marker.pose.orientation.w = 1.0
        object_marker.scale.x = 0.25
        object_marker.scale.y = 0.25
        object_marker.scale.z = 0.25
        object_marker.color.a = 1.0

        if loc.obj_type == "ar_code":
            object_marker.type = Marker.SPHERE
            object_marker.color.r = 1.0
            object_marker.color.g = 0.78
            object_marker.color.b = 0.0
        elif loc.obj_type == "hazmat_sign":
            object_marker.type = Marker.CUBE
            object_marker.color.r = 1.0
            object_marker.color.g = 0.35
            object_marker.color.b = 0.0
        else:
            object_marker.type = Marker.SPHERE
            object_marker.color.r = 1.0
            object_marker.color.g = 0.0
            object_marker.color.b = 0.0

        text_marker = Marker()
        text_marker.header = loc.header
        text_marker.ns = "rrl_object_labels"
        text_marker.id = marker_id * 2 + 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = float(loc.x)
        text_marker.pose.position.y = float(loc.y)
        text_marker.pose.position.z = float(loc.z) + 0.35
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.22
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"{loc.obj_type}:{loc.name}"

        self.markers.extend([object_marker, text_marker])
        self.marker_pub.publish(MarkerArray(markers=self.markers))

    def _is_new_marker(self, loc: ObjectLocalization) -> bool:
        for obj_type, name, x, y, z in self.marker_records:
            if obj_type != loc.obj_type or name != loc.name:
                continue
            distance = np.linalg.norm([loc.x - x, loc.y - y, loc.z - z])
            if distance < self.marker_min_distance:
                return False
        return True


def main() -> None:
    rclpy.init()
    node = ObjectLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
