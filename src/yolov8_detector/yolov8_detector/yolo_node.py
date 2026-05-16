import os
import sys
from typing import Set

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rrl_interfaces.msg import ObjectDetection, ObjectDetectionArray

# Force CPU by default unless overridden by parameter.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HAZMAT_CLASSES: Set[str] = {
    "blasting_agents",
    "corrosive",
    "dangerous_when_wet",
    "explosives",
    "flammable_gas",
    "flammable_solid",
    "fuel_oil",
    "inhalation_hazard",
    "non_flammable_gas",
    "organic_peroxide",
    "oxidizer",
    "oxygen",
    "poison",
    "radioactive",
    "spontaneously_combustible",
}

REAL_OBJECT_CLASSES: Set[str] = {
    "Backpack",
    "baby_doll_face",
    "fire_extinguisher",
    "hard_hat",
    "propen_tank",
}


class Yolov8DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("yolov8_detector")

        self.declare_parameter("model_path", "/home/sbuntu/rrl_bracu_alter_ws/models/best.pt")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("publish_annotated", False)
        self.declare_parameter("annotated_topic", "/yolo/annotated")
        self.declare_parameter("enable_aruco", True)
        self.declare_parameter(
            "yolo_site_packages",
            "/home/sbuntu/yoloenv/lib/python3.12/site-packages",
        )

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.confidence = self.get_parameter("confidence").get_parameter_value().double_value
        device = self.get_parameter("device").get_parameter_value().string_value
        self.publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value
        annotated_topic = self.get_parameter("annotated_topic").get_parameter_value().string_value
        self.enable_aruco = self.get_parameter("enable_aruco").get_parameter_value().bool_value
        yolo_site_packages = self.get_parameter("yolo_site_packages").get_parameter_value().string_value

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            if yolo_site_packages and os.path.isdir(yolo_site_packages):
                sys.path.insert(0, yolo_site_packages)
                self.get_logger().warn(
                    f"ultralytics not found in ROS Python; trying fallback: {yolo_site_packages}"
                )
                try:
                    from ultralytics import YOLO
                except ImportError:
                    self.get_logger().error(
                        "ultralytics is not available in ROS Python or the configured yolo_site_packages."
                    )
                    raise exc
            else:
                self.get_logger().error(
                    "ultralytics is not installed and yolo_site_packages does not exist."
                )
                raise exc

        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.device = device if device else None
        self.bridge = CvBridge()

        self.detections_pub = self.create_publisher(ObjectDetectionArray, "yolo/detections", 10)
        self.annotated_pub = None
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(Image, annotated_topic, 10)

        self.sub = self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)

        self.aruco_detector = None
        self.aruco_dictionary = None
        self.aruco_parameters = None
        if self.enable_aruco:
            if hasattr(__import__("cv2"), "aruco"):
                import cv2

                self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                if hasattr(cv2.aruco, "DetectorParameters"):
                    self.aruco_parameters = cv2.aruco.DetectorParameters()
                else:
                    self.aruco_parameters = cv2.aruco.DetectorParameters_create()

                if hasattr(cv2.aruco, "ArucoDetector"):
                    self.aruco_detector = cv2.aruco.ArucoDetector(
                        self.aruco_dictionary,
                        self.aruco_parameters,
                    )
            else:
                self.get_logger().warn("OpenCV aruco module not available; disabling ArUco detection.")
                self.enable_aruco = False

        self.get_logger().info(f"YOLOv8 model loaded: {model_path}")
        self.get_logger().info(f"Subscribed to: {image_topic}")

    def _obj_type(self, class_name: str) -> str:
        if class_name in HAZMAT_CLASSES:
            return "hazmat_sign"
        if class_name in REAL_OBJECT_CLASSES:
            return "real_object"
        return "real_object"

    def on_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # pragma: no cover - cv_bridge raises generic exceptions
            self.get_logger().warn(f"Failed to convert image: {exc}")
            return

        results = self.model(frame, conf=self.confidence, verbose=False, device=self.device)

        detections_msg = ObjectDetectionArray()
        detections_msg.header = msg.header

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                det = ObjectDetection()
                det.header = msg.header
                det.obj_type = self._obj_type(class_name)
                det.name = class_name
                det.confidence = confidence
                det.xmin = x1
                det.ymin = y1
                det.xmax = x2
                det.ymax = y2
                detections_msg.detections.append(det)

        if self.enable_aruco and self.aruco_detector is not None:
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.aruco_detector is not None:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    self.aruco_dictionary,
                    parameters=self.aruco_parameters,
                )
            if ids is not None:
                for marker_id, corner in zip(ids.flatten(), corners):
                    pts = corner[0]
                    x_vals = pts[:, 0]
                    y_vals = pts[:, 1]
                    det = ObjectDetection()
                    det.header = msg.header
                    det.obj_type = "ar_code"
                    det.name = str(int(marker_id))
                    det.confidence = 1.0
                    det.xmin = int(x_vals.min())
                    det.ymin = int(y_vals.min())
                    det.xmax = int(x_vals.max())
                    det.ymax = int(y_vals.max())
                    detections_msg.detections.append(det)

        if detections_msg.detections:
            self.detections_pub.publish(detections_msg)

        if self.annotated_pub is not None:
            annotated = results[0].plot()
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)


def main() -> None:
    rclpy.init()
    node = Yolov8DetectorNode()
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
