from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import re

import rclpy
from rclpy.node import Node

from rrl_interfaces.msg import ObjectLocalization


@dataclass
class DetectionRecord:
    detection_id: int
    timestamp: str
    obj_type: str
    name: str
    x: float
    y: float
    z: float
    robot: str
    mode: str


@dataclass
class DetectionStore:
    min_distance_threshold: float
    detections: List[DetectionRecord] = field(default_factory=list)
    detection_counter: int = 0

    def is_duplicate(self, record: DetectionRecord) -> bool:
        for det in self.detections:
            if det.obj_type == record.obj_type and det.name == record.name:
                return True
        return False

    def add(self, record: DetectionRecord) -> Optional[DetectionRecord]:
        if self.is_duplicate(record):
            return None
        self.detections.append(record)
        return record


class CsvLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("csv_logger")

        self.declare_parameter("localizations_topic", "/localization/objects")
        self.declare_parameter("team_name", "MyTeam")
        self.declare_parameter("file_team_name", "")
        self.declare_parameter("country", "MyCountry")
        self.declare_parameter("mission", "Prelim1")
        self.declare_parameter("robot", "robot1")
        self.declare_parameter("mode", "T")
        self.declare_parameter("output_dir", "/home/sbuntu/rrl_bracu_alter_ws/exports")
        self.declare_parameter("min_distance", 0.3)

        topic = self.get_parameter("localizations_topic").get_parameter_value().string_value
        self.team_name = self.get_parameter("team_name").get_parameter_value().string_value
        self.file_team_name = self.get_parameter("file_team_name").get_parameter_value().string_value
        self.country = self.get_parameter("country").get_parameter_value().string_value
        self.mission = self.get_parameter("mission").get_parameter_value().string_value
        self.robot = self.get_parameter("robot").get_parameter_value().string_value
        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        min_distance = self.get_parameter("min_distance").get_parameter_value().double_value

        self.start_datetime = datetime.now()
        self.start_date = self.start_datetime.strftime("%Y-%m-%d")
        self.start_time = self.start_datetime.strftime("%H:%M:%S")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.store = DetectionStore(min_distance_threshold=min_distance)

        self.create_subscription(ObjectLocalization, topic, self.on_localization, 10)
        self.get_logger().info(f"Logging localizations from: {topic}")
        self.save_csv()

        rclpy.get_default_context().on_shutdown(self.save_csv)

    def on_localization(self, msg: ObjectLocalization) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")

        record = DetectionRecord(
            detection_id=self.store.detection_counter + 1,
            timestamp=timestamp,
            obj_type=msg.obj_type,
            name=msg.name,
            x=round(float(msg.x), 4),
            y=round(float(msg.y), 4),
            z=round(float(msg.z), 4),
            robot=self.robot,
            mode=self.mode,
        )

        added = self.store.add(record)
        if added is None:
            return

        self.store.detection_counter += 1
        self.get_logger().info(
            f"[NEW #{record.detection_id}] {record.obj_type}: {record.name} at "
            f"({record.x:.2f}, {record.y:.2f}, {record.z:.2f})"
        )
        self.save_csv()

    def save_csv(self) -> None:
        year = self.start_datetime.year
        start_time_str = self.start_datetime.strftime("%H-%M-%S")
        team_token = self.file_team_name.strip() or self.team_name
        team_token = re.sub(r"[^A-Za-z0-9_-]+", "", team_token)
        filename = f"RoboCup{year}-{team_token}-{self.mission}-{start_time_str}-pois.csv"
        filepath = self.output_dir / filename

        with open(filepath, "w", newline="") as f:
            f.write("\"pois\"\n")
            f.write("\"1.3\"\n")
            f.write(f"\"{self.team_name}\"\n")
            f.write(f"\"{self.country}\"\n")
            f.write(f"\"{self.start_date}\"\n")
            f.write(f"\"{self.start_time}\"\n")
            f.write(f"\"{self.mission}\"\n")
            f.write("detection,time,type,name,x,y,z,robot,mode\n")

            for det in self.store.detections:
                f.write(
                    f"{det.detection_id},{det.timestamp},\"{det.obj_type}\","
                    f"\"{det.name}\",{det.x},{det.y},{det.z},\"{det.robot}\",{det.mode}\n"
                )


def main() -> None:
    rclpy.init()
    node = CsvLoggerNode()
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
