#!/usr/bin/env python3
"""
RoboCup Rescue League 2026 - Full Object Detection System
==========================================================
Detects ALL required object types:
  1. hazmat_sign  - Using custom YOLOv8 hazmat model
  2. ar_code      - Using OpenCV ArUco detector
  3. real_object  - Using YOLOv8 COCO model (80 common objects)
  4. heat_sig     - Placeholder (requires thermal camera)

Outputs CSV in official RoboCup 2026 format.

Hardware: Raspberry Pi 4B + USB Webcam (or laptop camera)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import time
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Force CPU to avoid CUDA compatibility issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ============================================================================
# DETECTION DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    """Represents a unique detected object."""
    detection_id: int
    timestamp: str
    obj_type: str  # "hazmat_sign", "ar_code", "real_object", "heat_sig"
    name: str      # e.g., "poison", "42", "gloves"
    x: float       # meters
    y: float       # meters
    z: float       # meters
    robot: str
    mode: str      # "A" or "T"


@dataclass
class DetectionManager:
    """Manages detections and prevents duplicates."""
    detections: List[Detection] = field(default_factory=list)
    detection_counter: int = 0
    min_distance_threshold: float = 0.3  # meters
    
    def is_duplicate(self, x: float, y: float, z: float, name: str, obj_type: str) -> bool:
        """Check if detection is too close to an existing one of same type and name."""
        for det in self.detections:
            if det.name == name and det.obj_type == obj_type:
                dist = np.sqrt((det.x - x)**2 + (det.y - y)**2 + (det.z - z)**2)
                if dist < self.min_distance_threshold:
                    return True
        return False
    
    def add_detection(self, obj_type: str, name: str, x: float, y: float, z: float,
                      robot: str = "robot1", mode: str = "T") -> Optional[Detection]:
        """Add a new detection if not duplicate."""
        if self.is_duplicate(x, y, z, name, obj_type):
            return None
        
        self.detection_counter += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        detection = Detection(
            detection_id=self.detection_counter,
            timestamp=timestamp,
            obj_type=obj_type,
            name=name,
            x=round(x, 4),
            y=round(y, 4),
            z=round(z, 4),
            robot=robot,
            mode=mode
        )
        self.detections.append(detection)
        print(f"[NEW #{detection.detection_id}] {obj_type}: {name} at ({x:.2f}, {y:.2f}, {z:.2f})m")
        return detection
    
    def save_csv(self, filepath: str, team_name: str, country: str, mission: str):
        """Save detections to CSV in RoboCup 2026 format."""
        now = datetime.now()
        
        with open(filepath, 'w', newline='') as f:
            # Header section
            f.write('"pois"\n')
            f.write('"1.3"\n')
            f.write(f'"{team_name}"\n')
            f.write(f'"{country}"\n')
            f.write(f'"{now.strftime("%Y-%m-%d")}"\n')
            f.write(f'"{now.strftime("%H:%M:%S")}"\n')
            f.write(f'"{mission}"\n')
            
            # Column headers
            f.write("detection,time,type,name,x,y,z,robot,mode\n")
            
            # Detection rows
            for det in self.detections:
                f.write(f'{det.detection_id},{det.timestamp},"{det.obj_type}","{det.name}",'
                        f'{det.x},{det.y},{det.z},"{det.robot}",{det.mode}\n')
        
        print(f"\n✅ Saved {len(self.detections)} detections to: {filepath}")
    
    def get_summary(self) -> Dict[str, int]:
        """Get count of detections by type."""
        summary = {"hazmat_sign": 0, "ar_code": 0, "real_object": 0, "heat_sig": 0}
        for det in self.detections:
            if det.obj_type in summary:
                summary[det.obj_type] += 1
        return summary


# ============================================================================
# CAMERA HANDLER
# ============================================================================

class WebcamCamera:
    """Handles standard webcam streaming."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        
        # Camera intrinsics (approximate)
        self.fx = 600
        self.fy = 600
        self.cx = width / 2
        self.cy = height / 2
        
    def start(self):
        """Initialize and start the webcam."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.width = actual_w
        self.height = actual_h
        self.cx = actual_w / 2
        self.cy = actual_h / 2
        
        for _ in range(10):
            self.cap.read()
        
        print(f"✅ Webcam started: {actual_w}x{actual_h}")
        
    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def estimate_3d_position(self, cx: int, cy: int, bbox_size: float, 
                              real_size: float = 0.25) -> tuple:
        """Estimate 3D position from bounding box size."""
        if bbox_size > 0:
            z = (real_size * self.fx) / bbox_size
        else:
            z = 1.0
        
        z = max(0.2, min(z, 10.0))
        x = (cx - self.cx) * z / self.fx
        y = (cy - self.cy) * z / self.fy
        
        return (x, y, z)
    
    def stop(self):
        if self.cap:
            self.cap.release()
            print("🛑 Webcam stopped")


# ============================================================================
# ARUCO DETECTOR
# ============================================================================

class ArUcoDetector:
    """Detects ArUco markers using OpenCV."""
    
    def __init__(self):
        # Try different ArUco dictionaries (competition may use any)
        self.aruco_dicts = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        }
        
        # Use 4x4_50 as default (common in competitions)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Approximate marker size (meters) - adjust based on competition
        self.marker_size = 0.10  # 10cm
        
        print("✅ ArUco detector initialized (DICT_4X4_50)")
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect ArUco markers in frame.
        Returns list of {id, corners, center, size}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        detections = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i][0]
                
                # Calculate center
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))
                
                # Calculate size (average of width and height)
                width = np.linalg.norm(corner[0] - corner[1])
                height = np.linalg.norm(corner[1] - corner[2])
                size = (width + height) / 2
                
                detections.append({
                    "id": int(marker_id),
                    "corners": corner,
                    "center": (center_x, center_y),
                    "size": size
                })
        
        return detections
    
    def draw(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detected markers on frame."""
        for det in detections:
            corners = det["corners"].astype(int)
            
            # Draw polygon
            cv2.polylines(frame, [corners], True, (255, 0, 255), 2)
            
            # Draw ID
            cx, cy = det["center"]
            cv2.putText(frame, f"AR:{det['id']}", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return frame


# ============================================================================
# MAIN DETECTOR
# ============================================================================

class FullObjectDetector:
    """Main detection system for all RoboCup object types."""
    
    # COCO classes that are relevant as "real_object" in rescue scenarios
    RELEVANT_COCO_CLASSES = {
        0: "person",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        39: "bottle",
        41: "cup",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        56: "chair",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        73: "book",
        74: "clock",
        76: "scissors",
        77: "teddy bear",
    }
    
    def __init__(self, hazmat_model_path: str, camera_id: int = 0,
                 team_name: str = "MyTeam", country: str = "MyCountry",
                 mission: str = "Prelim1", robot_name: str = "robot1",
                 conf_threshold: float = 0.3, enable_coco: bool = True):
        
        self.team_name = team_name
        self.country = country
        self.mission = mission
        self.robot_name = robot_name
        self.conf_threshold = conf_threshold
        self.enable_coco = enable_coco
        
        print("🚀 Initializing RoboCup Full Object Detection System...")
        print("="*60)
        
        # Load hazmat model
        print("   [1/4] Loading YOLOv8 hazmat model...")
        self.hazmat_model = YOLO(hazmat_model_path)
        self.hazmat_classes = self.hazmat_model.names
        print(f"         Hazmat classes: {len(self.hazmat_classes)}")
        
        # Load COCO model for real objects
        if self.enable_coco:
            print("   [2/4] Loading YOLOv8 COCO model (for real objects)...")
            self.coco_model = YOLO("yolov8n.pt")  # Nano model for speed
            print(f"         COCO classes: 80 (filtering to {len(self.RELEVANT_COCO_CLASSES)} relevant)")
        else:
            self.coco_model = None
            print("   [2/4] COCO model disabled")
        
        # Initialize ArUco detector
        print("   [3/4] Initializing ArUco detector...")
        self.aruco_detector = ArUcoDetector()
        
        # Start camera
        print("   [4/4] Starting webcam...")
        self.camera = WebcamCamera(camera_id=camera_id, width=640, height=480)
        self.camera.start()
        
        self.detection_manager = DetectionManager()
        self.running = False
        self.start_time = datetime.now()
        
        print("="*60)
        print("✅ All detectors ready!\n")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with all detectors."""
        annotated = frame.copy()
        
        # 1. Detect hazmat signs
        hazmat_results = self.hazmat_model(frame, conf=self.conf_threshold, 
                                            verbose=False, device='cpu')
        annotated = self._process_yolo_results(
            annotated, hazmat_results, self.hazmat_classes,
            obj_type="hazmat_sign", color=(0, 255, 0), real_size=0.25
        )
        
        # 2. Detect real objects (COCO)
        if self.coco_model is not None:
            coco_results = self.coco_model(frame, conf=self.conf_threshold,
                                           verbose=False, device='cpu')
            annotated = self._process_yolo_results(
                annotated, coco_results, self.coco_model.names,
                obj_type="real_object", color=(255, 165, 0), real_size=0.15,
                filter_classes=self.RELEVANT_COCO_CLASSES
            )
        
        # 3. Detect ArUco markers
        aruco_detections = self.aruco_detector.detect(frame)
        for det in aruco_detections:
            cx, cy = det["center"]
            x, y, z = self.camera.estimate_3d_position(cx, cy, det["size"], 
                                                        real_size=0.10)
            
            self.detection_manager.add_detection(
                obj_type="ar_code",
                name=str(det["id"]),
                x=x, y=y, z=z,
                robot=self.robot_name,
                mode="T"
            )
        annotated = self.aruco_detector.draw(annotated, aruco_detections)
        
        # Draw status overlay
        summary = self.detection_manager.get_summary()
        status1 = f"Total: {len(self.detection_manager.detections)} | "
        status1 += f"Hazmat:{summary['hazmat_sign']} AR:{summary['ar_code']} Obj:{summary['real_object']}"
        status2 = "'q'=quit 's'=save 'c'=clear"
        
        cv2.putText(annotated, status1, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated, status2, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return annotated
    
    def _process_yolo_results(self, frame: np.ndarray, results, class_names: dict,
                               obj_type: str, color: tuple, real_size: float,
                               filter_classes: dict = None) -> np.ndarray:
        """Process YOLO results and draw on frame."""
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                cls_id = int(box.cls[0])
                
                # Filter to relevant classes if specified
                if filter_classes is not None and cls_id not in filter_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = class_names[cls_id]
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bbox_size = ((x2 - x1) + (y2 - y1)) / 2
                
                x, y, z = self.camera.estimate_3d_position(cx, cy, bbox_size, real_size)
                
                self.detection_manager.add_detection(
                    obj_type=obj_type,
                    name=cls_name,
                    x=x, y=y, z=z,
                    robot=self.robot_name,
                    mode="T"
                )
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        return frame
    
    def run(self):
        """Main detection loop."""
        self.running = True
        print("="*60)
        print("🔍 FULL OBJECT DETECTION RUNNING")
        print("   Detecting: hazmat_sign, ar_code, real_object")
        print("   Press 'q' to quit and save CSV")
        print("   Press 's' to save CSV without quitting")
        print("   Press 'c' to clear all detections")
        print("="*60 + "\n")
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while self.running:
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                annotated = self.process_frame(frame)
                
                # FPS display
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    fps = frame_count / elapsed
                    cv2.putText(annotated, f"FPS: {fps:.1f}", 
                                (annotated.shape[1] - 80, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow("RoboCup Full Detector", annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Quitting...")
                    break
                elif key == ord('s'):
                    self.save_results()
                elif key == ord('c'):
                    print("\n🗑️  Clearing all detections...")
                    self.detection_manager.detections.clear()
                    self.detection_manager.detection_counter = 0
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        finally:
            self.cleanup()
    
    def save_results(self):
        """Save detection results to CSV."""
        start_time_str = self.start_time.strftime("%H-%M-%S")
        year = self.start_time.year
        filename = f"RoboCup{year}-{self.team_name}-{self.mission}-{start_time_str}-pois.csv"
        
        output_dir = Path(__file__).parent / "detection_results"
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename
        
        self.detection_manager.save_csv(
            filepath=str(filepath),
            team_name=self.team_name,
            country=self.country,
            mission=self.mission
        )
        
        # Print summary
        summary = self.detection_manager.get_summary()
        print(f"   Summary: Hazmat={summary['hazmat_sign']}, AR={summary['ar_code']}, "
              f"Objects={summary['real_object']}, Heat={summary['heat_sig']}")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.save_results()
        self.camera.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="RoboCup Rescue Full Object Detector")
    parser.add_argument("--hazmat-model", type=str,
                        default="/home/sbuntu/Downloads/hazmat_50e_export/runs/weights/best.pt",
                        help="Path to YOLOv8 hazmat model")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID")
    parser.add_argument("--team", type=str, default="MyTeam",
                        help="Team name")
    parser.add_argument("--country", type=str, default="MyCountry",
                        help="Country")
    parser.add_argument("--mission", type=str, default="Prelim1",
                        help="Mission name")
    parser.add_argument("--robot", type=str, default="robot1",
                        help="Robot name")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold")
    parser.add_argument("--no-coco", action="store_true",
                        help="Disable COCO object detection (faster)")
    
    args = parser.parse_args()
    
    detector = FullObjectDetector(
        hazmat_model_path=args.hazmat_model,
        camera_id=args.camera,
        team_name=args.team,
        country=args.country,
        mission=args.mission,
        robot_name=args.robot,
        conf_threshold=args.conf,
        enable_coco=not args.no_coco
    )
    
    detector.run()


if __name__ == "__main__":
    main()
