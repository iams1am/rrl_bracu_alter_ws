#!/usr/bin/env python3
"""
RoboCup Rescue League 2026 - Hazmat Detection System
=====================================================
Detects hazmat signs using YOLOv8 + standard webcam.
Outputs CSV in official RoboCup 2026 format.

Hardware: Raspberry Pi 4B + USB Webcam (or laptop camera)
Note: Without depth camera, Z coordinate is estimated from bbox size.
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
from typing import List, Optional

# Force CPU to avoid CUDA compatibility issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@dataclass
class Detection:
    """Represents a unique detected object."""
    detection_id: int
    timestamp: str
    obj_type: str  # "hazmat_sign", "ar_code", "real_object", "heat_sig"
    name: str      # e.g., "poison", "radioactive"
    x: float       # meters (estimated)
    y: float       # meters (estimated)
    z: float       # meters (estimated from bbox size)
    robot: str
    mode: str      # "A" or "T"


@dataclass
class DetectionManager:
    """Manages detections and prevents duplicates."""
    detections: List[Detection] = field(default_factory=list)
    detection_counter: int = 0
    min_distance_threshold: float = 0.3  # meters (tighter for estimated coords)
    
    def is_duplicate(self, x: float, y: float, z: float, name: str) -> bool:
        """Check if detection is too close to an existing one of same type."""
        for det in self.detections:
            if det.name == name:
                dist = np.sqrt((det.x - x)**2 + (det.y - y)**2 + (det.z - z)**2)
                if dist < self.min_distance_threshold:
                    return True
        return False
    
    def add_detection(self, obj_type: str, name: str, x: float, y: float, z: float,
                      robot: str = "robot1", mode: str = "T") -> Optional[Detection]:
        """Add a new detection if not duplicate. Returns detection if added."""
        if self.is_duplicate(x, y, z, name):
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
        print(f"[NEW] #{detection.detection_id} {name} at ({x:.2f}, {y:.2f}, {z:.2f})m")
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


class WebcamCamera:
    """Handles standard webcam streaming."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        
        # Camera intrinsics (approximate for standard webcam)
        # These are used to estimate 3D position from 2D bbox
        self.fx = 600  # focal length x (pixels) - approximate
        self.fy = 600  # focal length y (pixels) - approximate
        self.cx = width / 2   # principal point x
        self.cy = height / 2  # principal point y
        
        # Approximate real-world size of hazmat signs (meters)
        # Standard hazmat placard is about 0.25m x 0.25m (10 inches)
        self.hazmat_real_size = 0.25
        
    def start(self):
        """Initialize and start the webcam."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Get actual resolution (may differ from requested)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.width = actual_w
        self.height = actual_h
        self.cx = actual_w / 2
        self.cy = actual_h / 2
        
        # Warm up camera
        for _ in range(10):
            self.cap.read()
        
        print(f"✅ Webcam started: {actual_w}x{actual_h}")
        
    def get_frame(self):
        """Get a frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def estimate_3d_position(self, cx: int, cy: int, bbox_width: int, bbox_height: int) -> tuple:
        """
        Estimate 3D position from bounding box.
        Uses the known real-world size of hazmat signs to estimate depth.
        
        Returns (x, y, z) in meters (camera frame).
        """
        # Use average of bbox width and height for size estimation
        bbox_size = (bbox_width + bbox_height) / 2
        
        # Estimate depth (Z) from bbox size using similar triangles:
        # real_size / depth = bbox_size / focal_length
        # depth = real_size * focal_length / bbox_size
        if bbox_size > 0:
            z = (self.hazmat_real_size * self.fx) / bbox_size
        else:
            z = 1.0  # default 1 meter
        
        # Clamp depth to reasonable range
        z = max(0.2, min(z, 10.0))
        
        # Estimate X and Y from pixel position:
        # x = (px - cx) * z / fx
        # y = (py - cy) * z / fy
        x = (cx - self.cx) * z / self.fx
        y = (cy - self.cy) * z / self.fy
        
        return (x, y, z)
    
    def stop(self):
        """Stop the webcam."""
        if self.cap:
            self.cap.release()
            print("🛑 Webcam stopped")


class HazmatDetector:
    """Main detection system combining YOLOv8 + Webcam."""
    
    def __init__(self, model_path: str, camera_id: int = 0,
                 team_name: str = "MyTeam", country: str = "MyCountry", 
                 mission: str = "Prelim1", robot_name: str = "robot1", 
                 conf_threshold: float = 0.3):
        
        self.team_name = team_name
        self.country = country
        self.mission = mission
        self.robot_name = robot_name
        self.conf_threshold = conf_threshold
        
        # Initialize components
        print("🚀 Initializing Hazmat Detection System (CPU Mode)...")
        
        print("   Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"   Classes: {list(self.class_names.values())}")
        
        print("   Starting webcam...")
        self.camera = WebcamCamera(camera_id=camera_id, width=640, height=480)
        self.camera.start()
        
        self.detection_manager = DetectionManager()
        self.running = False
        
        # Start time for session
        self.start_time = datetime.now()
        
    def process_frame(self, color_image: np.ndarray) -> np.ndarray:
        """Run detection on a frame and return annotated image."""
        
        # Run YOLOv8 inference on CPU
        results = self.model(color_image, conf=self.conf_threshold, verbose=False, device='cpu')
        
        annotated_frame = color_image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                
                # Get center and size of bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                
                # Estimate 3D coordinates
                x, y, z = self.camera.estimate_3d_position(cx, cy, bbox_w, bbox_h)
                
                # Try to add as new detection (rejected if duplicate)
                self.detection_manager.add_detection(
                    obj_type="hazmat_sign",
                    name=cls_name,
                    x=x, y=y, z=z,
                    robot=self.robot_name,
                    mode="T"
                )
                
                # Draw bounding box (green)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with estimated 3D coordinates
                label = f"{cls_name} ({conf:.2f})"
                coords = f"~({x:.2f}, {y:.2f}, {z:.2f})m"
                
                cv2.putText(annotated_frame, label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_frame, coords, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw status overlay
        status = f"Detections: {len(self.detection_manager.detections)} | 'q'=quit 's'=save 'c'=clear"
        cv2.putText(annotated_frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Note about estimated coordinates
        cv2.putText(annotated_frame, "* Coords estimated (no depth sensor)", (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return annotated_frame
    
    def run(self):
        """Main detection loop."""
        self.running = True
        print("\n" + "="*60)
        print("🔍 HAZMAT DETECTION RUNNING (Webcam + CPU Mode)")
        print("   Press 'q' to quit and save CSV")
        print("   Press 's' to save CSV without quitting")
        print("   Press 'c' to clear all detections")
        print("="*60 + "\n")
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while self.running:
                # Get frame
                color_image = self.camera.get_frame()
                if color_image is None:
                    continue
                
                # Process frame
                annotated = self.process_frame(color_image)
                
                # Calculate and display FPS
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    fps = frame_count / elapsed
                    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Show frame
                cv2.imshow("RoboCup Hazmat Detector", annotated)
                
                # Handle key presses
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
            print("\n🛑 Interrupted by user")
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
        
    def cleanup(self):
        """Clean up resources and save final results."""
        self.running = False
        self.save_results()
        self.camera.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="RoboCup Rescue Hazmat Detector (Webcam)")
    parser.add_argument("--model", type=str, 
                        default="/home/sbuntu/Downloads/hazmat_50e_export/runs/weights/best.pt",
                        help="Path to YOLOv8 model")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (0 = default webcam)")
    parser.add_argument("--team", type=str, default="MyTeam",
                        help="Team name for CSV")
    parser.add_argument("--country", type=str, default="MyCountry",
                        help="Country for CSV")
    parser.add_argument("--mission", type=str, default="Prelim1",
                        help="Mission name for CSV")
    parser.add_argument("--robot", type=str, default="robot1",
                        help="Robot name for CSV")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold (0-1)")
    
    args = parser.parse_args()
    
    detector = HazmatDetector(
        model_path=args.model,
        camera_id=args.camera,
        team_name=args.team,
        country=args.country,
        mission=args.mission,
        robot_name=args.robot,
        conf_threshold=args.conf
    )
    
    detector.run()


if __name__ == "__main__":
    main()
