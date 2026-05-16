#!/usr/bin/env python3
"""
RoboCup Rescue League 2026 - Official Detection System
=======================================================
Detects ALL required object types using train3 YOLOv8 model:

  HAZMAT SIGNS (type="hazmat_sign"):
    - blasting_agents, corrosive, dangerous_when_wet, explosives
    - flammable_gas, flammable_solid, fuel_oil, inhalation_hazard
    - non_flammable_gas, organic_peroxide, oxidizer, oxygen
    - poison, radioactive, spontaneously_combustible

  REAL OBJECTS (type="real_object"):
    - Backpack, baby_doll_face, fire_extinguisher, hard_hat, propen_tank

  AR CODES (type="ar_code"):
    - ArUco markers detected via OpenCV

Outputs CSV in official RoboCup 2026 format.
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
from typing import List, Optional, Dict, Set

# Force CPU to avoid CUDA compatibility issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ============================================================================
# TRAIN3 MODEL CLASS CATEGORIZATION
# ============================================================================

# Classes from train3 model that are HAZMAT SIGNS
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

# Classes from train3 model that are REAL OBJECTS
REAL_OBJECT_CLASSES: Set[str] = {
    "Backpack",
    "baby_doll_face",
    "fire_extinguisher",
    "hard_hat",
    "propen_tank",
}

# Estimated real-world sizes (meters) for depth estimation
OBJECT_SIZES: Dict[str, float] = {
    # Hazmat signs - standard placard ~25cm
    "blasting_agents": 0.25,
    "corrosive": 0.25,
    "dangerous_when_wet": 0.25,
    "explosives": 0.25,
    "flammable_gas": 0.25,
    "flammable_solid": 0.25,
    "fuel_oil": 0.25,
    "inhalation_hazard": 0.25,
    "non_flammable_gas": 0.25,
    "organic_peroxide": 0.25,
    "oxidizer": 0.25,
    "oxygen": 0.25,
    "poison": 0.25,
    "radioactive": 0.25,
    "spontaneously_combustible": 0.25,
    # Real objects - approximate sizes
    "Backpack": 0.40,
    "baby_doll_face": 0.15,
    "fire_extinguisher": 0.50,
    "hard_hat": 0.25,
    "propen_tank": 0.30,
}


# ============================================================================
# DETECTION DATA STRUCTURES
# ============================================================================

@dataclass
class Detection:
    """Represents a unique detected object per RoboCup 2026 spec."""
    detection_id: int
    timestamp: str
    obj_type: str  # "hazmat_sign", "ar_code", "real_object", "heat_sig"
    name: str      # e.g., "poison", "42", "Backpack"
    x: float       # meters
    y: float       # meters
    z: float       # meters
    robot: str
    mode: str      # "A" or "T"


@dataclass
class DetectionManager:
    """Manages detections and prevents duplicates per RoboCup rules."""
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
        """Add a new detection if not duplicate. Returns detection if added."""
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
        print(f"[NEW #{detection.detection_id}] {obj_type}: \"{name}\" at ({x:.2f}, {y:.2f}, {z:.2f})m")
        return detection
    
    def save_csv(self, filepath: str, team_name: str, country: str, mission: str,
                 start_date: str, start_time: str):
        """Save detections to CSV in exact RoboCup 2026 format."""
        
        with open(filepath, 'w', newline='') as f:
            # Header section (exactly as per RoboCup spec)
            f.write('"pois"\n')
            f.write('"1.3"\n')
            f.write(f'"{team_name}"\n')
            f.write(f'"{country}"\n')
            f.write(f'"{start_date}"\n')
            f.write(f'"{start_time}"\n')
            f.write(f'"{mission}"\n')
            
            # Column headers (no quotes per spec)
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
    
    def print_all_detections(self):
        """Print all detections in a table format."""
        print("\n" + "="*80)
        print("ALL DETECTED OBJECTS")
        print("="*80)
        print(f"{'#':<4} {'Time':<10} {'Type':<15} {'Name':<25} {'X':>8} {'Y':>8} {'Z':>8}")
        print("-"*80)
        for det in self.detections:
            print(f"{det.detection_id:<4} {det.timestamp:<10} {det.obj_type:<15} {det.name:<25} "
                  f"{det.x:>8.2f} {det.y:>8.2f} {det.z:>8.2f}")
        print("="*80)


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
        # Use 4x4_50 as default (common in competitions)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_size = 0.10  # 10cm
        
        print("✅ ArUco detector initialized (DICT_4X4_50)")
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detect ArUco markers in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        detections = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i][0]
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))
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
    
    def draw(self, frame: np.ndarray, detections: List[dict], coords_list: List[tuple]) -> np.ndarray:
        """Draw detected markers on frame."""
        for det, coords in zip(detections, coords_list):
            corners = det["corners"].astype(int)
            cv2.polylines(frame, [corners], True, (255, 0, 255), 2)
            
            cx, cy = det["center"]
            x, y, z = coords
            cv2.putText(frame, f"AR:{det['id']}", (cx - 30, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame, f"({x:.1f},{y:.1f},{z:.1f})m", (cx - 40, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 255), 1)
        
        return frame


# ============================================================================
# MAIN RRL2026 DETECTOR
# ============================================================================

class RRL2026Detector:
    """Official RoboCup Rescue League 2026 Detection System."""
    
    def __init__(self, model_path: str, camera_id: int = 0,
                 team_name: str = "MyTeam", country: str = "MyCountry",
                 mission: str = "Prelim1", robot_name: str = "robot1",
                 conf_threshold: float = 0.25):
        
        self.team_name = team_name
        self.country = country
        self.mission = mission
        self.robot_name = robot_name
        self.conf_threshold = conf_threshold
        
        # Record start time for CSV
        self.start_datetime = datetime.now()
        self.start_date = self.start_datetime.strftime("%Y-%m-%d")
        self.start_time = self.start_datetime.strftime("%H:%M:%S")
        
        print("🤖 RoboCup Rescue League 2026 - Object Detection System")
        print("="*60)
        print(f"   Team: {team_name} ({country})")
        print(f"   Mission: {mission}")
        print(f"   Robot: {robot_name}")
        print(f"   Start: {self.start_date} {self.start_time}")
        print("="*60)
        
        # Load train3 model (has both hazmat and real objects)
        print("\n[1/3] Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # Categorize classes
        hazmat_count = sum(1 for name in self.class_names.values() if name in HAZMAT_CLASSES)
        object_count = sum(1 for name in self.class_names.values() if name in REAL_OBJECT_CLASSES)
        print(f"       Loaded {len(self.class_names)} classes:")
        print(f"       - {hazmat_count} hazmat signs")
        print(f"       - {object_count} real objects")
        
        # Initialize ArUco detector
        print("\n[2/3] Initializing ArUco detector...")
        self.aruco_detector = ArUcoDetector()
        
        # Start camera
        print("\n[3/3] Starting webcam...")
        self.camera = WebcamCamera(camera_id=camera_id, width=640, height=480)
        self.camera.start()
        
        self.detection_manager = DetectionManager()
        self.running = False
        
        print("\n" + "="*60)
        print("✅ System Ready!")
        print("="*60)
    
    def get_object_type(self, class_name: str) -> str:
        """Determine if class is hazmat_sign or real_object."""
        if class_name in HAZMAT_CLASSES:
            return "hazmat_sign"
        elif class_name in REAL_OBJECT_CLASSES:
            return "real_object"
        else:
            return "real_object"  # Default to real_object for unknown
    
    def get_display_color(self, obj_type: str) -> tuple:
        """Get BGR color for display based on object type."""
        colors = {
            "hazmat_sign": (0, 255, 0),    # Green
            "real_object": (0, 165, 255),   # Orange
            "ar_code": (255, 0, 255),       # Magenta
            "heat_sig": (0, 0, 255),        # Red
        }
        return colors.get(obj_type, (255, 255, 255))
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with all detectors."""
        annotated = frame.copy()
        
        # 1. Run YOLOv8 detection (hazmat signs + real objects)
        results = self.model(frame, conf=self.conf_threshold, verbose=False, device='cpu')
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bbox_size = ((x2 - x1) + (y2 - y1)) / 2
                
                # Get object type and real size
                obj_type = self.get_object_type(cls_name)
                real_size = OBJECT_SIZES.get(cls_name, 0.25)
                
                # Estimate 3D position
                x, y, z = self.camera.estimate_3d_position(cx, cy, bbox_size, real_size)
                
                # Add detection
                self.detection_manager.add_detection(
                    obj_type=obj_type,
                    name=cls_name,
                    x=x, y=y, z=z,
                    robot=self.robot_name,
                    mode="T"
                )
                
                # Draw bounding box
                color = self.get_display_color(obj_type)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw labels
                type_label = "H" if obj_type == "hazmat_sign" else "O"
                label = f"[{type_label}] {cls_name} ({conf:.2f})"
                coords = f"({x:.1f}, {y:.1f}, {z:.1f})m"
                
                cv2.putText(annotated, label, (x1, y1 - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.putText(annotated, coords, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        
        # 2. Detect ArUco markers
        aruco_detections = self.aruco_detector.detect(frame)
        aruco_coords = []
        for det in aruco_detections:
            cx, cy = det["center"]
            x, y, z = self.camera.estimate_3d_position(cx, cy, det["size"], real_size=0.10)
            aruco_coords.append((x, y, z))
            
            self.detection_manager.add_detection(
                obj_type="ar_code",
                name=str(det["id"]),
                x=x, y=y, z=z,
                robot=self.robot_name,
                mode="T"
            )
        annotated = self.aruco_detector.draw(annotated, aruco_detections, aruco_coords)
        
        # Draw status overlay
        summary = self.detection_manager.get_summary()
        h, w = annotated.shape[:2]
        
        # Background for status
        cv2.rectangle(annotated, (0, 0), (w, 70), (0, 0, 0), -1)
        
        # Status text
        status1 = f"RRL2026 | {self.team_name} | {self.mission} | Total: {len(self.detection_manager.detections)}"
        status2 = f"Hazmat: {summary['hazmat_sign']} | AR: {summary['ar_code']} | Objects: {summary['real_object']}"
        status3 = "[q]=Quit+Save  [s]=Save  [c]=Clear  [p]=Print All"
        
        cv2.putText(annotated, status1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, status2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, status3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Legend
        cv2.putText(annotated, "[H]=Hazmat", (w-150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(annotated, "[O]=Object", (w-150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
        cv2.putText(annotated, "[AR]=ArUco", (w-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
        
        return annotated
    
    def run(self):
        """Main detection loop."""
        self.running = True
        print("\n" + "="*60)
        print("🔍 DETECTION RUNNING")
        print("   Detecting: hazmat_sign, ar_code, real_object")
        print("="*60)
        print("\nKEYS:")
        print("   [q] = Quit and save CSV")
        print("   [s] = Save CSV snapshot")
        print("   [c] = Clear all detections")
        print("   [p] = Print all detections to console")
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
                                (annotated.shape[1] - 70, annotated.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                cv2.imshow("RRL2026 Detector", annotated)
                
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
                elif key == ord('p'):
                    self.detection_manager.print_all_detections()
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        finally:
            self.cleanup()
    
    def save_results(self):
        """Save detection results to CSV in RoboCup format."""
        start_time_str = self.start_datetime.strftime("%H-%M-%S")
        year = self.start_datetime.year
        
        # Filename per RoboCup spec: RoboCup[Year]-[Teamname]-[Mission]-[Start Time]-pois.csv
        filename = f"RoboCup{year}-{self.team_name}-{self.mission}-{start_time_str}-pois.csv"
        
        output_dir = Path(__file__).parent / "detection_results"
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename
        
        self.detection_manager.save_csv(
            filepath=str(filepath),
            team_name=self.team_name,
            country=self.country,
            mission=self.mission,
            start_date=self.start_date,
            start_time=self.start_time
        )
        
        # Print summary
        summary = self.detection_manager.get_summary()
        print(f"   Summary: Hazmat={summary['hazmat_sign']}, AR={summary['ar_code']}, "
              f"Objects={summary['real_object']}, Heat={summary['heat_sig']}")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.save_results()
        self.detection_manager.print_all_detections()
        self.camera.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="RoboCup Rescue League 2026 - Official Object Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
  python robocup_rrl2026_detector.py --team "MyTeam" --country "USA" --mission "Final1"

DETECTED OBJECT TYPES:
  hazmat_sign  : blasting_agents, corrosive, dangerous_when_wet, explosives,
                 flammable_gas, flammable_solid, fuel_oil, inhalation_hazard,
                 non_flammable_gas, organic_peroxide, oxidizer, oxygen,
                 poison, radioactive, spontaneously_combustible
  
  real_object  : Backpack, baby_doll_face, fire_extinguisher, hard_hat, propen_tank
  
  ar_code      : ArUco markers (IDs 0-49)
        """
    )
    parser.add_argument("--model", type=str,
                        default="/home/sbuntu/Downloads/train3/weights/best.pt",
                        help="Path to YOLOv8 model (default: train3)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (default: 0)")
    parser.add_argument("--team", type=str, default="MyTeam",
                        help="Team name for CSV")
    parser.add_argument("--country", type=str, default="MyCountry",
                        help="Country for CSV")
    parser.add_argument("--mission", type=str, default="Prelim1",
                        help="Mission name (e.g., Prelim1, Final1)")
    parser.add_argument("--robot", type=str, default="robot1",
                        help="Robot name for CSV")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    
    detector = RRL2026Detector(
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
