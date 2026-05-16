# RoboCup Rescue League 2026 - Hazmat Detection System

Detects hazmat signs using YOLOv8 + standard webcam.
Outputs CSV in official **RoboCup 2026 format**.

## Hardware
- Raspberry Pi 4B (Ubuntu 24.04) or any Linux PC
- USB Webcam (or laptop camera)
- RPLIDAR A3 (for SLAM - separate)

## Quick Start

```bash
# Activate virtual environment
source /home/sbuntu/yoloenv/bin/activate

# Run with defaults (camera 0)
python /home/sbuntu/Downloads/hazmat_50e_export/robocup_hazmat_detector.py

# Run with your team info
python /home/sbuntu/Downloads/hazmat_50e_export/robocup_hazmat_detector.py \
    --team "YourTeamName" \
    --country "YourCountry" \
    --mission "Prelim1" \
    --robot "robot1" \
    --conf 0.3 \
    --camera 0
```

## Controls
| Key | Action |
|-----|--------|
| **q** | Quit and save CSV |
| **s** | Save CSV without quitting |
| **c** | Clear all detections |

## Output
CSV files are saved to: `detection_results/RoboCup2026-TeamName-Mission-HH-MM-SS-pois.csv`

### Example CSV Output:
```
"pois"
"1.3"
"MyTeam"
"MyCountry"
"2026-01-26"
"14:30:00"
"Prelim1"
detection,time,type,name,x,y,z,robot,mode
1,14:28:01,"hazmat_sign","poison",-0.52,-0.08,1.23,"robot1",T
2,14:28:05,"hazmat_sign","radioactive",0.31,0.12,0.95,"robot1",T
```

## Hazmat Classes (16)
- corrosive, dangerous-when-wet, explosive, flammable
- flammable-gas, flammable-liquid, flammable-solid
- infectious-substance, inhalation-hazard, non-flammable-gas
- organic-peroxide, oxidizer, oxygen, poison
- radioactive, spontaneously-combustible

## Options
| Flag | Default | Description |
|------|---------|-------------|
| --model | .../best.pt | Path to YOLOv8 model |
| --camera | 0 | Camera ID (0 = default) |
| --team | MyTeam | Team name |
| --country | MyCountry | Country |
| --mission | Prelim1 | Mission name |
| --robot | robot1 | Robot name |
| --conf | 0.3 | Detection confidence threshold |

## Notes
- **Coordinates are ESTIMATED** without a depth camera
  - Z (depth) is estimated from bounding box size assuming standard hazmat sign size (25cm)
  - X, Y are estimated from pixel position + estimated depth
- Detections within 0.3m of each other (same class) are considered duplicates
- For accurate coordinates, use RealSense D435i version of this script

## Upgrading to RealSense D435i
When you're ready for accurate depth-based coordinates, install pyrealsense2:
```bash
pip install pyrealsense2
```
Then use the RealSense version of the detector (ask for it!).
