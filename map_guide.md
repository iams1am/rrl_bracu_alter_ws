# 2D SLAM Mapping Guide (Cartographer)
**System:** ROS 2  
**Hardware:** IMU & RPLIDAR

This document outlines the procedure for initializing the robot, running SLAM (Simultaneous Localization and Mapping), visualization in RViz, and saving the resulting map.

---

## 0. Clone the Repository
First, clone the workspace repository:

```bash
git clone https://github.com/Peru00/BracU_Alter.git
```

Then build the workspace:

```bash
colcon build --symlink-install
```

---

## 1. Hardware Initialization
To ensure proper device recognition, connect the hardware in the following specific order:

1. **Connect the IMU** to the USB port first.
2. **Connect the RPLIDAR** to the USB port second.

---

## 2. Launching SLAM
Open a terminal and execute the following command to source the workspace and launch the 2D Cartographer node:

```bash
source /home/bracualter-base/alter_ws/install/setup.bash && ros2 launch cartographer_slam cartographer_2d.launch.py
```

---

## 3. Visualization (RViz)
Once the launch file is running, configure RViz to view the mapping process:

- **Add Topic:** Click "Add" and select the `Map` topic.
- **Set Fixed Frame:** In the "Global Options" or "Displays" panel, change the Fixed Frame to `base_link`.

---

## 4. Saving the Map
When you are satisfied with the generated map, run the following command in a new terminal to save it as a GeoTIFF:

```bash
ros2 run cartographer_slam save_map_geotiff.py -o /home/bracualter-base/alter_ws/map.tif
```

---

## 5. Troubleshooting: IMU & Lidar Conflicts

### Symptoms
- IMU is not publishing data.
- RPLIDAR is rotating slowly.
- The map is not updating or has very poor quality.

### Root Cause
Both the IMU and RPLIDAR use the same Silicon Labs CP210x USB-to-serial chip. They share the identical Vendor ID (`10c4`) and Product ID (`ea60`).

Because the IDs are identical, the default system rules (udev) cannot distinguish between them. This results in the system assigning the wrong symlinks (e.g., the system thinks the Lidar is the IMU).

### The Solution: Unique Udev Rules
You must update the `imu_usb.rules` file to identify the device by its Serial Number rather than just the Vendor/Product ID.

#### 1. Update the Rule
Modify your `imu_usb.rules` file to include the `ATTRS{serial}` tag. For example, if your IMU serial is `0001`, the rule ensures `imu_usb` is assigned only to that specific device.

#### 2. Resulting Logic
- `imu_usb` → Assigned to IMU (Serial `0001`)
- The RPLIDAR will default to the remaining port (or can be assigned its own rule using its specific serial number).

#### Verification
To test if the conflict is resolved, keep both devices connected and run the IMU visualization:

```bash
ros2 launch wit_ros2_imu rviz_and_imu.launch.py
```

---

## How to Find Your Device Serial Number
If your IMU serial number is not `0001`, you need to find the correct one to update your rules file.

1. Unplug the RPLIDAR (keep only the IMU connected).
2. Run the following command:

```bash
udevadm info -a -n /dev/ttyUSB0 | grep serial
```

Copy the output serial number and update your `imu_usb.rules` file accordingly.
