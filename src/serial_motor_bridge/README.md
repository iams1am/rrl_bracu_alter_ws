# Serial Motor Bridge for Differential Drive Robot

A ROS2 package that bridges joystick teleop commands to an Arduino Uno controlling a Cytron MD20A motor driver for differential drive robots.

## Hardware Requirements

- Arduino Uno
- Cytron MD20A Motor Driver (or compatible)
- Joystick (USB gamepad)
- Differential drive robot with two DC motors

## Wiring Diagram

### Arduino to Cytron MD20A

| Arduino Pin | MD20A Pin | Description        |
|-------------|-----------|-------------------|
| Pin 11       | PWM1      | Left Motor PWM    |
| Pin 13       | DIR1      | Left Motor Dir    |
| Pin 10       | PWM2      | Right Motor PWM   |
| Pin 12       | DIR2      | Right Motor Dir   |
| GND         | GND       | Common Ground     |

### Power Connections

- Motor power supply (12V-24V) to MD20A power terminals
- Motors connected to MD20A motor outputs

## Installation

### 1. Install Dependencies

```bash
sudo apt install ros-${ROS_DISTRO}-joy ros-${ROS_DISTRO}-teleop-twist-joy
pip3 install pyserial
```

### 2. Build the Package

```bash
cd ~/alter_ws
colcon build --packages-select serial_motor_bridge
source install/setup.bash
```

### 3. Upload Arduino Sketch

1. Open Arduino IDE
2. Open `arduino/differential_drive_controller/differential_drive_controller.ino`
3. Select Arduino Uno board
4. Upload to Arduino

### 4. Set Serial Port Permissions

```bash
sudo usermod -a -G dialout $USER
# Log out and log back in for changes to take effect
```

## Usage

### Launch Serial Motor Bridge Only

```bash
ros2 launch serial_motor_bridge serial_motor_bridge.launch.py
# Or with custom port:
ros2 launch serial_motor_bridge serial_motor_bridge.launch.py serial_port:=/dev/ttyACM0
```

### Launch with Joystick Teleop

```bash
ros2 launch serial_motor_bridge teleop_joy.launch.py debug_serial:=true
# Or with custom port:
ros2 launch serial_motor_bridge teleop_joy.launch.py serial_port:=/dev/ttyACM0 debug_serial:=true
```

### Manual Testing with Keyboard

If you don't have a joystick, you can test with keyboard teleop:

```bash
# Terminal 1: Launch serial bridge with debug output
ros2 launch serial_motor_bridge serial_motor_bridge.launch.py debug_serial:=true

# Terminal 2: Run keyboard teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Configuration

### Robot Parameters (`config/serial_motor_params.yaml`)
ACM0"  # Arduino Uno native USB
```yaml
serial_motor_bridge:
  ros__parameters:
    serial_port: "/dev/ttyUSB0"  # Arduino serial port
    baud_rate: 115200
    wheel_base: 0.3              # Distance between wheels (meters)
    max_linear_speed: 1.0        # Max linear speed (m/s)
    max_angular_speed: 2.0       # Max angular speed (rad/s)
    max_pwm: 255                 # Max PWM value
    timeout: 0.5                 # Safety timeout (seconds)
```

### Joystick Configuration (`config/teleop_joy_params.yaml`)

Adjust axis and button mappings for your joystick:

```yaml
teleop_twist_joy_node:
  ros__parameters:
    axis_linear:
      x: 1                # Left stick Y-axis
    axis_angular:
      yaw: 3              # Right stick X-axis
    enable_button: 4      # L1/LB button (must hold to drive)
    enable_turbo_button: 5
```

## Joystick Controls

| Control              | Action                    |
|---------------------|---------------------------|
| Left Stick Y-axis   | Forward/Backward          |
| Right Stick X-axis  | Turn Left/Right           |
| L1/LB (hold)        | Enable movement           |
| R1/RB (hold)        | Enable turbo mode         |

## Troubleshooting

### Find Arduino Serial Port

```bash
# List serial ports
ls /dev/tty*

# Common ports:
# - /dev/ttyACM0 (Arduino Uno native USB - default)
# - /dev/ttyUSB0 (USB-to-Serial adapter)
```

### Check Joystick Input

```bash
# Test joystick
ros2 run joy joy_node
ros2 topic echo /joy
```

### Check cmd_vel Topic

```bash
ros2 topic echo /cmd_vel
```

### Serial Communication Test

```python
# Python test script
import serial
import struct

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
# Send stop command
data = struct.pack('>BhhB', 0xAA, 0, 0, 0x55)
ser.write(data)
ser.close()
```

## Protocol Specification

### Serial Packet Format (6 bytes)

| Byte | Description         | Value Range        |
|------|--------------------|--------------------|
| 0    | Start byte         | 0xAA               |
| 1-2  | Left PWM (int16)   | -255 to 255        |
| 3-4  | Right PWM (int16)  | -255 to 255        |
| 5    | End byte           | 0x55               |

- PWM values are big-endian signed 16-bit integers
- Positive values = forward, Negative values = reverse

## License

Apache-2.0
