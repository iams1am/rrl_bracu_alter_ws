#!/usr/bin/env python3
"""
Serial Motor Bridge Node for Differential Drive Robot

Subscribes to /cmd_vel topic and sends motor commands to Arduino
via serial communication for Cytron MD20A motor driver control.

Motor Driver Pins:
- PWM1, DIR1: Left Motor
- PWM2, DIR2: Right Motor
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import struct
import time


class SerialMotorBridge(Node):
    def __init__(self):
        super().__init__('serial_motor_bridge')
        
        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('wheel_base', 0.3)  # Distance between wheels in meters
        self.declare_parameter('max_linear_speed', 1.0)  # Max linear speed m/s
        self.declare_parameter('max_angular_speed', 2.0)  # Max angular speed rad/s
        self.declare_parameter('max_pwm', 255)  # Max PWM value
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('timeout', 0.5)  # Command timeout in seconds
        self.declare_parameter('debug_serial', False)  # Enable serial output mirroring
        
        # Get parameters
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.max_pwm = self.get_parameter('max_pwm').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.timeout = self.get_parameter('timeout').value
        self.debug_serial = self.get_parameter('debug_serial').value
        
        # Initialize serial connection
        self.serial_conn = None
        self.connect_serial()
        
        # Subscribe to cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10
        )
        
        # Timer for timeout check (stop motors if no command received)
        self.last_cmd_time = time.time()
        self.timer = self.create_timer(0.1, self.timeout_check)
        
        # Serial read timer for debugging
        if self.debug_serial:
            self.serial_read_timer = self.create_timer(0.05, self.read_serial_output)
        
        self.get_logger().info(f'Serial Motor Bridge started on {self.serial_port}')
        self.get_logger().info(f'Subscribing to {self.cmd_vel_topic}')
        if self.debug_serial:
            self.get_logger().info('Serial debug output ENABLED')
    
    def connect_serial(self):
        """Establish serial connection with Arduino"""
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=1
            )
            time.sleep(2)  # Wait for Arduino to reset
            self.get_logger().info(f'Connected to Arduino on {self.serial_port}')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to connect to serial port: {e}')
            self.serial_conn = None
    
    def cmd_vel_callback(self, msg: Twist):
        """Process incoming velocity commands"""
        self.last_cmd_time = time.time()
        
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        # Differential drive kinematics
        # v_left = linear_x - (angular_z * wheel_base / 2)
        # v_right = linear_x + (angular_z * wheel_base / 2)
        
        left_speed = linear_x - (angular_z * self.wheel_base / 2.0)
        right_speed = linear_x + (angular_z * self.wheel_base / 2.0)
        
        # Normalize speeds to max linear speed
        max_speed = max(abs(left_speed), abs(right_speed), self.max_linear_speed)
        if max_speed > self.max_linear_speed:
            left_speed = left_speed / max_speed * self.max_linear_speed
            right_speed = right_speed / max_speed * self.max_linear_speed
        
        # Convert to PWM values (-255 to 255)
        left_pwm = int((left_speed / self.max_linear_speed) * self.max_pwm)
        right_pwm = int((right_speed / self.max_linear_speed) * self.max_pwm)
        
        # Clamp PWM values
        left_pwm = max(-self.max_pwm, min(self.max_pwm, left_pwm))
        right_pwm = max(-self.max_pwm, min(self.max_pwm, right_pwm))
        
        self.send_motor_command(left_pwm, right_pwm)
    
    def send_motor_command(self, left_pwm: int, right_pwm: int):
        """Send motor command to Arduino via serial"""
        if self.serial_conn is None or not self.serial_conn.is_open:
            self.get_logger().warn('Serial connection not available, attempting reconnect...')
            self.connect_serial()
            return
        
        try:
            # Protocol: <START_BYTE><LEFT_PWM_HIGH><LEFT_PWM_LOW><RIGHT_PWM_HIGH><RIGHT_PWM_LOW><END_BYTE>
            # Using signed 16-bit integers for PWM values
            # Start byte: 0xAA, End byte: 0x55
            
            start_byte = 0xAA
            end_byte = 0x55
            
            # Pack as signed 16-bit integers (big-endian)
            data = struct.pack('>BhhB', start_byte, left_pwm, right_pwm, end_byte)
            self.serial_conn.write(data)
            
            self.get_logger().debug(f'Sent: L={left_pwm}, R={right_pwm}')
            
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write error: {e}')
            self.serial_conn = None
    
    def timeout_check(self):
        """Stop motors if no command received within timeout period"""
        if time.time() - self.last_cmd_time > self.timeout:
            self.send_motor_command(0, 0)
    
    def read_serial_output(self):
        """Read and display serial output from Arduino (debug mode)"""
        if self.serial_conn is None or not self.serial_conn.is_open:
            return
        
        try:
            while self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.get_logger().info(f'[ARDUINO] {line}')
        except Exception as e:
            self.get_logger().debug(f'Serial read error: {e}')
    
    def destroy_node(self):
        """Clean up on node shutdown"""
        # Stop motors before closing
        self.send_motor_command(0, 0)
        
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info('Serial connection closed')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = SerialMotorBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
