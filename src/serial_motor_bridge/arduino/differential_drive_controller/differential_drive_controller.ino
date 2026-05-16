/*
 * Differential Drive Motor Controller for Arduino Uno
 * 
 * Controls Cytron MD20A motor driver via serial commands from ROS2
 * 
 * Motor Driver Connections:
 * - PWM1 (Left Motor PWM)  -> Pin 11 (PWM capable)
 * - DIR1 (Left Motor Dir)  -> Pin 13
 * - PWM2 (Right Motor PWM) -> Pin 10 (PWM capable)
 * - DIR2 (Right Motor Dir) -> Pin 12
 * 
 * Serial Protocol:
 * Start Byte: 0xAA
 * Left PWM:   2 bytes (signed 16-bit, big-endian) -255 to 255
 * Right PWM:  2 bytes (signed 16-bit, big-endian) -255 to 255
 * End Byte:   0x55
 * 
 * Total: 6 bytes per command
 */

// Motor Driver Pin Definitions
#define PWM1_PIN 11   // Left motor PWM
#define DIR1_PIN 13   // Left motor direction
#define PWM2_PIN 10   // Right motor PWM
#define DIR2_PIN 12   // Right motor direction

// Serial Communication
#define BAUD_RATE 115200
#define START_BYTE 0xAA
#define END_BYTE 0x55
#define PACKET_SIZE 6

// Safety timeout (ms) - stop motors if no command received
#define COMMAND_TIMEOUT 500

// Set to true to test motors on startup (set false for normal operation)
#define ENABLE_STARTUP_TEST false

// Variables
byte serialBuffer[PACKET_SIZE];
int bufferIndex = 0;
unsigned long lastCommandTime = 0;
int16_t leftPWM = 0;
int16_t rightPWM = 0;
boolean motorsRunning = false;

void setup() {
  // Initialize serial communication
  Serial.begin(BAUD_RATE);
  
  // Configure motor driver pins - set all LOW initially
  pinMode(PWM1_PIN, OUTPUT);
  pinMode(DIR1_PIN, OUTPUT);
  pinMode(PWM2_PIN, OUTPUT);
  pinMode(DIR2_PIN, OUTPUT);
  
  // Force all pins LOW on startup to prevent floating states
  digitalWrite(PWM1_PIN, LOW);
  digitalWrite(DIR1_PIN, LOW);
  digitalWrite(PWM2_PIN, LOW);
  digitalWrite(DIR2_PIN, LOW);
  analogWrite(PWM1_PIN, 0);
  analogWrite(PWM2_PIN, 0);
  
  delay(500);  // Wait for power to stabilize
  
  // Verify motors are stopped
  stopMotors();
  
  lastCommandTime = millis();
  motorsRunning = false;
}

void loop() {
  // Process incoming serial data
  processSerial();
  
  // Safety check - stop motors if no command received within timeout
  unsigned long now = millis();
  if (now - lastCommandTime > COMMAND_TIMEOUT) {
    if (motorsRunning) {
      stopMotors();
      motorsRunning = false;
    }
  }
  
  delay(10);  // Small delay to prevent overwhelming the motor driver
}

void processSerial() {
  while (Serial.available() > 0) {
    byte inByte = Serial.read();
    
    // Look for start byte
    if (bufferIndex == 0) {
      if (inByte == START_BYTE) {
        serialBuffer[bufferIndex++] = inByte;
      }
    }
    else {
      serialBuffer[bufferIndex++] = inByte;
      
      // Check if we have a complete packet
      if (bufferIndex >= PACKET_SIZE) {
        // Verify end byte
        if (serialBuffer[PACKET_SIZE - 1] == END_BYTE) {
          parseCommand();
        }
        bufferIndex = 0;  // Reset buffer for next packet
      }
    }
    
    // Prevent buffer overflow
    if (bufferIndex >= PACKET_SIZE) {
      bufferIndex = 0;
    }
  }
}

void parseCommand() {
  // Extract left PWM (bytes 1-2, big-endian signed 16-bit)
  leftPWM = (int16_t)(((uint16_t)serialBuffer[1] << 8) | (uint16_t)serialBuffer[2]);
  
  // Extract right PWM (bytes 3-4, big-endian signed 16-bit)
  rightPWM = (int16_t)(((uint16_t)serialBuffer[3] << 8) | (uint16_t)serialBuffer[4]);
  
  // Track if motors should be running
  if (leftPWM != 0 || rightPWM != 0) {
    motorsRunning = true;
  } else {
    motorsRunning = false;
  }
  
  // Apply motor commands
  setMotor(PWM1_PIN, DIR1_PIN, leftPWM);
  setMotor(PWM2_PIN, DIR2_PIN, rightPWM);
  
  // Update command timestamp
  lastCommandTime = millis();
}

void setMotor(int pwmPin, int dirPin, int16_t pwmValue) {
  // Ensure direction pin is set before PWM to avoid spikes
  if (pwmValue >= 0) {
    digitalWrite(dirPin, HIGH);  // Forward
    pwmValue = constrain(pwmValue, 0, 255);
  } else {
    digitalWrite(dirPin, LOW);   // Reverse
    pwmValue = constrain(-pwmValue, 0, 255);
  }
  
  // Apply PWM
  analogWrite(pwmPin, pwmValue);
}

void stopMotors() {
  // Force all pins to safe state
  digitalWrite(PWM1_PIN, LOW);
  digitalWrite(DIR1_PIN, LOW);
  digitalWrite(PWM2_PIN, LOW);
  digitalWrite(DIR2_PIN, LOW);
  analogWrite(PWM1_PIN, 0);
  analogWrite(PWM2_PIN, 0);
  leftPWM = 0;
  rightPWM = 0;
  motorsRunning = false;
}
