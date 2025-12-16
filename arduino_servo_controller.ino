/*
 * arduino_servo_controller.ino
 * 
 * Arduino sketch for receiving servo angle commands from a PC via Serial
 * and controlling two servos for the safetronics security system.
 * 
 * Hardware connections:
 * - Servo 1 (Door Lock) signal wire -> Pin 10
 * - Servo 2 (Pan/Tracking) signal wire -> Pin 9
 * - Both servos power -> 5V (or external power for larger servos)
 * - Both servos ground -> GND
 * 
 * Serial protocol:
 * - Format: "S1,angle\n" or "S2,angle\n"
 * - Example: "S1,90\n" sets servo 1 (door lock) to 90 degrees (unlocked)
 * - Example: "S2,0\n" sets servo 2 (tracking) to 0 degrees
 * - Single number "90\n" controls servo 2 (backward compatible)
 */

#include <Servo.h>

// Pin definitions
const int SERVO1_PIN = 10;  // Door lock servo
const int SERVO2_PIN = 9;   // Pan/tracking servo

// Servo objects
Servo doorLockServo;  // Servo 1
Servo panServo;       // Servo 2

// Buffer for incoming serial data
const int BUFFER_SIZE = 8;
char inputBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// Current servo positions
int servo1Angle = 0;   // Door lock starts locked (0 degrees)
int servo2Angle = 90;  // Pan servo starts centered (90 degrees)

// Timing for non-blocking operations
unsigned long lastUpdateTime = 0;
const unsigned long UPDATE_INTERVAL = 20; // milliseconds

void setup() {
    // Initialize serial communication at 9600 baud
    Serial.begin(9600);
    
    // Attach servos to pins
    doorLockServo.attach(SERVO1_PIN);
    panServo.attach(SERVO2_PIN);
    
    // Set initial positions
    doorLockServo.write(servo1Angle);  // Door locked
    panServo.write(servo2Angle);        // Camera centered
    
    // Clear input buffer
    memset(inputBuffer, 0, BUFFER_SIZE);
    
    // Send ready signal
    Serial.println("SERVO_READY");
    Serial.println("S1:LOCKED(0)");
    Serial.println("S2:CENTER(90)");
}

void loop() {
    // Non-blocking serial read
    readSerial();
    
    // Non-blocking servo update (if needed for smooth movement)
    updateServo();
}

void readSerial() {
    // Check if data is available on serial port
    while (Serial.available() > 0) {
        char inChar = Serial.read();
        
        // Check for newline (end of command)
        if (inChar == '\n' || inChar == '\r') {
            if (bufferIndex > 0) {
                // Null-terminate the string
                inputBuffer[bufferIndex] = '\0';
                
                // Parse and process the command
                processCommand(inputBuffer);
                
                // Reset buffer
                bufferIndex = 0;
                memset(inputBuffer, 0, BUFFER_SIZE);
            }
        } else {
            // Add character to buffer if space available
            if (bufferIndex < BUFFER_SIZE - 1) {
                inputBuffer[bufferIndex] = inChar;
                bufferIndex++;
            }
        }
    }
}

void processCommand(const char* command) {
    // Check if command format is "S1,angle" or "S2,angle"
    if (command[0] == 'S' && (command[1] == '1' || command[1] == '2') && command[2] == ',') {
        // Parse servo number and angle
        int servoNum = command[1] - '0';  // Convert '1' or '2' to 1 or 2
        int angle = atoi(&command[3]);    // Parse angle after "S1," or "S2,"
        
        // Validate angle range (0-180 degrees)
        if (angle >= 0 && angle <= 180) {
            if (servoNum == 1) {
                // Control door lock servo
                servo1Angle = angle;
                doorLockServo.write(servo1Angle);
                Serial.print("ACK:S1:");
                Serial.println(servo1Angle);
            } else if (servoNum == 2) {
                // Control pan/tracking servo
                servo2Angle = angle;
                panServo.write(servo2Angle);
                Serial.print("ACK:S2:");
                Serial.println(servo2Angle);
            }
        } else {
            Serial.println("ERR:INVALID_ANGLE");
        }
    } else {
        // Backward compatibility: single number controls servo 2 (pan)
        int angle = atoi(command);
        
        if (angle >= 0 && angle <= 180) {
            servo2Angle = angle;
            panServo.write(servo2Angle);
            Serial.print("ACK:S2:");
            Serial.println(servo2Angle);
        } else {
            Serial.println("ERR:INVALID_ANGLE");
        }
    }
}

void updateServo() {
    // This function can be extended for smooth movement
    // Currently servo is updated immediately in processCommand
    
    unsigned long currentTime = millis();
    
    if (currentTime - lastUpdateTime >= UPDATE_INTERVAL) {
        lastUpdateTime = currentTime;
        
        // Additional servo processing can be added here
        // For example: smooth interpolation between angles
    }
}
