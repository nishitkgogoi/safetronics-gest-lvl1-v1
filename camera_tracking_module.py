import cv2
import time
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
import json
import serial
import sys
from collections import deque

# Import the other modules
from face_recognition_module import FaceRecognition
from data_storage_module import DataStorage
from gesture_recognition_module import GestureRecognizer

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('security_system.log', encoding='utf-8')
    ]
)

class CameraTrackingSystem:
    def __init__(self, serial_port='COM11', baud_rate=9600):
        self.logger = self.setup_logging()
        
        # Initialize modules
        self.face_recognition = FaceRecognition()
        self.data_storage = DataStorage()
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 20
        
        # System states
        self.system_active = False
        self.door_unlocked = False
        self.tracking_unknown = False
        self.suspicious_activity_detected = False
        
        # Tracking variables
        self.unknown_person_detected_time = None
        self.owner_detected_time = None
        self.last_alert_time = None
        
        # Servo control
        self.servo1_position = 0  # 0=locked, 90=unlocked
        self.servo2_position = 90  # 90=center (pan servo)
        
        # Serial connection for Arduino
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial_connection = None
        self._init_serial_connection()
        
        # Motion detection
        self.previous_frame = None
        self.motion_threshold = 1000
        
        # Alert cooldown (prevent spam)
        self.alert_cooldown = 60  # seconds

        # Gesture detection buffer and state
        self.gesture_buffer = deque(maxlen=int(self.fps * 3))  # keep last ~3 seconds
        self.gesture_recognizer = None
        self._gesture_analysis_in_progress = False
        self.gesture_last_alert_time = None
        self.gesture_alert_cooldown = 30  # seconds between gesture alerts
        self.gesture_overlay = None
        self.gesture_overlay_until = 0
        
        self.logger.info("Camera Tracking System initialized")
    
    def _init_serial_connection(self):
        """Initialize serial connection to Arduino with error handling."""
        try:
            self.serial_connection = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=1
            )
            # Allow Arduino to reset after serial connection
            time.sleep(2)
            self.logger.info(f"Serial connection established on {self.serial_port} at {self.baud_rate} baud")
        except serial.SerialException as e:
            self.logger.warning(f"Could not establish serial connection: {e}. Running in simulation mode.")
            self.serial_connection = None
        except Exception as e:
            self.logger.warning(f"Unexpected error initializing serial: {e}. Running in simulation mode.")
            self.serial_connection = None
    
    def _send_servo_command(self, servo_num, angle):
        """Send servo command to Arduino via serial connection.
        
        Args:
            servo_num: Servo number (1 for door lock, 2 for pan/tracking)
            angle: Integer value between 0 and 180 degrees.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        # Clamp angle to valid range
        angle = max(0, min(180, int(angle)))
        
        if self.serial_connection is None:
            self.logger.debug(f"Serial not connected. Simulated servo {servo_num} angle: {angle}°")
            return False
        
        try:
            # Send command in format "S1,angle\n" or "S2,angle\n"
            command = f"S{servo_num},{angle}\n"
            self.serial_connection.write(command.encode('utf-8'))
            self.serial_connection.flush()
            self.logger.debug(f"Sent command: {command.strip()} to Arduino")
            return True
        except serial.SerialException as e:
            self.logger.error(f"Serial communication error: {e}")
            self._handle_serial_disconnect()
            return False
        except Exception as e:
            self.logger.error(f"Error sending servo command: {e}")
            return False
    
    def _handle_serial_disconnect(self):
        """Handle serial disconnection gracefully."""
        self.logger.warning("Serial connection lost. Attempting to reconnect...")
        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
            except Exception:
                pass
        self.serial_connection = None
        # Try to reconnect
        self._init_serial_connection()
    
    def setup_logging(self):
        """Setup logging for camera tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('camera_tracking.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def initialize_camera(self):
        """Initialize camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"Cannot open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test camera
            ret, test_frame = self.cap.read()
            if not ret:
                self.logger.error("Cannot read from camera")
                return False
            
            self.logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            return False
    
    def control_servo1(self, unlock=True):
        """Control door lock servo (Servo 1)"""
        try:
            angle = 90 if unlock else 0
            self.servo1_position = angle
            
            # Send command to Arduino for Servo 1
            self._send_servo_command(1, angle)
            
            if unlock:
                self.logger.info("DOOR UNLOCKED - Servo 1 rotated 90°")
                self.data_storage.log_security_event(
                    "DOOR_UNLOCKED", 
                    "owner", 
                    100, 
                    "front_door",
                    "Door unlocked via face recognition"
                )
            else:
                self.logger.info("DOOR LOCKED - Servo 1 rotated 0°")
                self.data_storage.log_security_event(
                    "DOOR_LOCKED", 
                    "system", 
                    100, 
                    "front_door",
                    "Door locked by system"
                )
            
            self.door_unlocked = unlock
            return True
            
        except Exception as e:
            self.logger.error(f"Error controlling servo 1: {e}")
            return False
    
    def control_servo2(self, angle):
        """Control tracking servo (Servo 2) by sending pan angle to Arduino.
        
        Args:
            angle: Integer value between 0 and 180 degrees for pan position.
                   90 is center, 0 is full left, 180 is full right.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Clamp angle to valid range
            angle = max(0, min(180, int(angle)))
            self.servo2_position = angle
            
            # Send command to Arduino for Servo 2
            self._send_servo_command(2, angle)
            
            self.logger.info(f"Tracking servo moved to {angle}°")
            
            # Log tracking activity
            self.data_storage.log_security_event(
                "TRACKING_MOVEMENT",
                "unknown",
                0,
                "front_door",
                f"Tracking servo moved to {angle}° to follow unknown person"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error controlling servo 2: {e}")
            return False
    
    def detect_suspicious_activity(self, frame):
        """Detect suspicious activities using motion analysis"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.previous_frame is None:
                self.previous_frame = gray
                return False
            
            # Compute difference between frames
            frame_diff = cv2.absdiff(self.previous_frame, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Dilate to fill holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            suspicious_motion = False
            motion_areas = []
            
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Significant movement
                    (x, y, w, h) = cv2.boundingRect(contour)
                    motion_areas.append((x, y, w, h))
                    suspicious_motion = True
            
            self.previous_frame = gray
            
            # Check if we should trigger alert
            if suspicious_motion and not self.door_unlocked:
                current_time = time.time()
                
                # Check alert cooldown
                if self.last_alert_time is None or (current_time - self.last_alert_time) > self.alert_cooldown:
                    self.trigger_security_alert("SUSPICIOUS_MOTION", motion_areas)
                    self.last_alert_time = current_time
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in suspicious activity detection: {e}")
            return False
    
    def trigger_security_alert(self, alert_type, details=None):
        """Trigger security alerts and notifications"""
        try:
            self.logger.warning(f"🚨 SECURITY ALERT: {alert_type}")
            
            # Determine severity and description
            if alert_type == "SUSPICIOUS_MOTION":
                severity = "HIGH"
                description = "Suspicious movement detected near entrance"
                action_taken = "GSM alerts sent, relay activated"
            elif alert_type == "INTRUDER_DETECTED":
                severity = "CRITICAL"
                description = "Unknown person attempting access"
                action_taken = "Tracking activated, alerts sent"
            else:
                severity = "MEDIUM"
                description = f"Security event: {alert_type}"
                action_taken = "System monitoring"
            
            # Log alert
            self.data_storage.log_alert(alert_type, severity, description, action_taken)
            
            # Simulate GSM alert (replace with actual GSM code)
            self.send_gsm_alert(alert_type, description)
            
            # Simulate relay activation (replace with actual GPIO code)
            self.activate_security_relay()
            
            self.suspicious_activity_detected = True
            
        except Exception as e:
            self.logger.error(f"Error triggering security alert: {e}")
    
    def send_gsm_alert(self, alert_type, description):
        """Simulate sending GSM alerts"""
        try:
            # Simulate sending SMS (replace with actual GSM module code)
            police_message = f"SECURITY ALERT: {alert_type} - {description} - Location: Your Address"
            owner_message = f"HOME SECURITY: {alert_type} - {description}"
            
            self.logger.info(f"📱 SMS to POLICE: {police_message}")
            self.logger.info(f"📱 SMS to OWNER: {owner_message}")
            
            # Log the alert
            self.data_storage.log_security_event(
                "GSM_ALERT_SENT",
                "system",
                100,
                "front_door",
                f"Alerts sent for: {alert_type}"
            )
            
        except Exception as e:
            self.logger.error(f"Error sending GSM alert: {e}")
    
    def activate_security_relay(self):
        """Activate security relay"""
        try:
            # Simulate relay activation (replace with actual GPIO code)
            self.logger.info("🔒 SECURITY RELAY ACTIVATED - Additional security measures enabled")
            
            self.data_storage.log_security_event(
                "RELAY_ACTIVATED",
                "system",
                100,
                "front_door",
                "Security relay activated for enhanced protection"
            )
            
        except Exception as e:
            self.logger.error(f"Error activating relay: {e}")
    
    def track_unknown_person(self, frame, face_bbox):
        """Track unknown person and adjust camera/servo to center face in X-axis.
        
        Implements closed-loop tracking by calculating the pan angle required
        to center the detected face in the frame and sending it to the Arduino.
        
        Args:
            frame: The current video frame.
            face_bbox: Tuple (x, y, w, h) of the detected face bounding box.
        
        Returns:
            int: The calculated pan angle sent to the servo.
        """
        # Tracking constants
        MAX_CORRECTION_ANGLE = 45.0  # Maximum angle adjustment per frame (degrees)
        TRACKING_DEAD_ZONE_PIXELS = 20  # Pixel threshold to prevent servo jitter
        DIRECTION_DISPLAY_THRESHOLD = 50  # Pixel threshold for UI direction display
        
        try:
            x, y, w, h = face_bbox
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate face center X coordinate
            face_center_x = x + w / 2
            
            # Calculate frame center
            frame_center_x = frame_w / 2  # 320 for 640px width
            
            # Calculate error (how far face is from center)
            error_x = face_center_x - frame_center_x
            
            # Calculate pan angle correction
            # Map error to angle: negative error = face is left, need to pan left (decrease angle)
            # Servo range: 0-180, center at 90
            # Scale factor: convert pixel error to angle adjustment
            # Max error is half frame width (320px) -> max correction of MAX_CORRECTION_ANGLE degrees
            scale_factor = MAX_CORRECTION_ANGLE / (frame_w / 2)  # ~0.14 degrees per pixel
            # Negate because positive error (face to right of center) requires
            # decreasing servo angle to pan camera right and bring face to center
            angle_correction = -error_x * scale_factor
            
            # Calculate new servo angle
            new_angle = self.servo2_position + angle_correction
            
            # Clamp to valid servo range
            new_angle = max(0, min(180, int(new_angle)))
            
            # Determine direction for display purposes
            if error_x < -DIRECTION_DISPLAY_THRESHOLD:
                direction = "LEFT"
            elif error_x > DIRECTION_DISPLAY_THRESHOLD:
                direction = "RIGHT"
            else:
                direction = "CENTER"
            
            # Only send command if there's significant correction needed (dead zone to prevent jitter)
            if abs(error_x) > TRACKING_DEAD_ZONE_PIXELS:
                self.control_servo2(new_angle)
            
            # Draw tracking info
            cv2.putText(frame, f"TRACKING: {direction} | Angle: {new_angle}°", 
                       (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return new_angle
            
        except Exception as e:
            self.logger.error(f"Error in person tracking: {e}")
            return self.servo2_position

    def _analyze_gesture_async(self, frames, bbox=None):
        """Background gesture analysis for a short clip of frames.

        NOTE: This method is retained for legacy single-process analysis but
        is not used when running the external gesture worker architecture."""
        try:
            self._gesture_analysis_in_progress = True

            # Lazy init of recognizer
            if self.gesture_recognizer is None:
                try:
                    self.gesture_recognizer = GestureRecognizer()
                except Exception as e:
                    self.logger.error(f"GestureRecognizer unavailable: {e}")
                    self._gesture_analysis_in_progress = False
                    return

            # Analyze
            suspicious, label, conf, details = self.gesture_recognizer.analyze(frames, bbox=bbox)

            self.logger.info(f"Gesture analysis result: suspicious={suspicious}, label={label}, conf={conf}, details={details}")

            if suspicious:
                now = time.time()
                # Check cooldown
                if self.gesture_last_alert_time is None or (now - self.gesture_last_alert_time) > self.gesture_alert_cooldown:
                    # Save clip
                    saved = None
                    try:
                        saved = self.data_storage.save_evidence_clip(frames, "suspicious_gesture", "unknown", fps=self.fps)
                    except Exception as e:
                        self.logger.error(f"Failed saving gesture clip: {e}")

                    # Trigger alert
                    self.trigger_security_alert("SUSPICIOUS_GESTURE", f"{label} ({conf:.2f})")
                    self.data_storage.log_security_event("SUSPICIOUS_GESTURE", "unknown", conf * 100, "front_door", f"Label={label}, clip={saved}")

                    # Overlay message for a short time
                    try:
                        self.gesture_overlay = f"{label} {int(conf*100)}%"
                        self.gesture_overlay_until = time.time() + 5
                    except Exception:
                        pass

                    self.gesture_last_alert_time = now

        except Exception as e:
            self.logger.error(f"Error in gesture analysis thread: {e}")
        finally:
            self._gesture_analysis_in_progress = False

    def _start_gesture_result_poller(self):
        """Start a background thread that polls the gesture worker results folder."""
        try:
            if hasattr(self, '_gesture_poller_thread') and self._gesture_poller_thread.is_alive():
                return

            def poller():
                self.logger.info("Gesture result poller started")
                while self.system_active:
                    try:
                        self._poll_gesture_results()
                    except Exception as e:
                        self.logger.error(f"Error polling gesture results: {e}")
                    time.sleep(1.0)
                self.logger.info("Gesture result poller stopped")

            self._gesture_poller_thread = threading.Thread(target=poller, daemon=True)
            self._gesture_poller_thread.start()
        except Exception as e:
            self.logger.error(f"Failed to start gesture poller: {e}")

    def _poll_gesture_results(self):
        """Check `evidence/gesture_queue/results` for result JSONs and act on them."""
        import glob, json, shutil, os
        results_dir = 'evidence/gesture_queue/results'
        for path in glob.glob(os.path.join(results_dir, '*.json')):
            try:
                with open(path, 'r') as f:
                    res = json.load(f)

                label = res.get('label')
                conf = float(res.get('confidence', 0))
                clip = res.get('clip')
                suspicious = bool(res.get('suspicious', False))

                # Process only suspicious results
                if suspicious:
                    now = time.time()
                    if self.gesture_last_alert_time is None or (now - self.gesture_last_alert_time) > self.gesture_alert_cooldown:
                        self.logger.warning(f"Gesture worker reported suspicious gesture: {label} ({conf:.2f})")
                        # Trigger an alert and log event
                        self.trigger_security_alert('SUSPICIOUS_GESTURE', f"{label} ({conf:.2f})")
                        self.data_storage.log_security_event('SUSPICIOUS_GESTURE', 'unknown', conf * 100, 'front_door', f"Label={label}, clip={clip}")
                        # Overlay message briefly
                        self.gesture_overlay = f"{label} {int(conf*100)}%"
                        self.gesture_overlay_until = time.time() + 5
                        self.gesture_last_alert_time = now

                # Move processed files to processed folder
                processed_dir = 'evidence/gesture_queue/processed'
                base = os.path.basename(path)
                shutil.move(path, os.path.join(processed_dir, base))
            except Exception as e:
                self.logger.error(f"Error handling gesture result {path}: {e}")
                try:
                    # Move problematic file to processed to avoid retry loops
                    processed_dir = 'evidence/gesture_queue/processed'
                    base = os.path.basename(path)
                    shutil.move(path, os.path.join(processed_dir, base))
                except Exception:
                    pass
    
    def run_security_system(self):
        """Main security system loop"""
        try:
            if not self.initialize_camera():
                return
            
            # Start result poller thread for gesture worker
            self._start_gesture_result_poller()

            self.system_active = True
            self.logger.info("Security System STARTED - Press 'q' to quit, 't' to train, 'l' to lock door")
            
            while self.system_active:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Can't receive frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Keep a rolling buffer of recent frames for gesture analysis
                try:
                    self.gesture_buffer.append(frame.copy())
                except Exception:
                    pass
                
                # Face recognition
                processed_frame, owner_detected, face_results, confidence = self.face_recognition.recognize_face(frame)
                
                # Handle owner detection
                if owner_detected and not self.door_unlocked:
                    self.control_servo1(unlock=True)
                    self.owner_detected_time = time.time()
                    self.data_storage.save_evidence_image(frame, "owner_detected", "owner")
                
                # Handle unknown persons
                unknown_faces = [face for face in face_results if "UNKNOWN" in face['name']]
                if unknown_faces and not owner_detected:
                    if not self.tracking_unknown:
                        self.unknown_person_detected_time = time.time()
                        self.tracking_unknown = True
                        self.logger.warning("Unknown person detected - starting tracking")
                        
                        # Trigger intruder alert
                        self.trigger_security_alert("INTRUDER_DETECTED")
                    
                    # Track the first unknown face
                    self.track_unknown_person(processed_frame, unknown_faces[0]['bbox'])
                    
                    # Save evidence
                    self.data_storage.save_evidence_image(frame, "intruder_detected", "unknown")

                    # Queue gesture clip for external worker and poll results
                    try:
                        now = time.time()
                        if self.gesture_last_alert_time is None or (now - self.gesture_last_alert_time) > self.gesture_alert_cooldown:
                            # Save a short clip for gesture analysis (~2s)
                            recent_frames = list(self.gesture_buffer)[-int(self.fps * 2):]
                            clip_path = None
                            try:
                                # Save a temporary clip
                                clip_path = self.data_storage.save_evidence_clip(recent_frames, "gesture_candidate", "unknown", fps=self.fps)
                                if clip_path:
                                    queued = self.data_storage.queue_clip_for_gesture(clip_path, bbox=unknown_faces[0]['bbox'])
                                    if queued:
                                        self.logger.info(f"Clip queued for gesture worker: {queued}")
                            except Exception as e:
                                self.logger.error(f"Error saving or queuing gesture clip: {e}")
                    except Exception as e:
                        self.logger.error(f"Error preparing gesture clip: {e}")
                
                # Reset tracking if no unknown faces for a while
                elif self.tracking_unknown and not unknown_faces:
                    if time.time() - self.unknown_person_detected_time > 5:  # 5 seconds no detection
                        self.tracking_unknown = False
                        self.control_servo2(90)  # Reset tracking servo to center
                
                # Detect suspicious activity
                suspicious_detected = self.detect_suspicious_activity(frame)
                if suspicious_detected:
                    cv2.putText(processed_frame, "SUSPICIOUS ACTIVITY DETECTED!", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show gesture overlay if active
                try:
                    if self.gesture_overlay and time.time() < self.gesture_overlay_until:
                        cv2.putText(processed_frame, f"GESTURE: {self.gesture_overlay}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
                except Exception:
                    pass
                
                # Auto-lock door after 10 seconds if no owner present
                if self.door_unlocked and not owner_detected:
                    if time.time() - self.owner_detected_time > 10:
                        self.control_servo1(unlock=False)
                
                # Display system status
                status_text = f"Door: {'UNLOCKED' if self.door_unlocked else 'LOCKED'} | " \
                            f"Tracking: {'ON' if self.tracking_unknown else 'OFF'} | " \
                            f"Alerts: {'ACTIVE' if self.suspicious_activity_detected else 'NORMAL'}"
                
                cv2.putText(processed_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display controls help
                help_text = "Controls: Q=Quit, T=Train, L=Lock, S=Status"
                cv2.putText(processed_frame, help_text, (10, processed_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Show frame
                cv2.imshow('Security System - Camera Tracking', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.train_new_face()
                elif key == ord('l'):
                    self.control_servo1(unlock=False)
                elif key == ord('s'):
                    self.show_system_status()
            
            # Cleanup
            self.cleanup()
            
        except Exception as e:
            self.logger.error(f"Error in main security loop: {e}")
            self.cleanup()
    
    def train_new_face(self):
        """Train new face in a separate thread"""
        def train_thread():
            self.face_recognition.train_new_face("owner")
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        self.logger.info("Face training started in background...")
    
    def show_system_status(self):
        """Display current system status"""
        status = {
            "System Active": self.system_active,
            "Door Locked": not self.door_unlocked,
            "Tracking Unknown": self.tracking_unknown,
            "Suspicious Activity": self.suspicious_activity_detected,
            "Known Faces": self.face_recognition.get_known_faces(),
            "Servo 1 Position": self.servo1_position,
            "Servo 2 Position": self.servo2_position
        }
        
        self.logger.info("=== SYSTEM STATUS ===")
        for key, value in status.items():
            self.logger.info(f"{key}: {value}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.system_active = False
            
            if hasattr(self, 'cap'):
                self.cap.release()
            
            cv2.destroyAllWindows()
            
            # Close serial connection
            if self.serial_connection is not None:
                try:
                    self.serial_connection.close()
                    self.logger.info("Serial connection closed")
                except Exception as e:
                    self.logger.warning(f"Error closing serial connection: {e}")
                self.serial_connection = None
            
            # Ensure door is locked on exit
            if self.door_unlocked:
                self.control_servo1(unlock=False)
            
            self.logger.info("Security System stopped and cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Main execution
if __name__ == "__main__":
    security_system = CameraTrackingSystem()
    security_system.run_security_system()