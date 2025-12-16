from camera_tracking_module import CameraTrackingSystem
import cv2

print(" Starting Security System on Laptop...")
print("Press 'T' to train your face as owner")
print("Press 'Q' to quit")

security_system = CameraTrackingSystem()
security_system.run_security_system()