from camera_tracking_module import CameraTrackingSystem
import cv2
import sys

# Force UTF-8 on stdout/stderr to prevent Windows console encoding errors when logging emojis
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # Older Pythons or redirected streams may not support reconfigure; ignore failures
    pass

print(" Starting Security System on Laptop...")
print("Press 'T' to train your face as owner")
print("Press 'Q' to quit")

security_system = CameraTrackingSystem()
security_system.run_security_system()