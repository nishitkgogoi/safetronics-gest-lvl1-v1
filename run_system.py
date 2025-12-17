from camera_tracking_module import CameraTrackingSystem
import cv2
import sys
import argparse

# Force UTF-8 on stdout/stderr to prevent Windows console encoding errors when logging emojis
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # Older Pythons or redirected streams may not support reconfigure; ignore failures
    pass

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug overlays (optical flow, diagnostics)')
args = parser.parse_args()

print(" Starting Security System on Laptop...")
print("Press 'T' to train your face as owner")
print("Press 'Q' to quit")

security_system = CameraTrackingSystem(debug=args.debug)
security_system.run_security_system()