"""Headless smoke test for the full security system.

- Simulates a camera using frames from the synthetic clip generator.
- Patches OpenCV GUI calls to avoid requiring a display.
- Runs the main loop until frames are exhausted and prints/logs results.
"""
import logging
import time
import cv2
import os
import sys
# Ensure project root is on sys.path when running as script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force UTF-8 for stdout/stderr to avoid Windows cp1252 logging errors when printing emojis
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from camera_tracking_module import CameraTrackingSystem
from tests.fixtures.clip_utils import generate_force_impact, load_clip_frames

logging.basicConfig(level=logging.INFO)

# Patch OpenCV GUI functions to be headless
_cv2_imshow = cv2.imshow
_cv2_waitKey = cv2.waitKey
_cv2_destroyAllWindows = cv2.destroyAllWindows
cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda *a, **k: None

class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.opened = True
    def isOpened(self):
        return True
    def read(self):
        if self.idx >= len(self.frames):
            return False, None
        f = self.frames[self.idx]
        self.idx += 1
        return True, f
    def release(self):
        self.opened = False


def run_headless_smoke(max_loops=None):
    # Load a strong motion clip to exercise gesture & motion paths
    clip_path = generate_force_impact()
    frames = load_clip_frames(clip_path)

    # Make more frames by repeating to reach a few seconds
    frames = frames * 6

    # Initialize system and inject fake capture
    sys = CameraTrackingSystem()

    def fake_init_camera():
        sys.cap = FakeCapture(frames)
        return True

    sys.initialize_camera = fake_init_camera

    # Set shorter cooldowns for test speed
    sys.alert_cooldown = 2
    sys.gesture_alert_cooldown = 2

    start = time.time()
    sys.run_security_system()
    duration = time.time() - start
    print(f"Headless smoke test finished in {duration:.2f}s")


if __name__ == '__main__':
    run_headless_smoke()

# Restore patched functions on module exit
cv2.imshow = _cv2_imshow
cv2.waitKey = _cv2_waitKey
cv2.destroyAllWindows = _cv2_destroyAllWindows
