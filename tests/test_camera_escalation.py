import time
import types
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from camera_tracking_module import CameraTrackingSystem


def make_rapid_motion_frames(num_frames=6, frame_size=(240, 320)):
    frames = []
    xs = np.linspace(20, 260, num_frames).astype(int)
    for x in xs:
        f = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
        cv2.rectangle(f, (x, 60), (x + 50, 140), (255, 255, 255), -1)
        frames.append(f)
    return frames


def test_analyze_gesture_async_escalates_on_force_label(monkeypatch, tmp_path):
    ct = CameraTrackingSystem()

    # Patch data_storage to avoid file writes
    ct.data_storage.save_evidence_clip = MagicMock(return_value=str(tmp_path / 'clip.mp4'))
    ct.data_storage.log_security_event = MagicMock()

    # Patch control_servo1 and trigger_security_alert to monitor calls
    ct.control_servo1 = MagicMock()
    ct.trigger_security_alert = MagicMock()

    # Force the gesture recognizer to return a force_entry_suspected result
    fake_analyze = MagicMock(return_value=(True, 'force_entry_suspected', 0.92, {'suspicion_score': 0.92}))
    monkeypatch.setattr('gesture_recognition_module.GestureRecognizer.analyze', fake_analyze)

    frames = make_rapid_motion_frames()

    # Call the async analyzer (it runs in same thread here)
    ct._analyze_gesture_async(frames, bbox=None)

    # Expect door lock attempted
    ct.control_servo1.assert_called_with(unlock=False)
    # Expect escalation alert
    ct.trigger_security_alert.assert_called()
    args = ct.trigger_security_alert.call_args[0]
    assert 'FORCE_ENTRY_SUSPECTED' in args[0] or 'FORCE_ENTRY_SUSPECTED' == args[0]

    # Ensure event logged
    ct.data_storage.log_security_event.assert_called()
