import sys
import types
import numpy as np
import cv2
import pytest

# Create a fake mediapipe module to avoid heavy dependency in unit tests
fake_mp = types.SimpleNamespace()
# Provide minimal solutions namespace
fake_mp.solutions = types.SimpleNamespace()
# Provide dummy Pose and Hands classes with process() returning objects with no landmarks
class _FakeRes:
    def __init__(self):
        self.pose_landmarks = None
        self.multi_hand_landmarks = None

class _FakePose:
    def __init__(self, *args, **kwargs):
        pass
    def process(self, _):
        return _FakeRes()
    def close(self):
        pass

class _FakeHands:
    def __init__(self, *args, **kwargs):
        pass
    def process(self, _):
        return _FakeRes()
    def close(self):
        pass

fake_mp.solutions.pose = types.SimpleNamespace(Pose=_FakePose)
fake_mp.solutions.hands = types.SimpleNamespace(Hands=_FakeHands)

# Inject fake mediapipe module
sys.modules['mediapipe'] = fake_mp

from gesture_recognition_module import GestureRecognizer


def make_moving_frames(num_frames=6, frame_size=(240, 320), start_x=20, end_x=220):
    """Generate a short sequence where a white square moves rapidly across frames."""
    frames = []
    xs = np.linspace(start_x, end_x, num_frames).astype(int)
    for x in xs:
        f = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
        cv2.rectangle(f, (x, 80), (x + 40, 140), (255, 255, 255), -1)
        frames.append(f)
    return frames


def test_force_entry_detected_by_motion_and_score():
    gr = GestureRecognizer(history_len=8, wrist_speed_thresh=0.001, verbose=False)

    frames = make_moving_frames(num_frames=8, start_x=10, end_x=270)

    suspicious, label, conf, details = gr.analyze(frames, bbox=None)

    assert suspicious is True, "Moving sequence should be flagged as suspicious"
    # Either the label is the dedicated 'force_entry_suspected' or the combined suspicion_score meets the lower threshold
    assert (label == 'force_entry_suspected') or (details.get('suspicion_score', 0) >= 0.45)
    assert 'motion_mag' in details and details['motion_mag'] > 0
    assert 'suspicion_score' in details


def test_non_suspicious_static_sequence():
    gr = GestureRecognizer(history_len=8, wrist_speed_thresh=0.01, verbose=False)
    # static frames (no significant motion)
    f = np.zeros((240, 320, 3), dtype=np.uint8)
    frames = [f.copy() for _ in range(6)]

    suspicious, label, conf, details = gr.analyze(frames, bbox=None)

    assert suspicious is False
    assert details.get('motion_mag', 0) == 0 or details.get('suspicion_score', 0) < 0.2
