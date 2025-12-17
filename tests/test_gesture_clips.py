import sys
import types
import pytest
import numpy as np

# Inject a lightweight fake mediapipe like other tests do
fake_mp = types.SimpleNamespace()
fake_mp.solutions = types.SimpleNamespace()
class _FakeRes:
    def __init__(self):
        self.pose_landmarks = None
        self.multi_hand_landmarks = None
class _FakePose:
    def __init__(self,*a,**k): pass
    def process(self,_): return _FakeRes()
    def close(self): pass
class _FakeHands:
    def __init__(self,*a,**k): pass
    def process(self,_): return _FakeRes()
    def close(self): pass
fake_mp.solutions.pose = types.SimpleNamespace(Pose=_FakePose)
fake_mp.solutions.hands = types.SimpleNamespace(Hands=_FakeHands)
sys.modules['mediapipe'] = fake_mp

from gesture_recognition_module import GestureRecognizer
from tests.fixtures.clip_utils import generate_all_clips, load_clip_frames


@pytest.fixture(scope='module', autouse=True)
def ensure_clips():
    generate_all_clips()


@pytest.mark.parametrize(
    "clip_name,expect_suspicious",
    [
        ("force_impact", True),
        ("repetitive_bangs", True),
        ("tool_swing", True),  # tool swinging near door -> suspicious
        ("slow_walk", False),
        ("dim_lighting", True),
        ("occlusion", True),
        ("multi_person", False),
        ("crowd_crossing", False),
        ("overlap_tool_person", True),
    ],
)
def test_clips_trigger_expected_responses(clip_name, expect_suspicious):
    clips = generate_all_clips()
    path = clips[clip_name]
    frames = load_clip_frames(path)

    gr = GestureRecognizer(history_len=16, wrist_speed_thresh=0.01, verbose=False)
    suspicious, label, conf, details = gr.analyze(frames, bbox=None)

    # Basic expectations: force-like and bangs flagged suspicious; slow walk not
    assert suspicious is expect_suspicious
    # Ensure details contain motion metrics
    assert 'motion_mag' in details
    assert 'suspicion_score' in details
    # If suspicious, confidence should be > 0.05
    if expect_suspicious:
        assert conf > 0.05
