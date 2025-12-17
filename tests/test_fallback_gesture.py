from gesture_recognition_module import GestureRecognizer
from tests.fixtures.clip_utils import generate_tool_swing, load_clip_frames


def test_fallback_mode_detects_force_like_motion():
    gr = GestureRecognizer(history_len=8, verbose=False)
    # Force the fallback mode to ensure test does not depend on mediapipe
    gr._use_mediapipe = False
    gr.pose = None
    gr.hands = None

    path = generate_tool_swing()
    frames = load_clip_frames(path)
    suspicious, label, conf, details = gr.analyze(frames)
    assert 'motion_mag' in details
    # tool_swing should be seen as suspicious by motion heuristics
    assert suspicious is True
    assert label == 'force_entry_suspected' or details['motion_mag'] > 0.05
    gr.close()
