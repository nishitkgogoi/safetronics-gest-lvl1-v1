import os
from gesture_recognition_module import GestureRecognizer


def test_tasks_init_fails_gracefully_with_missing_models():
    cfg = {
        'gesture': {
            'use_tasks': True,
            'pose_model_path': 'nonexistent_pose.tflite',
            'hand_model_path': 'nonexistent_hand.tflite'
        }
    }
    gr = GestureRecognizer(history_len=4, verbose=False, config=cfg)
    # If model files don't exist it'll fall back to non-tasks mode
    assert getattr(gr, '_use_tasks', False) is False
    gr.close()
