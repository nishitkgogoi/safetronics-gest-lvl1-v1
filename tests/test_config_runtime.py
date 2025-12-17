import os
from gesture_recognition_module import GestureRecognizer
from config_loader import load_config


def test_reload_config_changes_threshold():
    cfg = load_config()
    gr = GestureRecognizer(history_len=4, verbose=False)
    orig = gr.motion_only_threshold
    new_cfg = cfg.copy()
    if 'gesture' not in new_cfg:
        new_cfg['gesture'] = {}
    new_cfg['gesture']['motion_only_threshold'] = 0.001
    gr.reload_config(new_cfg)
    assert gr.motion_only_threshold == 0.001
    # restore
    gr.reload_config(cfg)
    assert gr.motion_only_threshold == float(cfg.get('gesture', {}).get('motion_only_threshold', 0.06))
    gr.close()
