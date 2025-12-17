from config_loader import load_config


def test_config_loads_and_contains_gesture():
    cfg = load_config()
    assert 'gesture' in cfg
    g = cfg['gesture']
    assert 'motion_only_threshold' in g
    assert 'combined_threshold' in g
