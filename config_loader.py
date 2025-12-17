import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'security_config.json')


def load_config(path=None):
    p = path or CONFIG_PATH
    try:
        with open(p, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            return cfg
    except Exception:
        # Return defaults if config missing or invalid
        return {
            'gesture': {
                'motion_only_threshold': 0.06,
                'motion_weight': 0.3,
                'hands_weight': 0.3,
                'raised_weight': 0.2,
                'speed_weight': 0.2,
                'combined_threshold': 0.45,
                'motion_min_for_combined': 0.08,
                # Mediapipe tasks integration (optional): provide model file paths to enable
                'use_tasks': False,
                'pose_model_path': None,
                'hand_model_path': None,
                # If true, try to auto-discover packaged mediapipe model files
                'auto_discover_tasks': True
            }
        }


if __name__ == '__main__':
    print('Loaded config:', load_config())
