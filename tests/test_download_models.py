import os
import tempfile
import shutil
from tools.download_mediapipe_models import download_models, MODELS_DIR


def test_download_models_from_file_urls(tmp_path):
    # create dummy files to act as source models
    src_pose = tmp_path / 'pose_dummy.tflite'
    src_hand = tmp_path / 'hand_dummy.tflite'
    src_pose.write_bytes(b'pose')
    src_hand.write_bytes(b'hand')

    pose_url = f'file://{src_pose}'
    hand_url = f'file://{src_hand}'

    # Ensure models dir is cleaned
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)

    res = download_models(pose_url, hand_url)
    assert 'pose' in res and os.path.exists(res['pose'])
    assert 'hand' in res and os.path.exists(res['hand'])
    # Clean up
    shutil.rmtree(MODELS_DIR)
