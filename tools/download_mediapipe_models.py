"""Helper to download mediapipe task models into a project `models/` directory.

Usage:
    python -m tools.download_mediapipe_models --pose <url-or-model-name> --hand <url-or-model-name>

Notes:
- This script accepts either a direct URL (http/https/file) or a model-name from MODEL_REGISTRY.
- For tests we support file:// URLs which copy the local file into models/.
"""
import os
import argparse
import shutil
import urllib.request
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, 'models')

# Registry of example model URLs (users should replace with proper asset locations)
MODEL_REGISTRY = {
    'pose_landmarker': "https://storage.googleapis.com/mediapipe-assets/pose_landmarker_full.task",
    'hand_landmarker': "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
}


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def download_url_to(path, url):
    # support file:// for local files (useful in tests)
    p = urlparse(url)
    if p.scheme == 'file':
        # Convert file:// URL to local path robustly (cross-platform)
        if p.netloc:
            src = f"//{p.netloc}{p.path}"
        else:
            src = p.path
        try:
            from urllib.request import url2pathname
            src = url2pathname(src)
        except Exception:
            pass
        shutil.copy(src, path)
        return
    with urllib.request.urlopen(url) as r, open(path, 'wb') as f:
        f.write(r.read())


def resolve_model_identifier(identifier):
    if not identifier:
        return None
    if identifier in MODEL_REGISTRY:
        return MODEL_REGISTRY[identifier]
    return identifier


def download_models(pose_id=None, hand_id=None):
    ensure_models_dir()
    results = {}
    if pose_id:
        url = resolve_model_identifier(pose_id)
        out = os.path.join(MODELS_DIR, os.path.basename(urlparse(url).path))
        download_url_to(out, url)
        results['pose'] = out
    if hand_id:
        url = resolve_model_identifier(hand_id)
        out = os.path.join(MODELS_DIR, os.path.basename(urlparse(url).path))
        download_url_to(out, url)
        results['hand'] = out
    return results


def _main():
    p = argparse.ArgumentParser()
    p.add_argument('--pose', help='Pose model identifier or URL')
    p.add_argument('--hand', help='Hand model identifier or URL')
    args = p.parse_args()

    res = download_models(args.pose, args.hand)
    for k, v in res.items():
        print(f'Downloaded {k} -> {v}')


if __name__ == '__main__':
    _main()
