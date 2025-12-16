#!/usr/bin/env python3
"""Gesture worker: processes queued clips with MediaPipe in an isolated venv.

Monitors: evidence/gesture_queue/incoming
Writes results to: evidence/gesture_queue/results
Moves processed clips to: evidence/gesture_queue/processed
"""

import time
import os
import glob
import json
import shutil
import cv2
import logging

from gesture_recognition_module import GestureRecognizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INCOMING = 'evidence/gesture_queue/incoming'
PROCESSED = 'evidence/gesture_queue/processed'
RESULTS = 'evidence/gesture_queue/results'

# Minimum confidence to mark suspicious
SUSPICIOUS_CONF_THRESH = 0.25


def process_clip(clip_path, meta_path=None):
    try:
        logger.info(f"Processing clip: {clip_path}")
        cap = cv2.VideoCapture(clip_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        gr = GestureRecognizer()
        suspicious, label, conf, details = gr.analyze(frames)

        result = {
            'clip': clip_path,
            'label': label,
            'confidence': conf,
            'suspicious': bool(suspicious),
            'details': details,
            'timestamp': time.time()
        }

        # Write result JSON
        uid = os.path.splitext(os.path.basename(clip_path))[0]
        out_path = os.path.join(RESULTS, f"{uid}.json")
        with open(out_path, 'w') as f:
            json.dump(result, f)

        # Move clip and meta to processed
        try:
            shutil.move(clip_path, os.path.join(PROCESSED, os.path.basename(clip_path)))
        except Exception:
            pass
        if meta_path and os.path.exists(meta_path):
            try:
                shutil.move(meta_path, os.path.join(PROCESSED, os.path.basename(meta_path)))
            except Exception:
                pass

        logger.info(f"Wrote result: {out_path} -> suspicious={suspicious} label={label} conf={conf:.2f}")

    except Exception as e:
        logger.exception(f"Error processing clip {clip_path}: {e}")


def main(poll_interval=1.0):
    logger.info("Gesture worker started. Watching incoming folder for clips...")
    while True:
        try:
            for clip_path in glob.glob(os.path.join(INCOMING, '*.avi')):
                base = os.path.splitext(os.path.basename(clip_path))[0]
                meta_path = os.path.join(INCOMING, f"{base}.json")
                process_clip(clip_path, meta_path)
        except Exception as e:
            logger.exception(f"Worker loop error: {e}")
        time.sleep(poll_interval)


if __name__ == '__main__':
    os.makedirs(INCOMING, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    main()
