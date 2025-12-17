import os
import cv2
import numpy as np

CLIP_DIR = os.path.join(os.path.dirname(__file__), 'clips')


def ensure_clip_dir():
    os.makedirs(CLIP_DIR, exist_ok=True)


def _write_clip(path, frames, fps=10):
    # Use MJPG codec for wide compatibility
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def generate_force_impact(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'force_impact.avi')
    if os.path.exists(path):
        return path

    frames = []
    w, h = 320, 240
    # Create an abrupt fast motion across very few frames
    positions = [10, 100, 240, 20, 260]
    for x in positions:
        f = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(f, (x, 80), (x + 60, 160), (255, 255, 255), -1)
        frames.append(f)
    _write_clip(path, frames, fps=10)
    return path


def generate_slow_walk(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'slow_walk.avi')
    if os.path.exists(path):
        return path

    frames = []
    w, h = 320, 240
    xs = np.linspace(10, 260, 30).astype(int)
    for x in xs:
        f = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(f, (x, 80), (x + 40, 140), (255, 255, 255), -1)
        frames.append(f)
    _write_clip(path, frames, fps=10)
    return path


def generate_repetitive_bangs(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'repetitive_bangs.avi')
    if os.path.exists(path):
        return path

    frames = []
    w, h = 320, 240
    for i in range(24):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        # Bang every 2 frames
        if i % 2 == 0:
            cv2.rectangle(f, (150, 60), (210, 140), (255, 255, 255), -1)
        frames.append(f)
    _write_clip(path, frames, fps=10)
    return path


def generate_tool_swing(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'tool_swing.avi')
    # Always regenerate to pick up changes and ensure strong-motion clip is written
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    frames = []
    w, h = 320, 240
    # Use fewer frames and larger rectangle to create bigger frame-to-frame differences
    xs = np.linspace(40, 220, 8).astype(int)
    for i, x in enumerate(xs):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        # Larger rectangle to mimic a swinging tool (bigger area changes)
        cv2.rectangle(f, (x, 70), (x + 140, 160), (255, 255, 255), -1)
        frames.append(f)
    # Add a reverse swing for more abruptness
    xs_rev = xs[::-1]
    for x in xs_rev:
        f = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(f, (x, 70), (x + 140, 160), (255, 255, 255), -1)
        frames.append(f)
    _write_clip(path, frames, fps=10)
    return path


def generate_dim_lighting(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'dim_lighting.avi')
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    frames = []
    w, h = 320, 240
    # fewer frames with larger jumps and slightly brighter rectangle so motion is detectable even in dim scenes
    xs = [10, 100, 220, 40, 260]
    for x in xs:
        f = np.zeros((h, w, 3), dtype=np.uint8)
        # dim lighting: darker background, but rectangle has moderate brightness
        cv2.rectangle(f, (0, 0), (w, h), (15, 15, 15), -1)
        cv2.rectangle(f, (x, 80), (x + 80, 160), (120, 120, 120), -1)
        frames.append(f)
    _write_clip(path, frames, fps=10)
    return path


def generate_occlusion(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'occlusion.avi')
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    frames = []
    w, h = 320, 240
    # create abrupt movement in middle with partial occlusion bar not fully covering rectangle
    xs = [10, 40, 160, 240, 40]
    for x in xs:
        f = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(f, (x, 80), (x + 120, 170), (255, 255, 255), -1)
        # occluding bar only covers left side
        cv2.rectangle(f, (0, 110), (120, 170), (0, 0, 0), -1)
        frames.append(f)
    _write_clip(path, frames, fps=10)
    return path


def generate_multi_person(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'multi_person.avi')
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    frames = []
    w, h = 320, 240
    # Two people crossing paths (mild motion) - expect non-suspicious
    xs1 = np.linspace(10, 200, 12).astype(int)
    xs2 = np.linspace(200, 40, 12).astype(int)
    for x1, x2 in zip(xs1, xs2):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(f, (x1, 70), (x1 + 40, 130), (255, 255, 255), -1)
        cv2.rectangle(f, (x2, 100), (x2 + 40, 160), (255, 255, 255), -1)
        frames.append(f)
    _write_clip(path, frames, fps=10)
    return path


def generate_crowd_crossing(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'crowd_crossing.avi')
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    frames = []
    w, h = 320, 240
    # Many small people crossing (dense, but slow)
    for i in range(16):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(5):
            x = (i * 10 + j * 30) % (w - 40)
            cv2.rectangle(f, (x, 60 + j * 10), (x + 25, 90 + j * 10), (200, 200, 200), -1)
        frames.append(f)
    _write_clip(path, frames, fps=10)
    return path


def generate_overlap_tool_person(path=None):
    ensure_clip_dir()
    if path is None:
        path = os.path.join(CLIP_DIR, 'overlap_tool_person.avi')
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    frames = []
    w, h = 320, 240
    # Person plus a tool swing overlapping — more likely suspicious
    # Make the tool larger and swing faster to create a stronger motion signal
    xs_person = np.linspace(10, 200, 6).astype(int)
    xs_tool = np.linspace(220, 20, 6).astype(int)
    for xp, xt in zip(xs_person, xs_tool):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(f, (xp, 80), (xp + 40, 140), (255, 255, 255), -1)
        # Bigger tool area and taller to create more frame-to-frame differences
        cv2.rectangle(f, (xt, 60), (xt + 160, 160), (255, 255, 255), -1)
        frames.append(f)
    # Add a reverse swing to simulate return motion
    xs_tool_rev = xs_tool[::-1]
    for xp, xt in zip(xs_person, xs_tool_rev):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(f, (xp, 80), (xp + 40, 140), (255, 255, 255), -1)
        cv2.rectangle(f, (xt, 60), (xt + 160, 160), (255, 255, 255), -1)
        frames.append(f)
    _write_clip(path, frames, fps=12)
    return path



def generate_all_clips():
    return {
        'force_impact': generate_force_impact(),
        'slow_walk': generate_slow_walk(),
        'repetitive_bangs': generate_repetitive_bangs(),
        'tool_swing': generate_tool_swing(),
        'dim_lighting': generate_dim_lighting(),
        'occlusion': generate_occlusion(),
        'multi_person': generate_multi_person(),
        'crowd_crossing': generate_crowd_crossing(),
        'overlap_tool_person': generate_overlap_tool_person(),
    }


def load_clip_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    return frames
