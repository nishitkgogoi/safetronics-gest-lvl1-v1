import cv2
import numpy as np
import logging
from collections import deque


class GestureRecognizer:
    """A lightweight MediaPipe-based gesture recognizer PoC.

    Features (rule-based):
    - raised_arms: both wrists above respective shoulders
    - hands_on_face: hand close to nose/eyes
    - rapid_hand_movement: large wrist displacement across frames

    Usage:
        gr = GestureRecognizer()
        suspicious, label, conf, details = gr.analyze(frames, bbox=None)

    Note: This is a PoC for quick integration. For improved accuracy, consider
    collecting labeled clips and training a classifier on keypoint sequences.
    """

    def __init__(self, history_len=24, wrist_speed_thresh=0.06, verbose=False, config=None):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.NullHandler())

        # Load configuration (allows runtime tuning)
        try:
            if config is None:
                from config_loader import load_config
                cfg = load_config()
            else:
                cfg = config
            gcfg = cfg.get('gesture', {})
        except Exception:
            gcfg = {}

        # Configurable thresholds and weights (with sensible defaults)
        self.motion_only_threshold = float(gcfg.get('motion_only_threshold', 0.06))
        self.motion_weight = float(gcfg.get('motion_weight', 0.3))
        self.hands_weight = float(gcfg.get('hands_weight', 0.3))
        self.raised_weight = float(gcfg.get('raised_weight', 0.2))
        self.speed_weight = float(gcfg.get('speed_weight', 0.2))
        self.combined_threshold = float(gcfg.get('combined_threshold', 0.45))
        self.motion_min_for_combined = float(gcfg.get('motion_min_for_combined', 0.08))

        # Lazy import of mediapipe to allow running project without it until used
        self._use_mediapipe = False
        try:
            import mediapipe as mp
            # Validate expected API
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose') and hasattr(mp.solutions, 'hands'):
                self._use_mediapipe = True
                # Store references
                self.mp = mp
                self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                                                   min_detection_confidence=0.4,
                                                   min_tracking_confidence=0.4)
                self.hands = mp.solutions.hands.Hands(static_image_mode=False,
                                                      max_num_hands=2,
                                                      min_detection_confidence=0.4,
                                                      min_tracking_confidence=0.4)
            else:
                # Mediapipe installed but missing expected attributes — fall back
                self._use_mediapipe = False
        except Exception:
            # Mediapipe not available — use fallback heuristics
            self._use_mediapipe = False

        # If not using mediapipe, set placeholders for compatibility
        if not self._use_mediapipe:
            self.mp = None
            self.pose = None
            self.hands = None
            self.logger.info('GestureRecognizer: running in fallback mode (mediapipe unavailable)')

        # Temporal history of keypoints (normalized coordinates)
        self.history_len = history_len

        # Thresholds
        self.wrist_history = deque(maxlen=history_len)  # each entry: list of (lx,ly,rx,ry) or None

        self.wrist_speed_thresh = wrist_speed_thresh  # normalized units per frame
        # Exponential moving averages for smoothing
        self.motion_ema = 0.0
        self.confidence_ema = 0.0
        self.ema_alpha = float(gcfg.get('ema_alpha', 0.4))
        self.verbose = verbose

        # Config file watch attributes
        self._config_path = None
        try:
            import os
            import json
            self._config_path = os.path.join(os.path.dirname(__file__), 'config', 'security_config.json')
        except Exception:
            self._config_path = None
        self._config_mtime = None
        self._config_watcher_thread = None
        self._stop_watcher = False
        self.config_watch_interval_seconds = float(gcfg.get('config_watch_interval_seconds', 1))

        # Attempt to initialize Mediapipe Task-based landmarkers if requested via config
        try:
            self._use_tasks = False
            # config keys may have been loaded into gcfg
            pose_model = gcfg.get('pose_model_path')
            hand_model = gcfg.get('hand_model_path')
            use_tasks_cfg = bool(gcfg.get('use_tasks', False))
            # Only try to init tasks if the user opted in
            if use_tasks_cfg and hasattr(__import__('mediapipe'), 'tasks'):
                try:
                    self._init_tasks_landmarkers(pose_model, hand_model)
                except Exception as e:
                    self.logger.warning(f'Failed to initialize Mediapipe tasks landmarkers: {e} — falling back to other modes')
                    self._use_tasks = False
        except Exception:
            pass

        # Start config watcher (hot-reload of thresholds)
        try:
            self._start_config_watcher()
        except Exception:
            pass

    def _norm_to_pixel(self, lm, frame_shape):
        h, w = frame_shape[:2]
        return int(lm.x * w), int(lm.y * h)

    def analyze(self, frames, bbox=None):
        """Analyze a short list of BGR frames and return gesture suspicion.

        Args:
            frames: list of numpy BGR frames (np.ndarray)
            bbox: optional bounding box (x,y,w,h) to crop around person for better focus

        Returns:
            suspicious (bool), label (str), confidence (0..1 float), details (dict)
        """
        raised_votes = 0
        hands_on_face_votes = 0
        rapid_motion_votes = 0
        frames_with_data = 0

        # Reset wrist history for this clip
        self.wrist_history.clear()

        # Pre-compute optical flow-based motion magnitude across frame pairs for stronger motion cue
        motion_vals = []
        try:
            prev_gray_of = None
            for frame in frames:
                small = cv2.resize(frame, (160, 120))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray_of is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray_of, gray, None,
                                                        0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    # Normalize by frame diagonal to get scale-invariant measure
                    norm = np.hypot(gray.shape[0], gray.shape[1])
                    motion_vals.append(float(np.mean(mag)) / (norm + 1e-6))
                prev_gray_of = gray
        except Exception:
            motion_vals = []

        # Use the maximum optical-flow-based motion as a strong indicator of abrupt movement
        motion_mag_of = float(np.max(motion_vals)) if motion_vals else 0.0

        # Now process per-frame pose/heuristic keypoints
        for frame in frames:
            proc = frame
            if bbox is not None:
                x, y, w, h = bbox
                # Add padding and clamp
                pad_x = int(w * 0.25)
                pad_y = int(h * 0.4)
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(frame.shape[1], x + w + pad_x)
                y1 = min(frame.shape[0], y + h + pad_y)
                proc = frame[y0:y1, x0:x1]

            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

            # Default wrist coords (normalized relative to proc crop)
            left_wrist = None
            right_wrist = None
            shoulders = None
            nose = None

            # --- Tasks API path (if initialized) ---
            if getattr(self, '_use_tasks', False):
                try:
                    # Use temporary file + MediaPipe Image wrapper for compatibility
                    import tempfile
                    from mediapipe.tasks.python.vision.core.image import Image as MPImage
                    tf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    tp = tf.name
                    tf.close()
                    cv2.imwrite(tp, proc)

                    # Pose
                    lw_xy = None
                    rw_xy = None
                    if hasattr(self, '_tasks_pose') and self._tasks_pose:
                        try:
                            img = MPImage.create_from_file(tp)
                            res = self._tasks_pose.detect(img)
                            # try to extract landmarks using common attribute names
                            pl = None
                            for attr in ('pose_landmarks', 'landmarks'):
                                pl = getattr(res, attr, None)
                                if pl:
                                    break
                            if pl:
                                # pl may be a list of landmarks; select first set
                                if isinstance(pl, (list, tuple)):
                                    pl0 = pl[0]
                                else:
                                    pl0 = pl
                                # Try to get typical indexes
                                try:
                                    left = pl0[15]  # LEFT_WRIST index in many schemes
                                    right = pl0[16]
                                    lw_xy = (left.x, left.y)
                                    rw_xy = (right.x, right.y)
                                except Exception:
                                    # fallback: take two most extreme xs
                                    coords = [(lm.x, lm.y) for lm in pl0]
                                    coords_sorted = sorted(coords, key=lambda x: x[0])
                                    if coords_sorted:
                                        if len(coords_sorted) == 1:
                                            if coords_sorted[0][0] < 0.5:
                                                lw_xy = coords_sorted[0]
                                            else:
                                                rw_xy = coords_sorted[0]
                                        else:
                                            lw_xy, rw_xy = coords_sorted[0], coords_sorted[-1]
                        except Exception:
                            pass

                    # Hands
                    if hasattr(self, '_tasks_hand') and self._tasks_hand:
                        try:
                            img = MPImage.create_from_file(tp)
                            hres = self._tasks_hand.detect(img)
                            # HandLandmarkerResult may have hand_landmarks
                            hl = getattr(hres, 'hand_landmarks', None)
                            if hl:
                                # use first hand(s)
                                centers = []
                                for hand in hl:
                                    if hand:
                                        # wrist often index 0
                                        try:
                                            wlm = hand[0]
                                            centers.append((wlm.x, wlm.y))
                                        except Exception:
                                            pass
                                if len(centers) == 1:
                                    if centers[0][0] < 0.5:
                                        lw_xy = centers[0]
                                    else:
                                        rw_xy = centers[0]
                                elif len(centers) >= 2:
                                    centers_sorted = sorted(centers, key=lambda x: x[0])
                                    lw_xy, rw_xy = centers_sorted[0], centers_sorted[-1]
                        except Exception:
                            pass

                    # Clean up temp file
                    try:
                        import os
                        os.remove(tp)
                    except Exception:
                        pass

                    # approximate shoulders and nose if not found
                    if not shoulders:
                        try:
                            left_sh = type('S', (), {'x': 0.3, 'y': 0.4})
                            right_sh = type('S', (), {'x': 0.7, 'y': 0.4})
                            shoulders = (left_sh, right_sh)
                        except Exception:
                            shoulders = None
                    nose = None
                    self.wrist_history.append((lw_xy, rw_xy, nose, shoulders))
                    # move to next frame
                    continue
                except Exception:
                    # If anything fails in tasks path, fall back to other methods for this frame
                    pass

            if self._use_mediapipe:
                pose_res = self.pose.process(rgb)
                hands_res = self.hands.process(rgb)

                if pose_res.pose_landmarks:
                    lm = pose_res.pose_landmarks.landmark
                    # landmark indices per Mediapipe naming
                    try:
                        left_wrist = lm[self.mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                        right_wrist = lm[self.mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
                        left_sh = lm[self.mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                        right_sh = lm[self.mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                        nose = lm[self.mp.solutions.pose.PoseLandmark.NOSE]
                        shoulders = (left_sh, right_sh)
                    except Exception:
                        shoulders = None

                # Gather hand landmarks (choose averaged wrist points from hands API if present)
                hand_wrist_coords = []
                if hands_res.multi_hand_landmarks:
                    for hand_landmarks in hands_res.multi_hand_landmarks:
                        # wrist is landmark 0 in hands model
                        wlm = hand_landmarks.landmark[0]
                        hand_wrist_coords.append(wlm)

                # Convert to normalized coordinates relative to proc frame
                lw_xy = None
                rw_xy = None
                if left_wrist is not None:
                    lw_xy = (left_wrist.x, left_wrist.y)
                if right_wrist is not None:
                    rw_xy = (right_wrist.x, right_wrist.y)

                # Prefer hands API wrist if available (more robust for hand location)
                if len(hand_wrist_coords) == 1:
                    hw = hand_wrist_coords[0]
                    # Put it into either left or right based on x distance
                    if lw_xy is None:
                        lw_xy = (hw.x, hw.y)
                    elif rw_xy is None:
                        rw_xy = (hw.x, hw.y)
                elif len(hand_wrist_coords) == 2:
                    # assign by x: smaller x = left
                    if hand_wrist_coords[0].x < hand_wrist_coords[1].x:
                        lw_xy = (hand_wrist_coords[0].x, hand_wrist_coords[0].y)
                        rw_xy = (hand_wrist_coords[1].x, hand_wrist_coords[1].y)
                    else:
                        lw_xy = (hand_wrist_coords[1].x, hand_wrist_coords[1].y)
                        rw_xy = (hand_wrist_coords[0].x, hand_wrist_coords[0].y)

                self.wrist_history.append((lw_xy, rw_xy, nose, shoulders))
            else:
                # Fallback: simple bright-contour heuristic to estimate 'hands'
                gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                centers = []
                for c in contours:
                    M = cv2.moments(c)
                    if M.get('m00', 0) != 0:
                        cx = (M['m10'] / M['m00']) / proc.shape[1]
                        cy = (M['m01'] / M['m00']) / proc.shape[0]
                        centers.append((cx, cy))
                lw_xy = None
                rw_xy = None
                if len(centers) == 1:
                    if centers[0][0] < 0.5:
                        lw_xy = centers[0]
                    else:
                        rw_xy = centers[0]
                elif len(centers) == 2:
                    centers_sorted = sorted(centers, key=lambda x: x[0])
                    lw_xy, rw_xy = centers_sorted[0], centers_sorted[1]

                # approximate shoulders and nose relative to frame
                shoulders = None
                try:
                    left_sh = type('S', (), {'x': 0.3, 'y': 0.4})
                    right_sh = type('S', (), {'x': 0.7, 'y': 0.4})
                    shoulders = (left_sh, right_sh)
                except Exception:
                    shoulders = None

                nose = None
                self.wrist_history.append((lw_xy, rw_xy, nose, shoulders))

            # Evaluate per-frame cues
            per_frame_raised = False
            per_frame_hands_on_face = False

            if shoulders is not None:
                # compare wrist y to shoulder y: lower y value = higher in image
                try:
                    if lw_xy is not None and rw_xy is not None:
                        per_frame_raised = (lw_xy[1] < shoulders[0].y - 0.02) and (rw_xy[1] < shoulders[1].y - 0.02)
                    elif lw_xy is not None:
                        per_frame_raised = (lw_xy[1] < shoulders[0].y - 0.02)
                    elif rw_xy is not None:
                        per_frame_raised = (rw_xy[1] < shoulders[1].y - 0.02)
                except Exception:
                    per_frame_raised = False

            # Hands near face
            if nose is not None and (lw_xy is not None or rw_xy is not None):
                nose_xy = (nose.x, nose.y)
                close_thresh = 0.12  # normalized units
                if lw_xy is not None:
                    d = np.hypot(lw_xy[0] - nose_xy[0], lw_xy[1] - nose_xy[1])
                    if d < close_thresh:
                        per_frame_hands_on_face = True
                if rw_xy is not None:
                    d = np.hypot(rw_xy[0] - nose_xy[0], rw_xy[1] - nose_xy[1])
                    if d < close_thresh:
                        per_frame_hands_on_face = True

            if per_frame_raised:
                raised_votes += 1
            if per_frame_hands_on_face:
                hands_on_face_votes += 1

            frames_with_data += 1

        # Evaluate wrist speed across collected history
        rapid_count = 0
        for i in range(1, len(self.wrist_history)):
            prev = self.wrist_history[i - 1]
            cur = self.wrist_history[i]
            for j in range(2):  # left=0 right=1
                p = prev[j]
                c = cur[j]
                if p is not None and c is not None:
                    dist = np.hypot(c[0] - p[0], c[1] - p[1])
                    # Adjust threshold based on expected frame-to-frame motion (less strict)
                    if dist > self.wrist_speed_thresh:
                        rapid_count += 1

        if len(self.wrist_history) > 1:
            speed_ratio = rapid_count / (len(self.wrist_history) - 1)
        else:
            speed_ratio = 0.0

        # Combine optical-flow motion magnitude with the prior frame-diff heuristic
        try:
            # fallback simple diff-based motion as secondary signal
            prev_gray = None
            diff_vals = []
            for frame in frames:
                small = cv2.resize(frame, (160, 120))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    diff_vals.append(float(diff.mean()) / 255.0)
                prev_gray = gray
            motion_mag_diff = float(np.max(diff_vals)) if diff_vals else 0.0
        except Exception:
            motion_mag_diff = 0.0

        # Final motion magnitude: weighted combination (optical flow prioritized)
        motion_mag = max(motion_mag_of, motion_mag_diff)

        # Smooth motion via EMA to avoid spurious spikes
        try:
            self.motion_ema = (self.ema_alpha * motion_mag) + (1.0 - self.ema_alpha) * self.motion_ema
        except Exception:
            self.motion_ema = motion_mag

        # Decision logic
        suspicious = False
        label = "none"
        # Compute normalized vote fractions
        raised_frac = (raised_votes / frames_with_data) if frames_with_data else 0
        hands_frac = (hands_on_face_votes / frames_with_data) if frames_with_data else 0

        # If motion magnitude is large (optical flow or diff), treat it as a strong indicator
        # of a forceful/abrupt action. This provides sensitivity when pose/hand keypoints are not available.
        suspicion_score = 0.0
        # Use EMA-smoothed motion for thresholding to reduce false positives
        motion_for_decision = max(motion_mag, self.motion_ema)
        if motion_for_decision >= self.motion_only_threshold:
            suspicious = True
            label = "force_entry_suspected"
            confidence = min(0.95, min(0.99, motion_for_decision * 5.0))
        elif (
            self.hands_weight * hands_frac +
            self.raised_weight * raised_frac +
            self.speed_weight * speed_ratio +
            self.motion_weight * motion_mag
        ) >= self.combined_threshold and motion_mag > self.motion_min_for_combined:
            # Combined suspicion score (weights configurable)
            suspicion_score = (
                self.hands_weight * hands_frac +
                self.raised_weight * raised_frac +
                self.speed_weight * speed_ratio +
                self.motion_weight * motion_for_decision
            )
            suspicious = True
            label = "force_entry_suspected"
            confidence = min(0.99, suspicion_score)
        elif hands_frac >= 0.25:
            suspicious = True
            label = "hands_covering_face"
            confidence = min(0.95, hands_frac + 0.3)
        elif raised_frac >= 0.25:
            suspicious = True
            label = "raised_arms"
            confidence = min(0.9, raised_frac + 0.2)
        elif speed_ratio >= 0.3:
            suspicious = True
            label = "rapid_hand_movement"
            confidence = min(0.85, speed_ratio + 0.15)
        else:
            confidence = max(0.0, max(raised_frac, hands_frac, speed_ratio, motion_mag))

        details = {
            'frames_analyzed': frames_with_data,
            'raised_frac': raised_frac,
            'hands_frac': hands_frac,
            'speed_ratio': speed_ratio,
            'motion_mag': motion_mag,
            'motion_mag_raw': motion_mag,
            'motion_mag_ema': self.motion_ema,
            'suspicion_score': suspicion_score,
            'history_len': len(self.wrist_history)
        }

        if self.verbose:
            self.logger.info(f"Gesture analysis -> suspicious={suspicious}, label={label}, conf={confidence}, details={details}")

        # Smooth returned confidence to avoid flicker on overlays
        try:
            self.confidence_ema = (self.ema_alpha * float(confidence)) + (1.0 - self.ema_alpha) * self.confidence_ema
            smooth_conf = float(self.confidence_ema)
        except Exception:
            smooth_conf = float(confidence)

        return suspicious, label, float(smooth_conf), details

    def close(self):
        try:
            if self.pose:
                self.pose.close()
            if self.hands:
                self.hands.close()
        except Exception:
            pass
        # stop config watcher
        try:
            self._stop_config_watcher()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def _start_config_watcher(self):
        import threading, os, time
        if not self._config_path:
            return
        try:
            if self._config_watcher_thread and self._config_watcher_thread.is_alive():
                return

            def watcher():
                try:
                    while not self._stop_watcher:
                        try:
                            m = os.path.getmtime(self._config_path)
                            if self._config_mtime is None:
                                self._config_mtime = m
                            elif m != self._config_mtime:
                                self._config_mtime = m
                                # Reload config
                                try:
                                    from config_loader import load_config
                                    cfg = load_config()
                                    g = cfg.get('gesture', {})
                                    # update attributes
                                    self.motion_only_threshold = float(g.get('motion_only_threshold', self.motion_only_threshold))
                                    self.motion_weight = float(g.get('motion_weight', self.motion_weight))
                                    self.hands_weight = float(g.get('hands_weight', self.hands_weight))
                                    self.raised_weight = float(g.get('raised_weight', self.raised_weight))
                                    self.speed_weight = float(g.get('speed_weight', self.speed_weight))
                                    self.combined_threshold = float(g.get('combined_threshold', self.combined_threshold))
                                    self.motion_min_for_combined = float(g.get('motion_min_for_combined', self.motion_min_for_combined))
                                    self.config_watch_interval_seconds = float(g.get('config_watch_interval_seconds', self.config_watch_interval_seconds))
                                    # handle tasks model path changes
                                    pose_m = g.get('pose_model_path')
                                    hand_m = g.get('hand_model_path')
                                    use_tasks_cfg = bool(g.get('use_tasks', False))
                                    if use_tasks_cfg and (pose_m or hand_m):
                                        try:
                                            self._init_tasks_landmarkers(pose_m, hand_m)
                                        except Exception as e:
                                            self.logger.warning(f'Failed to re-init tasks landmarkers on config change: {e}')

                                    self.logger.info('GestureRecognizer: config reloaded from file')
                                except Exception as e:
                                    self.logger.error(f'Error reloading gesture config: {e}')
                        except Exception:
                            pass
                        time.sleep(self.config_watch_interval_seconds)
                except Exception:
                    pass

            self._config_watcher_thread = threading.Thread(target=watcher, daemon=True)
            self._config_watcher_thread.start()
        except Exception:
            pass

    def _stop_config_watcher(self):
        self._stop_watcher = True
        try:
            if self._config_watcher_thread:
                self._config_watcher_thread.join(timeout=1.0)
        except Exception:
            pass

    def _auto_discover_task_models(self):
        """Search the installed mediapipe package for bundled task model files.

        Returns a tuple (pose_path, hand_path) where each may be None if not found.
        """
        try:
            import mediapipe as mp
            mp_dir = os.path.dirname(mp.__file__)
            pose_candidate = None
            hand_candidate = None
            for root, _, files in os.walk(mp_dir):
                for f in files:
                    fn = f.lower()
                    if fn.endswith('.tflite') or fn.endswith('.task') or fn.endswith('.lite'):
                        if 'pose' in fn and not pose_candidate:
                            pose_candidate = os.path.join(root, f)
                        if ('hand' in fn or 'hand_landmarker' in fn) and not hand_candidate:
                            hand_candidate = os.path.join(root, f)
                # Early exit if both found
                if pose_candidate and hand_candidate:
                    break
            return pose_candidate, hand_candidate
        except Exception:
            return None, None

    def _init_tasks_landmarkers(self, pose_model_path=None, hand_model_path=None):
        """Attempt to initialize Mediapipe Tasks Pose/Hand landmarkers using model file paths.

        If any initialization step fails, the method raises an exception and leaves
        the recognizer in a non-tasks mode (caller should handle fallback).
        """
        try:
            import mediapipe as mp
            vision = mp.tasks.vision
        except Exception as e:
            raise RuntimeError('Mediapipe tasks vision API not available') from e

        # If no explicit model path provided and auto-discover enabled, search for packaged models
        if not pose_model_path and not hand_model_path:
            try:
                pose_model_path, hand_model_path = self._auto_discover_task_models()
            except Exception:
                pose_model_path, hand_model_path = None, None

        got_any = False
        # Pose landmarker
        if pose_model_path:
            try:
                if hasattr(vision.PoseLandmarker, 'create_from_model_path'):
                    self._tasks_pose = vision.PoseLandmarker.create_from_model_path(pose_model_path)
                    got_any = True
            except Exception as e:
                raise RuntimeError(f'Failed to initialize PoseLandmarker from {pose_model_path}: {e}') from e

        # Hand landmarker
        if hand_model_path:
            try:
                if hasattr(vision.HandLandmarker, 'create_from_model_path'):
                    self._tasks_hand = vision.HandLandmarker.create_from_model_path(hand_model_path)
                    got_any = True
            except Exception as e:
                raise RuntimeError(f'Failed to initialize HandLandmarker from {hand_model_path}: {e}') from e

        if not got_any:
            raise RuntimeError('No valid task-based landmarkers were initialized')

        self._use_tasks = True
        self.logger.info('GestureRecognizer: Mediapipe tasks-based landmarkers initialized')

    def reload_config(self, cfg=None):
        """Reload config from given dict or from config file."""
        try:
            if cfg is None:
                from config_loader import load_config
                cfg = load_config()
            g = cfg.get('gesture', {})
            self.motion_only_threshold = float(g.get('motion_only_threshold', self.motion_only_threshold))
            self.motion_weight = float(g.get('motion_weight', self.motion_weight))
            self.hands_weight = float(g.get('hands_weight', self.hands_weight))
            self.raised_weight = float(g.get('raised_weight', self.raised_weight))
            self.speed_weight = float(g.get('speed_weight', self.speed_weight))
            self.combined_threshold = float(g.get('combined_threshold', self.combined_threshold))
            self.motion_min_for_combined = float(g.get('motion_min_for_combined', self.motion_min_for_combined))
            self.config_watch_interval_seconds = float(g.get('config_watch_interval_seconds', self.config_watch_interval_seconds))
            self.logger.info('GestureRecognizer: config reloaded via method')
        except Exception as e:
            self.logger.error(f'Error reloading gesture config via method: {e}')

