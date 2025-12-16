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

    def __init__(self, history_len=16, wrist_speed_thresh=0.08, verbose=False):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.NullHandler())

        # Lazy import of mediapipe to allow running project without it until used
        try:
            import mediapipe as mp
        except Exception as e:
            raise RuntimeError(
                "mediapipe is required for GestureRecognizer. "
                "Install with: pip install mediapipe"
            ) from e

        # Store references
        self.mp = mp
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                                           min_detection_confidence=0.4,
                                           min_tracking_confidence=0.4)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False,
                                              max_num_hands=2,
                                              min_detection_confidence=0.4,
                                              min_tracking_confidence=0.4)

        # Temporal history of keypoints (normalized coordinates)
        self.history_len = history_len
        self.wrist_history = deque(maxlen=history_len)  # each entry: list of (lx,ly,rx,ry) or None

        # Thresholds
        self.wrist_speed_thresh = wrist_speed_thresh  # normalized units per frame
        self.verbose = verbose

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

            pose_res = self.pose.process(rgb)
            hands_res = self.hands.process(rgb)

            # Default wrist coords (normalized relative to proc crop)
            left_wrist = None
            right_wrist = None
            shoulders = None
            nose = None

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
                    if dist > self.wrist_speed_thresh:
                        rapid_count += 1

        if len(self.wrist_history) > 1:
            speed_ratio = rapid_count / (len(self.wrist_history) - 1)
        else:
            speed_ratio = 0.0

        # Decision logic
        suspicious = False
        label = "none"
        # Compute normalized vote fractions
        raised_frac = (raised_votes / frames_with_data) if frames_with_data else 0
        hands_frac = (hands_on_face_votes / frames_with_data) if frames_with_data else 0

        # Prioritize hands on face and raised arms
        if hands_frac >= 0.25:
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
            confidence = max(0.0, max(raised_frac, hands_frac, speed_ratio))

        details = {
            'frames_analyzed': frames_with_data,
            'raised_frac': raised_frac,
            'hands_frac': hands_frac,
            'speed_ratio': speed_ratio,
            'history_len': len(self.wrist_history)
        }

        if self.verbose:
            self.logger.info(f"Gesture analysis -> suspicious={suspicious}, label={label}, conf={confidence}, details={details}")

        return suspicious, label, float(confidence), details

    def close(self):
        try:
            if self.pose:
                self.pose.close()
            if self.hands:
                self.hands.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
