"""
src/utils.py
────────────
Shared helper functions.

IMPORTANT: extract_landmarks() / process_frame() are used by BOTH the
training pipeline (src/extract_landmarks.py) and the webcam / web app
(src/webcam.py, app.py). Using the exact same function everywhere
guarantees the model always sees features in the same format it was
trained on.

A NOTE ON THE MEDIAPIPE VERSION:
Older MediaPipe tutorials use `mp.solutions.hands`. That API has been
removed from current MediaPipe releases. This file uses MediaPipe's
current "Tasks" API instead (`mediapipe.tasks.python.vision.HandLandmarker`),
which is what `pip install mediapipe` gives you today. The very first
time you run anything that imports this file, it will automatically
download a small (~8MB) hand-detection model file and cache it in
models/hand_landmarker.task — you need an internet connection for
that one-time download, but not afterward.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LANDMARKER_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model_downloaded():
    """Download the hand-detection model once and cache it locally."""
    if os.path.exists(LANDMARKER_MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading hand-detection model (one-time, ~8MB)...")
    urllib.request.urlretrieve(MODEL_URL, LANDMARKER_MODEL_PATH)
    print("Model downloaded to", LANDMARKER_MODEL_PATH)


_ensure_model_downloaded()

# Two detectors, tuned for two different jobs:
#
# IMAGE mode — used when preprocessing the training dataset. Every call is
# treated as a brand-new, unrelated photo, which is the correct (and more
# thorough) way to handle a folder of separate images.
_landmarker_image = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
)

# VIDEO mode — used for the live webcam/web app. Instead of searching the
# WHOLE frame for a hand every single time (slow), it tracks the hand it
# already found in the previous frame and only re-searches from scratch if
# it loses track. This is the main fix for webcam lag.
_landmarker_video = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
)
_video_frame_counter = 0  # VIDEO mode requires an increasing timestamp per call

# Standard 21-point hand skeleton connections, used only for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky finger
    (0, 17),                                 # palm base
]


def _normalize_landmarks(hand_landmarks):
    """
    Turn MediaPipe's 21 (x, y, z) hand points into a normalized feature vector.

    Normalization steps (this is the ONE place this logic lives):
      1. Shift every point so the wrist (landmark 0) becomes the origin.
         -> makes the features independent of WHERE the hand is in the frame.
      2. Scale every point by the distance from the wrist to the middle
         finger's base (landmark 9).
         -> makes the features independent of hand SIZE / distance from camera.

    `hand_landmarks` is a list of 21 points, each with .x, .y, .z.
    Returns a flat list of 63 numbers (21 points x 3 coordinates).
    """
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    wrist = points[0]
    points = points - wrist  # step 1: translate

    scale = np.linalg.norm(points[9])  # distance wrist -> middle finger MCP
    if scale < 1e-6:
        scale = 1e-6  # avoid divide-by-zero on a degenerate detection
    points = points / scale  # step 2: scale

    return points.flatten().tolist()


def _detect_hand_image(frame_bgr):
    """Detect a hand in one standalone image (used for training data)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return _landmarker_image.detect(mp_image)


def _detect_hand_video(frame_bgr):
    """Detect a hand in one frame of a live stream (used for webcam/web app)."""
    global _video_frame_counter
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    _video_frame_counter += 1
    return _landmarker_video.detect_for_video(mp_image, _video_frame_counter)


def extract_landmarks(frame_bgr, for_video=False):
    """
    Run hand detection on one still image and return a normalized 63-value
    feature list, or None if no hand was found. Used by the training
    pipeline (src/extract_landmarks.py).
    """
    result = _detect_hand_image(frame_bgr)

    if not result.hand_landmarks:
        return None

    return _normalize_landmarks(result.hand_landmarks[0])


def process_frame(frame_bgr):
    """
    Run hand detection ONCE (using the fast VIDEO-tracking mode, meant for
    a live webcam/browser stream) and return everything needed from a
    single frame: (features, raw_points).

    features   -> the 63-number normalized vector the ML model expects
                  (None if no hand found)
    raw_points -> the 21 (x, y) points in plain 0-1 image-relative
                  coordinates, meant for DRAWING only (e.g. on an HTML
                  canvas) — NOT normalized the way the model needs.
                  ([] if no hand found)

    Use this for any live camera loop (webcam.py, app.py) — VIDEO mode
    tracks the hand instead of re-searching the whole frame every time,
    and this only runs detection once per frame.
    """
    result = _detect_hand_video(frame_bgr)

    if not result.hand_landmarks:
        return None, []

    hand_landmarks = result.hand_landmarks[0]
    features = _normalize_landmarks(hand_landmarks)
    raw_points = [[lm.x, lm.y] for lm in hand_landmarks]
    return features, raw_points


def draw_landmarks_on_frame(frame_bgr):
    """Draw the detected hand skeleton on a copy of the frame (used by
    src/webcam.py's OpenCV window, which draws locally with no network
    round-trip so this is cheap there)."""
    result = _detect_hand_video(frame_bgr)
    annotated = frame_bgr.copy()
    if not result.hand_landmarks:
        return annotated

    height, width = annotated.shape[:2]
    for hand_landmarks in result.hand_landmarks:
        pixel_points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(annotated, pixel_points[start_idx], pixel_points[end_idx],
                      (0, 200, 0), 2)
        for x, y in pixel_points:
            cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    return annotated