"""
src/webcam.py
──────────────
Real-time ASL letter recognition from your webcam.

Uses the SAME extract_landmarks() function as training, so the model
always receives features in the format it learned from.

Stabilization (so the letter doesn't flicker every frame):
  - A prediction only "counts" if its confidence is above CONF_THRESHOLD.
  - We keep the last WINDOW_SIZE counted predictions in a rolling buffer.
  - The displayed letter only changes once the SAME letter appears at
    least MIN_AGREEMENT times in that buffer (simple majority vote).

Controls:
  q  -  quit
"""

import os
import pickle
from collections import deque, Counter

import cv2

from utils import process_frame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

CONF_THRESHOLD = 0.6   # ignore predictions the model isn't confident about
WINDOW_SIZE = 10       # how many recent confident predictions to remember
MIN_AGREEMENT = 6      # how many of those must agree before we switch letters


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder


def run_webcam():
    model, label_encoder = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    recent_predictions = deque(maxlen=WINDOW_SIZE)
    displayed_letter = ""

    print("Webcam started. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror, feels more natural
            status_text = "No hand detected"
            confidence = 0.0

            landmarks, frame = process_frame(frame)

            if landmarks is not None:
                probabilities = model.predict_proba([landmarks])[0]
                best_index = probabilities.argmax()
                confidence = float(probabilities[best_index])
                predicted_letter = label_encoder.classes_[best_index]

                if confidence >= CONF_THRESHOLD:
                    recent_predictions.append(predicted_letter)
                else:
                    status_text = "Low confidence"

                # Majority vote: only switch the displayed letter once
                # we've seen enough agreement in the recent buffer.
                if recent_predictions:
                    letter, count = Counter(recent_predictions).most_common(1)[0]
                    if count >= MIN_AGREEMENT:
                        displayed_letter = letter
            else:
                recent_predictions.clear()

            cv2.putText(frame, f"Letter: {displayed_letter}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if status_text != "No hand detected" or landmarks is None:
                cv2.putText(frame, status_text, (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("ASL Recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always release the camera and close windows, even on error/Ctrl+C.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()