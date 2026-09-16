"""
app.py
──────
Flask web app for the ASL project (browser-based alternative to
src/webcam.py). Uses the exact same feature extraction as training,
via src/utils.py.

Routes
------
GET  /                  → main UI
POST /predict           → accepts a base64 webcam frame, returns predicted letter
GET  /text_to_sign      → accepts ?text=ABC, returns list of sign image URLs
POST /speak             → accepts JSON {"text": "..."}, speaks via pyttsx3
"""

import os
import sys
import pickle
import base64
import io
import threading

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import process_frame

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "asl_alphabet_reduced")

clf, le = None, None


def load_model():
    global clf, le
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        print("Model loaded.")
    else:
        print("[WARN] Model not found. Run: python src/train.py")


load_model()


def _speak(text: str):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS error] {e}")


@app.route("/")
def index():
    return render_template("index.html")


def decode_uploaded_image(base64_text: str):
    if "," in base64_text:
        base64_text = base64_text.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_text)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "image": "<base64 data-URL>" }
    Returns JSON: { "letter": "A", "confidence": 0.97, "points": [[x,y], ...] }

    `points` are the 21 hand landmark positions as 0-1 fractions of the
    image, meant for the browser to draw the skeleton itself on a canvas
    overlay — much cheaper than the server re-encoding and sending back a
    whole annotated image on every request.
    """
    if clf is None:
        return jsonify({"error": "Model not loaded. Run python src/train.py first."}), 503

    data = request.get_json(force=True)
    base64_image = data.get("image", "")

    try:
        frame_bgr = decode_uploaded_image(base64_image)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400

    features, raw_points = process_frame(frame_bgr)

    if features is None:
        return jsonify({"letter": None, "confidence": 0.0, "points": [],
                         "status": "No hand detected"})

    probabilities = clf.predict_proba([features])[0]
    best_index = int(np.argmax(probabilities))
    predicted_letter = str(le.classes_[best_index])
    confidence = float(probabilities[best_index])

    return jsonify({
        "letter": predicted_letter,
        "confidence": round(confidence, 3),
        "points": raw_points,
        "status": "ok",
    })


@app.route("/text_to_sign")
def text_to_sign():
    """?text=HELLO -> { "letters": [{"char":"H","img_url":"/sign_img/H"}, ...] }"""
    text = request.args.get("text", "").upper()
    result = []
    for ch in text:
        if ch.isalpha():
            result.append({"char": ch, "img_url": f"/sign_img/{ch}"})
        elif ch == " ":
            result.append({"char": " ", "img_url": None})
    return jsonify({"letters": result})


@app.route("/sign_img/<letter>")
def sign_img(letter: str):
    """Serve one representative sign image for the given letter."""
    letter = letter.upper()
    letter_dir = os.path.join(TRAIN_DIR, letter)
    if not os.path.isdir(letter_dir):
        letter_dir = os.path.join(TRAIN_DIR, letter.lower())
    if not os.path.isdir(letter_dir):
        return ("Image not found", 404)

    for fname in sorted(os.listdir(letter_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            return send_from_directory(letter_dir, fname)
    return ("No images in folder", 404)


@app.route("/speak", methods=["POST"])
def speak():
    """JSON body: { "text": "Hello world" }"""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if text:
        threading.Thread(target=_speak, args=(text,), daemon=True).start()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)