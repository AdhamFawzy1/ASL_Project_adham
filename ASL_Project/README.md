# ASL Alphabet Recognition (MediaPipe + Classic ML)

A simple, beginner-friendly ASL alphabet recognizer:

```
Camera/Image → MediaPipe Hand Detection → Hand Landmarks →
Normalize → ML Classifier → Predicted Letter
```

No deep learning / image CNN — a hand has only 21 landmark points, so a
lightweight classifier (Random Forest / SVM / KNN / Logistic Regression /
Decision Tree — whichever wins the comparison) trained on those points is
plenty, and it's far faster and easier to understand.

## Project structure

```
ASL_Project/
├── dataset/
│   ├── asl_alphabet_train/     ← YOUR original 85,000-image dataset (you provide this)
│   ├── asl_alphabet_reduced/   ← created by reduce_dataset.py
│   ├── X.npy, y.npy            ← created by extract_landmarks.py
├── models/
│   ├── sign_model.pkl          ← created by train.py
│   └── label_encoder.pkl       ← created by train.py
├── results/
│   ├── metrics.txt             ← created by train.py
│   └── confusion_matrix.png    ← created by train.py
├── src/
│   ├── utils.py                ← MediaPipe + normalization (used by everything else)
│   ├── reduce_dataset.py       ← step 1: shrink dataset, balanced
│   ├── extract_landmarks.py    ← step 2: images -> feature vectors
│   ├── train.py                ← step 3: train + compare models, save best
│   └── webcam.py               ← step 4: real-time recognition
├── templates/index.html
├── app.py                      ← optional browser UI (Flask)
└── requirements.txt
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Important version note:** `mediapipe` recently removed its old
`mp.solutions.hands` API in favor of a newer "Tasks" API. This project
uses the classic, simpler `mp.solutions.hands` API on purpose (it's far
easier for a beginner to read), so **keep `mediapipe==0.10.9`** as pinned
in `requirements.txt`. If you `pip install --upgrade mediapipe`, the
project will break with `AttributeError: module 'mediapipe' has no
attribute 'solutions'`. If you ever want to upgrade, `src/utils.py` is
the only file that would need to change.

## 2. Put your dataset in place

You said you have ~85,000 images (about 3,000 per letter). Put that
folder here as:

```
dataset/asl_alphabet_train/<LETTER>/*.jpg
```

(one subfolder per letter, e.g. `A/`, `B/`, ... `Z/`, plus `space`,
`del`, `nothing` if your dataset has those — the scripts don't care
about the exact letter names, they just read whatever folders exist).

## 3. Reduce the dataset (keep it balanced)

This copies a random ~50% of each letter's images into a new folder,
so every class shrinks by the same proportion — no class gets left
behind:

```bash
python src/reduce_dataset.py --source dataset/asl_alphabet_train --dest dataset/asl_alphabet_reduced --fraction 0.5
```

Change `--fraction` if you want to keep more or less (e.g. `0.3` for ~30%).

## 4. Extract MediaPipe landmarks

Turns every remaining image into a 63-number feature vector (21 hand
points × x,y,z), normalized so it doesn't matter where the hand is in
the frame or how far from the camera:

```bash
python src/extract_landmarks.py --data dataset/asl_alphabet_reduced
```

This prints how many images per letter had a detectable hand. If a
letter loses a lot of images here, that letter's photos may be poorly
cropped or the hand isn't clearly visible — worth a look before training.

## 5. Train and compare models

```bash
python src/train.py
```

This trains Random Forest, SVM, KNN, Logistic Regression, and Decision
Tree on the **same** train/test split, prints a comparison table, and
automatically saves whichever model gets the best test accuracy to
`models/sign_model.pkl`. Full metrics (including per-letter precision/
recall/F1 and the most-confused letter pairs) go to `results/metrics.txt`,
and `results/confusion_matrix.png` shows the full confusion matrix.

**These are real numbers computed from your actual data — nothing here
is hard-coded or invented.** Open `results/metrics.txt` after training
to see your actual accuracy.

## 6. Run the webcam

```bash
python src/webcam.py
```

Hold your hand up and make a letter. Press `q` to quit.

To avoid the letter flickering between predictions, `webcam.py`:
- ignores any prediction below 60% confidence,
- keeps the last 10 confident predictions in a rolling window,
- only changes the displayed letter once 6 of those 10 agree.

You can tune `CONF_THRESHOLD`, `WINDOW_SIZE`, and `MIN_AGREEMENT` at the
top of `src/webcam.py` if it feels too slow or too twitchy to switch.

## (Optional) Browser version

```bash
python app.py
```
Then open `http://localhost:5000`. Same model, same features — just
running through a browser + webcam instead of an OpenCV window. Also
includes a "type a word, see the signs" feature and a speak button.

## Adding more training data later

Add more images into the matching folder under
`dataset/asl_alphabet_train/<LETTER>/`, then just re-run steps 3–5
(reduce → extract → train). You don't need to touch any code.

## Common problems

| Problem | Likely cause / fix |
|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | You have a newer mediapipe than `0.10.9`. Run `pip install mediapipe==0.10.9`. |
| Webcam window doesn't open / black screen | Another app may be using the camera, or try `cv2.VideoCapture(1)` in `webcam.py` if you have more than one camera. |
| "Model not found" in app.py / webcam.py | You need to run `src/train.py` first — it creates the `.pkl` files. |
| A specific letter never gets recognized | Check `results/metrics.txt` for that letter's precision/recall, and `results/confusion_matrix.png` for what it's confused with. Static hand shapes for letters like **M/N/S** or **U/V/R** are genuinely hard to tell apart from a still frame — more/better training photos for those letters usually helps most. |
| Predictions jump around a lot | Increase `MIN_AGREEMENT` or `CONF_THRESHOLD` in `src/webcam.py`. |

## Known limitations

- This is a **static hand-shape** classifier — it can't recognize
  letters that require motion (like **J** and **Z** in real ASL),
  since it only looks at one frame at a time.
- Visually similar static shapes (e.g. **M/N/S**, **U/V**, **K/V**) may
  be harder to tell apart using landmarks alone — check `metrics.txt`
  for your dataset's actual confusion pairs.
- Accuracy depends entirely on your dataset's lighting/background
  variety, since MediaPipe's landmark detection quality varies with
  image conditions.
