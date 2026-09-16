"""
src/extract_landmarks.py
──────────────────────────
Walks through the (reduced) image dataset, runs MediaPipe on every
image, and saves the resulting landmark features as two NumPy files:

    dataset/X.npy   -> one row of 63 numbers per image (the features)
    dataset/y.npy   -> the letter label for each row

Images where no hand is detected are skipped (and counted), since a
missing hand can't produce a training example.

Usage:
    python src/extract_landmarks.py --data dataset/asl_alphabet_reduced
"""

import os
import argparse
import numpy as np
import cv2

from utils import extract_landmarks


def build_feature_dataset(data_dir, out_dir):
    class_folders = sorted(
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    )

    if not class_folders:
        raise RuntimeError(f"No class folders found inside {data_dir}")

    features, labels = [], []
    skipped_per_class = {}

    for class_name in class_folders:
        class_dir = os.path.join(data_dir, class_name)
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        skipped = 0
        for fname in image_files:
            path = os.path.join(class_dir, fname)
            frame = cv2.imread(path)
            if frame is None:
                skipped += 1
                continue

            landmarks = extract_landmarks(frame, for_video=False)
            if landmarks is None:
                skipped += 1
                continue

            features.append(landmarks)
            labels.append(class_name)

        skipped_per_class[class_name] = skipped
        kept = len(image_files) - skipped
        print(f"  {class_name}: {kept} usable, {skipped} skipped (no hand detected)")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    total_skipped = sum(skipped_per_class.values())
    print(f"\nSaved {len(X)} feature vectors to {out_dir}/X.npy and y.npy")
    print(f"Total images skipped (no hand found): {total_skipped}")

    if total_skipped > 0.15 * (len(X) + total_skipped):
        print("\n[WARNING] More than 15% of images had no detectable hand.")
        print("Consider checking image quality/cropping in that dataset.")

    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks from image dataset.")
    parser.add_argument("--data", default="dataset/asl_alphabet_reduced",
                         help="Folder containing one subfolder per letter")
    parser.add_argument("--out", default="dataset",
                         help="Where to save X.npy and y.npy")
    args = parser.parse_args()

    build_feature_dataset(args.data, args.out)
