"""
src/reduce_dataset.py
──────────────────────
Cuts the dataset down to a smaller, BALANCED set by keeping a random
fraction of each letter's images (default: half).

Why random, not "first N": the images inside each class folder are
usually recorded in sequence (same lighting/pose for a while), so
just taking the first N would lose variety. A random sample keeps a
good mix of hand position, rotation, distance, and lighting.

This does NOT touch your original dataset — it copies the selected
files into a new folder.

Usage:
    python src/reduce_dataset.py --source dataset/asl_alphabet_train \
                                  --dest dataset/asl_alphabet_reduced \
                                  --fraction 0.5
"""

import os
import random
import shutil
import argparse


def reduce_dataset(source_dir, dest_dir, fraction=0.5, seed=42):
    random.seed(seed)  # same "random" split every time you run this

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    os.makedirs(dest_dir, exist_ok=True)

    class_folders = sorted(
        f for f in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, f))
    )

    if not class_folders:
        raise RuntimeError(f"No class folders found inside {source_dir}")

    print(f"Found {len(class_folders)} classes in {source_dir}\n")

    total_kept = 0
    for class_name in class_folders:
        src_class_dir = os.path.join(source_dir, class_name)
        dst_class_dir = os.path.join(dest_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        images = [
            f for f in os.listdir(src_class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)
        keep_count = max(1, int(len(images) * fraction))
        chosen = images[:keep_count]

        for fname in chosen:
            shutil.copy2(
                os.path.join(src_class_dir, fname),
                os.path.join(dst_class_dir, fname),
            )

        total_kept += keep_count
        print(f"  {class_name}: kept {keep_count} / {len(images)} images")

    print(f"\nDone. Kept {total_kept} images total in '{dest_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce dataset while keeping it balanced.")
    parser.add_argument("--source", default="dataset/asl_alphabet_train",
                         help="Folder containing one subfolder per letter")
    parser.add_argument("--dest", default="dataset/asl_alphabet_reduced",
                         help="Where to copy the reduced dataset")
    parser.add_argument("--fraction", type=float, default=0.5,
                         help="Fraction of images to keep per class (0.5 = half)")
    args = parser.parse_args()

    reduce_dataset(args.source, args.dest, args.fraction)
