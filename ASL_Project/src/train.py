"""
src/train.py
─────────────
Loads the landmark features (dataset/X.npy, dataset/y.npy), splits them
into train/test sets, trains several simple ML models, and picks the
best one based on REAL test-set performance (no invented numbers).

Saves:
    models/sign_model.pkl        -> the best trained model
    models/label_encoder.pkl     -> maps model output (0,1,2..) back to letters
    results/metrics.txt          -> full evaluation report
    results/confusion_matrix.png -> confusion matrix for the best model

Usage:
    python src/train.py --data dataset
"""

import os
import time
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # so this works without a display (servers, CI, etc.)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


MODELS = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}


def train_and_compare(data_dir, models_dir, results_dir):
    X = np.load(os.path.join(data_dir, "X.npy"))
    y_raw = np.load(os.path.join(data_dir, "y.npy"))

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Classes: {len(le.classes_)}  ->  {list(le.classes_)}")
    print(f"Total samples: {len(X)}  |  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    report_lines = []
    report_lines.append("ASL Model Evaluation\n" + "=" * 40)
    report_lines.append(f"Classes ({len(le.classes_)}): {', '.join(le.classes_)}")
    report_lines.append(f"Total samples: {len(X)}")
    report_lines.append(f"Training samples: {len(X_train)}")
    report_lines.append(f"Testing samples: {len(X_test)}\n")

    # Samples per class (on the full dataset, before splitting)
    report_lines.append("Samples per class:")
    for cls in le.classes_:
        count = int(np.sum(y_raw == cls))
        report_lines.append(f"  {cls}: {count}")
    report_lines.append("")

    best_name, best_model, best_test_acc = None, None, -1
    comparison_rows = []

    for name, model in MODELS.items():
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        start = time.time()
        test_pred = model.predict(X_test)
        predict_time_ms = (time.time() - start) / len(X_test) * 1000

        test_acc = accuracy_score(y_test, test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, test_pred, average="weighted", zero_division=0
        )

        comparison_rows.append((name, train_acc, test_acc, precision, recall, f1, predict_time_ms))
        print(f"{name:22s} | train {train_acc:.3f} | test {test_acc:.3f} "
              f"| precision {precision:.3f} | recall {recall:.3f} | f1 {f1:.3f} "
              f"| {predict_time_ms:.3f} ms/pred")

        if test_acc > best_test_acc:
            best_name, best_model, best_test_acc = name, model, test_acc

    report_lines.append("Model comparison (same train/test split for all):")
    report_lines.append(
        f"{'Model':22s} {'Train Acc':>10s} {'Test Acc':>10s} "
        f"{'Precision':>10s} {'Recall':>9s} {'F1':>7s} {'ms/pred':>9s}"
    )
    for row in comparison_rows:
        name, tr, te, p, r, f1, ms = row
        report_lines.append(
            f"{name:22s} {tr*100:9.2f}% {te*100:9.2f}% {p*100:9.2f}% {r*100:8.2f}% {f1*100:6.2f}% {ms:8.3f}"
        )
    report_lines.append("")

    # Final report for the best model
    best_train_pred = best_model.predict(X_train)
    best_test_pred = best_model.predict(X_test)
    best_train_acc = accuracy_score(y_train, best_train_pred)
    best_test_acc = accuracy_score(y_test, best_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, best_test_pred, average="weighted", zero_division=0
    )

    report_lines.append(f"BEST MODEL: {best_name}")
    report_lines.append(f"Training Accuracy: {best_train_acc*100:.2f}%")
    report_lines.append(f"Testing Accuracy:  {best_test_acc*100:.2f}%")
    report_lines.append(f"Precision (weighted): {precision*100:.2f}%")
    report_lines.append(f"Recall (weighted):    {recall*100:.2f}%")
    report_lines.append(f"F1 Score (weighted):  {f1*100:.2f}%\n")

    report_lines.append("Per-letter performance:")
    report_lines.append(
        classification_report(y_test, best_test_pred, target_names=le.classes_, zero_division=0)
    )

    # Confusion matrix
    os.makedirs(results_dir, exist_ok=True)
    cm = confusion_matrix(y_test, best_test_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close(fig)

    # Which letters got confused with which (helps satisfy requirement #6)
    report_lines.append("Most confused letter pairs (true -> predicted, count):")
    confusions = []
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], le.classes_[i], le.classes_[j]))
    confusions.sort(reverse=True)
    for count, true_letter, pred_letter in confusions[:15]:
        report_lines.append(f"  {true_letter} -> {pred_letter}: {count} times")

    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(report_lines))

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "sign_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\nBest model: {best_name} (test accuracy {best_test_acc*100:.2f}%)")
    print(f"Saved model to {models_dir}/sign_model.pkl")
    print(f"Saved report to {results_dir}/metrics.txt")
    print(f"Saved confusion matrix to {results_dir}/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare ASL models.")
    parser.add_argument("--data", default="dataset", help="Folder with X.npy and y.npy")
    parser.add_argument("--models-out", default="models")
    parser.add_argument("--results-out", default="results")
    args = parser.parse_args()

    train_and_compare(args.data, args.models_out, args.results_out)
