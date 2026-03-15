#!/usr/bin/env python3
"""
Generate project plots for COMP 4449 final deliverables.

Outputs:
- plots/classification_class_distribution.png
- plots/detection_instance_distribution.png
- plots/sample_images_by_class.png
- plots/yolo_training_curves.png
- plots/classification_confusion_matrix.png
- plots/yolo_image_level_confusion_matrix.png
- plots/yolo_predictions_grid.png

Assumptions:
- Current working directory is your project root (the folder that contains NEU-DET/, artifacts_phase2/, artifacts_yolo/)
- Classification final confusion matrix exists if you already ran the classification final pipeline.
- YOLO results.csv exists if you already ran YOLO training.
- YOLO best.pt exists if you already ran YOLO training.
"""

from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Use only if available; prediction grid will be skipped otherwise.
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = Path(".").resolve()

TRAIN_IMAGES_DIR = PROJECT_ROOT / "NEU-DET" / "train" / "images"
VAL_IMAGES_DIR = PROJECT_ROOT / "NEU-DET" / "validation" / "images"

TRAIN_ANN_DIR = PROJECT_ROOT / "NEU-DET" / "train" / "annotations"
VAL_ANN_DIR = PROJECT_ROOT / "NEU-DET" / "validation" / "annotations"

PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_CM_PATH = PROJECT_ROOT / "artifacts_phase2" / "final_efficientnetb0_transfer" / "final_confusion_matrix.npy"
YOLO_CM_PATH = PROJECT_ROOT / "artifacts_yolo" / "final_test_confusion_matrix.npy"

YOLO_RESULTS_CSV = PROJECT_ROOT / "artifacts_yolo" / "yolo_runs" / "neu_det_yolo" / "results.csv"
YOLO_WEIGHTS = PROJECT_ROOT / "artifacts_yolo" / "yolo_runs" / "neu_det_yolo" / "weights" / "best.pt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# -----------------------------
# Helpers
# -----------------------------
def get_class_names(images_dir: Path) -> List[str]:
    return sorted([p.name for p in images_dir.iterdir() if p.is_dir()])


def list_images_by_class(images_dir: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for cls in get_class_names(images_dir):
        class_dir = images_dir / cls
        out[cls] = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    return out


def parse_voc_objects(xml_path: Path) -> List[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is not None and name_node.text is not None:
            names.append(name_node.text.strip())
    return names


def plot_bar_chart(labels: List[str], values: List[int], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(val),
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# 1) Classification class distribution
# -----------------------------
def make_classification_distribution_plot() -> None:
    train_map = list_images_by_class(TRAIN_IMAGES_DIR)
    class_names = list(train_map.keys())
    counts = [len(train_map[c]) for c in class_names]

    plot_bar_chart(
        class_names,
        counts,
        title="Classification Dataset Distribution (Train Images)",
        ylabel="Number of Images",
        out_path=PLOTS_DIR / "classification_class_distribution.png",
    )


# -----------------------------
# 2) Detection instance distribution from XML
# -----------------------------
def make_detection_instance_distribution_plot() -> None:
    class_names = get_class_names(TRAIN_IMAGES_DIR)
    counts = {cls: 0 for cls in class_names}

    for xml_path in sorted(TRAIN_ANN_DIR.glob("*.xml")):
        for cls in parse_voc_objects(xml_path):
            if cls in counts:
                counts[cls] += 1

    plot_bar_chart(
        list(counts.keys()),
        list(counts.values()),
        title="Detection Dataset Distribution (Train Instances from XML)",
        ylabel="Number of Annotated Instances",
        out_path=PLOTS_DIR / "detection_instance_distribution.png",
    )


# -----------------------------
# 3) Sample images grid
# -----------------------------
def make_sample_images_grid(samples_per_class: int = 3) -> None:
    image_map = list_images_by_class(TRAIN_IMAGES_DIR)
    class_names = list(image_map.keys())

    rows = len(class_names)
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows))

    if rows == 1:
        axes = np.array([axes])

    for r, cls in enumerate(class_names):
        images = image_map[cls]
        chosen = images[:samples_per_class] if len(images) >= samples_per_class else images

        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")

            if c < len(chosen):
                img = Image.open(chosen[c]).convert("RGB")
                ax.imshow(img)
                if c == 0:
                    ax.set_title(cls, loc="left", fontsize=10, fontweight="bold")

    plt.suptitle("Sample Defect Images by Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sample_images_by_class.png", dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# 4) YOLO training curves
# -----------------------------
def make_yolo_training_curves() -> None:
    import csv

    if not YOLO_RESULTS_CSV.exists():
        print(f"[SKIP] YOLO results.csv not found: {YOLO_RESULTS_CSV}")
        return

    rows = []
    with open(YOLO_RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[SKIP] YOLO results.csv is empty")
        return

    epochs = []
    train_box_loss = []
    train_cls_loss = []
    val_map50 = []
    val_map = []
    precision = []
    recall = []

    for i, row in enumerate(rows, start=1):
        epochs.append(i)

        def get_float(key: str) -> float:
            val = row.get(key, "")
            try:
                return float(val)
            except Exception:
                return np.nan

        train_box_loss.append(get_float("train/box_loss"))
        train_cls_loss.append(get_float("train/cls_loss"))
        val_map50.append(get_float("metrics/mAP50(B)"))
        val_map.append(get_float("metrics/mAP50-95(B)"))
        precision.append(get_float("metrics/precision(B)"))
        recall.append(get_float("metrics/recall(B)"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_box_loss, label="train box loss")
    axes[0, 0].plot(epochs, train_cls_loss, label="train cls loss")
    axes[0, 0].set_title("YOLO Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, val_map50, label="mAP@0.5")
    axes[0, 1].plot(epochs, val_map, label="mAP@0.5:0.95")
    axes[0, 1].set_title("YOLO Validation mAP")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, precision, label="precision")
    axes[1, 0].plot(epochs, recall, label="recall")
    axes[1, 0].set_title("YOLO Precision / Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.7,
        "Recommended use in slides:\n"
        "- show mAP improvement across training\n"
        "- mention convergence behavior\n"
        "- mention final precision / recall",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "yolo_training_curves.png", dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# 5) Classification confusion matrix
# -----------------------------
def make_classification_confusion_matrix() -> None:
    if not CLASSIFICATION_CM_PATH.exists():
        print(f"[SKIP] Classification confusion matrix not found: {CLASSIFICATION_CM_PATH}")
        return

    class_names = get_class_names(TRAIN_IMAGES_DIR)
    cm = np.load(CLASSIFICATION_CM_PATH)

    plot_confusion_matrix(
        cm,
        class_names,
        title="EfficientNet Classification Confusion Matrix",
        out_path=PLOTS_DIR / "classification_confusion_matrix.png",
    )


# -----------------------------
# 6) YOLO image-level confusion matrix
# -----------------------------
def make_yolo_confusion_matrix() -> None:
    if not YOLO_CM_PATH.exists():
        print(f"[SKIP] YOLO confusion matrix not found: {YOLO_CM_PATH}")
        return

    class_names = get_class_names(TRAIN_IMAGES_DIR)
    cm = np.load(YOLO_CM_PATH)

    plot_confusion_matrix(
        cm,
        class_names,
        title="YOLO Image-Level Confusion Matrix",
        out_path=PLOTS_DIR / "yolo_image_level_confusion_matrix.png",
    )


# -----------------------------
# 7) YOLO qualitative predictions
# -----------------------------
def find_image_recursive(images_root: Path, stem: str) -> Path | None:
    matches = list(images_root.rglob(f"{stem}.*"))
    matches = [m for m in matches if m.is_file() and m.suffix.lower() in IMG_EXTS]
    return matches[0] if matches else None


def make_yolo_prediction_grid(n_images: int = 6, conf: float = 0.25) -> None:
    if not ULTRALYTICS_AVAILABLE:
        print("[SKIP] ultralytics not installed, skipping YOLO qualitative predictions")
        return

    if not YOLO_WEIGHTS.exists():
        print(f"[SKIP] YOLO weights not found: {YOLO_WEIGHTS}")
        return

    # Use validation annotations as source for deterministic sampling.
    xml_files = sorted(VAL_ANN_DIR.glob("*.xml"))
    if not xml_files:
        print("[SKIP] No validation XML files found")
        return

    selected_xml = xml_files[:n_images]
    selected_images = []

    for xml in selected_xml:
        img_path = find_image_recursive(VAL_IMAGES_DIR, xml.stem)
        if img_path is not None:
            selected_images.append(img_path)

    if not selected_images:
        print("[SKIP] Could not resolve images for prediction grid")
        return

    model = YOLO(str(YOLO_WEIGHTS))
    preds = model.predict(
        source=[str(p) for p in selected_images],
        conf=conf,
        verbose=False,
        save=False,
    )

    cols = 2
    rows = math.ceil(len(preds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if idx >= len(preds):
                continue

            plotted = preds[idx].plot()  # returns BGR ndarray
            plotted = plotted[:, :, ::-1]  # BGR -> RGB
            ax.imshow(plotted)
            ax.set_title(selected_images[idx].name, fontsize=9)
            idx += 1

    plt.suptitle("YOLO Qualitative Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "yolo_predictions_grid.png", dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    random.seed(42)
    np.random.seed(42)

    make_classification_distribution_plot()
    make_detection_instance_distribution_plot()
    make_sample_images_grid(samples_per_class=3)
    make_yolo_training_curves()
    make_classification_confusion_matrix()
    make_yolo_confusion_matrix()
    make_yolo_prediction_grid(n_images=6, conf=0.25)

    print(f"Done. Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()