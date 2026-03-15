#!/usr/bin/env python3
"""
Phase II – Standalone YOLO Detection Pipeline for NEU-DET

This script is designed to be consistent with the documented Phase II method steps:
- training, validation, and testing of YOLO
- balanced training with stochastic augmentation
- standardized evaluation metrics
- final performance reporting

What it does
------------
1) Reads a VOC-style detection dataset:
       train/images, train/annotations
       validation/images, validation/annotations
2) Builds an explicit manifest from XML annotations
3) Splits the train set into train/val using stratified sampling on the image-level class
4) Converts VOC XML annotations into YOLO txt labels
5) Builds a balanced training set by replicating minority-class images/labels
6) Trains a YOLO detector with Ultralytics
7) Evaluates on validation and held-out test data
8) Exports:
       - YOLO detection metrics (precision, recall, mAP@0.5, mAP@0.5:0.95)
       - image-level classification-style metrics on the test set
         (accuracy, macro F1, weighted F1, per-class report, confusion matrix)

Why image-level metrics are included
------------------------------------
Your documented project steps explicitly mention:
- accuracy
- macro F1
- per-class results

Those are classification metrics, not primary detection metrics. To stay consistent with
your project method history, this script also computes image-level classification metrics by:
- using the ground-truth primary class for each image (from XML)
- using the top-confidence predicted class for that image

Requirements
------------
pip install ultralytics pandas scikit-learn pyyaml pillow

Examples
--------
# 1) Prepare the YOLO workspace only
python phase2_yolo_detection_pipeline_ready.py \
  --dataset_root "./NEU-DET" \
  --run prepare

# 2) Train and validate YOLO
python phase2_yolo_detection_pipeline_ready.py \
  --dataset_root "./NEU-DET" \
  --run train

# 3) Full pipeline: prepare -> train -> final test evaluation
python phase2_yolo_detection_pipeline_ready.py \
  --dataset_root "./NEU-DET" \
  --run all

# 4) Final-only evaluation using the best saved weights
python phase2_yolo_detection_pipeline_ready.py \
  --dataset_root "./NEU-DET" \
  --run final \
  --weights "./artifacts_yolo/yolo_runs/neu_det_yolo/weights/best.pt"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError("Ultralytics is not installed. Install it with: pip install ultralytics") from exc


def set_global_seed(seed: int) -> None:
    """Best-effort reproducibility across Python and NumPy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class Config:
    dataset_root: Path

    train_images_rel: Path = Path("train/images")
    train_annotations_rel: Path = Path("train/annotations")
    test_images_rel: Path = Path("validation/images")
    test_annotations_rel: Path = Path("validation/annotations")

    artifacts_dir: Path = Path("./artifacts_yolo")
    workspace_dir: Path = Path("./artifacts_yolo/yolo_workspace")
    runs_dir: Path = Path("./artifacts_yolo/yolo_runs")

    val_size: float = 0.20
    seed: int = 42
    rebalance_train: bool = False

    model_name: str = "yolov8n.pt"
    imgsz: int = 224
    epochs: int = 50
    batch: int = 16
    patience: int = 15
    device: str = "mps"

    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.0
    degrees: float = 10.0
    translate: float = 0.05
    scale: float = 0.10
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.5
    mixup: float = 0.0

    pred_conf: float = 0.25

    @property
    def train_images_dir(self) -> Path:
        return self.dataset_root / self.train_images_rel

    @property
    def train_annotations_dir(self) -> Path:
        return self.dataset_root / self.train_annotations_rel

    @property
    def test_images_dir(self) -> Path:
        return self.dataset_root / self.test_images_rel

    @property
    def test_annotations_dir(self) -> Path:
        return self.dataset_root / self.test_annotations_rel


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _safe_text(parent: ET.Element, tag: str, default: Optional[str] = None) -> Optional[str]:
    elem = parent.find(tag)
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def parse_voc_xml(xml_path: Path) -> Dict:
    """Parse one Pascal VOC XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = _safe_text(root, "filename", default=xml_path.stem)
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in annotation: {xml_path}")

    width = int(_safe_text(size, "width", default="0"))
    height = int(_safe_text(size, "height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in annotation: {xml_path}")

    objects = []
    for obj in root.findall("object"):
        name = _safe_text(obj, "name")
        bbox = obj.find("bndbox")
        if name is None or bbox is None:
            continue

        xmin = int(float(_safe_text(bbox, "xmin", default="0")))
        ymin = int(float(_safe_text(bbox, "ymin", default="0")))
        xmax = int(float(_safe_text(bbox, "xmax", default="0")))
        ymax = int(float(_safe_text(bbox, "ymax", default="0")))

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append({"name": name, "bbox": (xmin, ymin, xmax, ymax)})

    if not objects:
        raise ValueError(f"No valid objects found in annotation: {xml_path}")

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }




def build_detection_manifest(images_dir: Path, annotations_dir: Path, strict_matching: bool = False) -> pd.DataFrame:
    """Build an explicit image-level manifest from VOC annotations.

    Important:
    - The NEU-DET layout places images inside class subfolders under images_dir.
    - Annotations are flat XML files under annotations_dir.
    - XML <filename> may or may not include the class subfolder, so we index images recursively.
    - If strict_matching=False, unmatched annotations are skipped and logged instead of crashing.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    rows: List[Dict] = []
    skipped: List[str] = []
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML annotation files found under: {annotations_dir}")

    # Build recursive indexes once
    all_images = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not all_images:
        raise ValueError(f"No images found recursively under: {images_dir}")

    by_name: Dict[str, List[Path]] = {}
    by_stem: Dict[str, List[Path]] = {}
    for p in all_images:
        by_name.setdefault(p.name, []).append(p)
        by_stem.setdefault(p.stem, []).append(p)

    for xml_path in xml_files:
        parsed = parse_voc_xml(xml_path)
        candidate: Optional[Path] = None

        # 1) Exact recursive match on XML filename
        filename = parsed["filename"]
        if filename:
            exact_path = images_dir / filename
            if exact_path.exists():
                candidate = exact_path
            else:
                exact_matches = by_name.get(Path(filename).name, [])
                if len(exact_matches) == 1:
                    candidate = exact_matches[0]

        # 2) Fallback: match by annotation stem recursively
        if candidate is None:
            stem_matches = by_stem.get(xml_path.stem, [])
            if len(stem_matches) == 1:
                candidate = stem_matches[0]
            elif len(stem_matches) > 1:
                msg = f"Multiple image matches found for annotation stem '{xml_path.stem}': {stem_matches}"
                if strict_matching:
                    raise FileExistsError(msg)
                skipped.append(f"{xml_path.name} :: {msg}")
                continue

        if candidate is None:
            msg = f"Could not match image for annotation: {xml_path}. Checked recursive image index under {images_dir}"
            if strict_matching:
                raise FileNotFoundError(msg)
            skipped.append(f"{xml_path.name} :: unmatched")
            continue

        primary_class = parsed["objects"][0]["name"]
        classes = sorted(list({obj["name"] for obj in parsed["objects"]}))

        rows.append(
            {
                "image_path": str(candidate.resolve()),
                "xml_path": str(xml_path.resolve()),
                "image_name": candidate.name,
                "stem": candidate.stem,
                "width": parsed["width"],
                "height": parsed["height"],
                "primary_class": primary_class,
                "all_classes": classes,
                "n_objects": len(parsed["objects"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No matched annotations/images found under images_dir={images_dir} and annotations_dir={annotations_dir}"
        )

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} unmatched or ambiguous annotations under {annotations_dir}")
        preview = skipped[:10]
        for item in preview:
            print("  -", item)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    return df


def voc_bbox_to_yolo(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert VOC bbox to YOLO normalized bbox."""
    xmin, ymin, xmax, ymax = bbox
    xc = ((xmin + xmax) / 2.0) / width
    yc = ((ymin + ymax) / 2.0) / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    return xc, yc, bw, bh


def write_yolo_label_from_xml(xml_path: Path, label_out_path: Path, class_to_id: Dict[str, int]) -> None:
    """Convert VOC XML to one YOLO txt label file."""
    parsed = parse_voc_xml(xml_path)
    lines: List[str] = []
    for obj in parsed["objects"]:
        cls_name = obj["name"]
        if cls_name not in class_to_id:
            continue
        xc, yc, bw, bh = voc_bbox_to_yolo(obj["bbox"], parsed["width"], parsed["height"])
        lines.append(f"{class_to_id[cls_name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    label_out_path.parent.mkdir(parents=True, exist_ok=True)
    label_out_path.write_text("\n".join(lines), encoding="utf-8")


def make_balanced_train_manifest(df_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Balance the training manifest by replicating minority-class images to the max class size."""
    rng = np.random.default_rng(seed)
    counts = df_train["primary_class"].value_counts().sort_index()
    max_count = int(counts.max())
    balanced_parts: List[pd.DataFrame] = []

    for _, group in df_train.groupby("primary_class"):
        group = group.copy().reset_index(drop=True)
        n = len(group)
        group["replicate_id"] = 0
        balanced_parts.append(group)

        if n < max_count:
            needed = max_count - n
            sampled_idx = rng.choice(group.index.values, size=needed, replace=True)
            extra = group.iloc[sampled_idx].copy().reset_index(drop=True)
            extra["replicate_id"] = np.arange(1, len(extra) + 1)
            balanced_parts.append(extra)

    return pd.concat(balanced_parts, axis=0, ignore_index=True)


def clear_dir(path: Path) -> None:
    """Delete and recreate a directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_class_names(df_train: pd.DataFrame, df_test: pd.DataFrame) -> List[str]:
    """Build a stable sorted class list from train + test manifests."""
    names = sorted(set(df_train["primary_class"].tolist()) | set(df_test["primary_class"].tolist()))
    if not names:
        raise ValueError("Could not infer class names from manifests.")
    return names


def stratified_train_val_split(df_train_all: pd.DataFrame, val_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train into train/val in a stratified manner using the image-level primary class."""
    train_df, val_df = train_test_split(
        df_train_all,
        test_size=val_size,
        stratify=df_train_all["primary_class"],
        random_state=seed,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def materialize_split(
    df: pd.DataFrame,
    images_out_dir: Path,
    labels_out_dir: Path,
    class_to_id: Dict[str, int],
    allow_replication_suffix: bool,
) -> None:
    """Materialize one split to YOLO workspace directories."""
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    for row in df.itertuples(index=False):
        src_img = Path(row.image_path)
        src_xml = Path(row.xml_path)
        rep_id = getattr(row, "replicate_id", 0)

        if allow_replication_suffix and int(rep_id) > 0:
            dst_stem = f"{src_img.stem}__rep{int(rep_id):04d}"
        else:
            dst_stem = src_img.stem

        dst_img = images_out_dir / f"{dst_stem}{src_img.suffix}"
        dst_label = labels_out_dir / f"{dst_stem}.txt"

        shutil.copy2(src_img, dst_img)
        write_yolo_label_from_xml(src_xml, dst_label, class_to_id)


def write_data_yaml(data_yaml_path: Path, train_images: Path, val_images: Path, test_images: Path, class_names: List[str]) -> None:
    """Write YOLO data.yaml."""
    data = {
        "path": str(data_yaml_path.parent.resolve()),
        "train": str(train_images.resolve()),
        "val": str(val_images.resolve()),
        "test": str(test_images.resolve()),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def export_manifest_csv(path: Path, df: pd.DataFrame) -> None:
    """Save a manifest csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def remove_cache_files(root: Path) -> None:
    """Delete Ultralytics cache files to avoid stale image-path references across reruns."""
    if not root.exists():
        return
    for p in root.rglob("*.cache"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def verify_split_integrity(images_dir: Path, labels_dir: Path) -> Dict[str, int]:
    """
    Verify that every YOLO label file has a matching image stem and vice versa.
    Returns simple counts for logging.
    """
    image_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    label_files = [p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    labels_without_images = sorted(label_stems - image_stems)
    images_without_labels = sorted(image_stems - label_stems)

    if labels_without_images:
        raise FileNotFoundError(
            f"{labels_dir}: found label files with no matching image stems. Example(s): {labels_without_images[:10]}"
        )
    if images_without_labels:
        raise FileNotFoundError(
            f"{images_dir}: found image files with no matching label stems. Example(s): {images_without_labels[:10]}"
        )

    return {
        "n_images": len(image_files),
        "n_labels": len(label_files),
    }


def prepare_yolo_workspace(cfg: Config) -> Dict:
    """Prepare YOLO workspace, manifests, balancing summary, and data.yaml."""
    clear_dir(cfg.workspace_dir)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    train_all_df = build_detection_manifest(cfg.train_images_dir, cfg.train_annotations_dir)
    test_df = build_detection_manifest(cfg.test_images_dir, cfg.test_annotations_dir)

    class_names = discover_class_names(train_all_df, test_df)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    train_df, val_df = stratified_train_val_split(train_all_df, cfg.val_size, cfg.seed)

    if cfg.rebalance_train:
        train_df_balanced = make_balanced_train_manifest(train_df, cfg.seed)
    else:
        train_df_balanced = train_df.copy()
        train_df_balanced["replicate_id"] = 0

    val_df = val_df.copy()
    val_df["replicate_id"] = 0
    test_df = test_df.copy()
    test_df["replicate_id"] = 0

    train_images_out = cfg.workspace_dir / "train/images"
    train_labels_out = cfg.workspace_dir / "train/labels"
    val_images_out = cfg.workspace_dir / "val/images"
    val_labels_out = cfg.workspace_dir / "val/labels"
    test_images_out = cfg.workspace_dir / "test/images"
    test_labels_out = cfg.workspace_dir / "test/labels"

    materialize_split(train_df_balanced, train_images_out, train_labels_out, class_to_id, allow_replication_suffix=True)
    materialize_split(val_df, val_images_out, val_labels_out, class_to_id, allow_replication_suffix=False)
    materialize_split(test_df, test_images_out, test_labels_out, class_to_id, allow_replication_suffix=False)

    export_manifest_csv(cfg.artifacts_dir / "manifests/train_all_manifest.csv", train_all_df)
    export_manifest_csv(cfg.artifacts_dir / "manifests/train_manifest.csv", train_df)
    export_manifest_csv(cfg.artifacts_dir / "manifests/train_balanced_manifest.csv", train_df_balanced)
    export_manifest_csv(cfg.artifacts_dir / "manifests/val_manifest.csv", val_df)
    export_manifest_csv(cfg.artifacts_dir / "manifests/test_manifest.csv", test_df)

    balance_summary = {
        "train_before": train_df["primary_class"].value_counts().sort_index().to_dict(),
        "train_after": train_df_balanced["primary_class"].value_counts().sort_index().to_dict(),
        "val": val_df["primary_class"].value_counts().sort_index().to_dict(),
        "test": test_df["primary_class"].value_counts().sort_index().to_dict(),
        "class_names": class_names,
    }
    with open(cfg.artifacts_dir / "balance_summary.json", "w", encoding="utf-8") as f:
        json.dump(balance_summary, f, indent=2)

    data_yaml_path = cfg.workspace_dir / "data.yaml"
    write_data_yaml(data_yaml_path, train_images_out, val_images_out, test_images_out, class_names)

    remove_cache_files(cfg.workspace_dir)
    train_stats = verify_split_integrity(train_images_out, train_labels_out)
    val_stats = verify_split_integrity(val_images_out, val_labels_out)
    test_stats = verify_split_integrity(test_images_out, test_labels_out)

    return {
        "data_yaml": data_yaml_path,
        "class_names": class_names,
        "class_to_id": class_to_id,
        "train_manifest": train_df,
        "train_balanced_manifest": train_df_balanced,
        "val_manifest": val_df,
        "test_manifest": test_df,
        "balance_summary": balance_summary,
        "split_stats": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
    }


def _safe_result_metric(obj: object, attr: str, default: float = float("nan")) -> float:
    """Safely retrieve a float metric attribute from an Ultralytics result object."""
    value = getattr(obj, attr, default)
    try:
        return float(value)
    except Exception:
        return default


def extract_detection_metrics(results: object) -> Dict:
    """Extract standard YOLO detection metrics."""
    box = getattr(results, "box", None)
    if box is None:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "map50": float("nan"),
            "map50_95": float("nan"),
            "per_class_map50_95": [],
        }

    per_class = getattr(box, "maps", [])
    if hasattr(per_class, "tolist"):
        per_class = per_class.tolist()

    return {
        "precision": _safe_result_metric(box, "mp"),
        "recall": _safe_result_metric(box, "mr"),
        "map50": _safe_result_metric(box, "map50"),
        "map50_95": _safe_result_metric(box, "map"),
        "per_class_map50_95": per_class,
    }


def train_yolo_detector(cfg: Config, data_yaml: Path) -> Path:
    """Train YOLO and return the path to best weights."""
    remove_cache_files(cfg.workspace_dir)
    model = YOLO(cfg.model_name)
    train_result = model.train(
        data=str(data_yaml),
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        patience=cfg.patience,
        device=cfg.device,
        project=str(cfg.runs_dir),
        name="neu_det_yolo",
        exist_ok=True,
        seed=cfg.seed,
        pretrained=True,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
    )

    save_dir = Path(train_result.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected best weights not found: {best_weights}")
    return best_weights


def validate_yolo_detector(weights: Path, data_yaml: Path, split: str, cfg: Config) -> Dict:
    """Run YOLO validation on a split and return standardized detection metrics."""
    remove_cache_files(cfg.workspace_dir)
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=str(cfg.runs_dir),
        name=f"neu_det_yolo_val_{split}",
        exist_ok=True,
    )
    return extract_detection_metrics(results)


def load_test_ground_truth_from_manifest(test_manifest: pd.DataFrame, class_to_id: Dict[str, int]) -> pd.DataFrame:
    """Map test manifest primary class names to integer ids."""
    df = test_manifest.copy()
    df["y_true"] = df["primary_class"].map(class_to_id).astype(int)
    return df


def infer_image_level_predictions(weights: Path, test_manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Predict one class per test image using the highest-confidence detected box."""
    model = YOLO(str(weights))
    image_paths = test_manifest["image_path"].tolist()

    preds = model.predict(
        source=image_paths,
        imgsz=cfg.imgsz,
        conf=cfg.pred_conf,
        device=cfg.device,
        verbose=False,
        save=False,
        stream=False,
    )

    rows: List[Dict] = []
    for img_path, pred in zip(image_paths, preds):
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            pred_class = -1
            pred_conf = 0.0
        else:
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            best_idx = int(np.argmax(confs))
            pred_class = int(clses[best_idx])
            pred_conf = float(confs[best_idx])

        rows.append({"image_path": str(img_path), "y_pred": pred_class, "pred_conf": pred_conf})

    return pd.DataFrame(rows)


def compute_image_level_scores(
    test_manifest: pd.DataFrame,
    df_pred: pd.DataFrame,
    class_names: List[str],
    class_to_id: Dict[str, int],
) -> Dict:
    """Compute accuracy, macro F1, weighted F1, per-class report, and confusion matrix."""
    gt_df = load_test_ground_truth_from_manifest(test_manifest, class_to_id)
    merged = gt_df.merge(df_pred, on="image_path", how="left")
    if merged["y_pred"].isna().any():
        merged["y_pred"] = merged["y_pred"].fillna(-1)

    y_true = merged["y_true"].astype(int).to_numpy()
    y_pred = merged["y_pred"].astype(int).to_numpy()

    miss_class = len(class_names)
    y_pred_safe = np.where(y_pred < 0, miss_class, y_pred)

    labels_for_report = list(range(len(class_names)))
    accuracy = float(np.mean(y_true == y_pred_safe))
    macro_f1 = float(f1_score(y_true, y_pred_safe, average="macro", labels=labels_for_report, zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred_safe, average="weighted", labels=labels_for_report, zero_division=0))
    report = classification_report(
        y_true,
        y_pred_safe,
        labels=labels_for_report,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred_safe, labels=labels_for_report)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm,
        "n_no_detection": int(np.sum(y_pred < 0)),
    }


def save_json(path: Path, payload: Dict) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_prepare(cfg: Config) -> Dict:
    """Prepare YOLO workspace and manifests."""
    prepared = prepare_yolo_workspace(cfg)
    print("Prepared YOLO workspace:")
    print("  data.yaml:", prepared["data_yaml"])
    print("  class_names:", prepared["class_names"])
    print("  train counts before:", prepared["balance_summary"]["train_before"])
    print("  train counts after :", prepared["balance_summary"]["train_after"])
    print("  split stats:", prepared["split_stats"])
    return prepared


def run_train(cfg: Config) -> Dict:
    """Prepare workspace, train YOLO, validate on val, and export metrics."""
    prepared = prepare_yolo_workspace(cfg)
    best_weights = train_yolo_detector(cfg, prepared["data_yaml"])

    val_metrics = validate_yolo_detector(best_weights, prepared["data_yaml"], split="val", cfg=cfg)
    payload = {
        "best_weights": str(best_weights),
        "val_detection_metrics": val_metrics,
        "class_names": prepared["class_names"],
    }
    save_json(cfg.artifacts_dir / "train_summary.json", payload)

    print("Training complete.")
    print("Best weights:", best_weights)
    print("Validation detection metrics:", val_metrics)
    return payload


def run_final(cfg: Config, weights: Path) -> Dict:
    """Prepare workspace, evaluate YOLO on test, and compute image-level metrics."""
    prepared = prepare_yolo_workspace(cfg)
    test_detection_metrics = validate_yolo_detector(weights, prepared["data_yaml"], split="test", cfg=cfg)

    df_pred = infer_image_level_predictions(weights, prepared["test_manifest"], cfg)
    image_level_scores = compute_image_level_scores(
        prepared["test_manifest"],
        df_pred,
        prepared["class_names"],
        prepared["class_to_id"],
    )

    save_json(cfg.artifacts_dir / "final_test_detection_metrics.json", test_detection_metrics)
    save_json(cfg.artifacts_dir / "final_test_image_level_metrics.json", {
        "accuracy": image_level_scores["accuracy"],
        "macro_f1": image_level_scores["macro_f1"],
        "weighted_f1": image_level_scores["weighted_f1"],
        "report": image_level_scores["report"],
        "n_no_detection": image_level_scores["n_no_detection"],
    })
    np.save(cfg.artifacts_dir / "final_test_confusion_matrix.npy", image_level_scores["confusion_matrix"])

    payload = {
        "weights": str(weights),
        "test_detection_metrics": test_detection_metrics,
        "test_image_level_metrics": {
            "accuracy": image_level_scores["accuracy"],
            "macro_f1": image_level_scores["macro_f1"],
            "weighted_f1": image_level_scores["weighted_f1"],
            "n_no_detection": image_level_scores["n_no_detection"],
        },
    }

    print("Final test evaluation complete.")
    print("Detection metrics:", test_detection_metrics)
    print("Image-level metrics:", payload["test_image_level_metrics"])
    return payload


def run_all(cfg: Config) -> Dict:
    """Full pipeline: prepare -> train -> final test evaluation."""
    prepared = prepare_yolo_workspace(cfg)
    best_weights = train_yolo_detector(cfg, prepared["data_yaml"])

    val_metrics = validate_yolo_detector(best_weights, prepared["data_yaml"], split="val", cfg=cfg)
    test_detection_metrics = validate_yolo_detector(best_weights, prepared["data_yaml"], split="test", cfg=cfg)

    df_pred = infer_image_level_predictions(best_weights, prepared["test_manifest"], cfg)
    image_level_scores = compute_image_level_scores(
        prepared["test_manifest"],
        df_pred,
        prepared["class_names"],
        prepared["class_to_id"],
    )

    payload = {
        "best_weights": str(best_weights),
        "val_detection_metrics": val_metrics,
        "test_detection_metrics": test_detection_metrics,
        "test_image_level_metrics": {
            "accuracy": image_level_scores["accuracy"],
            "macro_f1": image_level_scores["macro_f1"],
            "weighted_f1": image_level_scores["weighted_f1"],
            "n_no_detection": image_level_scores["n_no_detection"],
        },
        "class_names": prepared["class_names"],
    }

    save_json(cfg.artifacts_dir / "all_pipeline_summary.json", payload)
    save_json(cfg.artifacts_dir / "final_test_image_level_metrics.json", {
        "accuracy": image_level_scores["accuracy"],
        "macro_f1": image_level_scores["macro_f1"],
        "weighted_f1": image_level_scores["weighted_f1"],
        "report": image_level_scores["report"],
        "n_no_detection": image_level_scores["n_no_detection"],
    })
    np.save(cfg.artifacts_dir / "final_test_confusion_matrix.npy", image_level_scores["confusion_matrix"])

    print("Full YOLO pipeline complete.")
    print("Best weights:", best_weights)
    print("Validation detection metrics:", val_metrics)
    print("Test detection metrics:", test_detection_metrics)
    print("Test image-level metrics:", payload["test_image_level_metrics"])
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone YOLO detection pipeline for NEU-DET.")
    p.add_argument("--dataset_root", type=str, required=True, help="Root of NEU-DET containing train/ and validation/")
    p.add_argument("--run", type=str, required=True, choices=["prepare", "train", "final", "all"], help="Pipeline stage to run")
    p.add_argument("--weights", type=str, default="", help="Path to YOLO weights for --run final")
    p.add_argument("--artifacts_dir", type=str, default="./artifacts_yolo", help="Artifacts output directory")
    p.add_argument("--workspace_dir", type=str, default="./artifacts_yolo/yolo_workspace", help="Workspace directory")
    p.add_argument("--runs_dir", type=str, default="./artifacts_yolo/yolo_runs", help="YOLO runs directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--val_size", type=float, default=0.20, help="Fraction of train used for internal validation")
    p.add_argument("--imgsz", type=int, default=224, help="Input image size")
    p.add_argument("--epochs", type=int, default=50, help="YOLO training epochs")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--device", type=str, default="mps", help="Ultralytics device, e.g. mps, cpu, 0")
    p.add_argument("--model_name", type=str, default="yolov8n.pt", help="Ultralytics model, e.g. yolov8n.pt")
    p.add_argument("--pred_conf", type=float, default=0.25, help="Prediction confidence threshold for image-level metrics")
    p.add_argument("--rebalance_train", action="store_true", help="Enable balanced training replication")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        dataset_root=Path(args.dataset_root),
        artifacts_dir=Path(args.artifacts_dir),
        workspace_dir=Path(args.workspace_dir),
        runs_dir=Path(args.runs_dir),
        seed=args.seed,
        val_size=args.val_size,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        model_name=args.model_name,
        pred_conf=args.pred_conf,
        rebalance_train=args.rebalance_train,
    )

    set_global_seed(cfg.seed)

    print("Dataset root:", cfg.dataset_root.resolve())
    print("Artifacts dir:", cfg.artifacts_dir.resolve())
    print("Workspace dir:", cfg.workspace_dir.resolve())
    print("Run mode:", args.run)

    if args.run == "prepare":
        run_prepare(cfg)
    elif args.run == "train":
        run_train(cfg)
    elif args.run == "final":
        if not args.weights:
            raise ValueError("--weights is required for --run final")
        run_final(cfg, Path(args.weights))
    elif args.run == "all":
        run_all(cfg)
    else:
        raise ValueError(f"Unsupported run mode: {args.run}")


if __name__ == "__main__":
    main()
