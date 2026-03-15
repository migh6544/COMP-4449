#!/usr/bin/env python3
"""
Phase II – Rigorous CV + Balanced Training Pipeline for NEU-DET (Classification)

Ready-to-run standalone script:
- Explicit manifest construction (filepath + label) from directory structure
- Stratified K-Fold cross-validation (fixes non-representative validation)
- Balanced training via per-class oversampling + stochastic augmentation ("random replication")
- Two model tracks:
    1) CNN v1 (from scratch)
    2) EfficientNetB0 transfer (two-stage fine-tuning)
- Full scoring: accuracy, macro/weighted F1, per-class report, confusion matrices
- Artifact export per fold and final test report

Coding standard:
- Single source of truth via Config dataclass
- Determinism via set_global_seed()
- No augmentation in eval pipelines
- Split integrity checks (fold must contain all classes)

Usage (examples):
  python phase2_rigorous_cv_pipeline_ready.py --train_dir "/path/to/train" --test_dir "/path/to/test" --run cv_cnn
  python phase2_rigorous_cv_pipeline_ready.py --train_dir "/path/to/train" --test_dir "/path/to/test" --run cv_transfer
  python phase2_rigorous_cv_pipeline_ready.py --train_dir "/path/to/train" --test_dir "/path/to/test" --run final --final_model cnn_v1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# -----------------------------
# 0) Reproducibility utilities
# -----------------------------
def set_global_seed(seed: int) -> None:
    """Best-effort reproducibility across Python hash, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# 1) Configuration
# -----------------------------
@dataclass(frozen=True)
class Config:
    train_dir: Path
    test_dir: Path

    img_height: int = 200
    img_width: int = 200
    batch_size: int = 32
    seed: int = 42

    n_splits: int = 5

    epochs_stage1: int = 20
    epochs_stage2: int = 15
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-5

    unfreeze_layers: int = 20

    artifacts_dir: Path = Path("./artifacts_phase2")


# -----------------------------
# 2) Manifest creation
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images_with_labels(base_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a manifest from a Keras-compatible directory structure.

    Expected:
      base_dir/
        class_a/
        class_b/
        ...

    Returns:
      df: columns [filepath, label, label_id]
      class_names: sorted list of class folder names
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    class_names = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class subfolders found under: {base_dir}")

    rows: List[Tuple[str, str]] = []
    for cname in class_names:
        for fp in (base_dir / cname).glob("*"):
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                rows.append((str(fp), cname))

    df = pd.DataFrame(rows, columns=["filepath", "label"])
    if df.empty:
        raise ValueError(f"No images found under: {base_dir} (expected extensions: {sorted(IMG_EXTS)})")

    df["label"] = df["label"].astype("category")
    df["label_id"] = df["label"].cat.codes.astype(int)

    return df, class_names


# -----------------------------
# 3) tf.data pipelines
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

AUGMENT = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.08),
        keras.layers.RandomZoom(0.10),
    ],
    name="train_augmentation",
)


def _decode_and_resize(img_bytes: tf.Tensor, img_height: int, img_width: int) -> tf.Tensor:
    """
    Decode image bytes safely and resize.

    Note: tf.image.decode_image returns dynamic shapes; we enforce channel shape for downstream layers.
    """
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _load_image(
    filepath: tf.Tensor,
    label_id: tf.Tensor,
    img_height: int,
    img_width: int,
    num_classes: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    img_bytes = tf.io.read_file(filepath)
    x = _decode_and_resize(img_bytes, img_height, img_width)
    y = tf.one_hot(label_id, depth=num_classes)
    return x, y


def make_eval_ds(df: pd.DataFrame, cfg: Config, num_classes: int, shuffle: bool = False) -> tf.data.Dataset:
    """Evaluation dataset: NO augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label_id"].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 4096), seed=cfg.seed, reshuffle_each_iteration=False)
    ds = ds.map(
        lambda fp, y: _load_image(fp, y, cfg.img_height, cfg.img_width, num_classes),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(cfg.batch_size).prefetch(AUTOTUNE)
    return ds


def make_train_ds_balanced(df: pd.DataFrame, cfg: Config, num_classes: int) -> tf.data.Dataset:
    """
    Balanced training dataset via:
      - per-class datasets repeated (oversampling)
      - equal-probability sampling across classes
      - train-only augmentation ("random replication")
    """
    per_class: List[tf.data.Dataset] = []
    for k in range(num_classes):
        df_k = df[df["label_id"] == k]
        if len(df_k) == 0:
            raise ValueError(f"Class id {k} has 0 samples in the provided training fold.")
        ds_k = tf.data.Dataset.from_tensor_slices((df_k["filepath"].values, df_k["label_id"].values))
        ds_k = ds_k.shuffle(buffer_size=min(len(df_k), 2048), seed=cfg.seed, reshuffle_each_iteration=True)
        ds_k = ds_k.repeat()
        per_class.append(ds_k)

    ds = tf.data.Dataset.sample_from_datasets(
        per_class,
        weights=[1.0 / num_classes] * num_classes,
        seed=cfg.seed,
    )

    ds = ds.map(
        lambda fp, y: _load_image(fp, y, cfg.img_height, cfg.img_width, num_classes),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.map(lambda x, y: (AUGMENT(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(AUTOTUNE)
    return ds


# -----------------------------
# 4) Models
# -----------------------------
def build_cnn_v1(cfg: Config, num_classes: int) -> keras.Model:
    """Compact CNN baseline (stable, regularized)."""
    inputs = keras.Input(shape=(cfg.img_height, cfg.img_width, 3))
    x = inputs

    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.30)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.40)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_v1_phase2")


def build_transfer_efficientnetb0(cfg: Config, num_classes: int) -> Tuple[keras.Model, keras.Model]:
    """EfficientNetB0 transfer model. Returns (model, backbone)."""
    inputs = keras.Input(shape=(cfg.img_height, cfg.img_width, 3))

    # Our pipeline emits [0,1] floats; EfficientNet preprocess expects [0,255].
    x = keras.layers.Lambda(lambda t: t * 255.0)(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    backbone.trainable = False

    x = backbone.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.30)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="efficientnetb0_transfer_phase2")
    return model, backbone


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


# -----------------------------
# 5) Scoring + artifact export
# -----------------------------
def predict_labels(model: keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true_int, y_pred_int) for a dataset."""
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    for xb, yb in ds:
        preds = model.predict(xb, verbose=0)
        y_true.append(np.argmax(yb.numpy(), axis=1))
        y_pred.append(np.argmax(preds, axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_scores(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm,
    }


def save_fold_artifacts(cfg: Config, run_name: str, fold: int, scores: Dict) -> None:
    fold_dir = cfg.artifacts_dir / run_name / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    with open(fold_dir / "classification_report.json", "w") as f:
        json.dump(scores["report"], f, indent=2)

    np.save(fold_dir / "confusion_matrix.npy", scores["confusion_matrix"])

    with open(fold_dir / "summary.json", "w") as f:
        json.dump(
            {
                "accuracy": scores["accuracy"],
                "macro_f1": scores["macro_f1"],
                "weighted_f1": scores["weighted_f1"],
            },
            f,
            indent=2,
        )


def assert_fold_has_all_classes(df_fold: pd.DataFrame, num_classes: int, context: str) -> None:
    """Hard fail if a fold is missing classes."""
    present = set(df_fold["label_id"].unique().tolist())
    missing = [k for k in range(num_classes) if k not in present]
    if missing:
        raise ValueError(f"{context}: missing class ids {missing}. This breaks stratified evaluation.")


# -----------------------------
# 6) Cross-validation runners
# -----------------------------
def run_cv_cnn_v1(df_train: pd.DataFrame, cfg: Config, class_names: List[str]) -> pd.DataFrame:
    run_name = "cnn_v1"
    num_classes = len(class_names)

    X = df_train["filepath"].values
    y = df_train["label_id"].values

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    results: List[Dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
        df_va = df_train.iloc[va_idx].reset_index(drop=True)

        assert_fold_has_all_classes(df_tr, num_classes, f"{run_name} fold {fold} train")
        assert_fold_has_all_classes(df_va, num_classes, f"{run_name} fold {fold} val")

        train_ds = make_train_ds_balanced(df_tr, cfg, num_classes)
        val_ds = make_eval_ds(df_va, cfg, num_classes, shuffle=False)

        model = build_cnn_v1(cfg, num_classes)
        compile_model(model, lr=cfg.lr_stage1)

        steps_per_epoch = int(np.ceil(len(df_tr) / cfg.batch_size))

        ckpt_path = cfg.artifacts_dir / run_name / f"fold_{fold:02d}_best.keras"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs_stage1,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
        )

        y_true, y_pred = predict_labels(model, val_ds)
        scores = compute_scores(y_true, y_pred, class_names)
        save_fold_artifacts(cfg, run_name, fold, scores)

        results.append(
            {
                "run": run_name,
                "fold": fold,
                "val_accuracy": scores["accuracy"],
                "val_macro_f1": scores["macro_f1"],
                "val_weighted_f1": scores["weighted_f1"],
                "n_train": len(df_tr),
                "n_val": len(df_va),
            }
        )

    df_res = pd.DataFrame(results)
    out = cfg.artifacts_dir / run_name / "cv_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out, index=False)
    return df_res


def run_cv_transfer_efficientnetb0(df_train: pd.DataFrame, cfg: Config, class_names: List[str]) -> pd.DataFrame:
    run_name = "efficientnetb0_transfer"
    num_classes = len(class_names)

    X = df_train["filepath"].values
    y = df_train["label_id"].values

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    results: List[Dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
        df_va = df_train.iloc[va_idx].reset_index(drop=True)

        assert_fold_has_all_classes(df_tr, num_classes, f"{run_name} fold {fold} train")
        assert_fold_has_all_classes(df_va, num_classes, f"{run_name} fold {fold} val")

        train_ds = make_train_ds_balanced(df_tr, cfg, num_classes)
        val_ds = make_eval_ds(df_va, cfg, num_classes, shuffle=False)

        model, backbone = build_transfer_efficientnetb0(cfg, num_classes)

        # Stage 1: frozen backbone
        compile_model(model, lr=cfg.lr_stage1)
        steps_per_epoch = int(np.ceil(len(df_tr) / cfg.batch_size))

        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs_stage1,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
        )

        # Stage 2: unfreeze last N layers
        backbone.trainable = True
        for layer in backbone.layers[:-cfg.unfreeze_layers]:
            layer.trainable = False

        compile_model(model, lr=cfg.lr_stage2)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs_stage2,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
        )

        y_true, y_pred = predict_labels(model, val_ds)
        scores = compute_scores(y_true, y_pred, class_names)
        save_fold_artifacts(cfg, run_name, fold, scores)

        results.append(
            {
                "run": run_name,
                "fold": fold,
                "val_accuracy": scores["accuracy"],
                "val_macro_f1": scores["macro_f1"],
                "val_weighted_f1": scores["weighted_f1"],
                "n_train": len(df_tr),
                "n_val": len(df_va),
                "unfreeze_layers": cfg.unfreeze_layers,
            }
        )

    df_res = pd.DataFrame(results)
    out = cfg.artifacts_dir / run_name / "cv_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out, index=False)
    return df_res


# -----------------------------
# 7) Final train + test eval
# -----------------------------
def train_final_and_test(
    model_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cfg: Config,
    class_names: List[str],
) -> Dict:
    """Retrain on full train pool, then evaluate once on fixed test set."""
    num_classes = len(class_names)
    steps_per_epoch = int(np.ceil(len(df_train) / cfg.batch_size))

    train_ds = make_train_ds_balanced(df_train, cfg, num_classes)
    test_ds = make_eval_ds(df_test, cfg, num_classes, shuffle=False)

    if model_name == "cnn_v1":
        model = build_cnn_v1(cfg, num_classes)
        compile_model(model, lr=cfg.lr_stage1)
        model.fit(train_ds, epochs=cfg.epochs_stage1, steps_per_epoch=steps_per_epoch, verbose=1)

    elif model_name == "efficientnetb0_transfer":
        model, backbone = build_transfer_efficientnetb0(cfg, num_classes)
        compile_model(model, lr=cfg.lr_stage1)
        model.fit(train_ds, epochs=cfg.epochs_stage1, steps_per_epoch=steps_per_epoch, verbose=1)

        backbone.trainable = True
        for layer in backbone.layers[:-cfg.unfreeze_layers]:
            layer.trainable = False
        compile_model(model, lr=cfg.lr_stage2)
        model.fit(train_ds, epochs=cfg.epochs_stage2, steps_per_epoch=steps_per_epoch, verbose=1)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    y_true, y_pred = predict_labels(model, test_ds)
    scores = compute_scores(y_true, y_pred, class_names)

    out_dir = cfg.artifacts_dir / f"final_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "final_test_report.json", "w") as f:
        json.dump(scores["report"], f, indent=2)
    np.save(out_dir / "final_confusion_matrix.npy", scores["confusion_matrix"])
    model.save(out_dir / "final_model.keras")

    with open(out_dir / "final_summary.json", "w") as f:
        json.dump(
            {
                "test_accuracy": scores["accuracy"],
                "test_macro_f1": scores["macro_f1"],
                "test_weighted_f1": scores["weighted_f1"],
            },
            f,
            indent=2,
        )

    return scores


# -----------------------------
# 8) CLI + Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase II rigorous CV pipeline (classification).")
    p.add_argument("--train_dir", type=str, required=True, help="Training directory with class subfolders.")
    p.add_argument("--test_dir", type=str, required=True, help="Test directory with class subfolders.")
    p.add_argument("--artifacts_dir", type=str, default="./artifacts_phase2", help="Output folder for artifacts.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--img_size", type=int, default=200, help="Square input size (HxW).")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    p.add_argument("--n_splits", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--epochs_stage1", type=int, default=20, help="Epochs for stage 1.")
    p.add_argument("--epochs_stage2", type=int, default=15, help="Epochs for stage 2 (transfer fine-tune).")
    p.add_argument("--lr_stage1", type=float, default=1e-3, help="Learning rate for stage 1.")
    p.add_argument("--lr_stage2", type=float, default=1e-5, help="Learning rate for stage 2.")
    p.add_argument("--unfreeze_layers", type=int, default=20, help="Transfer: number of backbone layers to unfreeze.")
    p.add_argument(
        "--run",
        type=str,
        required=True,
        choices=["cv_cnn", "cv_transfer", "final"],
        help="What to execute: cv_cnn | cv_transfer | final",
    )
    p.add_argument(
        "--final_model",
        type=str,
        default="cnn_v1",
        choices=["cnn_v1", "efficientnetb0_transfer"],
        help="For --run final only: which model to train and test.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        img_height=args.img_size,
        img_width=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        n_splits=args.n_splits,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        unfreeze_layers=args.unfreeze_layers,
        artifacts_dir=Path(args.artifacts_dir),
    )

    set_global_seed(cfg.seed)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    df_train, class_names = list_images_with_labels(cfg.train_dir)
    df_test, _ = list_images_with_labels(cfg.test_dir)

    print("\nClasses:", class_names)
    print("Train rows:", len(df_train), " | Test rows:", len(df_test))
    print("\nTrain counts:\n", df_train["label"].value_counts())
    print("\nTest counts:\n", df_test["label"].value_counts())

    if args.run == "cv_cnn":
        df_res = run_cv_cnn_v1(df_train, cfg, class_names)
        print("\nCV results (CNN v1):\n", df_res)
        print("\nSummary:\n", df_res[["val_accuracy", "val_macro_f1", "val_weighted_f1"]].describe())

    elif args.run == "cv_transfer":
        df_res = run_cv_transfer_efficientnetb0(df_train, cfg, class_names)
        print("\nCV results (EfficientNetB0):\n", df_res)
        print("\nSummary:\n", df_res[["val_accuracy", "val_macro_f1", "val_weighted_f1"]].describe())

    elif args.run == "final":
        scores = train_final_and_test(args.final_model, df_train, df_test, cfg, class_names)
        print("\nFINAL TEST:")
        print("  accuracy:", scores["accuracy"])
        print("  macro_f1:", scores["macro_f1"])
        print("  weighted_f1:", scores["weighted_f1"])
    else:
        raise ValueError("Unknown run mode.")


if __name__ == "__main__":
    main()
