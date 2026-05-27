#!/usr/bin/env python3
r"""
train_real_1dcnn_150x6.py

Trainiert ein 1D-CNN fuer die echte Sturzdetektions-App mit dem zuvor erzeugten Dataset:

Input:
    processed/fall_dataset_150x6.npz

Input-Shape pro Beispiel:
    150 x 6

Kanaele:
    ax, ay, az, gx, gy, gz

Klassen:
    0 = ruhig_alltag
    1 = normale_bewegung
    2 = fallaehnlich_aber_ok
    3 = sturz

Output:
    models/fall_detection_real_150x6.keras
    models/fall_detection_real_150x6_float32.tflite
    models/fall_detection_real_150x6_dynamic_quant.tflite
    models/metadata.json
    outputs/real_training_report.txt
    outputs/real_confusion_matrix.png
    outputs/real_training_accuracy.png
    outputs/real_training_loss.png

Installation:
    pip install numpy tensorflow matplotlib

Ausfuehren:
    cd C:\SturzDetektion\fall_realdata_project
    python .\scripts\train_real_1dcnn_150x6.py
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\SturzDetektion\fall_realdata_project")
DATASET_PATH = PROJECT_ROOT / "processed" / "fall_dataset_150x6.npz"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUTS_DIR / "real_training_report.txt"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW_SIZE = 150
N_CHANNELS = 6
N_CLASSES = 4
TARGET_HZ = 50

CHANNEL_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]
CLASS_NAMES = [
    "ruhig_alltag",
    "normale_bewegung",
    "fallaehnlich_aber_ok",
    "sturz",
]

# Training
BATCH_SIZE = 128
EPOCHS = 60
LEARNING_RATE = 1e-3

# Augmentation nur auf Training.
USE_AUGMENTATION = True
NOISE_STD = 0.035
SCALE_MIN = 0.92
SCALE_MAX = 1.08
GYRO_DROPOUT_PROB = 0.30
TIME_SHIFT_MAX = 10  # Samples nach links/rechts


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

REPORT_LINES: List[str] = []


def log(text: str = "") -> None:
    print(text)
    REPORT_LINES.append(str(text))


def write_report() -> None:
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))


# ---------------------------------------------------------------------
# Daten laden und splitten
# ---------------------------------------------------------------------

def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {DATASET_PATH}")

    data = np.load(DATASET_PATH, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    subjects = data["subjects"].astype(str)
    datasets = data["datasets"].astype(str)

    if x.shape[1:] != (WINDOW_SIZE, N_CHANNELS):
        raise ValueError(f"Falsche X-Shape: {x.shape}, erwartet (*, {WINDOW_SIZE}, {N_CHANNELS})")

    return x, y, subjects, datasets


def summarize_split(name: str, y: np.ndarray, datasets: np.ndarray, subjects: np.ndarray) -> None:
    log(f"\n{name}")
    log("-" * len(name))
    log(f"Samples: {len(y)}")
    log(f"Subjects: {len(set(subjects.tolist()))}")

    counts = Counter(y.tolist())
    for class_id, class_name in enumerate(CLASS_NAMES):
        log(f"  {class_id} {class_name:22s}: {counts.get(class_id, 0)}")

    ds_counts = Counter(datasets.tolist())
    log("  Nach Dataset:")
    for ds, count in sorted(ds_counts.items()):
        log(f"    {ds:10s}: {count}")


def subject_stratified_split(
    y: np.ndarray,
    subjects: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split nach Subjects, nicht nach Fenstern.

    Einfache Greedy-Variante:
    - Subjects werden gemischt
    - dann nacheinander dem Split zugeordnet, der noch am meisten Samples braucht

    Das ist nicht perfekt klassensstratifiziert, vermeidet aber Leakage zwischen
    Train/Val/Test durch identische Personen.
    """
    rng = np.random.default_rng(SEED)

    subject_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, s in enumerate(subjects):
        subject_to_indices[str(s)].append(idx)

    subject_items = list(subject_to_indices.items())
    rng.shuffle(subject_items)

    n_total = len(y)
    target_train = int(n_total * train_ratio)
    target_val = int(n_total * val_ratio)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for subject, idxs in subject_items:
        # Greedy nach aktuellem Fuellstand.
        if len(train_idx) < target_train:
            train_idx.extend(idxs)
        elif len(val_idx) < target_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------
# Normalisierung und Augmentation
# ---------------------------------------------------------------------

def compute_normalization(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    training: bool,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=min(len(y), 20000), seed=SEED, reshuffle_each_iteration=True)
        if USE_AUGMENTATION:
            ds = ds.map(augment_sample_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


@tf.function
def augment_sample_tf(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Leichte Zeitreihen-Augmentation auf normalisierten Daten."""
    x = tf.cast(x, tf.float32)

    # Leichtes Rauschen.
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=NOISE_STD, dtype=tf.float32)
    x = x + noise

    # Kanalweise leichte Skalierung.
    scales = tf.random.uniform([1, N_CHANNELS], minval=SCALE_MIN, maxval=SCALE_MAX, dtype=tf.float32)
    x = x * scales

    # Gyro-Dropout: gx/gy/gz gelegentlich auf 0 setzen.
    r = tf.random.uniform([], 0.0, 1.0)
    def drop_gyro() -> tf.Tensor:
        acc = x[:, 0:3]
        gyro = tf.zeros_like(x[:, 3:6])
        return tf.concat([acc, gyro], axis=1)
    x = tf.cond(r < GYRO_DROPOUT_PROB, drop_gyro, lambda: x)

    # Kleine Zeitverschiebung.
    shift = tf.random.uniform([], minval=-TIME_SHIFT_MAX, maxval=TIME_SHIFT_MAX + 1, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=0)

    return x, y


# ---------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------

def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, N_CHANNELS), name="sensor_window")

    x = tf.keras.layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    counts = Counter(y_train.tolist())
    total = len(y_train)
    weights: Dict[int, float] = {}
    for class_id in range(N_CLASSES):
        count = counts.get(class_id, 1)
        weights[class_id] = total / (N_CLASSES * count)
    return weights


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = cm[class_id, class_id]
        fp = cm[:, class_id].sum() - tp
        fn = cm[class_id, :].sum() - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        result[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[class_id, :].sum()),
        }
    return result


def log_classification_report(cm: np.ndarray) -> None:
    log("\nConfusion Matrix:")
    log(str(cm))

    report = classification_report(cm)
    log("\nKlassenauswertung:")
    for class_name, values in report.items():
        log(
            f"  {class_name:22s} "
            f"precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} "
            f"f1={values['f1']:.3f} "
            f"support={values['support']}"
        )


def plot_confusion_matrix(cm: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Echte Klasse")
    plt.xticks(range(N_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(N_CLASSES), CLASS_NAMES)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "real_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_history(history: tf.keras.callbacks.History) -> None:
    fig = plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "real_training_accuracy.png", dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "real_training_loss.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_tflite(model: tf.keras.Model) -> None:
    float_path = MODELS_DIR / "fall_detection_real_150x6_float32.tflite"
    quant_path = MODELS_DIR / "fall_detection_real_150x6_dynamic_quant.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(float_path, "wb") as f:
        f.write(tflite_model)

    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant = converter_quant.convert()
    with open(quant_path, "wb") as f:
        f.write(tflite_quant)

    log(f"\nTFLite exportiert:")
    log(f"  {float_path}")
    log(f"  {quant_path}")


def save_metadata(mean: np.ndarray, std: np.ndarray, train_subjects: Iterable[str], val_subjects: Iterable[str], test_subjects: Iterable[str]) -> None:
    metadata = {
        "sample_rate_hz": TARGET_HZ,
        "window_seconds": WINDOW_SIZE / TARGET_HZ,
        "window_size": WINDOW_SIZE,
        "channels": CHANNEL_NAMES,
        "classes": CLASS_NAMES,
        "normalization": {
            "mean": mean.reshape(-1).astype(float).tolist(),
            "std": std.reshape(-1).astype(float).tolist(),
        },
        "input_shape": [1, WINDOW_SIZE, N_CHANNELS],
        "output_shape": [1, N_CLASSES],
        "model_file": "fall_detection_real_150x6_float32.tflite",
        "note": "Android muss dieselbe Normalisierung verwenden: normalized = (raw - mean[channel]) / std[channel]",
        "split": {
            "train_subjects": sorted(set(map(str, train_subjects))),
            "val_subjects": sorted(set(map(str, val_subjects))),
            "test_subjects": sorted(set(map(str, test_subjects))),
        },
    }

    out_path = MODELS_DIR / "metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log(f"Metadata gespeichert: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    log("Train Real Fall Detection 1D-CNN 150x6")
    log("=" * 60)
    log(f"TensorFlow: {tf.__version__}")
    log(f"Dataset: {DATASET_PATH}")

    x, y, subjects, datasets = load_dataset()
    log(f"\nGeladen:")
    log(f"  X: {x.shape}")
    log(f"  y: {y.shape}")
    log(f"  subjects: {subjects.shape}")
    log(f"  datasets: {datasets.shape}")
    log(f"  NaN: {np.isnan(x).any()}")
    log(f"  Inf: {np.isinf(x).any()}")
    log(f"  X min/max: {float(np.min(x)):.5f} / {float(np.max(x)):.5f}")

    train_idx, val_idx, test_idx = subject_stratified_split(y, subjects)

    x_train_raw, y_train = x[train_idx], y[train_idx]
    x_val_raw, y_val = x[val_idx], y[val_idx]
    x_test_raw, y_test = x[test_idx], y[test_idx]

    subjects_train = subjects[train_idx]
    subjects_val = subjects[val_idx]
    subjects_test = subjects[test_idx]

    datasets_train = datasets[train_idx]
    datasets_val = datasets[val_idx]
    datasets_test = datasets[test_idx]

    summarize_split("Train Split", y_train, datasets_train, subjects_train)
    summarize_split("Validation Split", y_val, datasets_val, subjects_val)
    summarize_split("Test Split", y_test, datasets_test, subjects_test)

    mean, std = compute_normalization(x_train_raw)
    x_train = normalize(x_train_raw, mean, std)
    x_val = normalize(x_val_raw, mean, std)
    x_test = normalize(x_test_raw, mean, std)

    log("\nNormalisierung:")
    for i, ch in enumerate(CHANNEL_NAMES):
        log(f"  {ch}: mean={mean.reshape(-1)[i]:.6f}, std={std.reshape(-1)[i]:.6f}")

    train_ds = make_tf_dataset(x_train, y_train, training=True)
    val_ds = make_tf_dataset(x_val, y_val, training=False)
    test_ds = make_tf_dataset(x_test, y_test, training=False)

    model = build_model()
    model.summary(print_fn=log)

    class_weights = compute_class_weights(y_train)
    log("\nClass Weights:")
    for k, v in class_weights.items():
        log(f"  {k} {CLASS_NAMES[k]:22s}: {v:.4f}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "fall_detection_real_150x6_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    log(f"\nTest Loss: {test_loss:.4f}")
    log(f"Test Accuracy: {test_acc:.4f}")

    y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix_np(y_test, y_pred, N_CLASSES)
    log_classification_report(cm)
    plot_confusion_matrix(cm)

    keras_path = MODELS_DIR / "fall_detection_real_150x6.keras"
    model.save(keras_path)
    log(f"\nKeras Modell gespeichert: {keras_path}")

    export_tflite(model)
    save_metadata(mean, std, subjects_train, subjects_val, subjects_test)

    # Beispielvorhersagen aus dem Testsplit.
    log("\nBeispielvorhersagen:")
    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(len(x_test), size=min(5, len(x_test)), replace=False)
    for idx in sample_indices:
        probs = y_prob[idx]
        pred = int(np.argmax(probs))
        true = int(y_test[idx])
        prob_text = ", ".join(f"{CLASS_NAMES[i]}={probs[i]:.3f}" for i in range(N_CLASSES))
        log(f"  true={CLASS_NAMES[true]:22s} pred={CLASS_NAMES[pred]:22s} richtig={true == pred} | {prob_text}")

    log("\nFertig.")
    log(f"Report: {REPORT_PATH}")
    write_report()


if __name__ == "__main__":
    main()
