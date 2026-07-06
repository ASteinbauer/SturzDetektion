#!/usr/bin/env python3
"""
export_test_sequences.py

Extrahiert echte Sensor-Sequenzen aus dem verarbeiteten SisFall/UP-Fall/UniMiB-Dataset
und exportiert sie als JSON-Asset fuer die Android-App.

Nur Test-Subjects (aus metadata.json split) werden verwendet.

Output:
    outputs/test_sequences.json
    -> Copiere nach: app/src/main/assets/test_sequences.json
"""

from pathlib import Path
import json
import numpy as np
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATASET_PATH = PROJECT_ROOT / "processed" / "fall_dataset_150x6.npz"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "test_sequences.json"

CLASS_NAMES = ["ruhig_alltag", "normale_bewegung", "fallaehnlich_aber_ok", "sturz"]
CLASS_LABELS_DE = {
    0: "Ruhig sitzen",
    1: "Normales Gehen",
    2: "Fallaehnlich (harmlos)",
    3: "Echter Sturz",
}
CLASS_EXPECTED = {
    0: "Normal",
    1: "Normal",
    2: "Verdächtig",
    3: "Sturz",
}

SAMPLES_PER_CLASS = 3


def main():
    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data["x"]          # [N, 150, 6]
    y = data["y"]          # [N]
    subjects = data["subjects"]
    datasets = data["datasets"]

    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    test_subjects = set(meta["split"]["test_subjects"])

    print(f"Dataset: X={X.shape}, y={y.shape}")
    print(f"Test-Subjects: {len(test_subjects)}")

    test_mask = np.array([str(s) in test_subjects for s in subjects])
    X_t = X[test_mask]
    y_t = y[test_mask]
    sub_t = subjects[test_mask]
    ds_t = datasets[test_mask]

    print(f"Test-Samples: {len(X_t)}")
    print("Klassen-Verteilung:", Counter(y_t.tolist()))

    rng = np.random.default_rng(42)
    sequences = []

    for cls in range(4):
        cls_mask = y_t == cls
        cls_X = X_t[cls_mask]
        cls_sub = sub_t[cls_mask]
        cls_ds = ds_t[cls_mask]

        if len(cls_X) == 0:
            print(f"WARNUNG: Keine Samples fuer Klasse {cls}")
            continue

        # Diverse Auswahl aus verschiedenen Datasets
        unique_ds = list(dict.fromkeys(str(d) for d in cls_ds))
        selected = []
        for ds in unique_ds:
            ds_mask = np.array([str(d) == ds for d in cls_ds])
            ds_idx = np.where(ds_mask)[0]
            if len(ds_idx) > 0:
                selected.append(int(rng.choice(ds_idx)))
            if len(selected) >= SAMPLES_PER_CLASS:
                break

        # Auffuellen falls noetig
        all_idx = list(range(len(cls_X)))
        while len(selected) < min(SAMPLES_PER_CLASS, len(cls_X)):
            idx = int(rng.choice(all_idx))
            if idx not in selected:
                selected.append(idx)

        for idx in selected[:SAMPLES_PER_CLASS]:
            seq = cls_X[idx]  # [150, 6] in m/s2 und rad/s
            acc_mag = np.linalg.norm(seq[:, :3], axis=1)
            gyro_mag = np.linalg.norm(seq[:, 3:], axis=1)

            # Kaputte/leere Samples filtern
            if np.max(acc_mag) < 2.0:
                print(f"  Klasse {cls}: Ueberspringe degeneriertes Sample (idx={idx})")
                continue

            pre_accel = float(np.mean(np.abs(acc_mag[:40] - 9.81)))
            pre_gyro = float(np.mean(gyro_mag[:40]))

            entry = {
                "class": int(cls),
                "class_name": CLASS_NAMES[cls],
                "label": CLASS_LABELS_DE[cls],
                "expected": CLASS_EXPECTED[cls],
                "dataset": str(cls_ds[idx]),
                "subject": str(cls_sub[idx]),
                "max_accel_ms2": round(float(np.max(acc_mag)), 2),
                "max_gyro_rads": round(float(np.max(gyro_mag)), 3),
                "pre_accel_activity": round(pre_accel, 3),
                "pre_gyro_activity": round(pre_gyro, 3),
                "values": [[round(float(v), 5) for v in row] for row in seq],
            }
            sequences.append(entry)

            print(
                f"  Klasse {cls} ({CLASS_NAMES[cls]}): ds={cls_ds[idx]},"
                f" subject={cls_sub[idx]},"
                f" max_acc={np.max(acc_mag):.1f},"
                f" pre_acc={pre_accel:.3f},"
                f" pre_gyro={pre_gyro:.4f}"
            )

    output = {
        "version": 1,
        "window_size": 150,
        "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
        "sequences": sequences,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nGespeichert: {OUTPUT_PATH}  ({size_kb:.1f} KB)")
    print(f"Sequenzen: {len(sequences)}")
    print(f"\n-> Copiere nach: app/src/main/assets/test_sequences.json")


if __name__ == "__main__":
    main()