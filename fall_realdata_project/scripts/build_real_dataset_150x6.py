#!/usr/bin/env python3
r"""
build_real_dataset_150x6.py

Erzeugt ein gemeinsames Trainingsdataset fuer die Sturzdetektions-App:

Input pro Beispiel:
    150 x 6

Kanaele:
    ax, ay, az, gx, gy, gz

Zielklassen:
    0 = ruhig_alltag
    1 = normale_bewegung
    2 = fallaehnlich_aber_ok
    3 = sturz

Unterstuetzte Rohdaten in dieser Projektstruktur:
    C:\SturzDetektion\fall_realdata_project\raw\sisfall\SisFall_dataset
    C:\SturzDetektion\fall_realdata_project\raw\unimib\UniMiB
    C:\SturzDetektion\fall_realdata_project\raw\upfall\CompleteDataSet.csv

Output:
    processed/fall_dataset_150x6.npz
    processed/fall_dataset_150x6_metadata.json
    outputs/build_dataset_report.txt

Installation:
    pip install numpy pandas

Ausfuehren:
    cd C:\SturzDetektion\fall_realdata_project
    python .\scripts\build_real_dataset_150x6.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Projekt-Konfiguration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SISFALL_ROOT = PROJECT_ROOT / "raw" / "sisfall" / "SisFall_dataset"
UNIMIB_ROOT = PROJECT_ROOT / "raw" / "unimib" / "UniMiB"
UPFALL_CSV = PROJECT_ROOT / "raw" / "upfall" / "CompleteDataSet.csv"
PERSONAL_ROOT = PROJECT_ROOT / "raw" / "personal"

PROCESSED_DIR = PROJECT_ROOT / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NPZ = PROCESSED_DIR / "fall_dataset_150x6.npz"
OUTPUT_METADATA = PROCESSED_DIR / "fall_dataset_150x6_metadata.json"
OUTPUT_REPORT = OUTPUTS_DIR / "build_dataset_report.txt"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET_HZ = 50
WINDOW_SIZE = 150
N_CHANNELS = 6
WINDOW_SECONDS = WINDOW_SIZE / TARGET_HZ

CLASS_NAMES = [
    "ruhig_alltag",
    "normale_bewegung",
    "fallaehnlich_aber_ok",
    "sturz",
]

CHANNEL_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]
PERSONAL_REQUIRED_COLUMNS = ["timestamp", "label", "ax", "ay", "az", "gx", "gy", "gz"]

# Damit das Dataset nicht riesig wird. Bei Bedarf erhoehen.
MAX_WINDOWS_PER_CLASS = 10000

# Wenn True, werden bei Sturz-Sequenzen mehrere Fenster um den Impact erzeugt.
MULTI_OFFSET_FALL_WINDOWS = True


# ---------------------------------------------------------------------
# Label-Konstanten
# ---------------------------------------------------------------------

LABEL_RUHIG = 0
LABEL_NORMAL = 1
LABEL_FALLAEHNLICH = 2
LABEL_STURZ = 3

PERSONAL_LABEL_MAP = {
    "ruhig_alltag": LABEL_RUHIG,
    "normale_bewegung": LABEL_NORMAL,
    "fallaehnlich_aber_ok": LABEL_FALLAEHNLICH,
    "sturz": LABEL_STURZ,
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def log_line(report_lines: List[str], text: str = "") -> None:
    print(text)
    report_lines.append(str(text))


def safe_float_array(values: Iterable[float]) -> Optional[np.ndarray]:
    try:
        return np.array([float(v) for v in values], dtype=np.float32)
    except Exception:
        return None


def resample_linear(seq: np.ndarray, source_hz: float, target_hz: float = TARGET_HZ) -> np.ndarray:
    """Resample einer Sequenz [T, C] von source_hz nach target_hz."""
    if seq.ndim != 2:
        raise ValueError(f"seq muss 2D sein, bekommen: {seq.shape}")
    if len(seq) < 2:
        return seq.astype(np.float32)
    if abs(source_hz - target_hz) < 1e-6:
        return seq.astype(np.float32)

    duration = (len(seq) - 1) / float(source_hz)
    new_len = int(round(duration * target_hz)) + 1
    new_len = max(new_len, 2)

    old_t = np.linspace(0.0, duration, len(seq), dtype=np.float32)
    new_t = np.linspace(0.0, duration, new_len, dtype=np.float32)

    out = np.empty((new_len, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(new_t, old_t, seq[:, c]).astype(np.float32)
    return out


def resample_by_timestamps(seq: np.ndarray, timestamps: np.ndarray, target_hz: float = TARGET_HZ) -> np.ndarray:
    """Resample mit echten Sekunden-Zeitstempeln."""
    if len(seq) < 2:
        return seq.astype(np.float32)

    t = np.asarray(timestamps, dtype=np.float64)
    t = t - t[0]

    # Doppelte oder kaputte Timestamps entfernen.
    valid = np.isfinite(t)
    seq = seq[valid]
    t = t[valid]
    if len(seq) < 2:
        return seq.astype(np.float32)

    order = np.argsort(t)
    t = t[order]
    seq = seq[order]

    unique_mask = np.concatenate([[True], np.diff(t) > 1e-6])
    t = t[unique_mask]
    seq = seq[unique_mask]
    if len(seq) < 2:
        return seq.astype(np.float32)

    duration = float(t[-1])
    if duration <= 0:
        return seq.astype(np.float32)

    new_len = int(round(duration * target_hz)) + 1
    new_len = max(new_len, 2)
    new_t = np.linspace(0.0, duration, new_len, dtype=np.float64)

    out = np.empty((new_len, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(new_t, t, seq[:, c]).astype(np.float32)
    return out


def pad_or_crop_window(window: np.ndarray, size: int = WINDOW_SIZE) -> np.ndarray:
    """Bringt ein Fenster auf exakt WINDOW_SIZE."""
    if len(window) == size:
        return window.astype(np.float32)
    if len(window) > size:
        return window[:size].astype(np.float32)

    pad_len = size - len(window)
    if len(window) == 0:
        return np.zeros((size, N_CHANNELS), dtype=np.float32)
    pad = np.repeat(window[-1:, :], pad_len, axis=0)
    return np.vstack([window, pad]).astype(np.float32)


def acceleration_magnitude(seq: np.ndarray) -> np.ndarray:
    return np.linalg.norm(seq[:, 0:3], axis=1)


def extract_sliding_windows(seq: np.ndarray, stride: int = 75) -> List[np.ndarray]:
    windows: List[np.ndarray] = []
    if len(seq) < WINDOW_SIZE:
        windows.append(pad_or_crop_window(seq))
        return windows

    for start in range(0, len(seq) - WINDOW_SIZE + 1, stride):
        windows.append(seq[start : start + WINDOW_SIZE].astype(np.float32))
    return windows


def extract_impact_windows(seq: np.ndarray) -> List[np.ndarray]:
    """Fenster um den staerksten Beschleunigungspeak."""
    if len(seq) == 0:
        return []

    mag = acceleration_magnitude(seq)
    peak_idx = int(np.argmax(mag))

    # 1.2 Sekunden vor Peak, 1.8 Sekunden nach Peak.
    base_start = peak_idx - int(1.2 * TARGET_HZ)

    offsets = [0]
    if MULTI_OFFSET_FALL_WINDOWS:
        offsets = [-25, 0, 25]

    windows: List[np.ndarray] = []
    for off in offsets:
        start = base_start + off
        start = max(0, min(start, max(0, len(seq) - WINDOW_SIZE)))
        end = start + WINDOW_SIZE
        windows.append(pad_or_crop_window(seq[start:end]))

    return windows


def balance_dataset(
    x_list: List[np.ndarray],
    y_list: List[int],
    subject_list: List[str],
    dataset_list: List[str],
    max_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices_by_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(y_list):
        indices_by_class[int(y)].append(i)

    selected: List[int] = []
    rng = np.random.default_rng(SEED)
    for label in range(len(CLASS_NAMES)):
        idxs = indices_by_class.get(label, [])
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, size=max_per_class, replace=False).tolist()
        selected.extend(idxs)

    rng.shuffle(selected)

    x = np.stack([x_list[i] for i in selected]).astype(np.float32)
    y = np.array([y_list[i] for i in selected], dtype=np.int64)
    subjects = np.array([subject_list[i] for i in selected], dtype=object)
    datasets = np.array([dataset_list[i] for i in selected], dtype=object)
    return x, y, subjects, datasets


# ---------------------------------------------------------------------
# SisFall Loader
# ---------------------------------------------------------------------

# Hinweis:
# SisFall-Dateien enthalten 9 Spalten:
#   0..2: Accelerometer 1
#   3..5: Gyroscope
#   6..8: Accelerometer 2
# Wir nutzen fuer das App-Modell: Acc1 + Gyro.
# Skalierung ist naeherungsweise:
#   Acc1 raw / 256 * 9.81 => m/s^2
#   Gyro raw / 14.375 * pi/180 => rad/s
# Falls du spaeter andere Skalierung testen willst, passe diese Konstanten an.
SISFALL_SOURCE_HZ = 200
SISFALL_ACC_SCALE = 9.81 / 256.0
SISFALL_GYRO_SCALE = math.pi / 180.0 / 14.375


def parse_sisfall_txt(path: Path) -> Optional[np.ndarray]:
    rows: List[np.ndarray] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip().replace(";", "")
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                if len(parts) < 9:
                    continue
                arr = safe_float_array(parts[:9])
                if arr is not None and arr.shape[0] == 9:
                    rows.append(arr)
    except Exception:
        return None

    if not rows:
        return None

    raw = np.stack(rows).astype(np.float32)

    acc = raw[:, 0:3] * SISFALL_ACC_SCALE
    gyro = raw[:, 3:6] * SISFALL_GYRO_SCALE
    seq = np.concatenate([acc, gyro], axis=1).astype(np.float32)
    return seq


def map_sisfall_label(activity_code: str) -> int:
    # Alle Fxx sind Stuerze.
    if activity_code.startswith("F"):
        return LABEL_STURZ

    # D01-D04 sind einfache Bewegungen wie Gehen/Joggen.
    normal_codes = {"D01", "D02", "D03", "D04"}

    # D05-D19 sind ADLs, darunter Treppen, Hinsetzen/Aufstehen, Hinlegen,
    # Buecken, Springen etc. Fuer eine Sturz-App sind viele davon fall-aehnlich.
    if activity_code in normal_codes:
        return LABEL_NORMAL

    return LABEL_FALLAEHNLICH


def load_sisfall(report_lines: List[str]) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    subject_list: List[str] = []
    dataset_list: List[str] = []

    if not SISFALL_ROOT.exists():
        log_line(report_lines, f"SisFall nicht gefunden: {SISFALL_ROOT}")
        return x_list, y_list, subject_list, dataset_list

    txt_files = sorted(p for p in SISFALL_ROOT.rglob("*.txt") if p.name.lower() != "readme.txt")
    log_line(report_lines, f"SisFall TXT-Dateien: {len(txt_files)}")

    for i, path in enumerate(txt_files, start=1):
        name_parts = path.stem.split("_")
        if len(name_parts) < 3:
            continue
        activity_code = name_parts[0]
        subject = name_parts[1]
        label = map_sisfall_label(activity_code)

        seq = parse_sisfall_txt(path)
        if seq is None or len(seq) < 10:
            continue

        seq = resample_linear(seq, SISFALL_SOURCE_HZ, TARGET_HZ)

        if label == LABEL_STURZ:
            windows = extract_impact_windows(seq)
        else:
            windows = extract_sliding_windows(seq, stride=75)

        for w in windows:
            x_list.append(w)
            y_list.append(label)
            subject_list.append(f"sisfall_{subject}")
            dataset_list.append("sisfall")

        if i % 500 == 0:
            log_line(report_lines, f"  SisFall verarbeitet: {i}/{len(txt_files)}")

    return x_list, y_list, subject_list, dataset_list


# ---------------------------------------------------------------------
# UniMiB Loader
# ---------------------------------------------------------------------

# UniMiB acc_data.npy hat shape [11771, 453] = 151 Samples * 3 Achsen.
# In dieser NPY-Version liegen die Achsen normalerweise blockweise:
#   x[0:151], y[151:302], z[302:453]
# Gyro fehlt, daher gx/gy/gz = 0.


def unimib_row_to_150x6(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.shape[0] != 453:
        raise ValueError(f"UniMiB row muss 453 Werte haben, bekommen: {row.shape}")

    acc = row.reshape(3, 151).T  # [151, 3]
    acc = acc[:WINDOW_SIZE, :]   # [150, 3]
    gyro = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
    return np.concatenate([acc, gyro], axis=1).astype(np.float32)


def map_unimib_label(activity_id: int) -> int:
    # 1 Standing up from sitting
    # 2 Standing up from laying
    # 3 Walking
    # 4 Running
    # 5 Going upstairs
    # 6 Jumping
    # 7 Going downstairs
    # 8 Lying down from standing
    # 9 Sitting down
    # 10-17 Falltypen
    if activity_id >= 10:
        return LABEL_STURZ
    if activity_id in {3, 4, 5, 7}:
        return LABEL_NORMAL
    # Aufstehen, Hinlegen, Hinsetzen, Springen = fall-aehnlich aber okay.
    return LABEL_FALLAEHNLICH


def load_unimib(report_lines: List[str]) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    subject_list: List[str] = []
    dataset_list: List[str] = []

    data_path = UNIMIB_ROOT / "acc_data.npy"
    labels_path = UNIMIB_ROOT / "acc_labels.npy"
    if not data_path.exists() or not labels_path.exists():
        log_line(report_lines, f"UniMiB nicht gefunden: {UNIMIB_ROOT}")
        return x_list, y_list, subject_list, dataset_list

    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    log_line(report_lines, f"UniMiB data shape: {data.shape}")
    log_line(report_lines, f"UniMiB labels shape: {labels.shape}")

    for i in range(len(data)):
        activity_id = int(labels[i, 0])
        subject_id = int(labels[i, 1])
        label = map_unimib_label(activity_id)

        try:
            window = unimib_row_to_150x6(data[i])
        except Exception:
            continue

        x_list.append(window)
        y_list.append(label)
        subject_list.append(f"unimib_S{subject_id:02d}")
        dataset_list.append("unimib")

    return x_list, y_list, subject_list, dataset_list


# ---------------------------------------------------------------------
# UP-Fall Loader
# ---------------------------------------------------------------------

# UP-Fall CompleteDataSet.csv hat zwei Header-Zeilen.
# Wir nutzen RightPocket, weil das der Smartphone-/Hosentaschenposition am naechsten kommt:
#   08,09,10 = RightPocketAccelerometer in g
#   11,12,13 = RightPocketAngularVelocity in deg/s

UPFALL_ACC_COLS = [8, 9, 10]
UPFALL_GYRO_COLS = [11, 12, 13]
UPFALL_TIME_COL = 0
UPFALL_SUBJECT_COL = 43
UPFALL_ACTIVITY_COL = 44
UPFALL_TRIAL_COL = 45


def map_upfall_label(activity_id: int) -> int:
    # UP-Fall Activity 1-5 = Falltypen.
    # 6 = Walking, 7 = Standing, 8 = Sitting,
    # 9 = Picking up object, 10 = Jumping, 11 = Laying.
    if activity_id in {1, 2, 3, 4, 5}:
        return LABEL_STURZ
    if activity_id == 6:
        return LABEL_NORMAL
    if activity_id in {7, 8, 11}:
        return LABEL_RUHIG
    if activity_id in {9, 10}:
        return LABEL_FALLAEHNLICH
    return LABEL_FALLAEHNLICH


def load_upfall(report_lines: List[str]) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    subject_list: List[str] = []
    dataset_list: List[str] = []

    if not UPFALL_CSV.exists():
        log_line(report_lines, f"UP-Fall nicht gefunden: {UPFALL_CSV}")
        return x_list, y_list, subject_list, dataset_list

    log_line(report_lines, "Lade UP-Fall CSV. Das kann kurz dauern ...")
    df = pd.read_csv(UPFALL_CSV, skiprows=[1])
    log_line(report_lines, f"UP-Fall shape: {df.shape}")

    # Numerische Spalten nach Position extrahieren.
    group_cols = [df.columns[UPFALL_SUBJECT_COL], df.columns[UPFALL_ACTIVITY_COL], df.columns[UPFALL_TRIAL_COL]]
    time_col = df.columns[UPFALL_TIME_COL]

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    log_line(report_lines, f"UP-Fall Sequenzen: {len(grouped)}")

    for (subject, activity, trial), g in grouped:
        try:
            subject_i = int(subject)
            activity_i = int(activity)
        except Exception:
            continue

        label = map_upfall_label(activity_i)

        acc_g = g.iloc[:, UPFALL_ACC_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        gyro_deg = g.iloc[:, UPFALL_GYRO_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        # NaNs grob behandeln.
        data = np.concatenate([acc_g * 9.81, gyro_deg * (math.pi / 180.0)], axis=1).astype(np.float32)
        if np.isnan(data).any():
            data = pd.DataFrame(data).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)

        # Zeitstempel in Sekunden.
        try:
            ts = pd.to_datetime(g[time_col], errors="coerce")
            seconds = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
            if np.isfinite(seconds).sum() >= 2 and np.nanmax(seconds) > 0:
                seq = resample_by_timestamps(data, seconds, TARGET_HZ)
            else:
                # Fallback: UP-Fall ist grob ca. 18-20 Hz.
                seq = resample_linear(data, source_hz=18.0, target_hz=TARGET_HZ)
        except Exception:
            seq = resample_linear(data, source_hz=18.0, target_hz=TARGET_HZ)

        if label == LABEL_STURZ:
            windows = extract_impact_windows(seq)
        else:
            windows = extract_sliding_windows(seq, stride=75)

        for w in windows:
            x_list.append(w)
            y_list.append(label)
            subject_list.append(f"upfall_S{subject_i:02d}")
            dataset_list.append("upfall")

    return x_list, y_list, subject_list, dataset_list


# ---------------------------------------------------------------------
# Personal Loader
# ---------------------------------------------------------------------

def personal_timestamps_to_seconds(values: pd.Series) -> Optional[np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() >= 2:
        raw = numeric.to_numpy(dtype=np.float64)
        raw = raw - raw[np.isfinite(raw)][0]
        finite = raw[np.isfinite(raw)]
        if len(finite) >= 2:
            diffs = np.diff(np.sort(np.unique(finite)))
            median_diff = float(np.median(diffs)) if len(diffs) else 0.0
            if median_diff > 0.5:
                raw = raw / 1000.0
            return raw

    dt = pd.to_datetime(values, errors="coerce")
    if dt.notna().sum() >= 2:
        return (dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)

    return None


def load_personal(report_lines: List[str]) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    subject_list: List[str] = []
    dataset_list: List[str] = []

    if not PERSONAL_ROOT.exists():
        log_line(report_lines, f"Personal-Daten nicht gefunden, uebersprungen: {PERSONAL_ROOT}")
        return x_list, y_list, subject_list, dataset_list

    csv_files = sorted(PERSONAL_ROOT.glob("*.csv"))
    log_line(report_lines, f"Personal CSV-Dateien: {len(csv_files)}")

    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            log_line(report_lines, f"  Personal uebersprungen ({path.name}): {exc}")
            continue

        missing = [col for col in PERSONAL_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            log_line(report_lines, f"  Personal uebersprungen ({path.name}), fehlende Spalten: {missing}")
            continue

        work = df[PERSONAL_REQUIRED_COLUMNS].copy()
        work["label"] = work["label"].astype(str).str.strip()
        work = work[work["label"].isin(PERSONAL_LABEL_MAP)]
        if work.empty:
            log_line(report_lines, f"  Personal uebersprungen ({path.name}): keine bekannten Labels")
            continue

        for col in CHANNEL_NAMES:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=CHANNEL_NAMES)
        if len(work) < 2:
            continue

        seconds = personal_timestamps_to_seconds(work["timestamp"])
        if seconds is None:
            log_line(report_lines, f"  Personal uebersprungen ({path.name}): Timestamps nicht lesbar")
            continue

        work = work.assign(_seconds=seconds)
        work = work[np.isfinite(work["_seconds"])].sort_values("_seconds")
        if len(work) < 2:
            continue

        segment_id = (work["label"] != work["label"].shift()).cumsum()
        for segment_number, segment in work.groupby(segment_id, sort=False):
            label_name = str(segment["label"].iloc[0])
            label = PERSONAL_LABEL_MAP[label_name]
            data = segment[CHANNEL_NAMES].to_numpy(dtype=np.float32)
            timestamps = segment["_seconds"].to_numpy(dtype=np.float64)
            if len(data) < 2:
                continue

            seq = resample_by_timestamps(data, timestamps, TARGET_HZ)
            if len(seq) < 2:
                continue

            if label == LABEL_STURZ:
                windows = extract_impact_windows(seq)
            else:
                windows = extract_sliding_windows(seq, stride=75)

            for w in windows:
                x_list.append(w)
                y_list.append(label)
                subject_list.append(f"personal_{path.stem}_{int(segment_number)}")
                dataset_list.append("personal")

    return x_list, y_list, subject_list, dataset_list


# ---------------------------------------------------------------------
# Checks und Report
# ---------------------------------------------------------------------

def summarize_counts(report_lines: List[str], y: Iterable[int], datasets: Iterable[str], title: str) -> None:
    y = list(map(int, y))
    datasets = list(datasets)
    log_line(report_lines, "")
    log_line(report_lines, title)
    log_line(report_lines, "-" * len(title))

    class_counts = Counter(y)
    for label_id, class_name in enumerate(CLASS_NAMES):
        log_line(report_lines, f"  {label_id} {class_name:22s}: {class_counts.get(label_id, 0)}")

    log_line(report_lines, "")
    log_line(report_lines, "Nach Dataset:")
    ds_counts = Counter(datasets)
    for ds, count in sorted(ds_counts.items()):
        log_line(report_lines, f"  {ds:10s}: {count}")

    log_line(report_lines, "")
    log_line(report_lines, "Nach Dataset und Klasse:")
    combo = Counter(zip(datasets, y))
    for ds in sorted(set(datasets)):
        parts = []
        for label_id, class_name in enumerate(CLASS_NAMES):
            parts.append(f"{class_name}={combo.get((ds, label_id), 0)}")
        log_line(report_lines, f"  {ds:10s}: " + ", ".join(parts))


def main() -> None:
    report_lines: List[str] = []
    log_line(report_lines, "Build Real Fall Dataset 150x6")
    log_line(report_lines, "=" * 60)
    log_line(report_lines, f"Projekt: {PROJECT_ROOT}")
    log_line(report_lines, f"Fenster: {WINDOW_SIZE} Samples = {WINDOW_SECONDS:.1f}s bei {TARGET_HZ} Hz")

    all_x: List[np.ndarray] = []
    all_y: List[int] = []
    all_subjects: List[str] = []
    all_datasets: List[str] = []

    for loader_name, loader_fn in [
        ("SisFall", load_sisfall),
        ("UniMiB", load_unimib),
        ("UP-Fall", load_upfall),
        ("Personal", load_personal),
    ]:
        log_line(report_lines, "")
        log_line(report_lines, f"Lade {loader_name} ...")
        x, y, subjects, datasets = loader_fn(report_lines)
        log_line(report_lines, f"{loader_name} Fenster: {len(x)}")
        summarize_counts(report_lines, y, datasets, f"{loader_name} Roh-Counts")

        all_x.extend(x)
        all_y.extend(y)
        all_subjects.extend(subjects)
        all_datasets.extend(datasets)

    if not all_x:
        raise RuntimeError("Keine Daten geladen. Pruefe Pfade und Rohdaten.")

    summarize_counts(report_lines, all_y, all_datasets, "Alle Daten vor Balancing")

    x_arr, y_arr, subjects_arr, datasets_arr = balance_dataset(
        all_x, all_y, all_subjects, all_datasets, max_per_class=MAX_WINDOWS_PER_CLASS
    )

    summarize_counts(report_lines, y_arr, datasets_arr, "Alle Daten nach Balancing")

    log_line(report_lines, "")
    log_line(report_lines, f"Final X shape: {x_arr.shape}")
    log_line(report_lines, f"Final y shape: {y_arr.shape}")
    log_line(report_lines, f"NaN in X: {np.isnan(x_arr).any()}")
    log_line(report_lines, f"Inf in X: {np.isinf(x_arr).any()}")
    log_line(report_lines, f"X min/max: {float(np.min(x_arr)):.5f} / {float(np.max(x_arr)):.5f}")

    np.savez_compressed(
        OUTPUT_NPZ,
        x=x_arr,
        y=y_arr,
        subjects=subjects_arr,
        datasets=datasets_arr,
        class_names=np.array(CLASS_NAMES, dtype=object),
        channel_names=np.array(CHANNEL_NAMES, dtype=object),
        window_size=np.array(WINDOW_SIZE, dtype=np.int64),
        target_hz=np.array(TARGET_HZ, dtype=np.int64),
    )

    metadata = {
        "window_size": WINDOW_SIZE,
        "target_hz": TARGET_HZ,
        "window_seconds": WINDOW_SECONDS,
        "channels": CHANNEL_NAMES,
        "classes": CLASS_NAMES,
        "input_shape": [1, WINDOW_SIZE, N_CHANNELS],
        "output_shape": [1, len(CLASS_NAMES)],
        "datasets": sorted(set(map(str, datasets_arr.tolist()))),
        "max_windows_per_class": MAX_WINDOWS_PER_CLASS,
        "notes": [
            "SisFall: Acc1 + Gyro genutzt, auf 50 Hz resampled.",
            "UP-Fall: RightPocket Acc + Gyro genutzt, auf 50 Hz resampled.",
            "UniMiB: Accelerometer-only, Gyro-Kanaele mit 0 gefuellt.",
            "Normalisierung wird spaeter im Trainingsskript aus dem Trainingssplit berechnet.",
        ],
    }
    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log_line(report_lines, "")
    log_line(report_lines, "Fertig.")
    log_line(report_lines, f"Dataset:  {OUTPUT_NPZ}")
    log_line(report_lines, f"Metadata: {OUTPUT_METADATA}")
    log_line(report_lines, f"Report:   {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
