from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\SturzDetektion\fall_realdata_project")
SISFALL_ROOT = PROJECT_ROOT / "raw" / "sisfall" / "SisFall_dataset"
UNIMIB_ROOT = PROJECT_ROOT / "raw" / "unimib" / "UniMiB"
UPFALL_CSV = PROJECT_ROOT / "raw" / "upfall" / "CompleteDataSet.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "dataset_detail_inspection.txt"


def write(out, text=""):
    print(text)
    out.write(str(text) + "\n")


def inspect_sisfall(out):
    write(out, "\n" + "#" * 100)
    write(out, "SISFALL DETAIL")
    write(out, "#" * 100)

    if not SISFALL_ROOT.exists():
        write(out, f"Pfad existiert nicht: {SISFALL_ROOT}")
        return

    txt_files = sorted(p for p in SISFALL_ROOT.rglob("*.txt") if p.name.lower() != "readme.txt")
    write(out, f"TXT-Dateien ohne Readme: {len(txt_files)}")

    subject_counter = Counter(p.parent.name for p in txt_files)
    activity_counter = Counter(p.name.split("_")[0] for p in txt_files if "_" in p.name)

    write(out, "\nSubjects, erste 20:")
    for subject, count in list(subject_counter.items())[:20]:
        write(out, f"  {subject}: {count}")

    write(out, "\nAktivitaetscodes:")
    for code, count in sorted(activity_counter.items()):
        write(out, f"  {code}: {count}")

    write(out, "\nBeispieldateien mit Shapes und Wertebereichen:")
    for path in txt_files[:10]:
        try:
            # SisFall-Zeilen enden mit Semikolon; delimiter Komma.
            arr = np.genfromtxt(path, delimiter=",", dtype=np.float32, autostrip=True)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # Falls Semikolon am Ende NaNs erzeugt, entfernen.
            arr = arr[:, ~np.all(np.isnan(arr), axis=0)]
            write(out, f"  {path.relative_to(SISFALL_ROOT)} shape={arr.shape} min={np.nanmin(arr):.2f} max={np.nanmax(arr):.2f}")
            write(out, f"    first_row={arr[0].tolist()}")
        except Exception as e:
            write(out, f"  Fehler bei {path}: {e}")


def inspect_unimib(out):
    write(out, "\n" + "#" * 100)
    write(out, "UNIMIB DETAIL")
    write(out, "#" * 100)

    if not UNIMIB_ROOT.exists():
        write(out, f"Pfad existiert nicht: {UNIMIB_ROOT}")
        return

    npy_files = sorted(UNIMIB_ROOT.glob("*.npy"))
    write(out, f"NPY-Dateien: {len(npy_files)}")

    for path in npy_files:
        try:
            arr = np.load(path, allow_pickle=True)
            write(out, f"\n{path.name}")
            write(out, f"  dtype={arr.dtype}")
            write(out, f"  shape={arr.shape}")

            if arr.dtype == object:
                flat = arr.reshape(-1)
                write(out, f"  object preview={flat[:10].tolist()}")
            else:
                finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.array([])
                if finite.size > 0:
                    write(out, f"  min={finite.min():.5f} max={finite.max():.5f} mean={finite.mean():.5f} std={finite.std():.5f}")
                preview = arr.reshape(-1)[:20]
                write(out, f"  preview={preview.tolist()}")

            # Label-Dateien genauer zählen.
            if "label" in path.name.lower() and arr.size > 0:
                if arr.ndim == 1:
                    counts = Counter(arr.tolist())
                    write(out, "  label counts:")
                    for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])):
                        write(out, f"    {k}: {v}")
                else:
                    write(out, "  first 10 label rows:")
                    for row in arr[:10]:
                        write(out, f"    {row.tolist() if hasattr(row, 'tolist') else row}")
        except Exception as e:
            write(out, f"Fehler bei {path.name}: {e}")


def inspect_upfall(out):
    write(out, "\n" + "#" * 100)
    write(out, "UPFALL DETAIL")
    write(out, "#" * 100)

    if not UPFALL_CSV.exists():
        write(out, f"Pfad existiert nicht: {UPFALL_CSV}")
        return

    # UP-Fall hat zwei Header-Zeilen. Wir laden Row 0 als grobe Namen und überspringen Row 1.
    try:
        df_head = pd.read_csv(UPFALL_CSV, nrows=5)
        write(out, "CSV normal gelesen, erste Spalten:")
        write(out, list(df_head.columns[:60]))
        write(out, f"Head shape={df_head.shape}")
    except Exception as e:
        write(out, f"Normaler CSV-Head fehlgeschlagen: {e}")

    try:
        df = pd.read_csv(UPFALL_CSV, skiprows=[1])
        write(out, f"\nCSV mit skiprows=[1] shape={df.shape}")
        write(out, "Spalten mit Index:")
        for idx, col in enumerate(df.columns):
            write(out, f"  {idx:02d}: {col}")

        # Wir interessieren uns für RightPocket: Index 8-13 nach der Zeitspalte.
        write(out, "\nErste 5 RightPocket-Rohzeilen per Positionsindex:")
        cols = [0, 8, 9, 10, 11, 12, 13, 43, 44, 45, 46]
        cols = [c for c in cols if c < len(df.columns)]
        write(out, df.iloc[:5, cols].to_string())

        for col_name in ["Subject", "Activity", "Trial", "Tag"]:
            if col_name in df.columns:
                counts = df[col_name].value_counts(dropna=False).sort_index()
                write(out, f"\nCounts fuer {col_name}:")
                write(out, counts.to_string())

        if all(c in df.columns for c in ["Subject", "Activity", "Trial"]):
            groups = df.groupby(["Subject", "Activity", "Trial"], dropna=False).size()
            write(out, f"\nAnzahl Sequenzen Subject/Activity/Trial: {len(groups)}")
            write(out, "Sequenzlaengen describe:")
            write(out, groups.describe().to_string())
            write(out, "\nErste 20 Sequenzen:")
            write(out, groups.head(20).to_string())
    except Exception as e:
        write(out, f"CSV Detailanalyse fehlgeschlagen: {e}")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        inspect_sisfall(out)
        inspect_unimib(out)
        inspect_upfall(out)

    print("\nFertig.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
