"""Download datasets for CompressorGuard. See DATA.md for details and manual fallbacks.

Usage:
    python scripts/download_data.py            # both datasets
    python scripts/download_data.py --only metropt
    python scripts/download_data.py --only ims
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"   # independent of the working directory


def download_metropt() -> None:
    out_dir = RAW / "metropt3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "MetroPT3(AirCompressor).csv"
    if out.exists():
        print(f"[metropt] already present: {out}")
        return
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        sys.exit("[metropt] pip install ucimlrepo  (or download manually, see DATA.md §1)")
    print("[metropt] fetching UCI dataset 791 ...")
    try:
        ds = fetch_ucirepo(id=791)
        ds.data.original.to_csv(out, index=False)
        print(f"[metropt] saved {out}")
    except Exception as e:  # large UCI datasets sometimes fail via the API
        sys.exit(f"[metropt] API download failed ({e}). Download manually from "
                 "https://archive.ics.uci.edu/dataset/791/metropt+3+dataset")


def download_ims() -> None:
    out_dir = RAW / "ims"
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        print(f"[ims] {out_dir} not empty, skipping download")
        return
    if shutil.which("kaggle") is None:
        sys.exit("[ims] pip install kaggle and set up ~/.kaggle/kaggle.json (see DATA.md §2)")
    print("[ims] downloading from Kaggle (large, may take a while) ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "vinayak123tyagi/bearing-dataset",
         "-p", str(out_dir), "--unzip"],
        check=True,
    )
    print(f"[ims] extracted to {out_dir}")
    # Same discovery logic the pipeline uses (handles e.g. 3rd_test/4th_test/txt nesting)
    sys.path.insert(0, str(RAW.parents[1] / "src"))
    from compressor_guard.vibration.io import find_test_folder
    for t in (1, 2, 3):
        try:
            print(f"[ims] test {t}: {find_test_folder(t, out_dir)}")
        except FileNotFoundError:
            print(f"[ims] test {t}: NOT FOUND, inspect {out_dir} manually")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["metropt", "ims"])
    args = ap.parse_args()
    if args.only in (None, "metropt"):
        download_metropt()
    if args.only in (None, "ims"):
        download_ims()
