from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


RADCNN_FILE_ID = "14u6mn3t8BanRQPwuWk2M0pSJxX-iRae5"
ZIP_PATH = ROOT_DIR / "external" / "radar_iq_datasets" / "data" / "radarcnn_dataset.zip"
EXTRACT_ROOT = ROOT_DIR / "external" / "radar_iq_datasets" / "data" / "radarcnn_unpacked"


def ensure_gdown():
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - simple runtime guard
        raise SystemExit("`gdown` is required. Install it with `./.venv/bin/python -m pip install gdown`.") from exc
    return gdown


def download_if_needed(zip_path: Path, file_id: str, force: bool) -> None:
    if zip_path.exists() and not force:
        print(f"Zip already present at {zip_path}")
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    gdown = ensure_gdown()
    print(f"Downloading RadarCNN dataset to {zip_path}")
    gdown.download(id=file_id, output=str(zip_path), quiet=False)


def extract_if_needed(zip_path: Path, extract_root: Path, force: bool) -> None:
    data_root = extract_root / "data"
    if data_root.exists() and not force:
        print(f"Extracted dataset already present at {data_root}")
        return

    if extract_root.exists() and force:
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} to {extract_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)


def summarize_dataset(data_root: Path) -> None:
    pickle_paths = list(data_root.rglob("*.pickle"))
    class_counts = Counter(path.parent.name for path in pickle_paths)
    split_counts = Counter(path.parent.parent.name for path in pickle_paths)
    print(f"Extracted root: {data_root}")
    print(f"Pickle files: {len(pickle_paths)}")
    print(f"Split counts: {dict(sorted(split_counts.items()))}")
    print(f"Class counts: {dict(sorted(class_counts.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the RadarCNN radar IQ dataset.")
    parser.add_argument("--file-id", default=RADCNN_FILE_ID)
    parser.add_argument("--zip-path", type=Path, default=ZIP_PATH)
    parser.add_argument("--extract-root", type=Path, default=EXTRACT_ROOT)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    download_if_needed(args.zip_path, args.file_id, force=args.force_download)
    extract_if_needed(args.zip_path, args.extract_root, force=args.force_extract)
    summarize_dataset(args.extract_root / "data")


if __name__ == "__main__":
    main()
