from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.run_experiment11_frozen_expert_residual_multidataset import (  # noqa: E402
    SEED,
    make_loader,
    make_pair_loader,
    train_residual_fusion,
    train_single_expert,
)
from experiments.run_experiment9_wavelet_multidataset import normalize_complex_rms, to_fft  # noqa: E402


DATASET_ZIP_PATH = ROOT_DIR / "external" / "radar_iq_datasets" / "data" / "radarcnn_dataset.zip"
DATASET_ROOT = ROOT_DIR / "external" / "radar_iq_datasets" / "data" / "radarcnn_unpacked" / "data"
RESULTS_PATH = Path(__file__).resolve().with_name("results.json")


def maybe_extract_dataset(dataset_root: Path, zip_path: Path) -> None:
    if dataset_root.exists():
        return
    if not zip_path.exists():
        raise FileNotFoundError(
            f"RadarCNN dataset not found. Expected either extracted data at {dataset_root} or zip at {zip_path}."
        )
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_root.parent)


def discover_paths(dataset_root: Path) -> tuple[list[str], list[Path], np.ndarray, list[Path], np.ndarray]:
    train_root = dataset_root / "train"
    test_root = dataset_root / "test"
    class_names = sorted(path.name for path in train_root.iterdir() if path.is_dir())

    train_paths: list[Path] = []
    train_labels: list[int] = []
    test_paths: list[Path] = []
    test_labels: list[int] = []

    for label, class_name in enumerate(class_names):
        class_train_paths = sorted((train_root / class_name).glob("*.pickle"))
        class_test_paths = sorted((test_root / class_name).glob("*.pickle"))
        train_paths.extend(class_train_paths)
        train_labels.extend([label] * len(class_train_paths))
        test_paths.extend(class_test_paths)
        test_labels.extend([label] * len(class_test_paths))

    return class_names, train_paths, np.asarray(train_labels, dtype=np.int64), test_paths, np.asarray(test_labels, dtype=np.int64)


def cap_paths_per_class(paths: list[Path], labels: np.ndarray, max_files_per_class: int | None) -> tuple[list[Path], np.ndarray]:
    if max_files_per_class is None:
        return paths, labels

    limited_paths: list[Path] = []
    limited_labels: list[int] = []
    counts: Counter[int] = Counter()
    for path, label in zip(paths, labels, strict=True):
        if counts[int(label)] >= max_files_per_class:
            continue
        counts[int(label)] += 1
        limited_paths.append(path)
        limited_labels.append(int(label))
    return limited_paths, np.asarray(limited_labels, dtype=np.int64)


def load_pickle_examples(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    keys = sorted(payload)
    examples = []
    for key in keys:
        sample = np.asarray(payload[key], dtype=np.complex64)
        iq = np.stack((sample.real, sample.imag), axis=0).astype(np.float32)
        examples.append(iq)
    array = np.stack(examples, axis=0)
    return normalize_complex_rms(array)


def load_split(paths: list[Path], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signals = []
    expanded_labels = []
    for path, label in zip(paths, labels, strict=True):
        file_signals = load_pickle_examples(path)
        signals.append(file_signals)
        expanded_labels.append(np.full(file_signals.shape[0], int(label), dtype=np.int64))

    X = np.concatenate(signals, axis=0).astype(np.float32)
    y = np.concatenate(expanded_labels, axis=0).astype(np.int64)
    return X, y


def build_dataset(
    dataset_root: Path,
    val_fraction: float,
    max_train_files_per_class: int | None,
    batch_size: int,
    epochs: int,
) -> dict:
    maybe_extract_dataset(dataset_root, DATASET_ZIP_PATH)
    class_names, train_paths, train_labels, test_paths, test_labels = discover_paths(dataset_root)
    train_paths, train_labels = cap_paths_per_class(train_paths, train_labels, max_train_files_per_class)

    train_paths_arr = np.asarray(train_paths, dtype=object)
    train_file_paths, val_file_paths, train_file_labels, val_file_labels = train_test_split(
        train_paths_arr,
        train_labels,
        test_size=val_fraction,
        random_state=SEED,
        stratify=train_labels,
    )

    iq_train, y_train = load_split(list(train_file_paths), train_file_labels)
    iq_val, y_val = load_split(list(val_file_paths), val_file_labels)
    iq_test, y_test = load_split(test_paths, test_labels)

    return {
        "name": "radarcnn_object_classification",
        "class_names": class_names,
        "iq_train": iq_train,
        "iq_val": iq_val,
        "iq_test": iq_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "batch_size": batch_size,
        "epochs": epochs,
        "dataset_root": str(dataset_root),
        "zip_path": str(DATASET_ZIP_PATH),
        "train_file_count": int(len(train_file_paths)),
        "val_file_count": int(len(val_file_paths)),
        "test_file_count": int(len(test_paths)),
        "val_fraction": float(val_fraction),
        "max_train_files_per_class": None if max_train_files_per_class is None else int(max_train_files_per_class),
    }


def run_dataset(dataset: dict) -> dict:
    class_names = dataset["class_names"]
    iq_train = dataset["iq_train"]
    iq_val = dataset["iq_val"]
    iq_test = dataset["iq_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    batch_size = dataset["batch_size"]
    epochs = dataset["epochs"]

    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    iq_train_loader = make_loader(iq_train, y_train, True, use_cuda, batch_size)
    iq_val_loader = make_loader(iq_val, y_val, False, use_cuda, batch_size)
    iq_test_loader = make_loader(iq_test, y_test, False, use_cuda, batch_size)
    fft_train_loader = make_loader(fft_train, y_train, True, use_cuda, batch_size)
    fft_val_loader = make_loader(fft_val, y_val, False, use_cuda, batch_size)
    fft_test_loader = make_loader(fft_test, y_test, False, use_cuda, batch_size)
    pair_train_loader = make_pair_loader(iq_train, fft_train, y_train, True, use_cuda, batch_size)
    pair_val_loader = make_pair_loader(iq_val, fft_val, y_val, False, use_cuda, batch_size)
    pair_test_loader = make_pair_loader(iq_test, fft_test, y_test, False, use_cuda, batch_size)

    results = {
        "train_examples": int(iq_train.shape[0]),
        "val_examples": int(iq_val.shape[0]),
        "test_examples": int(iq_test.shape[0]),
        "signal_length": int(iq_train.shape[-1]),
        "class_names": class_names,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "dataset_root": dataset["dataset_root"],
        "zip_path": dataset["zip_path"],
        "train_file_count": int(dataset["train_file_count"]),
        "val_file_count": int(dataset["val_file_count"]),
        "test_file_count": int(dataset["test_file_count"]),
        "val_fraction": float(dataset["val_fraction"]),
        "max_train_files_per_class": dataset["max_train_files_per_class"],
    }

    print(
        f"Dataset {dataset['name']}: train={iq_train.shape}, val={iq_val.shape}, test={iq_test.shape}, batch_size={batch_size}, epochs={epochs}",
        flush=True,
    )

    print("\nTraining IQ expert")
    iq_expert, iq_result = train_single_expert(iq_train_loader, iq_val_loader, iq_test_loader, y_test, class_names, epochs)
    results["iq_cnn"] = iq_result

    print("\nTraining FFT expert")
    fft_expert, fft_result = train_single_expert(fft_train_loader, fft_val_loader, fft_test_loader, y_test, class_names, epochs)
    results["fft_cnn"] = fft_result

    print("\nTraining frozen-expert residual fusion")
    residual_result = train_residual_fusion(
        iq_expert,
        fft_expert,
        pair_train_loader,
        pair_val_loader,
        pair_test_loader,
        y_test,
        class_names,
        epochs,
    )
    results["frozen_expert_residual_fusion"] = residual_result

    best_single = max(iq_result["test_acc"], fft_result["test_acc"])
    results["best_single_test_acc"] = float(best_single)
    results["improves_over_best_single"] = bool(residual_result["test_acc"] > best_single)
    results["matches_or_beats_best_single"] = bool(residual_result["test_acc"] >= best_single)
    results["delta_vs_best_single"] = float(residual_result["test_acc"] - best_single)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 11 frozen-expert residual fusion on the RadarCNN object-classification dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-files-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    dataset = build_dataset(
        dataset_root=args.dataset_root,
        val_fraction=args.val_fraction,
        max_train_files_per_class=args.max_train_files_per_class,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    if args.dry_run:
        summary = {
            "dataset": dataset["name"],
            "class_names": dataset["class_names"],
            "train_shape": list(dataset["iq_train"].shape),
            "val_shape": list(dataset["iq_val"].shape),
            "test_shape": list(dataset["iq_test"].shape),
            "train_file_count": dataset["train_file_count"],
            "val_file_count": dataset["val_file_count"],
            "test_file_count": dataset["test_file_count"],
        }
        print(json.dumps(summary, indent=2))
        return

    results = {
        "experiment": "frozen_expert_residual_radarcnn",
        "dataset": run_dataset(dataset),
        "runtime_seconds": time.perf_counter() - start,
    }
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
