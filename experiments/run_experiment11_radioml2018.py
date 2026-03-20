from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.run_experiment11_frozen_expert_residual_multidataset import (  # noqa: E402
    SEED,
    evaluate_predictions,
    train_residual_fusion,
    train_single_expert,
)
from experiments.run_experiment9_wavelet_multidataset import normalize_complex_rms  # noqa: E402


RADIOML2018_PATH = ROOT_DIR / "external" / "radioml_dataset" / "GOLD_XYZ_OSC.0001_1024.hdf5"
RESULTS_PATH = ROOT_DIR / "experiments" / "experiment11_radioml2018_results.json"

OFFICIAL_CLASS_NAMES = [
    "32PSK",
    "16APSK",
    "32QAM",
    "FM",
    "GMSK",
    "32APSK",
    "OQPSK",
    "8ASK",
    "BPSK",
    "8PSK",
    "AM-SSB-SC",
    "4ASK",
    "16PSK",
    "64APSK",
    "128QAM",
    "128APSK",
    "AM-DSB-SC",
    "AM-SSB-WC",
    "64QAM",
    "QPSK",
    "256QAM",
    "AM-DSB-WC",
    "OOK",
    "16QAM",
]

PAIR_BLOCK = 4096
NUM_CLASSES = 24
NUM_SNRS = 26
SIGNAL_LENGTH = 1024
SNR_VALUES = list(range(-20, 32, 2))


def to_fft_batch(signals: torch.Tensor) -> torch.Tensor:
    complex_signals = torch.complex(signals[:, 0, :], signals[:, 1, :])
    fft_signals = torch.fft.fftshift(torch.fft.fft(complex_signals, dim=-1), dim=-1)
    features = torch.stack((fft_signals.real, fft_signals.imag), dim=1).to(dtype=torch.float32)
    rms = torch.sqrt(torch.mean(features.square(), dim=(1, 2), keepdim=True) + 1e-8)
    return features / rms


def build_split_indices(train_per_pair: int, val_per_pair: int, test_per_pair: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_per_pair = train_per_pair + val_per_pair + test_per_pair
    if total_per_pair != PAIR_BLOCK:
        raise ValueError(f"Expected full-dataset split to use all {PAIR_BLOCK} samples per (class, SNR) block, got {total_per_pair}")

    train_indices = np.empty(NUM_CLASSES * NUM_SNRS * train_per_pair, dtype=np.int64)
    val_indices = np.empty(NUM_CLASSES * NUM_SNRS * val_per_pair, dtype=np.int64)
    test_indices = np.empty(NUM_CLASSES * NUM_SNRS * test_per_pair, dtype=np.int64)
    train_pos = 0
    val_pos = 0
    test_pos = 0
    rng = np.random.default_rng(SEED)

    for class_idx in range(NUM_CLASSES):
        for snr_idx in range(NUM_SNRS):
            block_start = (class_idx * NUM_SNRS + snr_idx) * PAIR_BLOCK
            perm = rng.permutation(PAIR_BLOCK)
            train_local = np.sort(perm[:train_per_pair])
            val_local = np.sort(perm[train_per_pair : train_per_pair + val_per_pair])
            test_local = np.sort(perm[train_per_pair + val_per_pair :])

            train_indices[train_pos : train_pos + train_per_pair] = block_start + train_local
            val_indices[val_pos : val_pos + val_per_pair] = block_start + val_local
            test_indices[test_pos : test_pos + test_per_pair] = block_start + test_local
            train_pos += train_per_pair
            val_pos += val_per_pair
            test_pos += test_per_pair

    return train_indices, val_indices, test_indices


def labels_from_indices(indices: np.ndarray) -> np.ndarray:
    return (indices // PAIR_BLOCK // NUM_SNRS).astype(np.int64)


class RadioML2018Dataset(Dataset):
    def __init__(self, h5_path: Path, indices: np.ndarray, labels: np.ndarray):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self._h5_file = None
        self._x_data = None

    def __len__(self) -> int:
        return len(self.indices)

    def _ensure_open(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._x_data = self._h5_file["X"]

    def __getitem__(self, idx: int):
        self._ensure_open()
        sample = self._x_data[int(self.indices[idx])].astype(np.float32)
        sample = np.transpose(sample, (1, 0))
        sample = normalize_complex_rms(sample[np.newaxis, ...])[0]
        return torch.from_numpy(sample), torch.tensor(int(self.labels[idx]), dtype=torch.long)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_x_data"] = None
        return state


class RadioML2018PairDataset(Dataset):
    def __init__(self, h5_path: Path, indices: np.ndarray, labels: np.ndarray):
        self.base = RadioML2018Dataset(h5_path=h5_path, indices=indices, labels=labels)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        iq, label = self.base[idx]
        fft = to_fft_batch(iq.unsqueeze(0)).squeeze(0)
        return iq, fft, label


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )


def run_streaming_dataset(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int,
    epochs: int,
) -> dict:
    y_train = labels_from_indices(train_indices)
    y_val = labels_from_indices(val_indices)
    y_test = labels_from_indices(test_indices)

    print("\n=== Dataset: radioml2018_full ===")
    print("train/val/test:", len(train_indices), len(val_indices), len(test_indices))

    iq_train_loader = make_loader(RadioML2018Dataset(RADIOML2018_PATH, train_indices, y_train), batch_size, True)
    iq_val_loader = make_loader(RadioML2018Dataset(RADIOML2018_PATH, val_indices, y_val), batch_size, False)
    iq_test_loader = make_loader(RadioML2018Dataset(RADIOML2018_PATH, test_indices, y_test), batch_size, False)

    fft_train_loader = make_loader(RadioML2018PairDataset(RADIOML2018_PATH, train_indices, y_train), batch_size, True)
    fft_val_loader = make_loader(RadioML2018PairDataset(RADIOML2018_PATH, val_indices, y_val), batch_size, False)
    fft_test_loader = make_loader(RadioML2018PairDataset(RADIOML2018_PATH, test_indices, y_test), batch_size, False)

    class FFTOnlyView(Dataset):
        def __init__(self, pair_dataset: Dataset):
            self.pair_dataset = pair_dataset

        def __len__(self) -> int:
            return len(self.pair_dataset)

        def __getitem__(self, idx: int):
            _, fft, label = self.pair_dataset[idx]
            return fft, label

    pair_train_dataset = RadioML2018PairDataset(RADIOML2018_PATH, train_indices, y_train)
    pair_val_dataset = RadioML2018PairDataset(RADIOML2018_PATH, val_indices, y_val)
    pair_test_dataset = RadioML2018PairDataset(RADIOML2018_PATH, test_indices, y_test)

    fft_train_loader = make_loader(FFTOnlyView(pair_train_dataset), batch_size, True)
    fft_val_loader = make_loader(FFTOnlyView(pair_val_dataset), batch_size, False)
    fft_test_loader = make_loader(FFTOnlyView(pair_test_dataset), batch_size, False)

    pair_train_loader = make_loader(pair_train_dataset, batch_size, True)
    pair_val_loader = make_loader(pair_val_dataset, batch_size, False)
    pair_test_loader = make_loader(pair_test_dataset, batch_size, False)

    results = {
        "train_examples": int(len(train_indices)),
        "val_examples": int(len(val_indices)),
        "test_examples": int(len(test_indices)),
        "signal_length": SIGNAL_LENGTH,
        "class_names": OFFICIAL_CLASS_NAMES,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "train_path": str(RADIOML2018_PATH),
        "val_path": str(RADIOML2018_PATH),
        "test_path": str(RADIOML2018_PATH),
    }

    print("\nTraining IQ expert")
    iq_expert, iq_result = train_single_expert(iq_train_loader, iq_val_loader, iq_test_loader, y_test, OFFICIAL_CLASS_NAMES, epochs)
    results["iq_cnn"] = iq_result

    print("\nTraining FFT expert")
    fft_expert, fft_result = train_single_expert(fft_train_loader, fft_val_loader, fft_test_loader, y_test, OFFICIAL_CLASS_NAMES, epochs)
    results["fft_cnn"] = fft_result

    print("\nTraining frozen-expert residual fusion")
    residual_result = train_residual_fusion(
        iq_expert,
        fft_expert,
        pair_train_loader,
        pair_val_loader,
        pair_test_loader,
        y_test,
        OFFICIAL_CLASS_NAMES,
        epochs,
    )
    results["frozen_expert_residual_fusion"] = residual_result
    best_single = max(iq_result["test_acc"], fft_result["test_acc"])
    results["best_single_test_acc"] = float(best_single)
    results["improves_over_best_single"] = bool(residual_result["test_acc"] > best_single)
    results["matches_or_beats_best_single"] = bool(residual_result["test_acc"] >= best_single)
    results["delta_vs_best_single"] = float(residual_result["test_acc"] - best_single)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 11 frozen-expert residual fusion on the full RadioML 2018 dataset with streaming HDF5 access.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--train-per-class-snr", type=int, default=3276)
    parser.add_argument("--val-per-class-snr", type=int, default=410)
    parser.add_argument("--test-per-class-snr", type=int, default=410)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()

    train_indices, val_indices, test_indices = build_split_indices(
        train_per_pair=args.train_per_class_snr,
        val_per_pair=args.val_per_class_snr,
        test_per_pair=args.test_per_class_snr,
    )

    dataset_start = time.perf_counter()
    dataset_results = run_streaming_dataset(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    dataset_results["runtime_seconds"] = time.perf_counter() - dataset_start

    results = {
        "experiment": "experiment11_radioml2018_full_streaming",
        "source_architecture": "frozen_expert_residual_multidataset",
        "source_script": str(ROOT_DIR / "experiments" / "run_experiment11_frozen_expert_residual_multidataset.py"),
        "dataset": {
            "name": "radioml2018_full",
            "train_path": str(RADIOML2018_PATH),
            "class_names": OFFICIAL_CLASS_NAMES,
            "class_name_mapping_note": (
                "Uses the public RML2018.01A classes.txt ordering commonly mirrored online; "
                "some third-party analysis reports that this ordering may not perfectly match the HDF5 label indices."
            ),
            "snr_values": SNR_VALUES,
            "train_per_class_snr": int(args.train_per_class_snr),
            "val_per_class_snr": int(args.val_per_class_snr),
            "test_per_class_snr": int(args.test_per_class_snr),
            "num_classes": NUM_CLASSES,
            "num_snr_levels": NUM_SNRS,
            "split_note": "Per-(class, SNR) random split using all 4096 examples per block with fixed seed 0.",
        },
        "results": dataset_results,
        "runtime_seconds": time.perf_counter() - start,
    }

    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
