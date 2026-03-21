from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.shared.story_datasets import (  # noqa: E402
    load_modulation_dataset,
    load_orbit_dataset,
    load_real_split,
    load_waveform_dataset,
    missing_story_experiment3_dependencies,
    to_fft,
)
from experiments.shared.repro import classification_metrics, configure_device, make_dataset_loader, make_tensor_loader, save_json, set_global_seed
from experiments.shared.story_models import ExpertCNN, FrozenExpertResidualFusion


SEED = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
ALPHA_PENALTY = 0.01
DELTA_PENALTY = 0.001
DELTA_SCALE = 2.0

REAL_TRAIN_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "train.h5"
REAL_VAL_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "val.h5"
REAL_TEST_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "test.h5"

AUG_REAL_TRAIN_PATH = ROOT_DIR / "data" / "real_mat_dataset_augmented" / "train.h5"
AUG_REAL_VAL_PATH = ROOT_DIR / "data" / "real_mat_dataset_augmented" / "val.h5"
AUG_REAL_TEST_PATH = ROOT_DIR / "data" / "real_mat_dataset_augmented" / "test.h5"

CAPTURED_TRAIN_PATH = ROOT_DIR / "data" / "captured_npy_dataset_experiment5" / "train.h5"
CAPTURED_VAL_PATH = ROOT_DIR / "data" / "captured_npy_dataset_experiment5" / "val.h5"
CAPTURED_TEST_PATH = ROOT_DIR / "data" / "captured_npy_dataset_experiment5" / "test.h5"

RESULTS_PATH = Path(__file__).resolve().parent / "artifacts" / "experiment11_frozen_expert_residual_multidataset_results.json"
DEFAULT_DATASET_ORDER = [
    "modulation_family",
    "waveform_family",
    "subghz_real_128",
    "subghz_real_512",
    "subghz_real_1024_40ep",
    "subghz_real_augmented_512",
    "orbit_rf",
    "captured_npy_real_128",
]


class PairDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return make_tensor_loader(X, y, shuffle=shuffle, use_cuda=use_cuda, batch_size=batch_size)


def make_pair_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return make_dataset_loader(PairDataset(iq, fft, y), shuffle=shuffle, use_cuda=use_cuda, batch_size=batch_size)


def train_single_expert(
    train_loader,
    val_loader,
    test_loader,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
):
    device, use_cuda = configure_device()
    model = ExpertCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train(True)
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    logits = model(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / max(total, 1)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{epochs} | val acc {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    result = {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
    }
    result.update(classification_metrics(y_test, y_pred, class_names))
    return model.cpu(), result


def compute_residual_loss(outputs, yb, criterion):
    final_loss = criterion(outputs["final_logits"], yb)
    alpha_penalty = outputs["alpha"].mean()
    delta_penalty = outputs["delta"].pow(2).mean()
    return final_loss + ALPHA_PENALTY * alpha_penalty + DELTA_PENALTY * delta_penalty


def train_residual_fusion(
    iq_expert: ExpertCNN,
    fft_expert: ExpertCNN,
    train_loader,
    val_loader,
    test_loader,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
):
    device, use_cuda = configure_device()
    model = FrozenExpertResidualFusion(iq_expert, fft_expert, num_classes=len(class_names), delta_scale=DELTA_SCALE).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_stats = None
    for epoch in range(1, epochs + 1):
        model.train(True)
        for iqb, fftb, yb in train_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
                loss = compute_residual_loss(outputs, yb, criterion)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        correct = 0
        total = 0
        alpha_sum = 0.0
        anchor_iq_sum = 0.0
        with torch.no_grad():
            for iqb, fftb, yb in val_loader:
                iqb = iqb.to(device, non_blocking=use_cuda)
                fftb = fftb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(iqb, fftb)
                correct += (outputs["final_logits"].argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
                alpha_sum += float(outputs["alpha"].sum().item())
                anchor_iq_sum += float(outputs["anchor_is_iq"].sum().item())
        val_acc = correct / max(total, 1)
        alpha_mean = alpha_sum / max(total, 1)
        anchor_iq_fraction = anchor_iq_sum / max(total, 1)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{epochs} | val acc {val_acc:.3f} | alpha {alpha_mean:.3f} | iq-anchor {anchor_iq_fraction:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_stats = {
                "best_val_alpha_mean": float(alpha_mean),
                "best_val_iq_anchor_fraction": float(anchor_iq_fraction),
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    alpha_sum = 0.0
    anchor_iq_sum = 0.0
    total = 0
    with torch.no_grad():
        for iqb, fftb, _ in test_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            alpha_sum += float(outputs["alpha"].sum().item())
            anchor_iq_sum += float(outputs["anchor_is_iq"].sum().item())
            total += iqb.size(0)
    y_pred = np.concatenate(preds)
    result = {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "test_alpha_mean": float(alpha_sum / max(total, 1)),
        "test_iq_anchor_fraction": float(anchor_iq_sum / max(total, 1)),
    }
    result.update(best_stats)
    result.update(classification_metrics(y_test, y_pred, class_names))
    return result


def build_h5_dataset(name: str, train_path: Path, val_path: Path, test_path: Path, max_windows_per_file: int, batch_size: int, epochs: int):
    iq_train, y_train, class_names = load_real_split(train_path, max_windows_per_file=max_windows_per_file)
    iq_val, y_val, _ = load_real_split(val_path, max_windows_per_file=max_windows_per_file)
    iq_test, y_test, _ = load_real_split(test_path, max_windows_per_file=max_windows_per_file)
    return {
        "name": name,
        "class_names": class_names,
        "iq_train": iq_train,
        "iq_val": iq_val,
        "iq_test": iq_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_windows_per_file": int(max_windows_per_file),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }


def run_dataset(dataset: dict):
    name = dataset["name"]
    class_names = dataset["class_names"]
    iq_train = dataset["iq_train"]
    iq_val = dataset["iq_val"]
    iq_test = dataset["iq_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    batch_size = dataset["batch_size"]
    epochs = dataset["epochs"]

    print(f"\n=== Dataset: {name} ===")
    print("iq train/val/test:", iq_train.shape, iq_val.shape, iq_test.shape)

    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)

    _, use_cuda = configure_device()
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
    }
    for key in ("max_windows_per_file", "max_packets_per_node_per_day", "num_common_nodes", "train_path", "val_path", "test_path"):
        if key in dataset:
            results[key] = dataset[key]

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


def build_story_datasets(orbit_max_packets_per_node: int = 256, dataset_names: set[str] | None = None) -> list[dict]:
    missing_dependencies = missing_story_experiment3_dependencies()
    if missing_dependencies:
        missing_text = "\n".join(f"- {path}" for path in missing_dependencies)
        raise FileNotFoundError(
            "Experiment 3 requires local datasets that are not present.\n"
            "Missing paths:\n"
            f"{missing_text}\n\n"
            "Run `./.venv/bin/python experiments/check_data.py` for a full audit and see docs/datasets.md for dataset placement details."
        )

    requested = dataset_names or set(DEFAULT_DATASET_ORDER)
    datasets = []

    if "modulation_family" in requested:
        modulation = load_modulation_dataset()
        modulation["epochs"] = 40
        datasets.append(modulation)

    if "waveform_family" in requested:
        waveform = load_waveform_dataset()
        waveform["epochs"] = 40
        datasets.append(waveform)

    if "subghz_real_128" in requested:
        datasets.append(
            build_h5_dataset("subghz_real_128", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=128, batch_size=256, epochs=20)
        )

    if "subghz_real_512" in requested:
        datasets.append(
            build_h5_dataset("subghz_real_512", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=512, batch_size=256, epochs=20)
        )

    if "subghz_real_1024_40ep" in requested:
        datasets.append(
            build_h5_dataset("subghz_real_1024_40ep", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=1024, batch_size=256, epochs=40)
        )

    if "subghz_real_augmented_512" in requested:
        datasets.append(
            build_h5_dataset(
                "subghz_real_augmented_512",
                AUG_REAL_TRAIN_PATH,
                AUG_REAL_VAL_PATH,
                AUG_REAL_TEST_PATH,
                max_windows_per_file=512,
                batch_size=256,
                epochs=20,
            )
        )

    if "orbit_rf" in requested:
        orbit = load_orbit_dataset(max_packets_per_node=orbit_max_packets_per_node)
        orbit["epochs"] = 20
        orbit["batch_size"] = 256
        datasets.append(orbit)

    if "captured_npy_real_128" in requested:
        datasets.append(
            build_h5_dataset(
                "captured_npy_real_128",
                CAPTURED_TRAIN_PATH,
                CAPTURED_VAL_PATH,
                CAPTURED_TEST_PATH,
                max_windows_per_file=128,
                batch_size=256,
                epochs=20,
            )
        )

    return datasets


def select_datasets(datasets: list[dict], dataset_names: list[str] | None) -> list[dict]:
    if dataset_names is None:
        return datasets

    selected = []
    dataset_by_name = {dataset["name"]: dataset for dataset in datasets}
    missing = [name for name in dataset_names if name not in dataset_by_name]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Unknown dataset names requested: {missing_text}")

    for name in dataset_names:
        selected.append(dataset_by_name[name])
    return selected


def run_experiment(
    results_path: Path = RESULTS_PATH,
    orbit_max_packets_per_node: int = 256,
    seed: int = SEED,
    dataset_names: list[str] | None = None,
) -> dict:
    set_global_seed(seed)
    start = time.perf_counter()

    requested_set = set(dataset_names) if dataset_names is not None else None
    datasets = select_datasets(
        build_story_datasets(orbit_max_packets_per_node=orbit_max_packets_per_node, dataset_names=requested_set),
        dataset_names,
    )

    results = {
        "experiment": "frozen_expert_residual_multidataset",
        "seed": int(seed),
        "architecture": {
            "description": "Frozen IQ and FFT experts, confidence-based expert anchor, bounded residual correction",
            "alpha_penalty": ALPHA_PENALTY,
            "delta_penalty": DELTA_PENALTY,
            "delta_scale": DELTA_SCALE,
        },
        "requested_datasets": [dataset["name"] for dataset in datasets],
        "datasets": {},
    }
    save_json(results_path, results)

    for dataset in datasets:
        dataset_start = time.perf_counter()
        dataset_results = run_dataset(dataset)
        dataset_results["runtime_seconds"] = time.perf_counter() - dataset_start
        results["datasets"][dataset["name"]] = dataset_results
        save_json(results_path, results)

    results["runtime_seconds"] = time.perf_counter() - start
    save_json(results_path, results)
    print(f"\nSaved results to {results_path}")
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run frozen-expert residual IQ+FFT fusion across all Experiment 5-comparable datasets.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--orbit-max-packets-per-node", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASET_ORDER,
        help="Optional subset of datasets to run, in the order provided.",
    )
    args = parser.parse_args()
    run_experiment(
        results_path=args.results_path,
        orbit_max_packets_per_node=args.orbit_max_packets_per_node,
        seed=args.seed,
        dataset_names=args.datasets,
    )


if __name__ == "__main__":
    main()
