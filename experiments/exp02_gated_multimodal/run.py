from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


EXPERIMENT1_ARTIFACTS_DIR = ROOT_DIR / "experiments" / "exp01_iq_vs_fft" / "artifacts"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODULATION_DATASET_PATH = EXPERIMENT1_ARTIFACTS_DIR / "modulation_time_features.h5"
WAVEFORM_DATASET_PATH = EXPERIMENT1_ARTIFACTS_DIR / "waveform_frequency_features.h5"
EXPERIMENT1_RESULTS_PATH = EXPERIMENT1_ARTIFACTS_DIR / "experiment1_results.json"
EXPERIMENT2_RESULTS_PATH = ROOT_DIR / "experiments" / "experiment1" / "experiment2_results.json"
RESULTS_PATH = ARTIFACTS_DIR / "experiment3_results.json"

SEED = 0
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AUX_LOSS_WEIGHT = 0.25
BRANCH_DROPOUT = 0.10


def stratify_or_none(y: np.ndarray):
    _, counts = np.unique(y, return_counts=True)
    return y if np.all(counts >= 2) else None


class MultimodalDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


def load_h5(path: Path, normalize_rms: bool):
    with h5py.File(path, "r") as h5_file:
        signals = h5_file["signals"][:].astype(np.float32)
        labels = h5_file["labels"][:].astype(np.int64)
        class_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h5_file.attrs["class_names"]
        ]
    if normalize_rms:
        rms = np.sqrt(np.mean(signals**2, axis=(1, 2), keepdims=True) + 1e-8)
        signals = signals / rms
    return signals, labels, class_names


def to_fft(signals: np.ndarray):
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    fft_signals = np.fft.fftshift(np.fft.fft(complex_signals, axis=-1), axes=-1)
    features = np.stack((fft_signals.real, fft_signals.imag), axis=1).astype(np.float32)
    rms = np.sqrt(np.mean(features**2, axis=(1, 2), keepdims=True) + 1e-8)
    return features / rms


def split_dataset(iq: np.ndarray, fft: np.ndarray, labels: np.ndarray):
    idx = np.arange(len(labels))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify_or_none(labels),
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=stratify_or_none(y_temp),
    )
    return (
        iq[idx_train],
        iq[idx_val],
        iq[idx_test],
        fft[idx_train],
        fft[idx_val],
        fft[idx_test],
        y_train,
        y_val,
        y_test,
    )


def make_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool):
    return make_dataset_loader(MultimodalDataset(iq, fft, y), shuffle=shuffle, use_cuda=use_cuda, batch_size=BATCH_SIZE)


def compute_loss(outputs, y_batch, criterion):
    final_loss = criterion(outputs["final_logits"], y_batch)
    iq_loss = criterion(outputs["iq_logits"], y_batch)
    fft_loss = criterion(outputs["fft_logits"], y_batch)
    fusion_loss = criterion(outputs["fusion_logits"], y_batch)
    return final_loss + AUX_LOSS_WEIGHT * (iq_loss + fft_loss + fusion_loss)


def run_epoch(model, loader, criterion, optimizer, scaler, device, use_cuda, training: bool):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    gate_sum = torch.zeros(3, dtype=torch.float64)

    for iq_batch, fft_batch, y_batch in loader:
        iq_batch = iq_batch.to(device, non_blocking=use_cuda)
        fft_batch = fft_batch.to(device, non_blocking=use_cuda)
        y_batch = y_batch.to(device, non_blocking=use_cuda)

        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iq_batch, fft_batch)
                loss = compute_loss(outputs, y_batch, criterion)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        final_logits = outputs["final_logits"]
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (final_logits.argmax(dim=1) == y_batch).sum().item()
        total_examples += y_batch.size(0)
        gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)

    gate_mean = (gate_sum / max(total_examples, 1)).tolist()
    return total_loss / total_examples, total_correct / total_examples, gate_mean


def evaluate_test(model, loader, device, use_cuda, class_names):
    model.eval()
    all_preds = []
    all_targets = []
    gate_sum = torch.zeros(3, dtype=torch.float64)
    total_examples = 0

    with torch.no_grad():
        for iq_batch, fft_batch, y_batch in loader:
            iq_batch = iq_batch.to(device, non_blocking=use_cuda)
            fft_batch = fft_batch.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iq_batch, fft_batch)
            logits = outputs["final_logits"]
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y_batch.numpy())
            gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)
            total_examples += y_batch.size(0)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return {
        **classification_metrics(y_true, y_pred, class_names),
        "mean_gate_weights": (gate_sum / max(total_examples, 1)).tolist(),
    }


def train_and_evaluate(
    iq_train: np.ndarray,
    iq_val: np.ndarray,
    iq_test: np.ndarray,
    fft_train: np.ndarray,
    fft_val: np.ndarray,
    fft_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
):
    device, use_cuda = configure_device()

    train_loader = make_loader(iq_train, fft_train, y_train, shuffle=True, use_cuda=use_cuda)
    val_loader = make_loader(iq_val, fft_val, y_val, shuffle=False, use_cuda=use_cuda)
    test_loader = make_loader(iq_test, fft_test, y_test, shuffle=False, use_cuda=use_cuda)

    model = GatedMultimodalIQFFTCNN(num_classes=len(class_names), branch_dropout=BRANCH_DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_gate_mean = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_gate = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_cuda, True
        )
        val_loss, val_acc, val_gate = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, use_cuda, False
        )
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.3f} "
            f"| gates train {[round(x, 3) for x in train_gate]} val {[round(x, 3) for x in val_gate]}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_gate_mean = val_gate
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_test(model, test_loader, device, use_cuda, class_names)
    return {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "test_acc": test_metrics["test_acc"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "best_val_gate_weights": best_gate_mean,
        "test_gate_weights": test_metrics["mean_gate_weights"],
    }


def run_dataset_experiment(dataset_name: str, dataset_path: Path, normalize_rms: bool):
    iq, labels, class_names = load_h5(dataset_path, normalize_rms=normalize_rms)
    fft = to_fft(iq)
    splits = split_dataset(iq, fft, labels)
    print(f"\n=== {dataset_name}: Gated IQ+FFT multimodal CNN ===")
    result = train_and_evaluate(*splits, class_names)
    return {"gated_multimodal_cnn": result, "class_names": class_names}


def run_experiment() -> dict:
    set_global_seed(SEED)

    if not MODULATION_DATASET_PATH.exists() or not WAVEFORM_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Experiment 1 synthetic artifacts were not found. "
            "Run `./.venv/bin/python experiments/exp01_iq_vs_fft/run.py` "
            "or `./.venv/bin/python experiments/recreate_main_story.py --mode synthetic-only` first."
        )

    start = time.perf_counter()
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")

    results = {
        "modulation_time_feature_dataset": run_dataset_experiment(
            dataset_name="Modulation dataset",
            dataset_path=MODULATION_DATASET_PATH,
            normalize_rms=False,
        ),
        "waveform_frequency_feature_dataset": run_dataset_experiment(
            dataset_name="Waveform-family dataset",
            dataset_path=WAVEFORM_DATASET_PATH,
            normalize_rms=True,
        ),
    }

    if EXPERIMENT1_RESULTS_PATH.exists():
        results["experiment1_baselines"] = json.loads(EXPERIMENT1_RESULTS_PATH.read_text())
    if EXPERIMENT2_RESULTS_PATH.exists():
        results["experiment2_multimodal"] = json.loads(EXPERIMENT2_RESULTS_PATH.read_text())

    results["runtime_seconds"] = time.perf_counter() - start
    save_json(RESULTS_PATH, results)
    print(f"\nSaved results to {RESULTS_PATH}")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
from experiments.shared.repro import classification_metrics, configure_device, make_dataset_loader, save_json, set_global_seed
from experiments.shared.story_models import GatedMultimodalIQFFTCNN
