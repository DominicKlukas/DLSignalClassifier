from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DATASET_PATH = ROOT_DIR / "experiments" / "experiment1" / "waveform_frequency_features.h5"
RESULTS_PATH = Path(__file__).resolve().with_name("results.json")
PLOT_PATH = Path(__file__).resolve().with_name("plot.png")

SEED = 0
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AUX_LOSS_WEIGHT = 0.25
BRANCH_DROPOUT = 0.10
DEFAULT_SNR_LEVELS: list[str] = ["clean", "30", "20", "10", "5", "0", "-5", "-10", "-15", "-20"]


def stratify_or_none(y: np.ndarray):
    _, counts = np.unique(y, return_counts=True)
    return y if np.all(counts >= 2) else None


class RepresentationCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MultimodalDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


class FeatureBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class GatedMultimodalIQFFTCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.iq_branch = FeatureBranch()
        self.fft_branch = FeatureBranch()
        self.iq_head = nn.Linear(256, num_classes)
        self.fft_head = nn.Linear(256, num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def _apply_branch_dropout(self, iq_features: torch.Tensor, fft_features: torch.Tensor):
        if not self.training or BRANCH_DROPOUT <= 0.0:
            return iq_features, fft_features
        batch = iq_features.shape[0]
        device = iq_features.device
        keep_iq = (torch.rand(batch, 1, device=device) > BRANCH_DROPOUT).float()
        keep_fft = (torch.rand(batch, 1, device=device) > BRANCH_DROPOUT).float()
        both_dropped = keep_iq + keep_fft == 0
        if both_dropped.any():
            choose_iq = (torch.rand(batch, 1, device=device) > 0.5).float()
            keep_iq = torch.where(both_dropped, choose_iq, keep_iq)
            keep_fft = torch.where(both_dropped, 1.0 - keep_iq, keep_fft)
        return iq_features * keep_iq, fft_features * keep_fft

    def forward(self, iq: torch.Tensor, fft: torch.Tensor):
        iq_features = self.iq_branch(iq)
        fft_features = self.fft_branch(fft)
        iq_features, fft_features = self._apply_branch_dropout(iq_features, fft_features)
        iq_logits = self.iq_head(iq_features)
        fft_logits = self.fft_head(fft_features)
        fused_features = torch.cat([iq_features, fft_features], dim=1)
        fusion_logits = self.fusion_head(fused_features)
        gate_weights = torch.softmax(self.gate(fused_features), dim=1)
        final_logits = (
            gate_weights[:, 0:1] * iq_logits
            + gate_weights[:, 1:2] * fft_logits
            + gate_weights[:, 2:3] * fusion_logits
        )
        return {
            "final_logits": final_logits,
            "iq_logits": iq_logits,
            "fft_logits": fft_logits,
            "fusion_logits": fusion_logits,
            "gate_weights": gate_weights,
        }


def load_h5(path: Path):
    with h5py.File(path, "r") as h5_file:
        signals = h5_file["signals"][:].astype(np.float32)
        labels = h5_file["labels"][:].astype(np.int64)
        class_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h5_file.attrs["class_names"]
        ]
    return normalize_iq_channels(signals), labels, class_names


def normalize_iq_channels(signals: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(signals**2, axis=(1, 2), keepdims=True) + 1e-8).astype(np.float32)
    return (signals / rms).astype(np.float32)


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


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=use_cuda, persistent_workers=False)


def make_multimodal_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool):
    return DataLoader(MultimodalDataset(iq, fft, y), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=use_cuda, persistent_workers=False)


def run_epoch_single(model, loader, criterion, optimizer, scaler, device, use_cuda, training: bool):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=use_cuda)
        y_batch = y_batch.to(device, non_blocking=use_cuda)
        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_examples += y_batch.size(0)
    return total_loss / total_examples, total_correct / total_examples


def evaluate_single(model, test_signals, y_test, class_names, device, use_cuda):
    test_loader = make_loader(test_signals, y_test, False, use_cuda)
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(X_batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    report = classification_report(y_test, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0, output_dict=True)
    return {
        "test_acc": float((y_pred == y_test).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def train_single(train_signals, val_signals, y_train, y_val, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_loader = make_loader(train_signals, y_train, True, use_cuda)
    val_loader = make_loader(val_signals, y_val, False, use_cuda)
    model = RepresentationCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda", enabled=use_cuda)
    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch_single(model, train_loader, criterion, optimizer, scaler, device, use_cuda, True)
        val_loss, val_acc = run_epoch_single(model, val_loader, criterion, optimizer, scaler, device, use_cuda, False)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device, use_cuda, {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
    }


def compute_gated_loss(outputs, y_batch, criterion):
    final_loss = criterion(outputs["final_logits"], y_batch)
    iq_loss = criterion(outputs["iq_logits"], y_batch)
    fft_loss = criterion(outputs["fft_logits"], y_batch)
    fusion_loss = criterion(outputs["fusion_logits"], y_batch)
    return final_loss + AUX_LOSS_WEIGHT * (iq_loss + fft_loss + fusion_loss)


def run_epoch_gated(model, loader, criterion, optimizer, scaler, device, use_cuda, training: bool):
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
                loss = compute_gated_loss(outputs, y_batch, criterion)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (outputs["final_logits"].argmax(dim=1) == y_batch).sum().item()
        total_examples += y_batch.size(0)
        gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)
    gate_mean = (gate_sum / max(total_examples, 1)).tolist()
    return total_loss / total_examples, total_correct / total_examples, gate_mean


def evaluate_test_gated(model, loader, device, use_cuda, class_names):
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
            all_preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            all_targets.append(y_batch.numpy())
            gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)
            total_examples += y_batch.size(0)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    report = classification_report(y_true, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0, output_dict=True)
    return {
        "test_acc": float((y_pred == y_true).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "mean_gate_weights": (gate_sum / max(total_examples, 1)).tolist(),
    }


def evaluate_gated(model, iq_test, fft_test, y_test, class_names, device, use_cuda):
    test_loader = make_multimodal_loader(iq_test, fft_test, y_test, False, use_cuda)
    return evaluate_test_gated(model, test_loader, device, use_cuda, class_names)


def train_gated(iq_train, iq_val, fft_train, fft_val, y_train, y_val, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_loader = make_multimodal_loader(iq_train, fft_train, y_train, True, use_cuda)
    val_loader = make_multimodal_loader(iq_val, fft_val, y_val, False, use_cuda)
    model = GatedMultimodalIQFFTCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda", enabled=use_cuda)
    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_gate = None
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_gate = run_epoch_gated(model, train_loader, criterion, optimizer, scaler, device, use_cuda, True)
        val_loss, val_acc, val_gate = run_epoch_gated(model, val_loader, criterion, optimizer, scaler, device, use_cuda, False)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"gates train {[round(x, 3) for x in train_gate]} val {[round(x, 3) for x in val_gate]}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_gate = val_gate
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device, use_cuda, {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "best_val_gate_weights": best_gate,
    }


def add_awgn(signals: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    signal_power = np.mean(np.abs(complex_signals) ** 2, axis=1).astype(np.float32)
    snr_linear = np.power(10.0, snr_db / 10.0, dtype=np.float32)
    sigma = np.sqrt(signal_power / np.maximum(snr_linear, 1e-8) / 2.0, dtype=np.float32)
    noise = (
        rng.normal(0.0, 1.0, size=complex_signals.shape).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=complex_signals.shape).astype(np.float32)
    ) * sigma[:, None]
    noisy = (complex_signals + noise).astype(np.complex64)
    noisy_iq = np.stack((noisy.real, noisy.imag), axis=1).astype(np.float32)
    return normalize_iq_channels(noisy_iq)


def maybe_noisy(signals: np.ndarray, snr_label: str, rng: np.random.Generator) -> np.ndarray:
    if snr_label == "clean":
        return signals.copy()
    return add_awgn(signals, float(snr_label), rng)


def plot_results(labels: list[str], curves: dict[str, list[float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    for name, values in curves.items():
        ax.plot(x, values, marker="o", linewidth=2, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Test-time AWGN Level")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Corrected Waveform-Family AWGN Ablation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithful test-only AWGN ablation on waveform-family dataset.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--plot-path", type=Path, default=PLOT_PATH)
    parser.add_argument("--snr-levels", nargs="+", default=DEFAULT_SNR_LEVELS)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")

    iq, labels, class_names = load_h5(DATASET_PATH)
    fft = to_fft(iq)
    splits = split_dataset(iq, fft, labels)
    iq_train, iq_val, iq_test, fft_train, fft_val, fft_test, y_train, y_val, y_test = splits
    print("train:", iq_train.shape, "val:", iq_val.shape, "test:", iq_test.shape)

    print("\n=== Clean training: IQ CNN ===")
    time_model, time_device, time_use_cuda, time_train = train_single(iq_train, iq_val, y_train, y_val, class_names)
    time_clean = time_train | evaluate_single(time_model, iq_test, y_test, class_names, time_device, time_use_cuda)
    print("\n=== Clean training: FFT CNN ===")
    fft_model, fft_device, fft_use_cuda, fft_train_result = train_single(fft_train, fft_val, y_train, y_val, class_names)
    fft_clean = fft_train_result | evaluate_single(fft_model, fft_test, y_test, class_names, fft_device, fft_use_cuda)
    print("\n=== Clean training: Gated multimodal CNN ===")
    gated_model, gated_device, gated_use_cuda, gated_train = train_gated(iq_train, iq_val, fft_train, fft_val, y_train, y_val, class_names)
    gated_clean = gated_train | evaluate_gated(gated_model, iq_test, fft_test, y_test, class_names, gated_device, gated_use_cuda)

    curves = {"IQ CNN": [], "FFT CNN": [], "Gated Multimodal CNN": []}
    rows = []
    for idx, snr_label in enumerate(args.snr_levels):
        rng = np.random.default_rng(SEED + idx)
        noisy_iq_test = maybe_noisy(iq_test, snr_label, rng)
        noisy_fft_test = to_fft(noisy_iq_test)
        if snr_label == "clean":
            time_acc = time_clean["test_acc"]
            fft_acc = fft_clean["test_acc"]
            gated_acc = gated_clean["test_acc"]
        else:
            time_acc = evaluate_single(time_model, noisy_iq_test, y_test, class_names, time_device, time_use_cuda)["test_acc"]
            fft_acc = evaluate_single(fft_model, noisy_fft_test, y_test, class_names, fft_device, fft_use_cuda)["test_acc"]
            gated_acc = evaluate_gated(gated_model, noisy_iq_test, noisy_fft_test, y_test, class_names, gated_device, gated_use_cuda)["test_acc"]
        curves["IQ CNN"].append(time_acc)
        curves["FFT CNN"].append(fft_acc)
        curves["Gated Multimodal CNN"].append(gated_acc)
        rows.append({"snr_label": snr_label, "time_cnn_acc": time_acc, "fft_cnn_acc": fft_acc, "gated_multimodal_acc": gated_acc})
        print(f"Level {snr_label:>5} | IQ {time_acc:.3f} | FFT {fft_acc:.3f} | Gated {gated_acc:.3f}")

    plot_results(list(args.snr_levels), curves, args.plot_path)
    results = {
        "dataset": {
            "dataset_path": str(DATASET_PATH),
            "class_names": class_names,
            "train_windows": int(iq_train.shape[0]),
            "val_windows": int(iq_val.shape[0]),
            "test_windows": int(iq_test.shape[0]),
        },
        "clean_reference": {
            "time_cnn": time_clean,
            "fft_cnn": fft_clean,
            "gated_multimodal_cnn": gated_clean,
        },
        "test_only_awgn_ablation": rows,
        "plot_path": str(args.plot_path.expanduser().resolve()),
        "runtime_seconds": time.perf_counter() - start,
    }
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.results_path}")
    print(f"Saved plot to {args.plot_path}")


if __name__ == "__main__":
    main()
