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
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_TRAIN_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "train.h5"
DEFAULT_VAL_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "val.h5"
DEFAULT_TEST_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "test.h5"
DEFAULT_RESULTS_PATH = ROOT_DIR / "experiments" / "experiment7_awgn_ablation_real_results.json"
DEFAULT_PLOT_PATH = ROOT_DIR / "experiments" / "experiment7_awgn_ablation_real.png"

SEED = 0
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 20
DEFAULT_MAX_WINDOWS_PER_FILE = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AUX_LOSS_WEIGHT = 0.25
DEFAULT_SNR_LEVELS = [30.0, 20.0, 10.0, 5.0, 0.0, -5.0]


class MultimodalDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


class TimeOrFFTCNN(nn.Module):
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
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


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

    def forward(self, iq: torch.Tensor, fft: torch.Tensor):
        iq_features = self.iq_branch(iq)
        fft_features = self.fft_branch(fft)
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


def to_fft(signals: np.ndarray) -> np.ndarray:
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    fft_signals = np.fft.fftshift(np.fft.fft(complex_signals, axis=-1), axes=-1)
    features = np.stack((fft_signals.real, fft_signals.imag), axis=1).astype(np.float32)
    rms = np.sqrt(np.mean(features**2, axis=(1, 2), keepdims=True) + 1e-8)
    return features / rms


def select_balanced_indices(source_files: np.ndarray, max_windows_per_file: int) -> np.ndarray:
    indices_by_file: dict[str, list[int]] = {}
    for idx, value in enumerate(source_files):
        key = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        indices_by_file.setdefault(key, []).append(idx)

    selected = []
    for key in sorted(indices_by_file):
        indices = np.asarray(indices_by_file[key], dtype=np.int64)
        if len(indices) <= max_windows_per_file:
            selected.extend(indices.tolist())
            continue
        positions = np.linspace(0, len(indices) - 1, num=max_windows_per_file, dtype=np.int64)
        selected.extend(indices[positions].tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def load_split(path: Path, max_windows_per_file: int):
    with h5py.File(path, "r") as h5_file:
        source_files = h5_file["metadata"]["source_file"][:]
        indices = select_balanced_indices(source_files, max_windows_per_file=max_windows_per_file)
        signals = h5_file["signals"][indices].astype(np.float32)
        labels = h5_file["labels"][indices].astype(np.int64)
        class_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h5_file.attrs["class_names"]
        ]
    return signals, labels, class_names


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def make_multimodal_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(MultimodalDataset(iq, fft, y), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def evaluate_single_mode(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device, use_cuda: bool) -> np.ndarray:
    loader = make_loader(X, y, False, use_cuda, batch_size)
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(Xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_gated(model: GatedMultimodalIQFFTCNN, iq: np.ndarray, fft: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device, use_cuda: bool) -> np.ndarray:
    loader = make_multimodal_loader(iq, fft, y, False, use_cuda, batch_size)
    model.eval()
    preds = []
    with torch.no_grad():
        for iqb, fftb, _ in loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_single_mode(X_train, X_val, y_train, y_val, class_names, batch_size: int, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = TimeOrFFTCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)
    train_loader = make_loader(X_train, y_train, True, use_cuda, batch_size)
    val_loader = make_loader(X_val, y_val, False, use_cuda, batch_size)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train(True)
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(Xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    logits = model(Xb)
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
    return model, {"best_epoch": best_epoch, "val_acc": float(best_val_acc), "device": device.type}


def compute_gated_loss(outputs, y_batch, criterion):
    return (
        criterion(outputs["final_logits"], y_batch)
        + AUX_LOSS_WEIGHT * criterion(outputs["iq_logits"], y_batch)
        + AUX_LOSS_WEIGHT * criterion(outputs["fft_logits"], y_batch)
        + AUX_LOSS_WEIGHT * criterion(outputs["fusion_logits"], y_batch)
    )


def train_gated(iq_train, iq_val, fft_train, fft_val, y_train, y_val, class_names, batch_size: int, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = GatedMultimodalIQFFTCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)
    train_loader = make_multimodal_loader(iq_train, fft_train, y_train, True, use_cuda, batch_size)
    val_loader = make_multimodal_loader(iq_val, fft_val, y_val, False, use_cuda, batch_size)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_gate = None
    for epoch in range(1, epochs + 1):
        model.train(True)
        for iqb, fftb, yb in train_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
                loss = compute_gated_loss(outputs, yb, criterion)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        correct = 0
        total = 0
        gate_sum = torch.zeros(3, dtype=torch.float64)
        with torch.no_grad():
            for iqb, fftb, yb in val_loader:
                iqb = iqb.to(device, non_blocking=use_cuda)
                fftb = fftb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(iqb, fftb)
                correct += (outputs["final_logits"].argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
                gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)
        val_acc = correct / max(total, 1)
        gate_mean = (gate_sum / max(total, 1)).tolist()
        scheduler.step()
        print(f"Epoch {epoch:02d}/{epochs} | val acc {val_acc:.3f} | gates {[round(x, 3) for x in gate_mean]}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_gate = gate_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "val_acc": float(best_val_acc), "best_val_gate_weights": best_gate, "device": device.type}


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
    rms = np.sqrt(np.mean(np.abs(noisy) ** 2, axis=1, keepdims=True) + 1e-8).astype(np.float32)
    noisy = noisy / rms
    return np.stack((noisy.real, noisy.imag), axis=1).astype(np.float32)


def plot_results(snr_levels: list[float], curves: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in curves.items():
        ax.plot(snr_levels, values, marker="o", linewidth=2, label=label)
    ax.set_xlabel("AWGN SNR (dB)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Experiment 7: AWGN Ablation on Real Dataset")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AWGN ablation on the real Sub-GHz benchmark.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--max-windows-per-file", type=int, default=DEFAULT_MAX_WINDOWS_PER_FILE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--snr-levels", type=float, nargs="+", default=DEFAULT_SNR_LEVELS)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    start = time.perf_counter()

    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")

    train_path = args.train_path.expanduser().resolve()
    val_path = args.val_path.expanduser().resolve()
    test_path = args.test_path.expanduser().resolve()

    iq_train, y_train, class_names = load_split(train_path, args.max_windows_per_file)
    iq_val, y_val, _ = load_split(val_path, args.max_windows_per_file)
    iq_test, y_test, _ = load_split(test_path, args.max_windows_per_file)
    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)

    print("train:", iq_train.shape)
    print("val:", iq_val.shape)
    print("test:", iq_test.shape)

    time_model, time_train = train_single_mode(iq_train, iq_val, y_train, y_val, class_names, args.batch_size, args.epochs)
    fft_model, fft_train_result = train_single_mode(fft_train, fft_val, y_train, y_val, class_names, args.batch_size, args.epochs)
    gated_model, gated_train = train_gated(
        iq_train, iq_val, fft_train, fft_val, y_train, y_val, class_names, args.batch_size, args.epochs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    curves = {"IQ CNN": [], "FFT CNN": [], "Gated Multimodal CNN": []}
    ablation_rows = []

    for snr_db in args.snr_levels:
        iq_noisy = add_awgn(iq_test, snr_db=snr_db, rng=rng)
        fft_noisy = to_fft(iq_noisy)

        time_pred = evaluate_single_mode(time_model, iq_noisy, y_test, args.batch_size, device, use_cuda)
        fft_pred = evaluate_single_mode(fft_model, fft_noisy, y_test, args.batch_size, device, use_cuda)
        gated_pred = evaluate_gated(gated_model, iq_noisy, fft_noisy, y_test, args.batch_size, device, use_cuda)

        time_acc = float((time_pred == y_test).mean())
        fft_acc = float((fft_pred == y_test).mean())
        gated_acc = float((gated_pred == y_test).mean())

        curves["IQ CNN"].append(time_acc)
        curves["FFT CNN"].append(fft_acc)
        curves["Gated Multimodal CNN"].append(gated_acc)
        ablation_rows.append(
            {
                "snr_db": float(snr_db),
                "time_cnn_acc": time_acc,
                "fft_cnn_acc": fft_acc,
                "gated_multimodal_acc": gated_acc,
            }
        )
        print(f"SNR {snr_db:>5.1f} dB | IQ {time_acc:.3f} | FFT {fft_acc:.3f} | Gated {gated_acc:.3f}")

    plot_results(list(args.snr_levels), curves, args.plot_path)

    results = {
        "dataset": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "max_windows_per_file": int(args.max_windows_per_file),
            "train_windows": int(iq_train.shape[0]),
            "val_windows": int(iq_val.shape[0]),
            "test_windows": int(iq_test.shape[0]),
            "class_names": class_names,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "time_cnn": time_train,
            "fft_cnn": fft_train_result,
            "gated_multimodal_cnn": gated_train,
        },
        "awgn_ablation": ablation_rows,
        "plot_path": str(args.plot_path.expanduser().resolve()),
        "runtime_seconds": time.perf_counter() - start,
    }
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.results_path}")
    print(f"Saved plot to {args.plot_path}")


if __name__ == "__main__":
    main()
