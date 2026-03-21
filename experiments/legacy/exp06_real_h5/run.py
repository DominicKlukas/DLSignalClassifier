from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_TRAIN_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "train.h5"
DEFAULT_VAL_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "val.h5"
DEFAULT_TEST_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "test.h5"
RESULTS_PATH = Path(__file__).resolve().with_name("results_default.json")

SEED = 0
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AUX_LOSS_WEIGHT = 0.25
DEFAULT_MAX_WINDOWS_PER_FILE = 256


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
    return np.asarray(selected, dtype=np.int64)


def load_split(path: Path, max_windows_per_file: int):
    with h5py.File(path, "r") as h5_file:
        source_files = h5_file["metadata"]["source_file"][:]
        indices = np.sort(select_balanced_indices(source_files, max_windows_per_file=max_windows_per_file))
        signals = h5_file["signals"][indices].astype(np.float32)
        labels = h5_file["labels"][indices].astype(np.int64)
        class_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h5_file.attrs["class_names"]
        ]
        protocols = h5_file["metadata"]["protocol"][indices]
        selected_files = source_files[indices]

    protocol_names = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in protocols]
    source_file_names = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in selected_files]
    return signals, labels, class_names, protocol_names, source_file_names


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def make_multimodal_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(MultimodalDataset(iq, fft, y), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def train_single_mode(X_train, X_val, X_test, y_train, y_val, y_test, class_names, batch_size: int, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = TimeOrFFTCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)
    train_loader = make_loader(X_train, y_train, True, use_cuda, batch_size)
    val_loader = make_loader(X_val, y_val, False, use_cuda, batch_size)
    test_loader = make_loader(X_test, y_test, False, use_cuda, batch_size)

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
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            Xb = Xb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(Xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        zero_division=0,
        output_dict=True,
    )
    return {
        "best_epoch": best_epoch,
        "val_acc": float(best_val_acc),
        "test_acc": float((y_pred == y_test).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def compute_gated_loss(outputs, y_batch, criterion):
    return (
        criterion(outputs["final_logits"], y_batch)
        + AUX_LOSS_WEIGHT * criterion(outputs["iq_logits"], y_batch)
        + AUX_LOSS_WEIGHT * criterion(outputs["fft_logits"], y_batch)
        + AUX_LOSS_WEIGHT * criterion(outputs["fusion_logits"], y_batch)
    )


def train_gated(iq_train, iq_val, iq_test, fft_train, fft_val, fft_test, y_train, y_val, y_test, class_names, batch_size: int, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = GatedMultimodalIQFFTCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)
    train_loader = make_multimodal_loader(iq_train, fft_train, y_train, True, use_cuda, batch_size)
    val_loader = make_multimodal_loader(iq_val, fft_val, y_val, False, use_cuda, batch_size)
    test_loader = make_multimodal_loader(iq_test, fft_test, y_test, False, use_cuda, batch_size)

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
    model.eval()
    preds = []
    gate_sum = torch.zeros(3, dtype=torch.float64)
    total = 0
    with torch.no_grad():
        for iqb, fftb, _ in test_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)
            total += iqb.size(0)
    y_pred = np.concatenate(preds)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        zero_division=0,
        output_dict=True,
    )
    return {
        "best_epoch": best_epoch,
        "val_acc": float(best_val_acc),
        "test_acc": float((y_pred == y_test).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "best_val_gate_weights": best_gate,
        "test_gate_weights": (gate_sum / max(total, 1)).tolist(),
    }


def class_window_counts(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(labels, minlength=len(class_names))
    return {class_names[idx]: int(counts[idx]) for idx in range(len(class_names))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare IQ, FFT, and gated multimodal CNNs on the real HDF5 dataset.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--max-windows-per-file", type=int, default=DEFAULT_MAX_WINDOWS_PER_FILE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()

    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"Using max {args.max_windows_per_file} windows per source file")

    train_path = args.train_path.expanduser().resolve()
    val_path = args.val_path.expanduser().resolve()
    test_path = args.test_path.expanduser().resolve()

    iq_train, y_train, class_names, train_protocols, train_files = load_split(train_path, args.max_windows_per_file)
    iq_val, y_val, _, val_protocols, val_files = load_split(val_path, args.max_windows_per_file)
    iq_test, y_test, _, test_protocols, test_files = load_split(test_path, args.max_windows_per_file)

    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)

    print("train:", iq_train.shape, class_window_counts(y_train, class_names))
    print("val:", iq_val.shape, class_window_counts(y_val, class_names))
    print("test:", iq_test.shape, class_window_counts(y_test, class_names))

    results = {
        "dataset": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "window_length": int(iq_train.shape[-1]),
            "max_windows_per_file": args.max_windows_per_file,
            "class_names": class_names,
            "train_windows": int(iq_train.shape[0]),
            "val_windows": int(iq_val.shape[0]),
            "test_windows": int(iq_test.shape[0]),
            "train_class_counts": class_window_counts(y_train, class_names),
            "val_class_counts": class_window_counts(y_val, class_names),
            "test_class_counts": class_window_counts(y_test, class_names),
            "train_source_files": int(len(set(train_files))),
            "val_source_files": int(len(set(val_files))),
            "test_source_files": int(len(set(test_files))),
        },
        "time_cnn": train_single_mode(
            iq_train, iq_val, iq_test, y_train, y_val, y_test, class_names, args.batch_size, args.epochs
        ),
        "fft_cnn": train_single_mode(
            fft_train, fft_val, fft_test, y_train, y_val, y_test, class_names, args.batch_size, args.epochs
        ),
        "gated_multimodal_cnn": train_gated(
            iq_train,
            iq_val,
            iq_test,
            fft_train,
            fft_val,
            fft_test,
            y_train,
            y_val,
            y_test,
            class_names,
            args.batch_size,
            args.epochs,
        ),
        "runtime_seconds": time.perf_counter() - start,
    }
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
