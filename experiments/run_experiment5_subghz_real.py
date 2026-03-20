from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SEED = 0
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
WINDOW_LENGTH = 4096
WINDOW_STRIDE = 2048
MAX_WINDOWS_PER_FILE = 64


CLASS_ALIASES = {
    "lora": "LoRa",
    "sigfox": "Sigfox",
    "noise": "Noise",
    "sun-fsk": "IEEE 802.15.4 SUN-FSK",
    "sun_fsk": "IEEE 802.15.4 SUN-FSK",
    "sunofdm": "IEEE 802.15.4 SUN-OFDM",
    "sun-ofdm": "IEEE 802.15.4 SUN-OFDM",
    "sun_ofdm": "IEEE 802.15.4 SUN-OFDM",
    "802.15.4g": "IEEE 802.15.4g",
    "802154g": "IEEE 802.15.4g",
    "ieee802154g": "IEEE 802.15.4g",
    "802.11ah": "IEEE 802.11ah",
    "80211ah": "IEEE 802.11ah",
    "ieee80211ah": "IEEE 802.11ah",
    "halow": "IEEE 802.11ah",
}

SUPPORTED_SUFFIXES = {".bin", ".dat", ".iq", ".raw", ".sigmf-data", ".npy", ".mat"}


def stratify_or_none(y: np.ndarray):
    _, counts = np.unique(y, return_counts=True)
    return y if np.all(counts >= 2) else None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def infer_class_from_path(path: Path) -> str | None:
    candidates = [part.lower() for part in path.parts] + [path.stem.lower()]
    for candidate in candidates:
        normalized = candidate.replace(" ", "").replace("_", "-")
        for alias, class_name in CLASS_ALIASES.items():
            if alias in normalized or alias in slugify(candidate):
                return class_name
    return None


def read_iq_file(path: Path, sample_format: str) -> np.ndarray:
    if path.suffix.lower() == ".mat":
        mat = scipy.io.loadmat(path)
        candidates = []
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.size < 2:
                continue
            candidates.append(arr)

        for arr in candidates:
            if np.iscomplexobj(arr):
                return np.asarray(arr).reshape(-1).astype(np.complex64)
            if arr.ndim == 2 and 2 in arr.shape:
                arr = arr.astype(np.float32)
                if arr.shape[-1] == 2:
                    return (arr[..., 0].reshape(-1) + 1j * arr[..., 1].reshape(-1)).astype(np.complex64)
                if arr.shape[0] == 2:
                    return (arr[0].reshape(-1) + 1j * arr[1].reshape(-1)).astype(np.complex64)

        raise ValueError(
            f"Could not infer IQ array from MAT file {path}. "
            "Expected a complex vector or a real array with one dimension of size 2."
        )

    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if np.iscomplexobj(data):
            return data.astype(np.complex64)
        if data.ndim == 2 and data.shape[-1] == 2:
            return (data[..., 0] + 1j * data[..., 1]).astype(np.complex64)
        raise ValueError(f"Unsupported npy layout for {path}")

    raw = np.fromfile(path, dtype={
        "u8": np.uint8,
        "i16": np.int16,
        "f32": np.float32,
    }[sample_format])
    if raw.size < 2:
        raise ValueError(f"File too small to contain IQ samples: {path}")
    if raw.size % 2 != 0:
        raw = raw[:-1]
    raw = raw.reshape(-1, 2)

    if sample_format == "u8":
        iq = (raw.astype(np.float32) - 127.5) / 128.0
    elif sample_format == "i16":
        iq = raw.astype(np.float32) / 32768.0
    else:
        iq = raw.astype(np.float32)

    return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def window_signal(signal: np.ndarray, length: int, stride: int, max_windows: int) -> np.ndarray:
    if signal.size < length:
        pad = np.zeros(length - signal.size, dtype=np.complex64)
        signal = np.concatenate([signal, pad])

    windows = []
    for start in range(0, max(signal.size - length + 1, 1), stride):
        segment = signal[start : start + length]
        if segment.size < length:
            segment = np.pad(segment, (0, length - segment.size))
        rms = np.sqrt(np.mean(np.abs(segment) ** 2) + 1e-8)
        windows.append((segment / rms).astype(np.complex64))
        if len(windows) >= max_windows:
            break

    if not windows:
        segment = signal[:length]
        rms = np.sqrt(np.mean(np.abs(segment) ** 2) + 1e-8)
        windows.append((segment / rms).astype(np.complex64))

    return np.stack(windows, axis=0)


@dataclass
class RecordingEntry:
    path: Path
    class_name: str
    class_index: int


def collect_recordings(dataset_root: Path) -> tuple[list[RecordingEntry], list[str]]:
    entries = []
    class_names = []
    class_to_index: dict[str, int] = {}
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        class_name = infer_class_from_path(path)
        if class_name is None:
            continue
        if class_name not in class_to_index:
            class_to_index[class_name] = len(class_names)
            class_names.append(class_name)
        entries.append(RecordingEntry(path=path, class_name=class_name, class_index=class_to_index[class_name]))
    return entries, class_names


def build_window_dataset(
    entries: list[RecordingEntry],
    sample_format: str,
    window_length: int,
    window_stride: int,
    max_windows_per_file: int,
):
    signals = []
    labels = []
    groups = []
    for group_idx, entry in enumerate(entries):
        signal = read_iq_file(entry.path, sample_format=sample_format)
        windows = window_signal(signal, length=window_length, stride=window_stride, max_windows=max_windows_per_file)
        stacked = np.stack((windows.real, windows.imag), axis=1).astype(np.float32)
        signals.append(stacked)
        labels.append(np.full(stacked.shape[0], entry.class_index, dtype=np.int64))
        groups.append(np.full(stacked.shape[0], group_idx, dtype=np.int64))
    return np.concatenate(signals, axis=0), np.concatenate(labels), np.concatenate(groups)


def train_val_test_split_by_group(labels: np.ndarray, groups: np.ndarray):
    unique_groups, first_idx = np.unique(groups, return_index=True)
    group_labels = labels[first_idx]
    g_train, g_temp = train_test_split(
        unique_groups,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify_or_none(group_labels),
    )
    temp_labels = group_labels[np.isin(unique_groups, g_temp)]
    g_val, g_test = train_test_split(
        g_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=stratify_or_none(temp_labels),
    )
    return np.isin(groups, g_train), np.isin(groups, g_val), np.isin(groups, g_test)


def to_fft(signals: np.ndarray):
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    fft_signals = np.fft.fftshift(np.fft.fft(complex_signals, axis=-1), axes=-1)
    features = np.stack((fft_signals.real, fft_signals.imag), axis=1).astype(np.float32)
    rms = np.sqrt(np.mean(features**2, axis=(1, 2), keepdims=True) + 1e-8)
    return features / rms


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


class MultimodalDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def make_multimodal_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool):
    dataset = MultimodalDataset(iq, fft, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def train_single_mode(X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = TimeOrFFTCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda", enabled=use_cuda)
    train_loader = make_loader(X_train, y_train, True, use_cuda)
    val_loader = make_loader(X_val, y_val, False, use_cuda)
    test_loader = make_loader(X_test, y_test, False, use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, EPOCHS + 1):
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:02d}/{EPOCHS} | val acc {val_acc:.3f}")

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
    report = classification_report(y_test, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0, output_dict=True)
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
        + 0.25 * criterion(outputs["iq_logits"], y_batch)
        + 0.25 * criterion(outputs["fft_logits"], y_batch)
        + 0.25 * criterion(outputs["fusion_logits"], y_batch)
    )


def train_gated(iq_train, iq_val, iq_test, fft_train, fft_val, fft_test, y_train, y_val, y_test, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = GatedMultimodalIQFFTCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda", enabled=use_cuda)
    train_loader = make_multimodal_loader(iq_train, fft_train, y_train, True, use_cuda)
    val_loader = make_multimodal_loader(iq_val, fft_val, y_val, False, use_cuda)
    test_loader = make_multimodal_loader(iq_test, fft_test, y_test, False, use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_gate = None
    for epoch in range(1, EPOCHS + 1):
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_gate = gate_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:02d}/{EPOCHS} | val acc {val_acc:.3f} | gates {[round(x, 3) for x in gate_mean]}")

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
    report = classification_report(y_test, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0, output_dict=True)
    return {
        "best_epoch": best_epoch,
        "val_acc": float(best_val_acc),
        "test_acc": float((y_pred == y_test).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "best_val_gate_weights": best_gate,
        "test_gate_weights": (gate_sum / max(total, 1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run real-data experiment on the Sub-GHz IQ signals dataset.")
    parser.add_argument("dataset_root", type=Path, help="Root directory of the downloaded Sub-GHz dataset.")
    parser.add_argument("--sample-format", choices=["u8", "i16", "f32"], default="u8", help="Raw IQ sample format.")
    parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH)
    parser.add_argument("--window-stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument("--max-windows-per-file", type=int, default=MAX_WINDOWS_PER_FILE)
    parser.add_argument("--results-path", type=Path, default=ROOT_DIR / "experiments" / "experiment5_subghz_results.json")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    entries, class_names = collect_recordings(args.dataset_root)
    if not entries:
        raise FileNotFoundError(
            f"No supported IQ files with recognizable class names were found under {args.dataset_root}. "
            "Expected class names like LoRa, Sigfox, 802.15.4g, SUN-OFDM, SUN-FSK, 802.11ah, or noise in paths."
        )

    print(f"Found {len(entries)} recording files across classes: {class_names}")
    signals, labels, groups = build_window_dataset(
        entries,
        sample_format=args.sample_format,
        window_length=args.window_length,
        window_stride=args.window_stride,
        max_windows_per_file=args.max_windows_per_file,
    )
    fft = to_fft(signals)
    train_mask, val_mask, test_mask = train_val_test_split_by_group(labels, groups)

    iq_train, iq_val, iq_test = signals[train_mask], signals[val_mask], signals[test_mask]
    fft_train, fft_val, fft_test = fft[train_mask], fft[val_mask], fft[test_mask]
    y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]

    print("windows:", signals.shape, "train/val/test:", iq_train.shape[0], iq_val.shape[0], iq_test.shape[0])

    start = time.perf_counter()
    results = {
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_format_assumption": args.sample_format,
        "class_names": class_names,
        "num_recordings": len(entries),
        "num_windows": int(signals.shape[0]),
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "max_windows_per_file": args.max_windows_per_file,
        "time_cnn": train_single_mode(iq_train, iq_val, iq_test, y_train, y_val, y_test, class_names),
        "fft_cnn": train_single_mode(fft_train, fft_val, fft_test, y_train, y_val, y_test, class_names),
        "gated_multimodal": train_gated(
            iq_train, iq_val, iq_test, fft_train, fft_val, fft_test, y_train, y_val, y_test, class_names
        ),
        "runtime_seconds": time.perf_counter() - start,
    }
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
