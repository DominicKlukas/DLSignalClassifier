from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SignalGenerator import SignalGenerator
from WaveformFamilyGenerator import WaveformFamilyGenerator


BASE_DIR = Path(__file__).resolve().parent / "experiment1"
BASE_DIR.mkdir(parents=True, exist_ok=True)

MODULATION_DATASET_PATH = BASE_DIR / "modulation_time_features.h5"
WAVEFORM_DATASET_PATH = BASE_DIR / "waveform_frequency_features.h5"
RESULTS_PATH = BASE_DIR / "experiment1_results.json"

SEED = 0
SIGNAL_LENGTH = 1024
SR_OUT = 8_000
MODULATION_CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4"]
WAVEFORM_CLASSES = ["CW", "AM", "FM", "OFDM", "LFM_CHIRP", "DSSS", "FHSS", "SC_BURST"]
SAMPLES_PER_CLASS = 500
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05


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


def save_h5(signals: np.ndarray, labels: np.ndarray, class_names: list[str], path: Path) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("signals", data=signals.astype(np.float32), compression="gzip", chunks=True)
        h5_file.create_dataset("labels", data=labels.astype(np.int64), compression="gzip", chunks=True)
        h5_file.attrs["signal_layout"] = "NCH"
        h5_file.attrs["signal_channels"] = np.asarray(["I", "Q"], dtype=string_dtype)
        h5_file.attrs["class_names"] = np.asarray(class_names, dtype=string_dtype)


def build_modulation_dataset(path: Path) -> None:
    generator = SignalGenerator(seed=SEED)
    all_signals = []
    all_labels = []
    for class_idx, modulation in enumerate(MODULATION_CLASSES):
        signals, _ = generator.generate_batch(
            signal_length=SIGNAL_LENGTH,
            num_signals=SAMPLES_PER_CLASS,
            modulation_types=[modulation],
            symbol_rate_bnds=[150.0, 1200.0],
            phase_shift_bnds=[-np.pi, np.pi],
            frequency_offset_bnds=[-80.0, 80.0],
            SNR_bnds=[0.0, 18.0],
            sr_out=SR_OUT,
        )
        batch = np.stack(
            [np.stack((np.real(signal), np.imag(signal)), axis=0) for signal in signals],
            axis=0,
        )
        all_signals.append(batch)
        all_labels.append(np.full(SAMPLES_PER_CLASS, class_idx, dtype=np.int64))

    signals = np.concatenate(all_signals, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    save_h5(signals, labels, MODULATION_CLASSES, path)


def build_waveform_dataset(path: Path) -> None:
    generator = WaveformFamilyGenerator(seed=SEED)
    all_signals = []
    all_labels = []
    for class_idx, waveform in enumerate(WAVEFORM_CLASSES):
        signals, _ = generator.generate_batch(
            signal_length=SIGNAL_LENGTH,
            num_signals=SAMPLES_PER_CLASS,
            waveform_types=[waveform],
            snr_db_bnds=[0.0, 18.0],
            center_frequency_bnds=[-1500.0, 1500.0],
            sample_rate_scale_choices=[1.0],
            sr_out=SR_OUT,
            occupied_bandwidth_bnds=[120.0, 900.0],
        )
        batch = np.stack(
            [np.stack((np.real(signal), np.imag(signal)), axis=0) for signal in signals],
            axis=0,
        )
        all_signals.append(batch)
        all_labels.append(np.full(SAMPLES_PER_CLASS, class_idx, dtype=np.int64))

    signals = np.concatenate(all_signals, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    save_h5(signals, labels, WAVEFORM_CLASSES, path)


def ensure_datasets():
    if not MODULATION_DATASET_PATH.exists():
        build_modulation_dataset(MODULATION_DATASET_PATH)
    if not WAVEFORM_DATASET_PATH.exists():
        build_waveform_dataset(WAVEFORM_DATASET_PATH)


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


def split_dataset(signals: np.ndarray, labels: np.ndarray):
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify_or_none(labels),
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=stratify_or_none(y_temp),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
        persistent_workers=False,
    )


def run_epoch(model, loader, criterion, optimizer, scaler, device, use_cuda, training: bool):
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


def train_and_evaluate(
    train_signals: np.ndarray,
    val_signals: np.ndarray,
    test_signals: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader = make_loader(train_signals, y_train, shuffle=True, use_cuda=use_cuda)
    val_loader = make_loader(val_signals, y_val, shuffle=False, use_cuda=use_cuda)
    test_loader = make_loader(test_signals, y_test, shuffle=False, use_cuda=use_cuda)

    model = RepresentationCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, use_cuda, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, use_cuda, False)
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

    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(X_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        zero_division=0,
        output_dict=True,
    )
    return {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "test_acc": float((y_pred == y_test).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def run_dataset_experiment(dataset_name: str, dataset_path: Path, normalize_rms: bool):
    signals, labels, class_names = load_h5(dataset_path, normalize_rms=normalize_rms)
    time_splits = split_dataset(signals, labels)
    fft_splits = split_dataset(to_fft(signals), labels)

    print(f"\n=== {dataset_name}: Time CNN ===")
    time_result = train_and_evaluate(*time_splits, class_names)
    print(f"\n=== {dataset_name}: FFT CNN ===")
    fft_result = train_and_evaluate(*fft_splits, class_names)
    return {"time_cnn": time_result, "fft_cnn": fft_result, "class_names": class_names}


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()

    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print("Preparing datasets...")
    ensure_datasets()

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
    results["runtime_seconds"] = time.perf_counter() - start

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {RESULTS_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
