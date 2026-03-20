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


BASE_DIR = Path(__file__).resolve().parent / "experiment1"
MODULATION_DATASET_PATH = BASE_DIR / "modulation_time_features.h5"
WAVEFORM_DATASET_PATH = BASE_DIR / "waveform_frequency_features.h5"
EXPERIMENT1_RESULTS_PATH = BASE_DIR / "experiment1_results.json"
EXPERIMENT2_RESULTS_PATH = BASE_DIR / "experiment2_results.json"
EXPERIMENT3_RESULTS_PATH = BASE_DIR / "experiment3_results.json"
RESULTS_PATH = BASE_DIR / "experiment4_results.json"

SEED = 0
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
N_FFT = 128
HOP_LENGTH = 32


def stratify_or_none(y: np.ndarray):
    _, counts = np.unique(y, return_counts=True)
    return y if np.all(counts >= 2) else None


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


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


def to_spectrogram(signals: np.ndarray):
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    tensor = torch.from_numpy(complex_signals.astype(np.complex64))
    window = torch.hann_window(N_FFT)
    stft = torch.stft(
        tensor,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True,
        center=True,
    )
    magnitude = torch.abs(stft)
    log_mag = torch.log1p(magnitude)
    log_mag = log_mag.unsqueeze(1)
    log_mag = log_mag / (log_mag.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return log_mag.numpy().astype(np.float32)


def split_dataset(features: np.ndarray, labels: np.ndarray):
    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
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


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader = make_loader(X_train, y_train, shuffle=True, use_cuda=use_cuda)
    val_loader = make_loader(X_val, y_val, shuffle=False, use_cuda=use_cuda)
    test_loader = make_loader(X_test, y_test, shuffle=False, use_cuda=use_cuda)

    model = SpectrogramCNN(num_classes=len(class_names)).to(device)
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
    spectrogram = to_spectrogram(signals)
    splits = split_dataset(spectrogram, labels)
    print(f"\n=== {dataset_name}: Spectrogram CNN ===")
    result = train_and_evaluate(*splits, class_names)
    return {"spectrogram_cnn": result, "class_names": class_names}


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not MODULATION_DATASET_PATH.exists() or not WAVEFORM_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Experiment 1 datasets were not found. Run experiments/run_experiment1_time_vs_fft.py first."
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

    for path_key, path in [
        ("experiment1_baselines", EXPERIMENT1_RESULTS_PATH),
        ("experiment2_multimodal", EXPERIMENT2_RESULTS_PATH),
        ("experiment3_gated_multimodal", EXPERIMENT3_RESULTS_PATH),
    ]:
        if path.exists():
            results[path_key] = json.loads(path.read_text())

    results["runtime_seconds"] = time.perf_counter() - start
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {RESULTS_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
