from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SEED = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AUX_LOSS_WEIGHT = 0.25

WAVEFORM_PATH = ROOT_DIR / "experiments" / "experiment1" / "waveform_frequency_features.h5"
MODULATION_PATH = ROOT_DIR / "experiments" / "experiment1" / "modulation_time_features.h5"
REAL_TRAIN_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "train.h5"
REAL_VAL_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "val.h5"
REAL_TEST_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "test.h5"
ORBIT_ROOT = ROOT_DIR / "orbit_rf_identification_dataset_updated"
RESULTS_PATH = ROOT_DIR / "experiments" / "experiment9_wavelet_multidataset_results.json"

DEFAULT_ORBIT_DAY_FILES = [
    "grid_2019_12_25.pkl",
    "grid_2020_02_03.pkl",
    "grid_2020_02_04.pkl",
    "grid_2020_02_05.pkl",
    "grid_2020_02_06.pkl",
]


def stratify_or_none(y: np.ndarray):
    _, counts = np.unique(y, return_counts=True)
    return y if np.all(counts >= 2) else None


def normalize_channel_rms(signals: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(signals**2, axis=(1, 2), keepdims=True) + 1e-8).astype(np.float32)
    return (signals / rms).astype(np.float32)


def normalize_complex_rms(signals: np.ndarray) -> np.ndarray:
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    rms = np.sqrt(np.mean(np.abs(complex_signals) ** 2, axis=1, keepdims=True) + 1e-8).astype(np.float32)
    complex_signals = complex_signals / rms
    return np.stack((complex_signals.real, complex_signals.imag), axis=1).astype(np.float32)


def to_fft(signals: np.ndarray) -> np.ndarray:
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    fft_signals = np.fft.fftshift(np.fft.fft(complex_signals, axis=-1), axes=-1)
    features = np.stack((fft_signals.real, fft_signals.imag), axis=1).astype(np.float32)
    rms = np.sqrt(np.mean(features**2, axis=(1, 2), keepdims=True) + 1e-8)
    return features / rms


def build_morlet_bank(scales: list[int], kernel_size: int, device: torch.device):
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    real_kernels = []
    imag_kernels = []
    for scale in scales:
        sigma = float(scale)
        x = t / sigma
        envelope = torch.exp(-0.5 * x**2)
        carrier = torch.exp(1j * 5.0 * x)
        wavelet = envelope * carrier
        wavelet = wavelet - wavelet.mean()
        norm = torch.sqrt(torch.sum(torch.abs(wavelet) ** 2) + 1e-8)
        wavelet = wavelet / norm
        real_kernels.append(wavelet.real)
        imag_kernels.append(wavelet.imag)
    real = torch.stack(real_kernels, dim=0).unsqueeze(1)
    imag = torch.stack(imag_kernels, dim=0).unsqueeze(1)
    return real, imag


def to_wavelet(
    signals: np.ndarray,
    batch_size: int = 256,
    scales: list[int] | None = None,
    kernel_size: int = 127,
) -> np.ndarray:
    if scales is None:
        scales = [2, 4, 8, 12, 16, 24]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_kernels, imag_kernels = build_morlet_bank(scales=scales, kernel_size=kernel_size, device=device)
    pad = kernel_size // 2
    outputs = []
    for start in range(0, len(signals), batch_size):
        batch = torch.from_numpy(signals[start : start + batch_size]).to(device=device, dtype=torch.float32)
        i_part = batch[:, 0:1, :]
        q_part = batch[:, 1:2, :]
        conv_real = F.conv1d(i_part, real_kernels, padding=pad) - F.conv1d(q_part, imag_kernels, padding=pad)
        conv_imag = F.conv1d(i_part, imag_kernels, padding=pad) + F.conv1d(q_part, real_kernels, padding=pad)
        magnitude = torch.sqrt(conv_real.square() + conv_imag.square() + 1e-8)
        magnitude = torch.log1p(magnitude)
        magnitude = magnitude / torch.sqrt(torch.mean(magnitude.square(), dim=(1, 2), keepdim=True) + 1e-8)
        outputs.append(magnitude.unsqueeze(1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


class PairDataset(Dataset):
    def __init__(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
        self.x1 = torch.from_numpy(x1)
        self.x2 = torch.from_numpy(x2)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x1[idx], self.x2[idx], self.y[idx]


class TripleDataset(Dataset):
    def __init__(self, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, y: np.ndarray):
        self.x1 = torch.from_numpy(x1)
        self.x2 = torch.from_numpy(x2)
        self.x3 = torch.from_numpy(x3)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x1[idx], self.x2[idx], self.x3[idx], self.y[idx]


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


class WaveletCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FeatureBranch1D(nn.Module):
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


class FeatureBranch2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class GatedIQFFTCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.iq_branch = FeatureBranch1D()
        self.fft_branch = FeatureBranch1D()
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
            "aux_logits": [iq_logits, fft_logits, fusion_logits],
            "gate_weights": gate_weights,
        }


class GatedIQFFTWaveletCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.iq_branch = FeatureBranch1D()
        self.fft_branch = FeatureBranch1D()
        self.wavelet_branch = FeatureBranch2D()
        self.iq_head = nn.Linear(256, num_classes)
        self.fft_head = nn.Linear(256, num_classes)
        self.wavelet_head = nn.Linear(64, num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 256 + 64, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )
        self.gate = nn.Sequential(
            nn.Linear(256 + 256 + 64, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )

    def forward(self, iq: torch.Tensor, fft: torch.Tensor, wavelet: torch.Tensor):
        iq_features = self.iq_branch(iq)
        fft_features = self.fft_branch(fft)
        wavelet_features = self.wavelet_branch(wavelet)
        iq_logits = self.iq_head(iq_features)
        fft_logits = self.fft_head(fft_features)
        wavelet_logits = self.wavelet_head(wavelet_features)
        fused_features = torch.cat([iq_features, fft_features, wavelet_features], dim=1)
        fusion_logits = self.fusion_head(fused_features)
        gate_weights = torch.softmax(self.gate(fused_features), dim=1)
        final_logits = (
            gate_weights[:, 0:1] * iq_logits
            + gate_weights[:, 1:2] * fft_logits
            + gate_weights[:, 2:3] * wavelet_logits
            + gate_weights[:, 3:4] * fusion_logits
        )
        return {
            "final_logits": final_logits,
            "aux_logits": [iq_logits, fft_logits, wavelet_logits, fusion_logits],
            "gate_weights": gate_weights,
        }


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def make_pair_loader(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(PairDataset(x1, x2, y), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def make_triple_loader(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(TripleDataset(x1, x2, x3, y), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    report = classification_report(y_true, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0, output_dict=True)
    return {
        "test_acc": float((y_pred == y_true).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def train_single_model(model: nn.Module, train_loader, val_loader, test_loader, y_test: np.ndarray, class_names: list[str], epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = model.to(device)
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
    result = {"best_epoch": int(best_epoch), "val_acc": float(best_val_acc)}
    result.update(evaluate_predictions(y_test, y_pred, class_names))
    return result


def compute_gated_loss(outputs, yb):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    final_loss = criterion(outputs["final_logits"], yb)
    aux_loss = sum(criterion(logits, yb) for logits in outputs["aux_logits"])
    return final_loss + AUX_LOSS_WEIGHT * aux_loss


def train_pair_gated(model: nn.Module, train_loader, val_loader, test_loader, y_test: np.ndarray, class_names: list[str], epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_gate = None
    for epoch in range(1, epochs + 1):
        model.train(True)
        for x1, x2, yb in train_loader:
            x1 = x1.to(device, non_blocking=use_cuda)
            x2 = x2.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(x1, x2)
                loss = compute_gated_loss(outputs, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        correct = 0
        total = 0
        gate_sum = None
        with torch.no_grad():
            for x1, x2, yb in val_loader:
                x1 = x1.to(device, non_blocking=use_cuda)
                x2 = x2.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(x1, x2)
                correct += (outputs["final_logits"].argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
                current = outputs["gate_weights"].detach().cpu().double().sum(dim=0)
                gate_sum = current if gate_sum is None else gate_sum + current
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
    gate_sum = None
    total = 0
    with torch.no_grad():
        for x1, x2, _ in test_loader:
            x1 = x1.to(device, non_blocking=use_cuda)
            x2 = x2.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(x1, x2)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            current = outputs["gate_weights"].detach().cpu().double().sum(dim=0)
            gate_sum = current if gate_sum is None else gate_sum + current
            total += x1.size(0)
    y_pred = np.concatenate(preds)
    result = {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "best_val_gate_weights": best_gate,
        "test_gate_weights": (gate_sum / max(total, 1)).tolist(),
    }
    result.update(evaluate_predictions(y_test, y_pred, class_names))
    return result


def train_triple_gated(model: nn.Module, train_loader, val_loader, test_loader, y_test: np.ndarray, class_names: list[str], epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_gate = None
    for epoch in range(1, epochs + 1):
        model.train(True)
        for x1, x2, x3, yb in train_loader:
            x1 = x1.to(device, non_blocking=use_cuda)
            x2 = x2.to(device, non_blocking=use_cuda)
            x3 = x3.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(x1, x2, x3)
                loss = compute_gated_loss(outputs, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        correct = 0
        total = 0
        gate_sum = None
        with torch.no_grad():
            for x1, x2, x3, yb in val_loader:
                x1 = x1.to(device, non_blocking=use_cuda)
                x2 = x2.to(device, non_blocking=use_cuda)
                x3 = x3.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(x1, x2, x3)
                correct += (outputs["final_logits"].argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
                current = outputs["gate_weights"].detach().cpu().double().sum(dim=0)
                gate_sum = current if gate_sum is None else gate_sum + current
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
    gate_sum = None
    total = 0
    with torch.no_grad():
        for x1, x2, x3, _ in test_loader:
            x1 = x1.to(device, non_blocking=use_cuda)
            x2 = x2.to(device, non_blocking=use_cuda)
            x3 = x3.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(x1, x2, x3)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            current = outputs["gate_weights"].detach().cpu().double().sum(dim=0)
            gate_sum = current if gate_sum is None else gate_sum + current
            total += x1.size(0)
    y_pred = np.concatenate(preds)
    result = {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "best_val_gate_weights": best_gate,
        "test_gate_weights": (gate_sum / max(total, 1)).tolist(),
    }
    result.update(evaluate_predictions(y_test, y_pred, class_names))
    return result


def load_waveform_dataset():
    with h5py.File(WAVEFORM_PATH, "r") as h5_file:
        signals = h5_file["signals"][:].astype(np.float32)
        labels = h5_file["labels"][:].astype(np.int64)
        class_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in h5_file.attrs["class_names"]]
    signals = normalize_channel_rms(signals)
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels, test_size=0.2, random_state=SEED, stratify=stratify_or_none(labels)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=stratify_or_none(y_temp)
    )
    return {
        "name": "waveform_family",
        "class_names": class_names,
        "iq_train": X_train,
        "iq_val": X_val,
        "iq_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "batch_size": 128,
        "epochs": 20,
    }


def load_modulation_dataset():
    with h5py.File(MODULATION_PATH, "r") as h5_file:
        signals = h5_file["signals"][:].astype(np.float32)
        labels = h5_file["labels"][:].astype(np.int64)
        class_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in h5_file.attrs["class_names"]]
    signals = normalize_channel_rms(signals)
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals, labels, test_size=0.2, random_state=SEED, stratify=stratify_or_none(labels)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=stratify_or_none(y_temp)
    )
    return {
        "name": "modulation_family",
        "class_names": class_names,
        "iq_train": X_train,
        "iq_val": X_val,
        "iq_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "batch_size": 128,
        "epochs": 20,
    }


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


def load_real_split(path: Path, max_windows_per_file: int):
    with h5py.File(path, "r") as h5_file:
        source_files = h5_file["metadata"]["source_file"][:]
        indices = np.sort(select_balanced_indices(source_files, max_windows_per_file=max_windows_per_file))
        signals = h5_file["signals"][indices].astype(np.float32)
        labels = h5_file["labels"][indices].astype(np.int64)
        class_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in h5_file.attrs["class_names"]]
    return signals, labels, class_names


def load_real_dataset(max_windows_per_file: int):
    iq_train, y_train, class_names = load_real_split(REAL_TRAIN_PATH, max_windows_per_file=max_windows_per_file)
    iq_val, y_val, _ = load_real_split(REAL_VAL_PATH, max_windows_per_file=max_windows_per_file)
    iq_test, y_test, _ = load_real_split(REAL_TEST_PATH, max_windows_per_file=max_windows_per_file)
    return {
        "name": "subghz_real",
        "class_names": class_names,
        "iq_train": iq_train,
        "iq_val": iq_val,
        "iq_test": iq_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "batch_size": 128,
        "epochs": 12,
        "max_windows_per_file": int(max_windows_per_file),
    }


def load_orbit_day(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def orbit_common_nodes(days: list[dict]) -> list[str]:
    common = None
    for day in days:
        nodes = set(day["node_list"])
        common = nodes if common is None else common & nodes
    return sorted(common)


def select_evenly_spaced(samples: np.ndarray, max_packets: int) -> np.ndarray:
    if len(samples) <= max_packets:
        return samples
    positions = np.linspace(0, len(samples) - 1, num=max_packets, dtype=np.int64)
    return samples[positions]


def build_orbit_split(day: dict, node_names: list[str], class_to_index: dict[str, int], max_packets_per_node: int):
    signals = []
    labels = []
    for node_name in node_names:
        idx = day["node_list"].index(node_name)
        packets = np.asarray(day["data"][idx], dtype=np.float32)
        packets = select_evenly_spaced(packets, max_packets=max_packets_per_node)
        packets = np.transpose(packets, (0, 2, 1))
        packets = normalize_complex_rms(packets)
        signals.append(packets)
        labels.append(np.full(packets.shape[0], class_to_index[node_name], dtype=np.int64))
    return np.concatenate(signals, axis=0), np.concatenate(labels, axis=0)


def load_orbit_dataset(max_packets_per_node: int):
    day_paths = [ORBIT_ROOT / name for name in DEFAULT_ORBIT_DAY_FILES]
    days = [load_orbit_day(path) for path in day_paths]
    common_nodes = orbit_common_nodes(days)
    class_names = common_nodes
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    iq_train_parts = []
    y_train_parts = []
    for day in days[:3]:
        x_day, y_day = build_orbit_split(day, common_nodes, class_to_index, max_packets_per_node)
        iq_train_parts.append(x_day)
        y_train_parts.append(y_day)
    iq_train = np.concatenate(iq_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    iq_val, y_val = build_orbit_split(days[3], common_nodes, class_to_index, max_packets_per_node)
    iq_test, y_test = build_orbit_split(days[4], common_nodes, class_to_index, max_packets_per_node)
    return {
        "name": "orbit_rf",
        "class_names": class_names,
        "iq_train": iq_train,
        "iq_val": iq_val,
        "iq_test": iq_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "batch_size": 256,
        "epochs": 12,
        "max_packets_per_node_per_day": int(max_packets_per_node),
        "num_common_nodes": len(common_nodes),
    }


def run_dataset_benchmark(dataset: dict):
    class_names = dataset["class_names"]
    iq_train = dataset["iq_train"]
    iq_val = dataset["iq_val"]
    iq_test = dataset["iq_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    batch_size = dataset["batch_size"]
    epochs = dataset["epochs"]

    print(f"\n=== Dataset: {dataset['name']} ===")
    print("iq train/val/test:", iq_train.shape, iq_val.shape, iq_test.shape)
    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)
    print("computing wavelet features...")
    wavelet_train = to_wavelet(iq_train, batch_size=batch_size)
    wavelet_val = to_wavelet(iq_val, batch_size=batch_size)
    wavelet_test = to_wavelet(iq_test, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    results = {
        "train_examples": int(iq_train.shape[0]),
        "val_examples": int(iq_val.shape[0]),
        "test_examples": int(iq_test.shape[0]),
        "signal_length": int(iq_train.shape[-1]),
        "class_names": class_names,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "wavelet_shape": list(wavelet_train.shape[1:]),
    }

    print("\nIQ CNN")
    results["iq_cnn"] = train_single_model(
        TimeOrFFTCNN(num_classes=len(class_names)),
        make_loader(iq_train, y_train, True, use_cuda, batch_size),
        make_loader(iq_val, y_val, False, use_cuda, batch_size),
        make_loader(iq_test, y_test, False, use_cuda, batch_size),
        y_test,
        class_names,
        epochs,
    )

    print("\nFFT CNN")
    results["fft_cnn"] = train_single_model(
        TimeOrFFTCNN(num_classes=len(class_names)),
        make_loader(fft_train, y_train, True, use_cuda, batch_size),
        make_loader(fft_val, y_val, False, use_cuda, batch_size),
        make_loader(fft_test, y_test, False, use_cuda, batch_size),
        y_test,
        class_names,
        epochs,
    )

    print("\nWavelet CNN")
    results["wavelet_cnn"] = train_single_model(
        WaveletCNN(num_classes=len(class_names)),
        make_loader(wavelet_train, y_train, True, use_cuda, batch_size),
        make_loader(wavelet_val, y_val, False, use_cuda, batch_size),
        make_loader(wavelet_test, y_test, False, use_cuda, batch_size),
        y_test,
        class_names,
        epochs,
    )

    print("\nGated IQ+FFT CNN")
    results["gated_iq_fft"] = train_pair_gated(
        GatedIQFFTCNN(num_classes=len(class_names)),
        make_pair_loader(iq_train, fft_train, y_train, True, use_cuda, batch_size),
        make_pair_loader(iq_val, fft_val, y_val, False, use_cuda, batch_size),
        make_pair_loader(iq_test, fft_test, y_test, False, use_cuda, batch_size),
        y_test,
        class_names,
        epochs,
    )

    print("\nGated IQ+FFT+Wavelet CNN")
    results["gated_iq_fft_wavelet"] = train_triple_gated(
        GatedIQFFTWaveletCNN(num_classes=len(class_names)),
        make_triple_loader(iq_train, fft_train, wavelet_train, y_train, True, use_cuda, batch_size),
        make_triple_loader(iq_val, fft_val, wavelet_val, y_val, False, use_cuda, batch_size),
        make_triple_loader(iq_test, fft_test, wavelet_test, y_test, False, use_cuda, batch_size),
        y_test,
        class_names,
        epochs,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark IQ, FFT, wavelet, and gated multimodal models across all repo datasets.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--real-max-windows-per-file", type=int, default=128)
    parser.add_argument("--orbit-max-packets-per-node", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()

    datasets = [
        load_modulation_dataset(),
        load_waveform_dataset(),
        load_real_dataset(max_windows_per_file=args.real_max_windows_per_file),
        load_orbit_dataset(max_packets_per_node=args.orbit_max_packets_per_node),
    ]

    results = {
        "experiment": "wavelet_multidataset_benchmark",
        "real_max_windows_per_file": int(args.real_max_windows_per_file),
        "orbit_max_packets_per_node": int(args.orbit_max_packets_per_node),
        "datasets": {},
    }
    for dataset in datasets:
        dataset_start = time.perf_counter()
        dataset_results = run_dataset_benchmark(dataset)
        dataset_results["runtime_seconds"] = time.perf_counter() - dataset_start
        results["datasets"][dataset["name"]] = dataset_results

    results["runtime_seconds"] = time.perf_counter() - start
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.results_path}")


if __name__ == "__main__":
    main()
