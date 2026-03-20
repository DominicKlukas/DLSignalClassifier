from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SEED = 0
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AUX_LOSS_WEIGHT = 0.25

DEFAULT_DATASET_ROOT = ROOT_DIR / "orbit_rf_identification_dataset_updated"
DEFAULT_RESULTS_PATH = ROOT_DIR / "experiments" / "experiment5_orbit_rf_results.json"
DEFAULT_DAY_FILES = [
    "grid_2019_12_25.pkl",
    "grid_2020_02_03.pkl",
    "grid_2020_02_04.pkl",
    "grid_2020_02_05.pkl",
    "grid_2020_02_06.pkl",
]


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


def load_day(path: Path):
    with open(path, "rb") as handle:
        day = pickle.load(handle)
    return day


def intersect_nodes(days: list[dict]) -> list[str]:
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


def build_split(day: dict, node_names: list[str], class_to_index: dict[str, int], max_packets_per_node: int):
    signals = []
    labels = []
    counts = {}
    for node_name in node_names:
        node_idx = day["node_list"].index(node_name)
        node_packets = np.asarray(day["data"][node_idx], dtype=np.float32)
        node_packets = select_evenly_spaced(node_packets, max_packets=max_packets_per_node)
        node_packets = np.transpose(node_packets, (0, 2, 1))
        node_packets = normalize_complex_rms(node_packets)
        signals.append(node_packets)
        labels.append(np.full(node_packets.shape[0], class_to_index[node_name], dtype=np.int64))
        counts[node_name] = int(node_packets.shape[0])
    return np.concatenate(signals, axis=0), np.concatenate(labels, axis=0), counts


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
    report = classification_report(
        y_test, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0, output_dict=True
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
    report = classification_report(
        y_test, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0, output_dict=True
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


def main():
    parser = argparse.ArgumentParser(description="Run an Experiment 5-style benchmark on the Orbit RF identification dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--day-files", nargs=5, default=DEFAULT_DAY_FILES)
    parser.add_argument("--max-packets-per-node-per-day", type=int, default=256)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset_root = args.dataset_root.expanduser().resolve()
    day_paths = [dataset_root / name for name in args.day_files]
    for path in day_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing Orbit dataset day file: {path}")

    days = [load_day(path) for path in day_paths]
    common_nodes = intersect_nodes(days)
    if not common_nodes:
        raise ValueError("No common transmitter IDs were found across the requested day files.")

    class_names = common_nodes
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    iq_train_parts = []
    y_train_parts = []
    train_counts = {}
    for day_idx, day in enumerate(days[:3]):
        X_day, y_day, counts = build_split(day, common_nodes, class_to_index, args.max_packets_per_node_per_day)
        iq_train_parts.append(X_day)
        y_train_parts.append(y_day)
        train_counts[day_paths[day_idx].name] = counts
    iq_train = np.concatenate(iq_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    iq_val, y_val, val_counts = build_split(days[3], common_nodes, class_to_index, args.max_packets_per_node_per_day)
    iq_test, y_test, test_counts = build_split(days[4], common_nodes, class_to_index, args.max_packets_per_node_per_day)

    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)

    print(f"Orbit day split using common transmitters: {len(common_nodes)} classes")
    print("train/val/test:", iq_train.shape, iq_val.shape, iq_test.shape)

    start = time.perf_counter()
    results = {
        "dataset_root": str(dataset_root),
        "day_files": args.day_files,
        "split_policy": "days 1-3 train, day 4 val, day 5 test, restricted to transmitters present in all five days",
        "class_names": class_names,
        "num_classes": len(class_names),
        "max_packets_per_node_per_day": int(args.max_packets_per_node_per_day),
        "train_packets": int(iq_train.shape[0]),
        "val_packets": int(iq_val.shape[0]),
        "test_packets": int(iq_test.shape[0]),
        "signal_length": int(iq_train.shape[-1]),
        "normalization": "per-packet complex RMS, shared across I/Q after packet selection",
        "train_day_packet_counts": train_counts,
        "val_day_packet_counts": val_counts,
        "test_day_packet_counts": test_counts,
        "time_cnn": train_single_mode(iq_train, iq_val, iq_test, y_train, y_val, y_test, class_names),
        "fft_cnn": train_single_mode(fft_train, fft_val, fft_test, y_train, y_val, y_test, class_names),
        "gated_multimodal": train_gated(
            iq_train, iq_val, iq_test, fft_train, fft_val, fft_test, y_train, y_val, y_test, class_names
        ),
        "runtime_seconds": time.perf_counter() - start,
    }
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.results_path}")


if __name__ == "__main__":
    main()
