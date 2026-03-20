import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


def stratify_or_none(y: np.ndarray):
    _, counts = np.unique(y, return_counts=True)
    return y if np.all(counts >= 2) else None


def load_dataset(dataset_path: Path):
    with h5py.File(dataset_path, "r") as h5_file:
        signals = h5_file["signals"][:].astype(np.float32)
        labels = h5_file["labels"][:].astype(np.int64)
        class_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h5_file.attrs["class_names"]
        ]
        metadata = {}
        if "metadata" in h5_file:
            for key in h5_file["metadata"].keys():
                metadata[key] = h5_file["metadata"][key][:]

    rms = np.sqrt(np.mean(signals**2, axis=(1, 2), keepdims=True) + 1e-8)
    signals = signals / rms
    return signals, labels, class_names, metadata


def make_splits(signals: np.ndarray, labels: np.ndarray, random_state: int = 0):
    X_train, X_temp, y_train, y_temp = train_test_split(
        signals,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify_or_none(labels),
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=stratify_or_none(y_temp),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


class IQAugment:
    def __init__(
        self,
        max_phase: float = math.pi,
        max_freq_ramp: float = 0.10,
        max_time_shift: int = 64,
        gain_jitter_db: float = 1.5,
        dc_offset: float = 0.05,
    ):
        self.max_phase = max_phase
        self.max_freq_ramp = max_freq_ramp
        self.max_time_shift = max_time_shift
        self.gain_jitter_db = gain_jitter_db
        self.dc_offset = dc_offset

    def __call__(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        x = x.clone()
        length = x.shape[-1]
        n = torch.arange(length, dtype=x.dtype)

        phase0 = float(rng.uniform(-self.max_phase, self.max_phase))
        omega = float(rng.uniform(-self.max_freq_ramp, self.max_freq_ramp))
        phase = phase0 + omega * n
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        i = x[0] * cos_phase - x[1] * sin_phase
        q = x[0] * sin_phase + x[1] * cos_phase
        x = torch.stack((i, q), dim=0)

        shift = int(rng.integers(-self.max_time_shift, self.max_time_shift + 1))
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)

        gain = 10.0 ** float(rng.uniform(-self.gain_jitter_db, self.gain_jitter_db) / 20.0)
        x = x * gain
        x[0] = x[0] + float(rng.uniform(-self.dc_offset, self.dc_offset))
        x[1] = x[1] + float(rng.uniform(-self.dc_offset, self.dc_offset))
        return x


class IQDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: IQAugment | None = None, seed: int = 0):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment is not None:
            x = self.augment(x, self.rng)
        return x, self.y[idx]


class VanillaSignalCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.real_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.imag_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.padding = padding
        nn.init.kaiming_uniform_(self.real_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.imag_weight, a=math.sqrt(5))

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        rr = F.conv1d(real, self.real_weight, padding=self.padding)
        ii = F.conv1d(imag, self.imag_weight, padding=self.padding)
        ri = F.conv1d(real, self.imag_weight, padding=self.padding)
        ir = F.conv1d(imag, self.real_weight, padding=self.padding)
        return rr - ii, ri + ir


class ComplexRadialGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-6)
        gate = torch.sigmoid(self.norm(magnitude))
        return real * gate, imag * gate


class ComplexBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = ComplexConv1d(in_channels, out_channels, kernel_size, padding)
        self.act1 = ComplexRadialGate(out_channels)
        self.conv2 = ComplexConv1d(out_channels, out_channels, kernel_size, padding)
        self.act2 = ComplexRadialGate(out_channels)
        self.pool = nn.AvgPool1d(2) if pool else None

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        real, imag = self.conv1(real, imag)
        real, imag = self.act1(real, imag)
        real, imag = self.conv2(real, imag)
        real, imag = self.act2(real, imag)
        if self.pool is not None:
            real = self.pool(real)
            imag = self.pool(imag)
        return real, imag


class InvariantTransition(nn.Module):
    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-6)
        rel_real = real[:, :, 1:] * real[:, :, :-1] + imag[:, :, 1:] * imag[:, :, :-1]
        rel_imag = imag[:, :, 1:] * real[:, :, :-1] - real[:, :, 1:] * imag[:, :, :-1]
        rel_real = F.pad(rel_real, (1, 0))
        rel_imag = F.pad(rel_imag, (1, 0))
        rel_scale = torch.sqrt(rel_real.square() + rel_imag.square() + 1e-6)
        return torch.cat((magnitude, rel_real / rel_scale, rel_imag / rel_scale), dim=1)


class PhaseEquivariantSignalCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.ModuleList(
            [
                ComplexBlock(1, 24, kernel_size=7, pool=True),
                ComplexBlock(24, 48, kernel_size=5, pool=True),
                ComplexBlock(48, 64, kernel_size=5, pool=False),
            ]
        )
        self.transition = InvariantTransition()
        self.real_features = nn.Sequential(
            nn.Conv1d(64 * 3, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
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
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        for block in self.stem:
            real, imag = block(real, imag)
        x = self.transition(real, imag)
        x = self.real_features(x)
        return self.classifier(x)


@dataclass
class ExperimentConfig:
    name: str
    model_kind: str
    use_augmentation: bool
    batch_size: int = 128
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05


def build_model(model_kind: str, num_classes: int) -> nn.Module:
    if model_kind == "vanilla":
        return VanillaSignalCNN(num_classes=num_classes)
    if model_kind == "phase_equivariant":
        return PhaseEquivariantSignalCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model kind: {model_kind}")


def load_model_checkpoint(checkpoint_path: Path, model_kind: str, num_classes: int, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_kind, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_loaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    use_augmentation: bool,
    use_cuda: bool,
):
    # Keep loaders notebook-friendly and compatible with restricted environments.
    num_workers = 0
    pin_memory = use_cuda
    persistent_workers = False
    augment = IQAugment() if use_augmentation else None

    train_loader = DataLoader(
        IQDataset(X_train, y_train, augment=augment, seed=12345),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        IQDataset(X_val, y_val, augment=None, seed=0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        IQDataset(X_test, y_test, augment=None, seed=0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, scaler, device, use_cuda, training: bool):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    start = time.perf_counter()

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

    elapsed = time.perf_counter() - start
    return (
        total_loss / max(total_examples, 1),
        total_correct / max(total_examples, 1),
        elapsed,
    )


def train_experiment(
    dataset_path: Path,
    config: ExperimentConfig,
    checkpoint_path: Path | None = None,
    random_state: int = 0,
):
    signals, labels, class_names, metadata = load_dataset(dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(signals, labels, random_state=random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader, test_loader = make_loaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        batch_size=config.batch_size,
        use_augmentation=config.use_augmentation,
        use_cuda=use_cuda,
    )

    model = build_model(config.model_kind, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = None
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, use_cuda, True)
        val_loss, val_acc, _ = run_epoch(model, val_loader, criterion, optimizer, scaler, device, use_cuda, False)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "class_names": class_names,
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "config": config.__dict__,
                    },
                    checkpoint_path,
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(X_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    test_acc = float((y_pred == y_true).mean())
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        zero_division=0,
        output_dict=True,
    )

    return {
        "config": config,
        "class_names": class_names,
        "metadata": metadata,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_acc": test_acc,
        "report": report,
    }


def evaluate_arrays(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    batch_size: int = 128,
):
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        IQDataset(X, y, augment=None, seed=0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        persistent_workers=False,
    )

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(X_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }
