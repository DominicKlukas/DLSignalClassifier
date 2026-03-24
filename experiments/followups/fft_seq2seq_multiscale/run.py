from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.signal as sc
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.shared.repro import configure_device, save_json, set_global_seed  # noqa: E402


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
DATASET_PATH = ARTIFACTS_DIR / "fft_magnitude_approximation_dataset.h5"
RESULTS_PATH = ARTIFACTS_DIR / "results.json"

SEED = 0
SIGNAL_LENGTH = 1024
TRAIN_SIZE = 24000
VAL_SIZE = 4000
TEST_SIZE = 4000
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 60
PROGRESS_EVERY = 25


@dataclass
class DatasetConfig:
    signal_length: int = SIGNAL_LENGTH
    train_size: int = TRAIN_SIZE
    val_size: int = VAL_SIZE
    test_size: int = TEST_SIZE
    seed: int = SEED


def to_unitary_fft_magnitude(signals: np.ndarray) -> np.ndarray:
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    fft_signals = np.fft.fftshift(np.fft.fft(complex_signals, axis=-1), axes=-1) / np.sqrt(signals.shape[-1])
    return np.abs(fft_signals).astype(np.float32)[:, None, :]


def normalize_rms(signal: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(signal) ** 2) + 1e-8)
    return (signal / rms).astype(np.complex64)


class DiverseSignalFactory:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def _time_axis(self, length: int) -> np.ndarray:
        return np.linspace(0.0, 1.0, num=length, endpoint=False, dtype=np.float64)

    def _complex_gaussian(self, length: int) -> np.ndarray:
        return (
            self.rng.normal(0.0, 1.0, size=length)
            + 1j * self.rng.normal(0.0, 1.0, size=length)
        ).astype(np.complex64)

    def _multi_tone(self, length: int) -> np.ndarray:
        t = self._time_axis(length)
        signal = np.zeros(length, dtype=np.complex128)
        num_tones = int(self.rng.integers(1, 7))
        for _ in range(num_tones):
            freq = self.rng.uniform(-0.48, 0.48)
            phase = self.rng.uniform(-math.pi, math.pi)
            amplitude = self.rng.uniform(0.2, 1.0)
            signal += amplitude * np.exp(1j * (2 * math.pi * freq * np.arange(length) + phase))
        signal += 0.1 * self._complex_gaussian(length)
        return signal.astype(np.complex64)

    def _linear_chirp(self, length: int) -> np.ndarray:
        t = self._time_axis(length)
        start_freq = self.rng.uniform(-0.4, 0.2)
        stop_freq = np.clip(start_freq + self.rng.uniform(0.1, 0.45) * self.rng.choice([-1.0, 1.0]), -0.48, 0.48)
        instantaneous_phase = 2 * math.pi * (
            start_freq * np.arange(length)
            + 0.5 * (stop_freq - start_freq) * (np.arange(length) ** 2) / max(length, 1)
        )
        envelope = 0.5 + 0.5 * np.sin(2 * math.pi * self.rng.uniform(0.5, 3.0) * t + self.rng.uniform(-math.pi, math.pi))
        return (envelope * np.exp(1j * instantaneous_phase)).astype(np.complex64)

    def _am_fm(self, length: int) -> np.ndarray:
        t = self._time_axis(length)
        carrier = self.rng.uniform(-0.35, 0.35)
        am_freq = self.rng.uniform(0.5, 8.0)
        fm_freq = self.rng.uniform(0.5, 10.0)
        am_depth = self.rng.uniform(0.2, 0.9)
        fm_dev = self.rng.uniform(0.02, 0.18)
        envelope = 1.0 + am_depth * np.sin(2 * math.pi * am_freq * t + self.rng.uniform(-math.pi, math.pi))
        phase = 2 * math.pi * carrier * np.arange(length) + fm_dev * np.sin(2 * math.pi * fm_freq * t)
        return (envelope * np.exp(1j * phase)).astype(np.complex64)

    def _burst_component(self, length: int) -> np.ndarray:
        t = self._time_axis(length)
        base = np.exp(1j * (2 * math.pi * self.rng.uniform(-0.45, 0.45) * np.arange(length) + self.rng.uniform(-math.pi, math.pi)))
        start = int(self.rng.integers(0, max(length // 2, 1)))
        width = int(self.rng.integers(max(length // 16, 8), max(length // 2, 16)))
        stop = min(length, start + width)
        window = np.zeros(length, dtype=np.float32)
        if stop > start:
            active = sc.windows.tukey(stop - start, alpha=float(self.rng.uniform(0.2, 0.9))).astype(np.float32)
            window[start:stop] = active
        amplitude = 0.5 + 0.5 * np.sin(2 * math.pi * self.rng.uniform(0.5, 5.0) * t)
        return (base * window * amplitude).astype(np.complex64)

    def _filtered_noise(self, length: int) -> np.ndarray:
        noise = self._complex_gaussian(length).astype(np.complex128)
        low = self.rng.uniform(0.02, 0.18)
        high = self.rng.uniform(low + 0.04, 0.48)
        taps = sc.firwin(63, [low, high], pass_zero=False)
        real = sc.lfilter(taps, [1.0], noise.real)
        imag = sc.lfilter(taps, [1.0], noise.imag)
        return (real + 1j * imag).astype(np.complex64)

    def _piecewise_phase(self, length: int) -> np.ndarray:
        segments = int(self.rng.integers(2, 6))
        boundaries = np.linspace(0, length, num=segments + 1, dtype=np.int64)
        signal = np.zeros(length, dtype=np.complex128)
        phase = self.rng.uniform(-math.pi, math.pi)
        for idx in range(segments):
            start, stop = int(boundaries[idx]), int(boundaries[idx + 1])
            freq = self.rng.uniform(-0.45, 0.45)
            phase_step = self.rng.uniform(-0.6, 0.6)
            n = np.arange(stop - start)
            signal[start:stop] = np.exp(1j * (phase + 2 * math.pi * freq * n))
            phase = float(np.angle(signal[stop - 1])) + phase_step
        return signal.astype(np.complex64)

    def _draw_component(self, length: int) -> np.ndarray:
        builders = (
            self._complex_gaussian,
            self._multi_tone,
            self._linear_chirp,
            self._am_fm,
            self._burst_component,
            self._filtered_noise,
            self._piecewise_phase,
        )
        builder = builders[int(self.rng.integers(0, len(builders)))]
        return builder(length)

    def generate_signal(self, length: int) -> tuple[np.ndarray, dict]:
        num_components = int(self.rng.integers(1, 4))
        signal = np.zeros(length, dtype=np.complex128)
        component_names = []
        for _ in range(num_components):
            component = self._draw_component(length)
            component_names.append(component.dtype.name)
            signal += self.rng.uniform(0.4, 1.0) * component

        if self.rng.random() < 0.35:
            impulse_count = int(self.rng.integers(1, 6))
            positions = self.rng.integers(0, length, size=impulse_count)
            signal[positions] += self.rng.normal(0.0, 2.0, size=impulse_count) + 1j * self.rng.normal(0.0, 2.0, size=impulse_count)

        if self.rng.random() < 0.5:
            signal *= np.exp(1j * self.rng.uniform(-math.pi, math.pi))

        noise_scale = self.rng.uniform(0.01, 0.2)
        signal += noise_scale * self._complex_gaussian(length)
        signal = normalize_rms(signal.astype(np.complex64))
        metadata = {
            "num_components": num_components,
            "noise_scale": float(noise_scale),
        }
        return signal, metadata


def write_split(group: h5py.Group, split_name: str, count: int, factory: DiverseSignalFactory, signal_length: int) -> None:
    signals = np.zeros((count, 2, signal_length), dtype=np.float32)
    metadata_noise = np.zeros(count, dtype=np.float32)
    metadata_components = np.zeros(count, dtype=np.int64)
    for idx in range(count):
        complex_signal, metadata = factory.generate_signal(signal_length)
        signals[idx, 0] = complex_signal.real.astype(np.float32)
        signals[idx, 1] = complex_signal.imag.astype(np.float32)
        metadata_noise[idx] = metadata["noise_scale"]
        metadata_components[idx] = metadata["num_components"]
    fft_targets = to_unitary_fft_magnitude(signals)
    split_group = group.create_group(split_name)
    split_group.create_dataset("signals", data=signals, compression="gzip", chunks=True)
    split_group.create_dataset("fft_targets", data=fft_targets, compression="gzip", chunks=True)
    metadata_group = split_group.create_group("metadata")
    metadata_group.create_dataset("noise_scale", data=metadata_noise, compression="gzip", chunks=True)
    metadata_group.create_dataset("num_components", data=metadata_components, compression="gzip", chunks=True)


def ensure_dataset(config: DatasetConfig, rebuild: bool = False) -> Path:
    if DATASET_PATH.exists() and not rebuild:
        return DATASET_PATH

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        DATASET_PATH.unlink()

    with h5py.File(DATASET_PATH, "w") as h5_file:
        h5_file.attrs["signal_layout"] = "NCH"
        h5_file.attrs["signal_channels"] = np.asarray(["I", "Q"], dtype=h5py.string_dtype(encoding="utf-8"))
        h5_file.attrs["target_description"] = "Magnitude of unitary fftshift(fft(I + jQ)) represented as one channel."
        h5_file.attrs["config_json"] = json.dumps(asdict(config))

    with h5py.File(DATASET_PATH, "a") as h5_file:
        split_specs = [
            ("train", config.train_size, config.seed),
            ("val", config.val_size, config.seed + 1),
            ("test", config.test_size, config.seed + 2),
        ]
        for split_name, count, seed in split_specs:
            factory = DiverseSignalFactory(seed=seed)
            write_split(h5_file, split_name, count, factory, config.signal_length)

    return DATASET_PATH


class FFTDataset(Dataset):
    def __init__(self, path: Path, split: str):
        self.path = path
        self.split = split
        self._h5_file = None
        self._signals = None
        self._targets = None
        with h5py.File(path, "r") as h5_file:
            self.length = int(h5_file[split]["signals"].shape[0])

    def _ensure_open(self) -> None:
        if self._h5_file is not None:
            return
        self._h5_file = h5py.File(self.path, "r")
        self._signals = self._h5_file[self.split]["signals"]
        self._targets = self._h5_file[self.split]["fft_targets"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        self._ensure_open()
        signal = self._signals[idx].astype(np.float32)
        fft_target = self._targets[idx].astype(np.float32)
        return torch.from_numpy(signal), torch.from_numpy(fft_target)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_signals"] = None
        state["_targets"] = None
        return state

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_sizes: tuple[int, ...] = (3, 9, 17, 33)):
        super().__init__()
        if channels % len(kernel_sizes) != 0:
            raise ValueError("channels must be divisible by number of branches")
        branch_channels = channels // len(kernel_sizes)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, branch_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.activation(x + self.mix(merged))


class MultiScaleFFTApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(2, 128, kernel_size=15),
            ConvNormAct(128, 128, kernel_size=7),
        )
        self.encoder1 = nn.Sequential(MultiScaleResidualBlock(128), MultiScaleResidualBlock(128))
        self.down1 = ConvNormAct(128, 256, kernel_size=5, stride=2)
        self.encoder2 = nn.Sequential(MultiScaleResidualBlock(256), MultiScaleResidualBlock(256))
        self.down2 = ConvNormAct(256, 512, kernel_size=5, stride=2)
        self.encoder3 = nn.Sequential(
            MultiScaleResidualBlock(512),
            MultiScaleResidualBlock(512),
            MultiScaleResidualBlock(512),
        )
        self.bottleneck = nn.Sequential(MultiScaleResidualBlock(512), MultiScaleResidualBlock(512))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvNormAct(512, 256, kernel_size=5),
        )
        self.decode2 = nn.Sequential(
            ConvNormAct(512, 256, kernel_size=3),
            MultiScaleResidualBlock(256),
            MultiScaleResidualBlock(256),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvNormAct(256, 128, kernel_size=5),
        )
        self.decode1 = nn.Sequential(
            ConvNormAct(256, 128, kernel_size=3),
            MultiScaleResidualBlock(128),
            MultiScaleResidualBlock(128),
        )
        self.refine = nn.Sequential(
            ConvNormAct(128, 128, kernel_size=7),
            MultiScaleResidualBlock(128),
        )
        self.output_head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x3 = self.bottleneck(x3)
        y2 = self.up2(x3)
        y2 = self.decode2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        y1 = self.decode1(torch.cat([y1, x1], dim=1))
        y = self.refine(y1)
        return self.output_head(y)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_loader(path: Path, split: str, batch_size: int, shuffle: bool, use_cuda: bool) -> DataLoader:
    return DataLoader(
        FFTDataset(path, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
    )


def sequence_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    target_norm = torch.sqrt(torch.mean(targets**2) + 1e-8)
    rel_rmse = torch.sqrt(torch.mean((predictions - targets) ** 2) + 1e-8) / target_norm
    pred_flat = predictions.reshape(predictions.shape[0], -1)
    target_flat = targets.reshape(targets.shape[0], -1)
    numerator = torch.sum(pred_flat * target_flat, dim=1)
    denominator = torch.sqrt(torch.sum(pred_flat**2, dim=1) * torch.sum(target_flat**2, dim=1) + 1e-8)
    cosine_similarity = float(torch.mean(numerator / denominator).item())
    return {
        "mse": float(mse),
        "mae": float(mae),
        "relative_rmse": float(rel_rmse.item()),
        "magnitude_cosine_similarity": cosine_similarity,
    }


def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((predictions - targets) ** 2)
    l1 = torch.mean(torch.abs(predictions - targets))
    log_mse = torch.mean((torch.log1p(predictions) - torch.log1p(targets)) ** 2)
    return mse + 0.25 * l1 + 0.2 * log_mse


def run_epoch(model, loader, optimizer, scaler, device, use_cuda: bool, training: bool, epoch_index: int, split_name: str):
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    prediction_batches = []
    target_batches = []
    num_batches = len(loader)

    for batch_index, (signals, targets) in enumerate(loader, start=1):
        signals = signals.to(device, non_blocking=use_cuda)
        targets = targets.to(device, non_blocking=use_cuda)
        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=use_cuda):
                predictions = model(signals)
                loss = loss_fn(predictions, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * signals.size(0)
        total_examples += signals.size(0)
        prediction_batches.append(predictions.detach().cpu())
        target_batches.append(targets.detach().cpu())
        if batch_index == 1 or batch_index % PROGRESS_EVERY == 0 or batch_index == num_batches:
            running_loss = total_loss / max(total_examples, 1)
            print(
                f"  epoch {epoch_index:03d} {split_name:<5} batch {batch_index:03d}/{num_batches:03d} "
                f"running_loss {running_loss:.5f}",
                flush=True,
            )

    predictions = torch.cat(prediction_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    metrics = sequence_metrics(predictions, targets)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def train_model(args, dataset_path: Path) -> dict:
    device, use_cuda = configure_device()
    train_loader = make_loader(dataset_path, "train", args.batch_size, True, use_cuda)
    val_loader = make_loader(dataset_path, "val", args.batch_size, False, use_cuda)
    test_loader = make_loader(dataset_path, "test", args.batch_size, False, use_cuda)

    model = MultiScaleFFTApproximator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-5,
    )
    scaler = GradScaler("cuda", enabled=use_cuda)

    history = []
    best_epoch = -1
    best_val_loss = float("inf")
    best_state = None
    checkpoint_path = CHECKPOINTS_DIR / "best_fft_approximator.pt"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, use_cuda, True, epoch, "train")
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, use_cuda, False, epoch, "val")
        scheduler.step(val_metrics["loss"])
        lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train": {key: float(value) for key, value in train_metrics.items()},
            "val": {key: float(value) for key, value in val_metrics.items()},
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train loss {train_metrics['loss']:.5f} mse {train_metrics['mse']:.5f} "
            f"| val loss {val_metrics['loss']:.5f} mse {val_metrics['mse']:.5f} rel_rmse {val_metrics['relative_rmse']:.5f} "
            f"| val cosine {val_metrics['magnitude_cosine_similarity']:.5f} | lr {lr:.6f}"
        )
        if val_metrics["loss"] < best_val_loss - 1e-8:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            torch.save(best_state, checkpoint_path)

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, optimizer, scaler, device, use_cuda, False, best_epoch, "test")
    return {
        "model": {
            "name": "multiscale_fft_magnitude_approximator",
            "num_trainable_parameters": int(count_trainable_parameters(model)),
            "signal_length": SIGNAL_LENGTH,
        },
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "history_tail": history[-10:],
        "checkpoint_path": str(checkpoint_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-scale convolutional seq2seq model to approximate FFT magnitude.")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--signal-length", type=int, default=SIGNAL_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    start = time.perf_counter()
    dataset_config = DatasetConfig(
        signal_length=args.signal_length,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=SEED,
    )
    dataset_path = ensure_dataset(dataset_config, rebuild=args.rebuild_dataset)
    result = train_model(args, dataset_path)
    result["dataset"] = {
        "path": str(dataset_path),
        **asdict(dataset_config),
        "target_description": "Magnitude of unitary fftshift(fft(I + jQ)) represented as one channel.",
    }
    result["runtime_seconds"] = time.perf_counter() - start
    save_json(RESULTS_PATH, result)
    print(f"Saved results to {RESULTS_PATH}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
