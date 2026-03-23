from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.exp03_frozen_expert_residual.run import (  # noqa: E402
    CAPTURED_TEST_PATH,
    CAPTURED_TRAIN_PATH,
    CAPTURED_VAL_PATH,
    DELTA_SCALE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    REAL_TEST_PATH,
    REAL_TRAIN_PATH,
    REAL_VAL_PATH,
    WEIGHT_DECAY,
    build_h5_dataset,
    compute_residual_loss,
    make_loader,
    make_pair_loader,
    train_residual_fusion,
    train_single_expert,
)
from experiments.shared.repro import classification_metrics, configure_device, save_json, set_global_seed  # noqa: E402
from experiments.shared.story_datasets import load_modulation_dataset, load_orbit_dataset, load_waveform_dataset, to_fft  # noqa: E402
from experiments.shared.story_models import ExpertCNN, FrozenExpertResidualFusion, GatedMultimodalIQFFTCNN  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results_clean.json"
DEFAULT_RESULTS_DIR = HERE / "results_by_dataset"

SEED = 0
DATASET_CHOICES = [
    "waveform_family",
    "modulation_family",
    "subghz_real_512",
    "captured_npy_real_128",
    "orbit_rf",
]
TRAIN_REGIME_CHOICES = ["clean", "augmented"]
BRANCH_DROPOUT = 0.10
AUX_LOSS_WEIGHT = 0.25

CORRUPTION_CONFIGS = {
    "awgn": [
        {"label": "clean", "snr_db": None},
        {"label": "20", "snr_db": 20.0},
        {"label": "10", "snr_db": 10.0},
        {"label": "0", "snr_db": 0.0},
        {"label": "-10", "snr_db": -10.0},
    ],
    "impulse_noise": [
        {"label": "clean", "prob": 0.0, "scale": 0.0},
        {"label": "0.002", "prob": 0.002, "scale": 3.0},
        {"label": "0.005", "prob": 0.005, "scale": 4.0},
        {"label": "0.010", "prob": 0.010, "scale": 5.0},
        {"label": "0.020", "prob": 0.020, "scale": 6.0},
    ],
    "narrowband_interferer": [
        {"label": "clean", "jsr_db": None},
        {"label": "-6", "jsr_db": -6.0},
        {"label": "0", "jsr_db": 0.0},
        {"label": "6", "jsr_db": 6.0},
        {"label": "12", "jsr_db": 12.0},
    ],
    "sample_rate_distortion": [
        {"label": "clean", "scale": 1.0},
        {"label": "0.98", "scale": 0.98},
        {"label": "0.95", "scale": 0.95},
        {"label": "1.05", "scale": 1.05},
        {"label": "1.10", "scale": 1.10},
    ],
    "clipping": [
        {"label": "clean", "threshold": None},
        {"label": "1.50", "threshold": 1.50},
        {"label": "1.25", "threshold": 1.25},
        {"label": "1.00", "threshold": 1.00},
        {"label": "0.75", "threshold": 0.75},
    ],
}


def make_rng(seed: int, offset: int) -> np.random.Generator:
    return np.random.default_rng(seed + offset)


def normalize_complex(signals: np.ndarray) -> np.ndarray:
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    rms = np.sqrt(np.mean(np.abs(complex_signals) ** 2, axis=1, keepdims=True) + 1e-8).astype(np.float32)
    complex_signals = complex_signals / rms
    return np.stack((complex_signals.real, complex_signals.imag), axis=1).astype(np.float32)


def batched_linear_resample(signals: np.ndarray, scales: np.ndarray) -> np.ndarray:
    batch, length = signals.shape
    center = 0.5 * (length - 1)
    coords = np.arange(length, dtype=np.float32)[None, :]
    positions = center + scales[:, None] * (coords - center)
    positions = np.clip(positions, 0.0, length - 1.001)

    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, length - 1)
    frac = (positions - left).astype(np.float32)
    batch_idx = np.arange(batch)[:, None]

    real = signals.real
    imag = signals.imag
    real_interp = (1.0 - frac) * real[batch_idx, left] + frac * real[batch_idx, right]
    imag_interp = (1.0 - frac) * imag[batch_idx, left] + frac * imag[batch_idx, right]
    return (real_interp + 1j * imag_interp).astype(np.complex64)


def apply_awgn(signals: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return signals.copy()
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    signal_power = np.mean(np.abs(complex_signals) ** 2, axis=1).astype(np.float32)
    snr_linear = np.power(10.0, snr_db / 10.0, dtype=np.float32)
    sigma = np.sqrt(signal_power / np.maximum(snr_linear, 1e-8) / 2.0, dtype=np.float32)
    noise = (
        rng.normal(0.0, 1.0, size=complex_signals.shape).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=complex_signals.shape).astype(np.float32)
    ) * sigma[:, None]
    noisy = (complex_signals + noise).astype(np.complex64)
    return normalize_complex(np.stack((noisy.real, noisy.imag), axis=1).astype(np.float32))


def apply_impulse_noise(signals: np.ndarray, prob: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    if prob <= 0.0:
        return signals.copy()
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    signal_rms = np.sqrt(np.mean(np.abs(complex_signals) ** 2, axis=1, keepdims=True) + 1e-8).astype(np.float32)
    mask = rng.random(size=complex_signals.shape) < prob
    impulses = (
        rng.normal(0.0, 1.0, size=complex_signals.shape).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=complex_signals.shape).astype(np.float32)
    ) * (scale * signal_rms)
    corrupted = complex_signals + impulses * mask
    return normalize_complex(np.stack((corrupted.real, corrupted.imag), axis=1).astype(np.float32))


def apply_narrowband_interferer(signals: np.ndarray, jsr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    if jsr_db is None:
        return signals.copy()
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    batch, length = complex_signals.shape
    signal_power = np.mean(np.abs(complex_signals) ** 2, axis=1, keepdims=True).astype(np.float32)
    jsr_linear = np.power(10.0, jsr_db / 10.0, dtype=np.float32)
    tone_amp = np.sqrt(signal_power * jsr_linear).astype(np.float32)
    freq = rng.uniform(-0.35 * math.pi, 0.35 * math.pi, size=(batch, 1)).astype(np.float32)
    phase = rng.uniform(-math.pi, math.pi, size=(batch, 1)).astype(np.float32)
    n = np.arange(length, dtype=np.float32)[None, :]
    tone = tone_amp * np.exp(1j * (freq * n + phase)).astype(np.complex64)
    corrupted = (complex_signals + tone).astype(np.complex64)
    return normalize_complex(np.stack((corrupted.real, corrupted.imag), axis=1).astype(np.float32))


def apply_sample_rate_distortion(signals: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-8:
        return signals.copy()
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    distorted = batched_linear_resample(complex_signals.astype(np.complex64), np.full((complex_signals.shape[0],), scale, dtype=np.float32))
    return normalize_complex(np.stack((distorted.real, distorted.imag), axis=1).astype(np.float32))


def apply_clipping(signals: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None:
        return signals.copy()
    complex_signals = signals[:, 0, :] + 1j * signals[:, 1, :]
    mag = np.abs(complex_signals)
    scale = np.minimum(1.0, threshold / np.maximum(mag, 1e-8))
    clipped = complex_signals * scale
    return normalize_complex(np.stack((clipped.real, clipped.imag), axis=1).astype(np.float32))


def corrupt_signals(signals: np.ndarray, corruption_name: str, params: dict, seed: int) -> np.ndarray:
    rng = make_rng(seed, offset=10_000 + abs(hash((corruption_name, params["label"]))) % 10_000)
    if corruption_name == "awgn":
        return apply_awgn(signals, params["snr_db"], rng)
    if corruption_name == "impulse_noise":
        return apply_impulse_noise(signals, params["prob"], params["scale"], rng)
    if corruption_name == "narrowband_interferer":
        return apply_narrowband_interferer(signals, params["jsr_db"], rng)
    if corruption_name == "sample_rate_distortion":
        return apply_sample_rate_distortion(signals, params["scale"])
    if corruption_name == "clipping":
        return apply_clipping(signals, params["threshold"])
    raise ValueError(f"Unknown corruption: {corruption_name}")


def build_augmented_split(signals: np.ndarray, seed: int, clean_fraction: float = 0.2) -> tuple[np.ndarray, list[dict]]:
    rng = make_rng(seed, offset=20_000)
    augmented = np.empty_like(signals)
    assignments: list[dict] = []
    corruption_names = list(CORRUPTION_CONFIGS.keys())

    for idx in range(signals.shape[0]):
        if rng.random() < clean_fraction:
            augmented[idx] = signals[idx]
            assignments.append({"corruption": "clean", "severity_label": "clean"})
            continue

        corruption_name = str(rng.choice(corruption_names))
        choices = CORRUPTION_CONFIGS[corruption_name][1:]
        params = dict(choices[int(rng.integers(0, len(choices)))])
        augmented[idx : idx + 1] = corrupt_signals(signals[idx : idx + 1], corruption_name, params, seed + idx * 17)
        assignments.append({"corruption": corruption_name, "severity_label": params["label"]})
    return augmented, assignments


def train_gated_fusion(
    iq_train: np.ndarray,
    iq_val: np.ndarray,
    iq_test: np.ndarray,
    fft_train: np.ndarray,
    fft_val: np.ndarray,
    fft_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
    batch_size: int,
) -> tuple[GatedMultimodalIQFFTCNN, dict]:
    device, use_cuda = configure_device()
    train_loader = make_pair_loader(iq_train, fft_train, y_train, True, use_cuda, batch_size)
    val_loader = make_pair_loader(iq_val, fft_val, y_val, False, use_cuda, batch_size)
    test_loader = make_pair_loader(iq_test, fft_test, y_test, False, use_cuda, batch_size)

    model = GatedMultimodalIQFFTCNN(num_classes=len(class_names), branch_dropout=BRANCH_DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_gate_mean = None
    for epoch in range(1, epochs + 1):
        model.train(True)
        for iqb, fftb, yb in train_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
                final_loss = criterion(outputs["final_logits"], yb)
                iq_loss = criterion(outputs["iq_logits"], yb)
                fft_loss = criterion(outputs["fft_logits"], yb)
                fusion_loss = criterion(outputs["fusion_logits"], yb)
                loss = final_loss + AUX_LOSS_WEIGHT * (iq_loss + fft_loss + fusion_loss)
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
        print(f"Epoch {epoch:02d}/{epochs} | gated val acc {val_acc:.3f} | gates {[round(x, 3) for x in gate_mean]}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_gate_mean = gate_mean
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
    result = {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "best_val_gate_weights": best_gate_mean,
        "test_gate_weights": (gate_sum / max(total, 1)).tolist(),
    }
    result.update(classification_metrics(y_test, y_pred, class_names))
    return model.cpu(), result


def evaluate_single_mode(model: ExpertCNN, signals: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> dict:
    device, use_cuda = configure_device()
    loader = make_loader(signals, y_true, False, use_cuda, batch_size=256)
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    model = model.cpu()
    return classification_metrics(y_true, y_pred, class_names)


def evaluate_gated(model: GatedMultimodalIQFFTCNN, iq: np.ndarray, fft: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> dict:
    device, use_cuda = configure_device()
    loader = make_pair_loader(iq, fft, y_true, False, use_cuda, batch_size=256)
    model = model.to(device)
    model.eval()
    preds = []
    gate_sum = torch.zeros(3, dtype=torch.float64)
    total = 0
    with torch.no_grad():
        for iqb, fftb, _ in loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)
            total += iqb.size(0)
    y_pred = np.concatenate(preds)
    model = model.cpu()
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["test_gate_weights"] = (gate_sum / max(total, 1)).tolist()
    return metrics


def evaluate_frozen_residual(
    model: FrozenExpertResidualFusion,
    iq: np.ndarray,
    fft: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
) -> dict:
    device, use_cuda = configure_device()
    loader = make_pair_loader(iq, fft, y_true, False, use_cuda, batch_size=256)
    model = model.to(device)
    model.eval()
    preds = []
    alpha_sum = 0.0
    anchor_iq_sum = 0.0
    total = 0
    with torch.no_grad():
        for iqb, fftb, _ in loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            alpha_sum += float(outputs["alpha"].sum().item())
            anchor_iq_sum += float(outputs["anchor_is_iq"].sum().item())
            total += iqb.size(0)
    y_pred = np.concatenate(preds)
    model = model.cpu()
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["test_alpha_mean"] = float(alpha_sum / max(total, 1))
    metrics["test_iq_anchor_fraction"] = float(anchor_iq_sum / max(total, 1))
    return metrics


def load_dataset(name: str) -> dict:
    if name == "waveform_family":
        dataset = load_waveform_dataset()
        dataset["epochs"] = 40
        dataset["batch_size"] = 128
        return dataset
    if name == "modulation_family":
        dataset = load_modulation_dataset()
        dataset["epochs"] = 40
        dataset["batch_size"] = 128
        return dataset
    if name == "subghz_real_512":
        return build_h5_dataset(
            "subghz_real_512",
            REAL_TRAIN_PATH,
            REAL_VAL_PATH,
            REAL_TEST_PATH,
            max_windows_per_file=512,
            batch_size=256,
            epochs=20,
        )
    if name == "captured_npy_real_128":
        return build_h5_dataset(
            "captured_npy_real_128",
            CAPTURED_TRAIN_PATH,
            CAPTURED_VAL_PATH,
            CAPTURED_TEST_PATH,
            max_windows_per_file=128,
            batch_size=256,
            epochs=20,
        )
    if name == "orbit_rf":
        dataset = load_orbit_dataset(max_packets_per_node=256)
        dataset["epochs"] = 20
        dataset["batch_size"] = 256
        return dataset
    raise ValueError(f"Unsupported dataset: {name}")


def default_results_path(dataset_name: str, train_regime: str, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    return results_dir / dataset_name / f"results_{train_regime}.json"


def summarize_corruption(results: dict) -> dict:
    summary = {}
    for corruption_name, entries in results.items():
        if corruption_name == "clean":
            continue
        summary[corruption_name] = {}
        for model_name in ("iq_cnn", "fft_cnn", "gated_multimodal_cnn", "frozen_expert_residual_fusion"):
            accs = [entry[model_name]["test_acc"] for entry in entries]
            summary[corruption_name][model_name] = {
                "mean_test_acc": float(np.mean(accs)),
                "min_test_acc": float(np.min(accs)),
                "max_test_acc": float(np.max(accs)),
            }
        deltas = [
            entry["frozen_expert_residual_fusion"]["test_acc"] - max(entry["iq_cnn"]["test_acc"], entry["fft_cnn"]["test_acc"])
            for entry in entries
        ]
        summary[corruption_name]["frozen_delta_vs_best_single"] = {
            "mean": float(np.mean(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
        }
    return summary


def plot_results(results: dict, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_labels = {
        "iq_cnn": "IQ",
        "fft_cnn": "FFT",
        "gated_multimodal_cnn": "Gated",
        "frozen_expert_residual_fusion": "Frozen Residual",
    }
    saved = []
    for corruption_name, entries in results.items():
        if corruption_name == "clean":
            continue
        labels = [entry["severity_label"] for entry in entries]
        plt.figure(figsize=(7, 4))
        for model_name, line_label in model_labels.items():
            accs = [entry[model_name]["test_acc"] for entry in entries]
            plt.plot(labels, accs, marker="o", label=line_label)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Test Accuracy")
        plt.xlabel("Severity")
        plt.title(f"{corruption_name.replace('_', ' ').title()} Robustness")
        plt.grid(alpha=0.3)
        plt.legend()
        path = output_dir / f"{corruption_name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(str(path))
    return saved


def run_experiment(dataset_name: str, seed: int, results_path: Path, train_regime: str) -> dict:
    set_global_seed(seed)
    start = time.perf_counter()

    dataset = load_dataset(dataset_name)
    iq_train = dataset["iq_train"]
    iq_val = dataset["iq_val"]
    iq_test = dataset["iq_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    class_names = dataset["class_names"]
    epochs = dataset["epochs"]
    batch_size = dataset["batch_size"]

    train_assignment_summary = None
    val_assignment_summary = None
    if train_regime == "augmented":
        iq_train, train_assignments = build_augmented_split(iq_train, seed=seed + 1, clean_fraction=0.2)
        iq_val, val_assignments = build_augmented_split(iq_val, seed=seed + 2, clean_fraction=0.2)
        train_assignment_summary = {}
        val_assignment_summary = {}
        for item in train_assignments:
            key = f"{item['corruption']}:{item['severity_label']}"
            train_assignment_summary[key] = train_assignment_summary.get(key, 0) + 1
        for item in val_assignments:
            key = f"{item['corruption']}:{item['severity_label']}"
            val_assignment_summary[key] = val_assignment_summary.get(key, 0) + 1

    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)

    print(f"\n=== {train_regime} training on {dataset_name} ===")
    iq_train_loader = make_loader(iq_train, y_train, True, torch.cuda.is_available(), batch_size)
    iq_val_loader = make_loader(iq_val, y_val, False, torch.cuda.is_available(), batch_size)
    iq_test_loader = make_loader(iq_test, y_test, False, torch.cuda.is_available(), batch_size)
    fft_train_loader = make_loader(fft_train, y_train, True, torch.cuda.is_available(), batch_size)
    fft_val_loader = make_loader(fft_val, y_val, False, torch.cuda.is_available(), batch_size)
    fft_test_loader = make_loader(fft_test, y_test, False, torch.cuda.is_available(), batch_size)
    pair_train_loader = make_pair_loader(iq_train, fft_train, y_train, True, torch.cuda.is_available(), batch_size)
    pair_val_loader = make_pair_loader(iq_val, fft_val, y_val, False, torch.cuda.is_available(), batch_size)
    pair_test_loader = make_pair_loader(iq_test, fft_test, y_test, False, torch.cuda.is_available(), batch_size)

    iq_expert, iq_clean = train_single_expert(iq_train_loader, iq_val_loader, iq_test_loader, y_test, class_names, epochs)
    fft_expert, fft_clean = train_single_expert(fft_train_loader, fft_val_loader, fft_test_loader, y_test, class_names, epochs)
    gated_model, gated_clean = train_gated_fusion(
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
        epochs,
        batch_size,
    )
    frozen_model = FrozenExpertResidualFusion(iq_expert, fft_expert, num_classes=len(class_names), delta_scale=DELTA_SCALE)
    frozen_clean = train_residual_fusion(
        iq_expert,
        fft_expert,
        pair_train_loader,
        pair_val_loader,
        pair_test_loader,
        y_test,
        class_names,
        epochs,
    )

    corruption_results = {
        "clean": {
            "severity_label": "clean",
            "iq_cnn": iq_clean,
            "fft_cnn": fft_clean,
            "gated_multimodal_cnn": gated_clean,
            "frozen_expert_residual_fusion": frozen_clean,
        }
    }

    for corruption_name, configs in CORRUPTION_CONFIGS.items():
        entries = []
        for idx, params in enumerate(configs):
            corrupted_iq = corrupt_signals(iq_test, corruption_name, params, seed + idx * 101)
            corrupted_fft = to_fft(corrupted_iq)
            entries.append(
                {
                    "severity_label": params["label"],
                    "iq_cnn": evaluate_single_mode(iq_expert, corrupted_iq, y_test, class_names),
                    "fft_cnn": evaluate_single_mode(fft_expert, corrupted_fft, y_test, class_names),
                    "gated_multimodal_cnn": evaluate_gated(gated_model, corrupted_iq, corrupted_fft, y_test, class_names),
                    "frozen_expert_residual_fusion": evaluate_frozen_residual(frozen_model, corrupted_iq, corrupted_fft, y_test, class_names),
                }
            )
        corruption_results[corruption_name] = entries

    plot_dir = results_path.parent / f"{results_path.stem.replace('results_', 'plots_')}"
    plot_paths = plot_results(corruption_results, plot_dir)
    results = {
        "experiment": "clean_train_corrupted_test_robustness",
        "dataset": dataset_name,
        "seed": seed,
        "train_regime": train_regime,
        "train_protocol": f"{train_regime}_train_{train_regime}_val_corrupted_test",
        "models": ["iq_cnn", "fft_cnn", "gated_multimodal_cnn", "frozen_expert_residual_fusion"],
        "corruption_results": corruption_results,
        "summary": summarize_corruption(corruption_results),
        "plot_paths": plot_paths,
        "runtime_seconds": time.perf_counter() - start,
    }
    if train_assignment_summary is not None:
        results["augmented_train_assignment_counts"] = train_assignment_summary
    if val_assignment_summary is not None:
        results["augmented_val_assignment_counts"] = val_assignment_summary
    save_json(results_path, results)
    print(f"Saved results to {results_path}")
    return results


def run_batch(dataset_names: list[str], seed: int, train_regime: str, results_dir: Path) -> dict:
    batch_start = time.perf_counter()
    summary = {
        "experiment": "corruption_robustness_batch",
        "seed": seed,
        "train_regime": train_regime,
        "datasets": {},
    }
    for dataset_name in dataset_names:
        dataset_start = time.perf_counter()
        results_path = default_results_path(dataset_name, train_regime, results_dir)
        result = run_experiment(dataset_name=dataset_name, seed=seed, results_path=results_path, train_regime=train_regime)
        summary["datasets"][dataset_name] = {
            "results_path": str(results_path),
            "runtime_seconds": time.perf_counter() - dataset_start,
            "summary": result["summary"],
        }
    summary["runtime_seconds"] = time.perf_counter() - batch_start
    summary_path = results_dir / f"batch_summary_{train_regime}.json"
    save_json(summary_path, summary)
    print(f"Saved batch summary to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean-train / corrupted-test robustness study for IQ, FFT, and fusion models.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="waveform_family")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, help="Optional list of datasets to run sequentially.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-regime", choices=TRAIN_REGIME_CHOICES, default="clean")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    if args.datasets:
        run_batch(dataset_names=args.datasets, seed=args.seed, train_regime=args.train_regime, results_dir=args.results_dir)
        return
    run_experiment(dataset_name=args.dataset, seed=args.seed, results_path=args.results_path, train_regime=args.train_regime)


if __name__ == "__main__":
    main()
