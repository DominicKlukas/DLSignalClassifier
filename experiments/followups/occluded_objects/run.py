from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SEED = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
ALPHA_PENALTY = 0.01
DELTA_PENALTY = 0.001
DELTA_SCALE = 2.0

DATASET_ZIP_PATH = ROOT_DIR / "external" / "radar_iq_datasets" / "data" / "occluded_object_classification.zip"
DATASET_ROOT = ROOT_DIR / "external" / "radar_iq_datasets" / "data" / "occluded_object_classification_unpacked" / "64GHz"
RESULTS_PATH = Path(__file__).resolve().with_name("results.json")


def maybe_extract_dataset(dataset_root: Path, zip_path: Path) -> None:
    if dataset_root.exists():
        return
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Occluded-object dataset not found. Expected either extracted data at {dataset_root} or zip at {zip_path}."
        )
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_root.parent)


def normalize_complex_rms(samples: np.ndarray) -> np.ndarray:
    complex_samples = samples[:, 0] + 1j * samples[:, 1]
    rms = np.sqrt(np.mean(np.abs(complex_samples) ** 2, axis=(1, 2, 3), keepdims=True) + 1e-8).astype(np.float32)
    complex_samples = complex_samples / rms
    return np.stack((complex_samples.real, complex_samples.imag), axis=1).astype(np.float32)


def load_pickle_sample(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    cube = np.zeros((20, 20, 100), dtype=np.complex64)
    for (row_idx, col_idx), values in payload.items():
        cube[int(row_idx) - 1, int(col_idx) - 21] = np.asarray(values, dtype=np.complex64)
    sample = np.stack((cube.real, cube.imag), axis=0).astype(np.float32)
    return normalize_complex_rms(sample[np.newaxis, ...])[0]


def to_fft(samples: np.ndarray) -> np.ndarray:
    complex_samples = samples[:, 0] + 1j * samples[:, 1]
    fft_samples = np.fft.fftn(complex_samples, axes=(-3, -2, -1))
    fft_samples = np.fft.fftshift(fft_samples, axes=(-3, -2, -1))
    features = np.stack((fft_samples.real, fft_samples.imag), axis=1).astype(np.float32)
    return normalize_complex_rms(features)


def load_dataset(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    class_names = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    signals = []
    labels = []
    file_names = []
    for label, class_name in enumerate(class_names):
        for path in sorted((dataset_root / class_name).glob("*.pickle")):
            signals.append(load_pickle_sample(path))
            labels.append(label)
            file_names.append(path.name)
    X = np.stack(signals, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, class_names, file_names


class PairDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


class Expert3DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(3, 3, 7), padding=(1, 1, 3)),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 5), padding=(1, 1, 2)),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FrozenExpertResidualFusion(nn.Module):
    def __init__(self, iq_expert: Expert3DCNN, fft_expert: Expert3DCNN, num_classes: int):
        super().__init__()
        self.iq_expert = iq_expert
        self.fft_expert = fft_expert
        for param in self.iq_expert.parameters():
            param.requires_grad = False
        for param in self.fft_expert.parameters():
            param.requires_grad = False
        self.iq_expert.eval()
        self.fft_expert.eval()

        context_dim = 128 + 128 + num_classes + num_classes + 4
        self.residual_net = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(128, 1)
        self.delta_head = nn.Linear(128, num_classes)

    def forward(self, iq: torch.Tensor, fft: torch.Tensor):
        with torch.no_grad():
            iq_features = self.iq_expert.encode(iq)
            fft_features = self.fft_expert.encode(fft)
            iq_logits = self.iq_expert.classify_features(iq_features)
            fft_logits = self.fft_expert.classify_features(fft_features)

        iq_probs = torch.softmax(iq_logits, dim=1)
        fft_probs = torch.softmax(fft_logits, dim=1)
        iq_conf = iq_probs.max(dim=1, keepdim=True).values
        fft_conf = fft_probs.max(dim=1, keepdim=True).values
        iq_entropy = -(iq_probs * torch.log(iq_probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        fft_entropy = -(fft_probs * torch.log(fft_probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)

        choose_iq = iq_conf >= fft_conf
        anchor_logits = torch.where(choose_iq, iq_logits, fft_logits)
        anchor_is_iq = choose_iq.float()

        context = torch.cat(
            [iq_features, fft_features, iq_logits, fft_logits, iq_conf, fft_conf, iq_entropy, fft_entropy],
            dim=1,
        )
        hidden = self.residual_net(context)
        alpha = torch.sigmoid(self.alpha_head(hidden))
        delta = DELTA_SCALE * torch.tanh(self.delta_head(hidden))
        final_logits = anchor_logits + alpha * delta
        return {
            "final_logits": final_logits,
            "alpha": alpha,
            "delta": delta,
            "anchor_is_iq": anchor_is_iq,
        }


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, use_cuda: bool):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
    )


def make_pair_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, use_cuda: bool):
    return DataLoader(
        PairDataset(iq, fft, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        zero_division=0,
        output_dict=True,
    )
    return {
        "test_acc": float((y_pred == y_true).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def train_single_expert(
    train_loader,
    eval_loader,
    y_eval: np.ndarray,
    class_names: list[str],
    epochs: int,
) -> tuple[Expert3DCNN, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = Expert3DCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

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
        scheduler.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in eval_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    logits = model(xb)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(preds)
        eval_acc = float((y_pred == y_eval).mean())
        print(f"Epoch {epoch:02d}/{epochs} | eval acc {eval_acc:.3f}", flush=True)

    result = {"best_epoch": int(epochs), "eval_acc": float(eval_acc)}
    result.update(evaluate_predictions(y_eval, y_pred, class_names))
    return model.cpu(), result


def compute_residual_loss(outputs, yb, criterion):
    final_loss = criterion(outputs["final_logits"], yb)
    alpha_penalty = outputs["alpha"].mean()
    delta_penalty = outputs["delta"].pow(2).mean()
    return final_loss + ALPHA_PENALTY * alpha_penalty + DELTA_PENALTY * delta_penalty


def train_residual_fusion(
    iq_expert: Expert3DCNN,
    fft_expert: Expert3DCNN,
    train_loader,
    eval_loader,
    y_eval: np.ndarray,
    class_names: list[str],
    epochs: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = FrozenExpertResidualFusion(iq_expert, fft_expert, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    for epoch in range(1, epochs + 1):
        model.train(True)
        for iqb, fftb, yb in train_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
                loss = compute_residual_loss(outputs, yb, criterion)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        preds = []
        alpha_sum = 0.0
        anchor_iq_sum = 0.0
        total = 0
        with torch.no_grad():
            for iqb, fftb, _ in eval_loader:
                iqb = iqb.to(device, non_blocking=use_cuda)
                fftb = fftb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(iqb, fftb)
                preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
                alpha_sum += float(outputs["alpha"].sum().item())
                anchor_iq_sum += float(outputs["anchor_is_iq"].sum().item())
                total += iqb.size(0)
        y_pred = np.concatenate(preds)
        eval_acc = float((y_pred == y_eval).mean())
        alpha_mean = float(alpha_sum / max(total, 1))
        anchor_iq_fraction = float(anchor_iq_sum / max(total, 1))
        print(
            f"Epoch {epoch:02d}/{epochs} | eval acc {eval_acc:.3f} | alpha {alpha_mean:.3f} | iq-anchor {anchor_iq_fraction:.3f}",
            flush=True,
        )

    result = {
        "best_epoch": int(epochs),
        "eval_acc": float(eval_acc),
        "test_alpha_mean": float(alpha_mean),
        "test_iq_anchor_fraction": float(anchor_iq_fraction),
    }
    result.update(evaluate_predictions(y_eval, y_pred, class_names))
    return result


def run_split(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    batch_size: int,
    epochs: int,
    test_fraction: float,
) -> dict:
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=SEED,
        stratify=y,
    )

    iq_train = X[train_idx]
    iq_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    fft_train = to_fft(iq_train)
    fft_test = to_fft(iq_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    iq_train_loader = make_loader(iq_train, y_train, batch_size, True, use_cuda)
    iq_test_loader = make_loader(iq_test, y_test, batch_size, False, use_cuda)
    fft_train_loader = make_loader(fft_train, y_train, batch_size, True, use_cuda)
    fft_test_loader = make_loader(fft_test, y_test, batch_size, False, use_cuda)
    pair_train_loader = make_pair_loader(iq_train, fft_train, y_train, batch_size, True, use_cuda)
    pair_test_loader = make_pair_loader(iq_test, fft_test, y_test, batch_size, False, use_cuda)

    results = {
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "input_shape": list(iq_train.shape[1:]),
        "class_names": class_names,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "test_fraction": float(test_fraction),
        "split_protocol": "paper_like_file_level_80_20_no_val",
    }

    print("\nTraining IQ expert", flush=True)
    iq_expert, iq_result = train_single_expert(iq_train_loader, iq_test_loader, y_test, class_names, epochs)
    results["iq_cnn"] = iq_result

    print("\nTraining FFT expert", flush=True)
    fft_expert, fft_result = train_single_expert(fft_train_loader, fft_test_loader, y_test, class_names, epochs)
    results["fft_cnn"] = fft_result

    print("\nTraining frozen-expert residual fusion", flush=True)
    residual_result = train_residual_fusion(
        iq_expert,
        fft_expert,
        pair_train_loader,
        pair_test_loader,
        y_test,
        class_names,
        epochs,
    )
    results["frozen_expert_residual_fusion"] = residual_result

    best_single = max(iq_result["test_acc"], fft_result["test_acc"])
    results["best_single_test_acc"] = float(best_single)
    results["improves_over_best_single"] = bool(residual_result["test_acc"] > best_single)
    results["matches_or_beats_best_single"] = bool(residual_result["test_acc"] >= best_single)
    results["delta_vs_best_single"] = float(residual_result["test_acc"] - best_single)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 11 on the occluded-object radar dataset using one pickle file as one radar sample.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    maybe_extract_dataset(args.dataset_root, DATASET_ZIP_PATH)
    X, y, class_names, file_names = load_dataset(args.dataset_root)
    print(
        f"Loaded {len(X)} radar samples from {len(file_names)} files with input shape {tuple(X.shape[1:])}",
        flush=True,
    )

    if args.dry_run:
        summary = {
            "dataset": "occluded_object_classification",
            "num_samples": int(len(X)),
            "input_shape": list(X.shape[1:]),
            "class_names": class_names,
        }
        print(json.dumps(summary, indent=2))
        return

    results = {
        "experiment": "frozen_expert_residual_occluded_objects",
        "dataset": run_split(X, y, class_names, batch_size=args.batch_size, epochs=args.epochs, test_fraction=args.test_fraction),
        "dataset_root": str(args.dataset_root),
        "zip_path": str(DATASET_ZIP_PATH),
        "runtime_seconds": time.perf_counter() - start,
    }
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.results_path}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
