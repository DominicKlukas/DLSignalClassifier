from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.run_experiment9_wavelet_multidataset import (  # noqa: E402
    load_modulation_dataset,
    load_orbit_dataset,
    load_real_split,
    load_waveform_dataset,
    to_fft,
)


SEED = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
ALPHA_PENALTY = 0.01
DELTA_PENALTY = 0.001
DELTA_SCALE = 2.0

REAL_TRAIN_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "train.h5"
REAL_VAL_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "val.h5"
REAL_TEST_PATH = ROOT_DIR / "data" / "real_mat_dataset" / "test.h5"

AUG_REAL_TRAIN_PATH = ROOT_DIR / "data" / "real_mat_dataset_augmented" / "train.h5"
AUG_REAL_VAL_PATH = ROOT_DIR / "data" / "real_mat_dataset_augmented" / "val.h5"
AUG_REAL_TEST_PATH = ROOT_DIR / "data" / "real_mat_dataset_augmented" / "test.h5"

CAPTURED_TRAIN_PATH = ROOT_DIR / "data" / "captured_npy_dataset_experiment5" / "train.h5"
CAPTURED_VAL_PATH = ROOT_DIR / "data" / "captured_npy_dataset_experiment5" / "val.h5"
CAPTURED_TEST_PATH = ROOT_DIR / "data" / "captured_npy_dataset_experiment5" / "test.h5"

RESULTS_PATH = ROOT_DIR / "experiments" / "experiment11_frozen_expert_residual_multidataset_results.json"


class PairDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


class ExpertCNN(nn.Module):
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features.unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FrozenExpertResidualFusion(nn.Module):
    def __init__(self, iq_expert: ExpertCNN, fft_expert: ExpertCNN, num_classes: int):
        super().__init__()
        self.iq_expert = iq_expert
        self.fft_expert = fft_expert
        for param in self.iq_expert.parameters():
            param.requires_grad = False
        for param in self.fft_expert.parameters():
            param.requires_grad = False
        self.iq_expert.eval()
        self.fft_expert.eval()

        context_dim = 256 + 256 + num_classes + num_classes + 4
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
            "anchor_logits": anchor_logits,
            "iq_logits": iq_logits,
            "fft_logits": fft_logits,
            "alpha": alpha,
            "delta": delta,
            "anchor_is_iq": anchor_is_iq,
        }


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def make_pair_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(PairDataset(iq, fft, y), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
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
    val_loader,
    test_loader,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = ExpertCNN(num_classes=len(class_names)).to(device)
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
    result = {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
    }
    result.update(evaluate_predictions(y_test, y_pred, class_names))
    return model.cpu(), result


def compute_residual_loss(outputs, yb, criterion):
    final_loss = criterion(outputs["final_logits"], yb)
    alpha_penalty = outputs["alpha"].mean()
    delta_penalty = outputs["delta"].pow(2).mean()
    return final_loss + ALPHA_PENALTY * alpha_penalty + DELTA_PENALTY * delta_penalty


def train_residual_fusion(
    iq_expert: ExpertCNN,
    fft_expert: ExpertCNN,
    train_loader,
    val_loader,
    test_loader,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
):
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

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    best_stats = None
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

        model.eval()
        correct = 0
        total = 0
        alpha_sum = 0.0
        anchor_iq_sum = 0.0
        with torch.no_grad():
            for iqb, fftb, yb in val_loader:
                iqb = iqb.to(device, non_blocking=use_cuda)
                fftb = fftb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(iqb, fftb)
                correct += (outputs["final_logits"].argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
                alpha_sum += float(outputs["alpha"].sum().item())
                anchor_iq_sum += float(outputs["anchor_is_iq"].sum().item())
        val_acc = correct / max(total, 1)
        alpha_mean = alpha_sum / max(total, 1)
        anchor_iq_fraction = anchor_iq_sum / max(total, 1)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{epochs} | val acc {val_acc:.3f} | alpha {alpha_mean:.3f} | iq-anchor {anchor_iq_fraction:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_stats = {
                "best_val_alpha_mean": float(alpha_mean),
                "best_val_iq_anchor_fraction": float(anchor_iq_fraction),
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    alpha_sum = 0.0
    anchor_iq_sum = 0.0
    total = 0
    with torch.no_grad():
        for iqb, fftb, _ in test_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            alpha_sum += float(outputs["alpha"].sum().item())
            anchor_iq_sum += float(outputs["anchor_is_iq"].sum().item())
            total += iqb.size(0)
    y_pred = np.concatenate(preds)
    result = {
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
        "test_alpha_mean": float(alpha_sum / max(total, 1)),
        "test_iq_anchor_fraction": float(anchor_iq_sum / max(total, 1)),
    }
    result.update(best_stats)
    result.update(evaluate_predictions(y_test, y_pred, class_names))
    return result


def build_h5_dataset(name: str, train_path: Path, val_path: Path, test_path: Path, max_windows_per_file: int, batch_size: int, epochs: int):
    iq_train, y_train, class_names = load_real_split(train_path, max_windows_per_file=max_windows_per_file)
    iq_val, y_val, _ = load_real_split(val_path, max_windows_per_file=max_windows_per_file)
    iq_test, y_test, _ = load_real_split(test_path, max_windows_per_file=max_windows_per_file)
    return {
        "name": name,
        "class_names": class_names,
        "iq_train": iq_train,
        "iq_val": iq_val,
        "iq_test": iq_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_windows_per_file": int(max_windows_per_file),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }


def run_dataset(dataset: dict):
    name = dataset["name"]
    class_names = dataset["class_names"]
    iq_train = dataset["iq_train"]
    iq_val = dataset["iq_val"]
    iq_test = dataset["iq_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    batch_size = dataset["batch_size"]
    epochs = dataset["epochs"]

    print(f"\n=== Dataset: {name} ===")
    print("iq train/val/test:", iq_train.shape, iq_val.shape, iq_test.shape)

    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    iq_train_loader = make_loader(iq_train, y_train, True, use_cuda, batch_size)
    iq_val_loader = make_loader(iq_val, y_val, False, use_cuda, batch_size)
    iq_test_loader = make_loader(iq_test, y_test, False, use_cuda, batch_size)
    fft_train_loader = make_loader(fft_train, y_train, True, use_cuda, batch_size)
    fft_val_loader = make_loader(fft_val, y_val, False, use_cuda, batch_size)
    fft_test_loader = make_loader(fft_test, y_test, False, use_cuda, batch_size)
    pair_train_loader = make_pair_loader(iq_train, fft_train, y_train, True, use_cuda, batch_size)
    pair_val_loader = make_pair_loader(iq_val, fft_val, y_val, False, use_cuda, batch_size)
    pair_test_loader = make_pair_loader(iq_test, fft_test, y_test, False, use_cuda, batch_size)

    results = {
        "train_examples": int(iq_train.shape[0]),
        "val_examples": int(iq_val.shape[0]),
        "test_examples": int(iq_test.shape[0]),
        "signal_length": int(iq_train.shape[-1]),
        "class_names": class_names,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
    }
    for key in ("max_windows_per_file", "max_packets_per_node_per_day", "num_common_nodes", "train_path", "val_path", "test_path"):
        if key in dataset:
            results[key] = dataset[key]

    print("\nTraining IQ expert")
    iq_expert, iq_result = train_single_expert(iq_train_loader, iq_val_loader, iq_test_loader, y_test, class_names, epochs)
    results["iq_cnn"] = iq_result

    print("\nTraining FFT expert")
    fft_expert, fft_result = train_single_expert(fft_train_loader, fft_val_loader, fft_test_loader, y_test, class_names, epochs)
    results["fft_cnn"] = fft_result

    print("\nTraining frozen-expert residual fusion")
    residual_result = train_residual_fusion(
        iq_expert,
        fft_expert,
        pair_train_loader,
        pair_val_loader,
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


def main():
    parser = argparse.ArgumentParser(description="Run frozen-expert residual IQ+FFT fusion across all Experiment 5-comparable datasets.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--orbit-max-packets-per-node", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()

    modulation = load_modulation_dataset()
    modulation["epochs"] = 40

    waveform = load_waveform_dataset()
    waveform["epochs"] = 40

    datasets = [
        modulation,
        waveform,
        build_h5_dataset("subghz_real_128", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=128, batch_size=256, epochs=20),
        build_h5_dataset("subghz_real_512", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=512, batch_size=256, epochs=20),
        build_h5_dataset("subghz_real_1024_40ep", REAL_TRAIN_PATH, REAL_VAL_PATH, REAL_TEST_PATH, max_windows_per_file=1024, batch_size=256, epochs=40),
        build_h5_dataset("subghz_real_augmented_512", AUG_REAL_TRAIN_PATH, AUG_REAL_VAL_PATH, AUG_REAL_TEST_PATH, max_windows_per_file=512, batch_size=256, epochs=20),
        load_orbit_dataset(max_packets_per_node=args.orbit_max_packets_per_node),
        build_h5_dataset("captured_npy_real_128", CAPTURED_TRAIN_PATH, CAPTURED_VAL_PATH, CAPTURED_TEST_PATH, max_windows_per_file=128, batch_size=256, epochs=20),
    ]

    # Match the original Orbit benchmark settings from the comparable-experiments file.
    datasets[6]["epochs"] = 20
    datasets[6]["batch_size"] = 256

    results = {
        "experiment": "frozen_expert_residual_multidataset",
        "architecture": {
            "description": "Frozen IQ and FFT experts, confidence-based expert anchor, bounded residual correction",
            "alpha_penalty": ALPHA_PENALTY,
            "delta_penalty": DELTA_PENALTY,
            "delta_scale": DELTA_SCALE,
        },
        "datasets": {},
    }

    for dataset in datasets:
        dataset_start = time.perf_counter()
        dataset_results = run_dataset(dataset)
        dataset_results["runtime_seconds"] = time.perf_counter() - dataset_start
        results["datasets"][dataset["name"]] = dataset_results

    results["runtime_seconds"] = time.perf_counter() - start
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
