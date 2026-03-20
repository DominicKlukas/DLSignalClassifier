from __future__ import annotations

import argparse
import json
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


from experiments.run_experiment11_frozen_expert_residual_multidataset import (  # noqa: E402
    REAL_TEST_PATH,
    REAL_TRAIN_PATH,
    REAL_VAL_PATH,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    LEARNING_RATE,
)
from experiments.run_experiment9_wavelet_multidataset import (  # noqa: E402
    load_modulation_dataset,
    load_real_split,
    load_waveform_dataset,
    to_fft,
)


SEED = 0
PROJECTION_DIM = 128
HEAD_HIDDEN_DIM = 128
HEAD_DROPOUT = 0.2
RESULTS_PATH = ROOT_DIR / "experiments" / "experiment12_fixed_feature_fusion_results.json"
MIXTURE_RESULTS_PATH = ROOT_DIR / "experiments" / "experiment11_frozen_expert_residual_multidataset_results.json"

TAP_SPECS = {
    "block1": 32,
    "block2": 64,
    "block3": 128,
    "final": 256,
}


class PairDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


class TapExpertCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.block4 = nn.Sequential(
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

    def encode_all(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        return {
            "block1": h1,
            "block2": h2,
            "block3": h3,
            "final": h4,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode_all(x)["final"]
        return self.classifier(features)


class FrozenTapFusion(nn.Module):
    def __init__(self, iq_expert: TapExpertCNN, fft_expert: TapExpertCNN, tap_name: str, num_classes: int):
        super().__init__()
        self.iq_expert = iq_expert
        self.fft_expert = fft_expert
        self.tap_name = tap_name
        for param in self.iq_expert.parameters():
            param.requires_grad = False
        for param in self.fft_expert.parameters():
            param.requires_grad = False
        self.iq_expert.eval()
        self.fft_expert.eval()

        tap_dim = TAP_SPECS[tap_name]
        self.iq_project = nn.Sequential(
            nn.LayerNorm(tap_dim),
            nn.Linear(tap_dim, PROJECTION_DIM),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
        )
        self.fft_project = nn.Sequential(
            nn.LayerNorm(tap_dim),
            nn.Linear(tap_dim, PROJECTION_DIM),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(PROJECTION_DIM * 2),
            nn.Linear(PROJECTION_DIM * 2, HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN_DIM, num_classes),
        )

    def pooled_tap(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.mean(dim=-1)

    def forward(self, iq: torch.Tensor, fft: torch.Tensor):
        with torch.no_grad():
            iq_taps = self.iq_expert.encode_all(iq)
            fft_taps = self.fft_expert.encode_all(fft)
        iq_features = self.pooled_tap(iq_taps[self.tap_name])
        fft_features = self.pooled_tap(fft_taps[self.tap_name])
        iq_proj = self.iq_project(iq_features)
        fft_proj = self.fft_project(fft_features)
        fused = torch.cat([iq_proj, fft_proj], dim=1)
        logits = self.classifier(fused)
        return {
            "final_logits": logits,
            "iq_proj": iq_proj,
            "fft_proj": fft_proj,
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


def train_expert(
    train_loader,
    val_loader,
    test_loader,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = TapExpertCNN(num_classes=len(class_names)).to(device)
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


def train_fusion_head(
    iq_expert: TapExpertCNN,
    fft_expert: TapExpertCNN,
    tap_name: str,
    train_loader,
    val_loader,
    test_loader,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = FrozenTapFusion(iq_expert, fft_expert, tap_name=tap_name, num_classes=len(class_names)).to(device)
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
    for epoch in range(1, epochs + 1):
        model.train(True)
        for iqb, fftb, yb in train_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
                loss = criterion(outputs["final_logits"], yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for iqb, fftb, yb in val_loader:
                iqb = iqb.to(device, non_blocking=use_cuda)
                fftb = fftb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(iqb, fftb)
                correct += (outputs["final_logits"].argmax(dim=1) == yb).sum().item()
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
        for iqb, fftb, _ in test_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    result = {
        "tap_name": tap_name,
        "best_epoch": int(best_epoch),
        "val_acc": float(best_val_acc),
    }
    result.update(evaluate_predictions(y_test, y_pred, class_names))
    return result


def build_subghz_dataset():
    iq_train, y_train, class_names = load_real_split(REAL_TRAIN_PATH, max_windows_per_file=128)
    iq_val, y_val, _ = load_real_split(REAL_VAL_PATH, max_windows_per_file=128)
    iq_test, y_test, _ = load_real_split(REAL_TEST_PATH, max_windows_per_file=128)
    return {
        "name": "subghz_real_128",
        "class_names": class_names,
        "iq_train": iq_train,
        "iq_val": iq_val,
        "iq_test": iq_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "batch_size": 256,
        "epochs": 20,
    }


def load_mixture_reference() -> dict[str, dict]:
    if not MIXTURE_RESULTS_PATH.exists():
        return {}
    data = json.loads(MIXTURE_RESULTS_PATH.read_text())
    return data.get("datasets", {})


def run_dataset(dataset: dict, mixture_reference: dict[str, dict]):
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

    print("\nTraining IQ expert")
    iq_expert, iq_result = train_expert(iq_train_loader, iq_val_loader, iq_test_loader, y_test, class_names, epochs)

    print("\nTraining FFT expert")
    fft_expert, fft_result = train_expert(fft_train_loader, fft_val_loader, fft_test_loader, y_test, class_names, epochs)

    best_single = max(iq_result["test_acc"], fft_result["test_acc"])
    tap_results = {}
    for tap_name in TAP_SPECS:
        print(f"\nTraining fusion head for tap {tap_name}")
        tap_result = train_fusion_head(
            iq_expert,
            fft_expert,
            tap_name,
            pair_train_loader,
            pair_val_loader,
            pair_test_loader,
            y_test,
            class_names,
            epochs,
        )
        tap_result["delta_vs_best_single"] = float(tap_result["test_acc"] - best_single)
        tap_results[tap_name] = tap_result

    best_tap_name = max(tap_results, key=lambda key: tap_results[key]["test_acc"])
    result = {
        "train_examples": int(iq_train.shape[0]),
        "val_examples": int(iq_val.shape[0]),
        "test_examples": int(iq_test.shape[0]),
        "signal_length": int(iq_train.shape[-1]),
        "class_names": class_names,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "iq_expert": iq_result,
        "fft_expert": fft_result,
        "best_single_test_acc": float(best_single),
        "best_tap_name": best_tap_name,
        "best_tap_result": tap_results[best_tap_name],
        "tap_results": tap_results,
    }
    if name in mixture_reference:
        mixture = mixture_reference[name]["frozen_expert_residual_fusion"]
        result["mixture_reference"] = {
            "test_acc": float(mixture["test_acc"]),
            "macro_f1": float(mixture["macro_f1"]),
            "weighted_f1": float(mixture["weighted_f1"]),
            "delta_vs_best_single": float(mixture_reference[name]["delta_vs_best_single"]),
        }
        result["best_tap_vs_mixture"] = float(result["best_tap_result"]["test_acc"] - mixture["test_acc"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Study fixed hidden-feature fusion depth with frozen IQ and FFT experts.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()

    waveform = load_waveform_dataset()
    waveform["epochs"] = 40
    subghz = build_subghz_dataset()

    mixture_reference = load_mixture_reference()
    datasets = [waveform, subghz]

    results = {
        "experiment": "fixed_hidden_feature_fusion_depth_study",
        "tap_specs": TAP_SPECS,
        "projection_dim": PROJECTION_DIM,
        "head_hidden_dim": HEAD_HIDDEN_DIM,
        "head_dropout": HEAD_DROPOUT,
        "datasets": {},
    }
    for dataset in datasets:
        dataset_start = time.perf_counter()
        dataset_result = run_dataset(dataset, mixture_reference)
        dataset_result["runtime_seconds"] = time.perf_counter() - dataset_start
        results["datasets"][dataset["name"]] = dataset_result

    results["runtime_seconds"] = time.perf_counter() - start
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
