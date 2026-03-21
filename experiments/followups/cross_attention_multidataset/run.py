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
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.run_experiment9_wavelet_multidataset import (  # noqa: E402
    AUX_LOSS_WEIGHT,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    WEIGHT_DECAY,
    load_modulation_dataset,
    load_orbit_dataset,
    load_real_dataset,
    load_waveform_dataset,
    to_fft,
    to_wavelet,
)


RESULTS_PATH = Path(__file__).resolve().with_name("results.json")
SEED = 0


class TripleDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, wavelet: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.wavelet = torch.from_numpy(wavelet)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.wavelet[idx], self.y[idx]


class SequenceEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 2, hidden_dim: int = 128, num_tokens: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(num_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x).transpose(1, 2)


class SequenceEncoder2D(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, token_grid: tuple[int, int] = (2, 8)):
        super().__init__()
        self.token_grid = token_grid
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(token_grid),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        batch, channels, height, width = features.shape
        return features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 128, num_heads: int = 4, num_fusion_tokens: int = 4):
        super().__init__()
        self.iq_encoder = SequenceEncoder1D(hidden_dim=hidden_dim, num_tokens=16)
        self.fft_encoder = SequenceEncoder1D(hidden_dim=hidden_dim, num_tokens=16)
        self.wavelet_encoder = SequenceEncoder2D(hidden_dim=hidden_dim, token_grid=(2, 8))

        self.modality_embeddings = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.fusion_tokens = nn.Parameter(torch.randn(1, num_fusion_tokens, hidden_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

        self.iq_head = nn.Linear(hidden_dim, num_classes)
        self.fft_head = nn.Linear(hidden_dim, num_classes)
        self.wavelet_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, iq: torch.Tensor, fft: torch.Tensor, wavelet: torch.Tensor):
        iq_tokens = self.iq_encoder(iq) + self.modality_embeddings[0].view(1, 1, -1)
        fft_tokens = self.fft_encoder(fft) + self.modality_embeddings[1].view(1, 1, -1)
        wavelet_tokens = self.wavelet_encoder(wavelet) + self.modality_embeddings[2].view(1, 1, -1)

        context = torch.cat([iq_tokens, fft_tokens, wavelet_tokens], dim=1)
        fusion_tokens = self.fusion_tokens.expand(iq.size(0), -1, -1)

        attended, _ = self.cross_attn(fusion_tokens, context, context, need_weights=False)
        fusion_tokens = self.cross_norm(fusion_tokens + attended)
        refined, _ = self.self_attn(fusion_tokens, fusion_tokens, fusion_tokens, need_weights=False)
        fusion_tokens = self.self_norm(fusion_tokens + refined)
        fusion_tokens = fusion_tokens + self.ffn(fusion_tokens)
        pooled = self.final_norm(fusion_tokens.mean(dim=1))

        iq_summary = iq_tokens.mean(dim=1)
        fft_summary = fft_tokens.mean(dim=1)
        wavelet_summary = wavelet_tokens.mean(dim=1)

        return {
            "final_logits": self.classifier(pooled),
            "aux_logits": [
                self.iq_head(iq_summary),
                self.fft_head(fft_summary),
                self.wavelet_head(wavelet_summary),
            ],
        }


def make_triple_loader(iq: np.ndarray, fft: np.ndarray, wavelet: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return DataLoader(TripleDataset(iq, fft, wavelet, y), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    report = classification_report(y_true, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0, output_dict=True)
    return {
        "test_acc": float((y_pred == y_true).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def compute_loss(outputs, yb, criterion):
    final_loss = criterion(outputs["final_logits"], yb)
    aux_loss = sum(criterion(logits, yb) for logits in outputs["aux_logits"])
    return final_loss + AUX_LOSS_WEIGHT * aux_loss


def train_cross_attention(
    iq_train: np.ndarray,
    iq_val: np.ndarray,
    iq_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    batch_size: int,
    epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)
    wavelet_train = to_wavelet(iq_train, batch_size=batch_size)
    wavelet_val = to_wavelet(iq_val, batch_size=batch_size)
    wavelet_test = to_wavelet(iq_test, batch_size=batch_size)

    train_loader = make_triple_loader(iq_train, fft_train, wavelet_train, y_train, True, use_cuda, batch_size)
    val_loader = make_triple_loader(iq_val, fft_val, wavelet_val, y_val, False, use_cuda, batch_size)
    test_loader = make_triple_loader(iq_test, fft_test, wavelet_test, y_test, False, use_cuda, batch_size)

    model = CrossAttentionFusionModel(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_cuda)

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train(True)
        for iqb, fftb, wavb, yb in train_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            wavb = wavb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb, wavb)
                loss = compute_loss(outputs, yb, criterion)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for iqb, fftb, wavb, yb in val_loader:
                iqb = iqb.to(device, non_blocking=use_cuda)
                fftb = fftb.to(device, non_blocking=use_cuda)
                wavb = wavb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(iqb, fftb, wavb)
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
        for iqb, fftb, wavb, _ in test_loader:
            iqb = iqb.to(device, non_blocking=use_cuda)
            fftb = fftb.to(device, non_blocking=use_cuda)
            wavb = wavb.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iqb, fftb, wavb)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    result = {"best_epoch": int(best_epoch), "val_acc": float(best_val_acc)}
    result.update(evaluate_predictions(y_test, y_pred, class_names))
    result["wavelet_shape"] = list(wavelet_train.shape[1:])
    return result


def run_dataset(dataset: dict):
    print(f"\n=== Dataset: {dataset['name']} ===")
    print("iq train/val/test:", dataset["iq_train"].shape, dataset["iq_val"].shape, dataset["iq_test"].shape)
    start = time.perf_counter()
    result = train_cross_attention(
        iq_train=dataset["iq_train"],
        iq_val=dataset["iq_val"],
        iq_test=dataset["iq_test"],
        y_train=dataset["y_train"],
        y_val=dataset["y_val"],
        y_test=dataset["y_test"],
        class_names=dataset["class_names"],
        batch_size=dataset["batch_size"],
        epochs=dataset["epochs"],
    )
    result["train_examples"] = int(dataset["iq_train"].shape[0])
    result["val_examples"] = int(dataset["iq_val"].shape[0])
    result["test_examples"] = int(dataset["iq_test"].shape[0])
    result["signal_length"] = int(dataset["iq_train"].shape[-1])
    result["batch_size"] = int(dataset["batch_size"])
    result["epochs"] = int(dataset["epochs"])
    result["runtime_seconds"] = time.perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser(description="Run a 3-modality cross-attention model across all benchmark datasets.")
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
        "experiment": "cross_attention_multidataset_benchmark",
        "real_max_windows_per_file": int(args.real_max_windows_per_file),
        "orbit_max_packets_per_node": int(args.orbit_max_packets_per_node),
        "datasets": {},
    }

    for dataset in datasets:
        results["datasets"][dataset["name"]] = run_dataset(dataset)

    results["runtime_seconds"] = time.perf_counter() - start
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.results_path}")


if __name__ == "__main__":
    main()
