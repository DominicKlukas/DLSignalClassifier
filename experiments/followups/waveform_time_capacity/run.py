from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from experiments.shared.repro import (  # noqa: E402
    classification_metrics,
    configure_device,
    make_dataset_loader,
    make_tensor_loader,
    save_json,
    set_global_seed,
)
from experiments.shared.story_datasets import load_waveform_dataset, to_fft  # noqa: E402
from experiments.shared.story_models import GatedMultimodalIQFFTCNN, RepresentationCNN  # noqa: E402


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
RESULTS_PATH = ARTIFACTS_DIR / "results.json"

SEED = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AUX_LOSS_WEIGHT = 0.25
BRANCH_DROPOUT = 0.10

DEFAULT_INITIAL_EPOCHS = 40
DEFAULT_EPOCH_CHUNK = 20
DEFAULT_MIN_EPOCHS = 40
DEFAULT_MAX_EPOCHS = 140
DEFAULT_PLATEAU_CHUNKS = 2
DEFAULT_MIN_DELTA = 5e-4
DEFAULT_BEST_EPOCH_MARGIN = 8


class PairDataset(Dataset):
    def __init__(self, iq: np.ndarray, fft: np.ndarray, y: np.ndarray):
        self.iq = torch.from_numpy(iq)
        self.fft = torch.from_numpy(fft)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.iq[idx], self.fft[idx], self.y[idx]


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_sizes: tuple[int, ...]):
        super().__init__()
        if channels % len(kernel_sizes) != 0:
            raise ValueError("channels must be divisible by the number of kernel sizes")
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


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiScaleTimeCNN(nn.Module):
    def __init__(self, num_classes: int, widths: tuple[int, int, int], blocks_per_stage: tuple[int, int, int]):
        super().__init__()
        kernel_sizes = (3, 9, 17, 33)
        self.stem = nn.Sequential(
            ConvNormAct(2, widths[0], kernel_size=15),
            ConvNormAct(widths[0], widths[0], kernel_size=7),
        )
        stages = []
        in_channels = widths[0]
        for stage_idx, (out_channels, num_blocks) in enumerate(zip(widths, blocks_per_stage)):
            if in_channels != out_channels:
                stages.append(ConvNormAct(in_channels, out_channels, kernel_size=1))
            for _ in range(num_blocks):
                stages.append(MultiScaleResidualBlock(out_channels, kernel_sizes))
            if stage_idx < len(widths) - 1:
                stages.append(nn.MaxPool1d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*stages, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.classifier = ClassificationHead(widths[-1], widths[-1] // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(x))
        return self.classifier(features)


class LargeKernelTimeCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        widths: tuple[int, int, int],
        stem_kernels: tuple[int, int, int],
        blocks_per_stage: tuple[int, int, int],
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(2, widths[0], kernel_size=stem_kernels[0]),
            ConvNormAct(widths[0], widths[0], kernel_size=stem_kernels[1]),
            ConvNormAct(widths[0], widths[0], kernel_size=stem_kernels[2]),
            nn.MaxPool1d(2),
        )
        stages = []
        in_channels = widths[0]
        for stage_idx, (out_channels, num_blocks) in enumerate(zip(widths, blocks_per_stage)):
            if in_channels != out_channels:
                stages.append(ConvNormAct(in_channels, out_channels, kernel_size=1))
            for block_idx in range(num_blocks):
                dilation = 1 if block_idx % 2 == 0 else 2
                stages.append(ResidualConvBlock(out_channels, kernel_size=7, dilation=dilation))
            if stage_idx < len(widths) - 1:
                stages.append(nn.MaxPool1d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*stages, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.classifier = ClassificationHead(widths[-1], widths[-1] // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(x))
        return self.classifier(features)


class LearnableFilterbankFrontEnd(nn.Module):
    def __init__(self, out_channels: int, kernel_sizes: tuple[int, ...]):
        super().__init__()
        if out_channels % len(kernel_sizes) != 0:
            raise ValueError("out_channels must be divisible by the number of filterbank branches")
        branch_channels = out_channels // len(kernel_sizes)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(2, branch_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.mix(combined)


class FilterbankTimeCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        front_channels: int,
        trunk_widths: tuple[int, int, int],
        trunk_blocks: tuple[int, int, int],
        filter_kernels: tuple[int, ...],
    ):
        super().__init__()
        self.front_end = LearnableFilterbankFrontEnd(front_channels, filter_kernels)
        stages = []
        in_channels = front_channels
        for stage_idx, (out_channels, num_blocks) in enumerate(zip(trunk_widths, trunk_blocks)):
            if in_channels != out_channels:
                stages.append(ConvNormAct(in_channels, out_channels, kernel_size=1))
            for block_idx in range(num_blocks):
                dilation = 2**min(block_idx, 2)
                stages.append(ResidualConvBlock(out_channels, kernel_size=5, dilation=dilation))
            if stage_idx < len(trunk_widths) - 1:
                stages.append(nn.MaxPool1d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*stages, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.classifier = ClassificationHead(trunk_widths[-1], trunk_widths[-1] // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(self.front_end(x))
        return self.classifier(features)


class ScalableConvFeatureExtractor1D(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], kernel_sizes: tuple[int, int, int, int] = (9, 7, 5, 3)):
        super().__init__()
        c1, c2, c3, c4 = channels
        k1, k2, k3, k4 = kernel_sizes
        self.net = nn.Sequential(
            nn.Conv1d(2, c1, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            nn.Conv1d(c3, c4, kernel_size=k4, padding=k4 // 2),
            nn.BatchNorm1d(c4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class ScaledGatedMultimodalIQFFTCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        branch_channels: tuple[int, int, int, int],
        fusion_hidden: int,
        gate_hidden: int,
        branch_dropout: float = 0.0,
    ):
        super().__init__()
        self.branch_dropout = branch_dropout
        self.iq_branch = ScalableConvFeatureExtractor1D(branch_channels)
        self.fft_branch = ScalableConvFeatureExtractor1D(branch_channels)
        feature_dim = self.iq_branch.out_dim
        fused_dim = feature_dim * 2

        self.iq_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(feature_dim, num_classes),
        )
        self.fft_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(feature_dim, num_classes),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(fusion_hidden, num_classes),
        )
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )

    def _apply_branch_dropout(self, iq_features: torch.Tensor, fft_features: torch.Tensor):
        if not self.training or self.branch_dropout <= 0.0:
            return iq_features, fft_features

        batch = iq_features.shape[0]
        device = iq_features.device
        keep_iq = (torch.rand(batch, 1, device=device) > self.branch_dropout).float()
        keep_fft = (torch.rand(batch, 1, device=device) > self.branch_dropout).float()
        both_dropped = keep_iq + keep_fft == 0
        if both_dropped.any():
            choose_iq = (torch.rand(batch, 1, device=device) > 0.5).float()
            keep_iq = torch.where(both_dropped, choose_iq, keep_iq)
            keep_fft = torch.where(both_dropped, 1.0 - keep_iq, keep_fft)

        return iq_features * keep_iq, fft_features * keep_fft

    def forward(self, iq: torch.Tensor, fft: torch.Tensor):
        iq_features = self.iq_branch(iq)
        fft_features = self.fft_branch(fft)
        iq_features, fft_features = self._apply_branch_dropout(iq_features, fft_features)
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


def build_model_registry() -> dict[str, dict[str, object]]:
    return {
        "time_baseline": {
            "family": "baseline",
            "scale": "baseline",
            "description": "Original time-domain CNN baseline from the repo.",
            "factory": lambda num_classes: RepresentationCNN(num_classes=num_classes),
        },
        "fft_baseline": {
            "family": "baseline",
            "scale": "reference",
            "description": "Original FFT-domain CNN reference baseline from the repo.",
            "factory": lambda num_classes: RepresentationCNN(num_classes=num_classes),
        },
        "gated": {
            "family": "fusion",
            "scale": "reference",
            "description": "Existing gated IQ+FFT multimodal fusion reference.",
            "factory": lambda num_classes: GatedMultimodalIQFFTCNN(num_classes=num_classes, branch_dropout=BRANCH_DROPOUT),
        },
        "gated_medium": {
            "family": "fusion",
            "scale": "medium",
            "description": "Capacity-matched gated IQ+FFT model sized to compare against multiscale_m.",
            "factory": lambda num_classes: ScaledGatedMultimodalIQFFTCNN(
                num_classes=num_classes,
                branch_channels=(96, 192, 384, 768),
                fusion_hidden=2048,
                gate_hidden=1024,
                branch_dropout=BRANCH_DROPOUT,
            ),
        },
        "gated_large": {
            "family": "fusion",
            "scale": "large",
            "description": "Capacity-matched gated IQ+FFT model sized to compare against multiscale_l.",
            "factory": lambda num_classes: ScaledGatedMultimodalIQFFTCNN(
                num_classes=num_classes,
                branch_channels=(128, 256, 512, 1024),
                fusion_hidden=3072,
                gate_hidden=1536,
                branch_dropout=BRANCH_DROPOUT,
            ),
        },
        "multiscale_s": {
            "family": "multiscale_time",
            "scale": "small",
            "description": "Multi-scale time CNN with parallel 3/9/17/33 kernels and a shallow trunk.",
            "factory": lambda num_classes: MultiScaleTimeCNN(num_classes, widths=(64, 128, 256), blocks_per_stage=(1, 1, 1)),
        },
        "multiscale_m": {
            "family": "multiscale_time",
            "scale": "medium",
            "description": "Multi-scale time CNN with wider stages and extra residual depth.",
            "factory": lambda num_classes: MultiScaleTimeCNN(num_classes, widths=(96, 192, 384), blocks_per_stage=(1, 2, 2)),
        },
        "multiscale_l": {
            "family": "multiscale_time",
            "scale": "large",
            "description": "Largest multi-scale time CNN with the widest stages and deepest residual trunk.",
            "factory": lambda num_classes: MultiScaleTimeCNN(num_classes, widths=(128, 256, 512), blocks_per_stage=(2, 2, 3)),
        },
        "largekernel_s": {
            "family": "large_kernel_time",
            "scale": "small",
            "description": "Time CNN with large early kernels (33/17/9) and a modest residual trunk.",
            "factory": lambda num_classes: LargeKernelTimeCNN(
                num_classes,
                widths=(64, 128, 256),
                stem_kernels=(33, 17, 9),
                blocks_per_stage=(1, 1, 1),
            ),
        },
        "largekernel_m": {
            "family": "large_kernel_time",
            "scale": "medium",
            "description": "Time CNN with wider early kernels (65/33/17) and deeper residual processing.",
            "factory": lambda num_classes: LargeKernelTimeCNN(
                num_classes,
                widths=(96, 192, 384),
                stem_kernels=(65, 33, 17),
                blocks_per_stage=(1, 2, 2),
            ),
        },
        "largekernel_l": {
            "family": "large_kernel_time",
            "scale": "large",
            "description": "Largest large-kernel time CNN with very wide early filters (129/65/33).",
            "factory": lambda num_classes: LargeKernelTimeCNN(
                num_classes,
                widths=(128, 256, 512),
                stem_kernels=(129, 65, 33),
                blocks_per_stage=(2, 2, 3),
            ),
        },
        "filterbank_s": {
            "family": "filterbank_time",
            "scale": "small",
            "description": "Time CNN with a learnable filterbank front end using wide kernels before a small residual trunk.",
            "factory": lambda num_classes: FilterbankTimeCNN(
                num_classes,
                front_channels=96,
                trunk_widths=(96, 192, 256),
                trunk_blocks=(1, 1, 1),
                filter_kernels=(17, 33, 65),
            ),
        },
        "filterbank_m": {
            "family": "filterbank_time",
            "scale": "medium",
            "description": "Time CNN with a larger learnable filterbank and deeper residual trunk.",
            "factory": lambda num_classes: FilterbankTimeCNN(
                num_classes,
                front_channels=144,
                trunk_widths=(144, 288, 384),
                trunk_blocks=(1, 2, 2),
                filter_kernels=(17, 33, 65),
            ),
        },
        "filterbank_l": {
            "family": "filterbank_time",
            "scale": "large",
            "description": "Largest learnable-filterbank time CNN with more filters and the deepest trunk.",
            "factory": lambda num_classes: FilterbankTimeCNN(
                num_classes,
                front_channels=192,
                trunk_widths=(192, 384, 512),
                trunk_blocks=(2, 2, 3),
                filter_kernels=(17, 33, 65, 129),
            ),
        },
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_single_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    return make_tensor_loader(X, y, shuffle=shuffle, use_cuda=use_cuda, batch_size=batch_size)


def make_pair_loader(iq: np.ndarray, fft: np.ndarray, y: np.ndarray, shuffle: bool, use_cuda: bool, batch_size: int):
    dataset = PairDataset(iq, fft, y)
    return make_dataset_loader(dataset, shuffle=shuffle, use_cuda=use_cuda, batch_size=batch_size)


def compute_gated_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    final_loss = criterion(outputs["final_logits"], targets)
    iq_loss = criterion(outputs["iq_logits"], targets)
    fft_loss = criterion(outputs["fft_logits"], targets)
    fusion_loss = criterion(outputs["fusion_logits"], targets)
    return final_loss + AUX_LOSS_WEIGHT * (iq_loss + fft_loss + fusion_loss)


def run_single_epoch(model, loader, criterion, optimizer, scaler, device, use_cuda: bool, training: bool):
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

    average_loss = total_loss / max(total_examples, 1)
    average_acc = total_correct / max(total_examples, 1)
    return average_loss, average_acc


def run_gated_epoch(model, loader, criterion, optimizer, scaler, device, use_cuda: bool, training: bool):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    gate_sum = torch.zeros(3, dtype=torch.float64)

    for iq_batch, fft_batch, y_batch in loader:
        iq_batch = iq_batch.to(device, non_blocking=use_cuda)
        fft_batch = fft_batch.to(device, non_blocking=use_cuda)
        y_batch = y_batch.to(device, non_blocking=use_cuda)

        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iq_batch, fft_batch)
                loss = compute_gated_loss(outputs, y_batch, criterion)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        logits = outputs["final_logits"]
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_examples += y_batch.size(0)
        gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)

    average_loss = total_loss / max(total_examples, 1)
    average_acc = total_correct / max(total_examples, 1)
    gate_mean = (gate_sum / max(total_examples, 1)).tolist()
    return average_loss, average_acc, gate_mean


def evaluate_single(model, loader, y_true: np.ndarray, class_names: list[str], device, use_cuda: bool):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(X_batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return classification_metrics(y_true, y_pred, class_names)


def evaluate_gated(model, loader, y_true: np.ndarray, class_names: list[str], device, use_cuda: bool):
    model.eval()
    preds = []
    gate_sum = torch.zeros(3, dtype=torch.float64)
    total_examples = 0
    with torch.no_grad():
        for iq_batch, fft_batch, _ in loader:
            iq_batch = iq_batch.to(device, non_blocking=use_cuda)
            fft_batch = fft_batch.to(device, non_blocking=use_cuda)
            with autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(iq_batch, fft_batch)
            preds.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
            gate_sum += outputs["gate_weights"].detach().cpu().double().sum(dim=0)
            total_examples += iq_batch.size(0)
    y_pred = np.concatenate(preds)
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["test_gate_weights"] = (gate_sum / max(total_examples, 1)).tolist()
    return metrics


def make_single_state(model, optimizer, scheduler, scaler) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def build_training_summary(history: list[dict], best_epoch: int, stop_reason: str, converged: bool) -> dict:
    val_losses = [float(item["val_loss"]) for item in history]
    best_val_loss = min(val_losses) if val_losses else None
    return {
        "epochs_ran": len(history),
        "best_epoch": int(best_epoch),
        "best_val_loss": best_val_loss,
        "stop_reason": stop_reason,
        "converged": bool(converged),
        "history_tail": history[-5:],
    }


def should_stop(history: list[dict], args, plateau_count: int) -> tuple[bool, int, str | None]:
    current_epoch = len(history)
    if current_epoch < args.min_epochs:
        return False, plateau_count, None
    if current_epoch >= args.max_epochs:
        return True, plateau_count, "reached_max_epochs"

    best_idx = min(range(len(history)), key=lambda idx: history[idx]["val_loss"])
    best_epoch = best_idx + 1
    if current_epoch - best_epoch <= args.best_epoch_margin:
        return False, 0, None

    chunk_start = max(0, current_epoch - args.epoch_chunk)
    previous_losses = [entry["val_loss"] for entry in history[:chunk_start]]
    recent_losses = [entry["val_loss"] for entry in history[chunk_start:]]
    if not previous_losses:
        return False, 0, None

    previous_best = min(previous_losses)
    recent_best = min(recent_losses)
    if previous_best - recent_best > args.min_delta:
        return False, 0, None

    plateau_count += 1
    if plateau_count >= args.plateau_chunks:
        return True, plateau_count, "validation_loss_plateau"
    return False, plateau_count, None


def train_single_model(model_name: str, model_factory, train_loader, val_loader, test_loader, y_test, class_names, args):
    device, use_cuda = configure_device()
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    checkpoint = None if args.restart else load_checkpoint(checkpoint_path)

    model = model_factory(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        threshold=1e-3,
        min_lr=1e-5,
    )
    scaler = GradScaler("cuda", enabled=use_cuda)

    history: list[dict] = []
    best_state = None
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    plateau_count = 0
    stop_reason = "completed"

    if checkpoint is not None:
        model.load_state_dict(checkpoint["train_state"]["model"])
        optimizer.load_state_dict(checkpoint["train_state"]["optimizer"])
        scheduler.load_state_dict(checkpoint["train_state"]["scheduler"])
        scaler.load_state_dict(checkpoint["train_state"]["scaler"])
        history = checkpoint["history"]
        best_state = checkpoint["best_state"]
        best_val_loss = checkpoint["best_val_loss"]
        best_val_acc = checkpoint["best_val_acc"]
        best_epoch = checkpoint["best_epoch"]
        plateau_count = checkpoint.get("plateau_count", 0)
        stop_reason = checkpoint.get("stop_reason", stop_reason)
        if checkpoint.get("converged", False):
            model.load_state_dict(best_state)
            test_metrics = evaluate_single(model, test_loader, y_test, class_names, device, use_cuda)
            return {
                "model_name": model_name,
                "checkpoint_path": str(checkpoint_path),
                "adaptive_training": build_training_summary(history, best_epoch, stop_reason, True),
                "best_val_acc": float(best_val_acc),
                **test_metrics,
            }

    target_epoch = min(args.max_epochs, max(args.initial_epochs, len(history) + args.epoch_chunk))
    while len(history) < target_epoch:
        epoch = len(history) + 1
        train_loss, train_acc = run_single_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_cuda, True
        )
        val_loss, val_acc = run_single_epoch(
            model, val_loader, criterion, optimizer, scaler, device, use_cuda, False
        )
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": current_lr,
            }
        )
        print(
            f"[{model_name}] Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | lr {current_lr:.6f}"
        )
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_epoch = epoch
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        converged, plateau_count, detected_stop_reason = should_stop(history, args, plateau_count)
        stop_reason = detected_stop_reason or stop_reason
        save_checkpoint(
            checkpoint_path,
            {
                "history": history,
                "best_state": best_state,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "plateau_count": plateau_count,
                "stop_reason": stop_reason,
                "converged": converged,
                "train_state": make_single_state(model, optimizer, scheduler, scaler),
            },
        )
        if converged:
            break
        if len(history) >= target_epoch:
            target_epoch = min(args.max_epochs, len(history) + args.epoch_chunk)

    if stop_reason == "completed" and len(history) >= args.max_epochs:
        stop_reason = "reached_max_epochs"
    model.load_state_dict(best_state)
    test_metrics = evaluate_single(model, test_loader, y_test, class_names, device, use_cuda)
    converged = stop_reason in {"validation_loss_plateau", "reached_max_epochs"}
    save_checkpoint(
        checkpoint_path,
        {
            "history": history,
            "best_state": best_state,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "plateau_count": plateau_count,
            "stop_reason": stop_reason,
            "converged": converged,
            "train_state": make_single_state(model, optimizer, scheduler, scaler),
        },
    )
    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "adaptive_training": build_training_summary(history, best_epoch, stop_reason, converged),
        "best_val_acc": float(best_val_acc),
        **test_metrics,
    }


def train_gated_model(model_name: str, model_factory, train_loader, val_loader, test_loader, y_test, class_names, args):
    device, use_cuda = configure_device()
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    checkpoint = None if args.restart else load_checkpoint(checkpoint_path)

    model = model_factory(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        threshold=1e-3,
        min_lr=1e-5,
    )
    scaler = GradScaler("cuda", enabled=use_cuda)

    history: list[dict] = []
    best_state = None
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    best_val_gate = None
    plateau_count = 0
    stop_reason = "completed"

    if checkpoint is not None:
        model.load_state_dict(checkpoint["train_state"]["model"])
        optimizer.load_state_dict(checkpoint["train_state"]["optimizer"])
        scheduler.load_state_dict(checkpoint["train_state"]["scheduler"])
        scaler.load_state_dict(checkpoint["train_state"]["scaler"])
        history = checkpoint["history"]
        best_state = checkpoint["best_state"]
        best_val_loss = checkpoint["best_val_loss"]
        best_val_acc = checkpoint["best_val_acc"]
        best_epoch = checkpoint["best_epoch"]
        best_val_gate = checkpoint.get("best_val_gate")
        plateau_count = checkpoint.get("plateau_count", 0)
        stop_reason = checkpoint.get("stop_reason", stop_reason)
        if checkpoint.get("converged", False):
            model.load_state_dict(best_state)
            test_metrics = evaluate_gated(model, test_loader, y_test, class_names, device, use_cuda)
            return {
                "model_name": model_name,
                "checkpoint_path": str(checkpoint_path),
                "adaptive_training": build_training_summary(history, best_epoch, stop_reason, True),
                "best_val_acc": float(best_val_acc),
                "best_val_gate_weights": best_val_gate,
                **test_metrics,
            }

    target_epoch = min(args.max_epochs, max(args.initial_epochs, len(history) + args.epoch_chunk))
    while len(history) < target_epoch:
        epoch = len(history) + 1
        train_loss, train_acc, train_gate = run_gated_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_cuda, True
        )
        val_loss, val_acc, val_gate = run_gated_epoch(
            model, val_loader, criterion, optimizer, scaler, device, use_cuda, False
        )
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": current_lr,
                "train_gate_weights": [float(x) for x in train_gate],
                "val_gate_weights": [float(x) for x in val_gate],
            }
        )
        print(
            f"[{model_name}] Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | gates {[round(x, 3) for x in val_gate]} | lr {current_lr:.6f}"
        )
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_epoch = epoch
            best_val_gate = [float(x) for x in val_gate]
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        converged, plateau_count, detected_stop_reason = should_stop(history, args, plateau_count)
        stop_reason = detected_stop_reason or stop_reason
        save_checkpoint(
            checkpoint_path,
            {
                "history": history,
                "best_state": best_state,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "best_val_gate": best_val_gate,
                "plateau_count": plateau_count,
                "stop_reason": stop_reason,
                "converged": converged,
                "train_state": make_single_state(model, optimizer, scheduler, scaler),
            },
        )
        if converged:
            break
        if len(history) >= target_epoch:
            target_epoch = min(args.max_epochs, len(history) + args.epoch_chunk)

    if stop_reason == "completed" and len(history) >= args.max_epochs:
        stop_reason = "reached_max_epochs"
    model.load_state_dict(best_state)
    test_metrics = evaluate_gated(model, test_loader, y_test, class_names, device, use_cuda)
    converged = stop_reason in {"validation_loss_plateau", "reached_max_epochs"}
    save_checkpoint(
        checkpoint_path,
        {
            "history": history,
            "best_state": best_state,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "best_val_gate": best_val_gate,
            "plateau_count": plateau_count,
            "stop_reason": stop_reason,
            "converged": converged,
            "train_state": make_single_state(model, optimizer, scheduler, scaler),
        },
    )
    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "adaptive_training": build_training_summary(history, best_epoch, stop_reason, converged),
        "best_val_acc": float(best_val_acc),
        "best_val_gate_weights": best_val_gate,
        **test_metrics,
    }


def run_experiment(args) -> dict:
    set_global_seed(SEED)
    start = time.perf_counter()
    dataset = load_waveform_dataset()
    class_names = dataset["class_names"]

    iq_train = dataset["iq_train"]
    iq_val = dataset["iq_val"]
    iq_test = dataset["iq_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]

    fft_train = to_fft(iq_train)
    fft_val = to_fft(iq_val)
    fft_test = to_fft(iq_test)

    _, use_cuda = configure_device()
    batch_size = args.batch_size or dataset["batch_size"]

    train_loader_time = make_single_loader(iq_train, y_train, True, use_cuda, batch_size)
    val_loader_time = make_single_loader(iq_val, y_val, False, use_cuda, batch_size)
    test_loader_time = make_single_loader(iq_test, y_test, False, use_cuda, batch_size)

    train_loader_fft = make_single_loader(fft_train, y_train, True, use_cuda, batch_size)
    val_loader_fft = make_single_loader(fft_val, y_val, False, use_cuda, batch_size)
    test_loader_fft = make_single_loader(fft_test, y_test, False, use_cuda, batch_size)

    train_loader_pair = make_pair_loader(iq_train, fft_train, y_train, True, use_cuda, batch_size)
    val_loader_pair = make_pair_loader(iq_val, fft_val, y_val, False, use_cuda, batch_size)
    test_loader_pair = make_pair_loader(iq_test, fft_test, y_test, False, use_cuda, batch_size)

    model_registry = build_model_registry()

    results = {
        "experiment": "waveform_time_capacity",
        "dataset": {
            "name": dataset["name"],
            "train_examples": int(iq_train.shape[0]),
            "val_examples": int(iq_val.shape[0]),
            "test_examples": int(iq_test.shape[0]),
            "signal_length": int(iq_train.shape[-1]),
            "num_classes": len(class_names),
            "class_names": class_names,
            "batch_size": int(batch_size),
        },
        "training_policy": {
            "initial_epochs": int(args.initial_epochs),
            "epoch_chunk": int(args.epoch_chunk),
            "min_epochs": int(args.min_epochs),
            "max_epochs": int(args.max_epochs),
            "plateau_chunks": int(args.plateau_chunks),
            "min_delta": float(args.min_delta),
            "best_epoch_margin": int(args.best_epoch_margin),
            "optimizer": "AdamW",
            "lr_scheduler": "ReduceLROnPlateau",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "results": {},
    }

    selected_models = set(args.models)
    for model_name in args.models:
        if model_name not in selected_models:
            continue
        spec = model_registry[model_name]
        print(f"\n=== waveform_family: {model_name} ===")

        if model_name == "fft_baseline":
            temp_model = spec["factory"](len(class_names))
            run_result = train_single_model(
                model_name,
                spec["factory"],
                train_loader_fft,
                val_loader_fft,
                test_loader_fft,
                y_test,
                class_names,
                args,
            )
        elif model_name in {"gated", "gated_medium", "gated_large"}:
            temp_model = spec["factory"](len(class_names))
            run_result = train_gated_model(
                model_name,
                spec["factory"],
                train_loader_pair,
                val_loader_pair,
                test_loader_pair,
                y_test,
                class_names,
                args,
            )
        else:
            temp_model = spec["factory"](len(class_names))
            run_result = train_single_model(
                model_name,
                spec["factory"],
                train_loader_time,
                val_loader_time,
                test_loader_time,
                y_test,
                class_names,
                args,
            )

        parameter_count = count_trainable_parameters(temp_model)
        results["results"][model_name] = {
            "family": spec["family"],
            "scale": spec["scale"],
            "description": spec["description"],
            "num_trainable_parameters": int(parameter_count),
            **run_result,
        }

    results["runtime_seconds"] = time.perf_counter() - start
    save_json(RESULTS_PATH, results)
    print(f"\nSaved results to {RESULTS_PATH}")
    print(json.dumps(results, indent=2))
    return results


def parse_args():
    model_registry = build_model_registry()
    parser = argparse.ArgumentParser(description="Waveform-family follow-up: test whether a wider/deeper time CNN can close the FFT gap.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(model_registry.keys()),
        choices=list(model_registry.keys()),
        help="Subset of models to train.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--initial-epochs", type=int, default=DEFAULT_INITIAL_EPOCHS)
    parser.add_argument("--epoch-chunk", type=int, default=DEFAULT_EPOCH_CHUNK)
    parser.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--plateau-chunks", type=int, default=DEFAULT_PLATEAU_CHUNKS)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--best-epoch-margin", type=int, default=DEFAULT_BEST_EPOCH_MARGIN)
    parser.add_argument("--restart", action="store_true", help="Ignore any saved checkpoint state and start fresh.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    run_experiment(args)


if __name__ == "__main__":
    main()
