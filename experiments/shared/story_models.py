from __future__ import annotations

import torch
from torch import nn


class ConvFeatureExtractor1D(nn.Module):
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


def build_classifier_head(num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.35),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )


class RepresentationCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = ConvFeatureExtractor1D()
        self.classifier = build_classifier_head(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class GatedMultimodalIQFFTCNN(nn.Module):
    def __init__(self, num_classes: int, branch_dropout: float = 0.0):
        super().__init__()
        self.branch_dropout = branch_dropout
        self.iq_branch = ConvFeatureExtractor1D()
        self.fft_branch = ConvFeatureExtractor1D()
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


class ExpertCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = ConvFeatureExtractor1D()
        self.classifier = build_classifier_head(num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features.unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FrozenExpertResidualFusion(nn.Module):
    def __init__(self, iq_expert: ExpertCNN, fft_expert: ExpertCNN, num_classes: int, delta_scale: float):
        super().__init__()
        self.iq_expert = iq_expert
        self.fft_expert = fft_expert
        self.delta_scale = delta_scale
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
        delta = self.delta_scale * torch.tanh(self.delta_head(hidden))
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
