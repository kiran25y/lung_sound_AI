"""Class-imbalance-aware losses. 'normal' dominates lung-sound datasets, so plain
cross-entropy learns to predict 'normal'. Focal / class-balanced CE fix this."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(cfg, class_counts: np.ndarray, device) -> nn.Module:
    weight = None
    if cfg.train.get("class_weights") == "auto":
        inv = 1.0 / np.clip(class_counts, 1, None)
        w = inv / inv.sum() * len(class_counts)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    if cfg.train.get("loss", "focal") == "focal":
        return FocalLoss(gamma=float(cfg.train.get("focal_gamma", 2.0)), weight=weight)
    return nn.CrossEntropyLoss(weight=weight)
