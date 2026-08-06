"""Phase 2.5 — Student CNN for ESP32-S3.

Depthwise-separable CNN operating on a [1, n_mels, frames] log-Mel spectrogram.
width=1.0 -> ~1.2M params; width=0.75 -> ~0.6M. INT8-friendly (ReLU6, no exotic ops)
so it maps cleanly onto ESP-NN / TFLite Micro kernels.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..mixstyle import MixStyle


def _round(ch: float, divisor: int = 8) -> int:
    new = max(divisor, int(ch + divisor / 2) // divisor * divisor)
    if new < 0.9 * ch:
        new += divisor
    return int(new)


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class StudentCNN(nn.Module):
    """Returns (logits, embedding). Embedding feeds the contrastive loss."""

    def __init__(self, num_classes=3, width=1.0, dropout=0.1,
                 mixstyle: MixStyle | None = None):
        super().__init__()
        c = [_round(w * width) for w in (32, 64, 128, 128, 256)]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[0]), nn.ReLU6(inplace=True))
        self.mixstyle = mixstyle           # applied after early blocks (style lives there)
        self.block1 = DepthwiseSeparable(c[0], c[1], stride=2)
        self.block2 = DepthwiseSeparable(c[1], c[2], stride=2)
        self.block3 = DepthwiseSeparable(c[2], c[3], stride=1)
        self.block4 = DepthwiseSeparable(c[3], c[4], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_dim = c[4]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c[4], num_classes)

    def forward(self, x, return_embedding: bool = False):
        x = self.stem(x)
        x = self.block1(x)
        if self.mixstyle is not None:
            x = self.mixstyle(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        emb = self.pool(x).flatten(1)
        logits = self.head(self.dropout(emb))
        if return_embedding:
            return logits, emb
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
