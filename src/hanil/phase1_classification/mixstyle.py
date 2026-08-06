"""Phase 2.3 — MixStyle domain generalisation (Experiment 2).

Mixes feature-map channel statistics (mean/std) between samples in a batch so the
backbone stops relying on device-specific style (stethoscope vs smartphone) and
learns clinically relevant content. Reference: Zhou et al., ICLR 2021.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)
        self._activated = True

    def set_activated(self, flag: bool) -> None:
        self._activated = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]. Active in training only.
        if not self.training or not self._activated or torch.rand(1).item() > self.p:
            return x
        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        lam = self.beta.sample((b, 1, 1, 1)).to(x.device)
        perm = torch.randperm(b, device=x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sig_mix = sig * lam + sig[perm] * (1 - lam)
        return x_norm * sig_mix + mu_mix
