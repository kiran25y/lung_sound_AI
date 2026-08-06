"""Phase 2.3 — Stethoscope-guided supervised contrastive loss (Experiment 2).

Pulls same-class embeddings together and pushes different-class apart, *across*
devices, using the clean stethoscope domain as the anchor for the noisier
smartphone domain. Complements MixStyle for cross-device generalisation.
Reference: supervised contrastive learning (Khosla 2020) adapted with a
domain-anchoring weight.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class StethoGuidedSupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, anchor_domain: int = 0,
                 anchor_boost: float = 1.5):
        super().__init__()
        self.t = temperature
        self.anchor_domain = anchor_domain      # 0 = stethoscope
        self.anchor_boost = anchor_boost

    def forward(self, feats: torch.Tensor, labels: torch.Tensor,
                domains: torch.Tensor) -> torch.Tensor:
        """feats [B, D] (will be L2-normalised), labels [B], domains [B]."""
        feats = F.normalize(feats, dim=1)
        b = feats.size(0)
        sim = feats @ feats.t() / self.t
        # numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim)
        self_mask = torch.eye(b, device=feats.device, dtype=torch.bool)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        # weight positives that involve a stethoscope anchor more heavily
        anchor = (domains == self.anchor_domain).float()
        w = 1.0 + (self.anchor_boost - 1.0) * anchor.unsqueeze(1)  # [B,1]
        denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = sim - torch.log(denom)

        pos_counts = pos_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (pos_mask * log_prob * w).sum(dim=1) / pos_counts
        loss = -mean_log_prob_pos
        # ignore rows with no positive pair
        valid = pos_mask.sum(dim=1) > 0
        return loss[valid].mean() if valid.any() else feats.sum() * 0.0
