"""SpecAugment — cheap frequency/time masking on the log-Mel spectrogram."""
from __future__ import annotations
import random
import torch


class SpecAugment:
    def __init__(self, cfg):
        s = cfg.specaugment
        self.enable = bool(s["enable"])
        self.freq_masks = int(s["freq_mask_param"]) if False else int(s["freq_masks"])
        self.freq_mask_param = int(s["freq_mask_param"])
        self.time_masks = int(s["time_masks"])
        self.time_mask_param = int(s["time_mask_param"])

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """spec [B, 1, F, T] -> masked in place-safe copy (training only)."""
        if not self.enable:
            return spec
        spec = spec.clone()
        b, _, F, T = spec.shape
        for i in range(b):
            for _ in range(self.freq_masks):
                f = random.randint(0, self.freq_mask_param)
                f0 = random.randint(0, max(0, F - f))
                spec[i, :, f0:f0 + f, :] = 0.0
            for _ in range(self.time_masks):
                t = random.randint(0, self.time_mask_param)
                t0 = random.randint(0, max(0, T - t))
                spec[i, :, :, t0:t0 + t] = 0.0
        return spec
