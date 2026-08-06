"""Log-Mel front end — the single, portable feature definition.

CRITICAL: the ESP32 fixed-point front end (see phase4_deploy/esp32/README.md) must
reproduce these exact parameters, or on-device accuracy will collapse even if the
network is perfect. Keep n_fft/hop/n_mels/f_min/f_max identical on both sides.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio


class LogMel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f = cfg.features
        self.sample_rate = cfg.audio["sample_rate"]
        self.log_offset = float(f["log_offset"])
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(f["n_fft"]),
            win_length=int(f["win_length"]),
            hop_length=int(f["hop_length"]),
            f_min=float(f["f_min"]),
            f_max=float(f["f_max"]),
            n_mels=int(f["n_mels"]),
            power=2.0,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] or [B, 1, T] -> logmel [B, 1, n_mels, frames]."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        mel = self.mel(wav)                       # [B, n_mels, frames]
        logmel = torch.log(mel + self.log_offset)
        # per-utterance standardisation (mean/var normalisation)
        mu = logmel.mean(dim=(1, 2), keepdim=True)
        sd = logmel.std(dim=(1, 2), keepdim=True) + 1e-5
        logmel = (logmel - mu) / sd
        return logmel.unsqueeze(1)                 # add channel dim
