"""Phase 2.2 — Noise-robust augmentation (Experiment 1).

Mixes clean lung sound with an open-source noise clip at a target SNR sampled
uniformly from cfg.noise.snr_db (default 5–20 dB), per clip, per epoch.

Licensing guard: when cfg.noise.tier == 'shippable', only commercial-safe corpora
(MUSAN noise, self-recorded, DEMAND) may be used. Research-tier corpora (ESC-50,
UrbanSound8K, WHAM!, DESED) raise unless tier == 'research'. See LICENSING.md.
"""
from __future__ import annotations
import glob
import os
import random
from typing import List
import torch
import torchaudio

SHIPPABLE_SOURCES = {"musan_noise", "self_recorded", "demand"}
RESEARCH_SOURCES = {"esc50", "urbansound8k", "wham", "desed_synthetic"}


class NoiseMixer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg.audio["sample_rate"]
        self.enable = bool(cfg.noise["enable"])
        self.prob = float(cfg.noise["prob"])
        self.snr_low, self.snr_high = cfg.noise["snr_db"]
        self.tier = cfg.noise.get("tier", "shippable")
        sources = cfg.noise.get("sources", ["musan_noise", "self_recorded"])
        self._check_licensing(sources)
        self.files = self._index_noise_files(cfg, sources)
        if self.enable and not self.files:
            raise FileNotFoundError(
                "Noise augmentation enabled but no noise wavs found. "
                "Populate data/noise/<source>/ (see data/README.md).")

    def _check_licensing(self, sources: List[str]) -> None:
        if self.tier == "shippable":
            bad = [s for s in sources if s in RESEARCH_SOURCES]
            if bad:
                raise ValueError(
                    f"noise.tier='shippable' but research-only sources requested: {bad}. "
                    f"Either drop them or set noise.tier='research' (do NOT ship that model).")

    def _index_noise_files(self, cfg, sources) -> List[str]:
        roots = {
            "musan_noise": cfg.paths["noise_dir"],
            "self_recorded": cfg.paths.get("self_noise_dir", "data/noise/self_recorded"),
            "desed_synthetic": cfg.paths.get("desed_dir", "data/noise/desed_synthetic"),
        }
        files: List[str] = []
        for s in sources:
            root = roots.get(s, os.path.join("data/noise", s))
            files += glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
        return files

    def _load_noise(self, num_samples: int) -> torch.Tensor:
        path = random.choice(self.files)
        from ...common.audio import _read_audio
        noise, sr = _read_audio(path)
        if noise.shape[0] > 1:
            noise = noise.mean(0, keepdim=True)
        if sr != self.sr:
            noise = torchaudio.functional.resample(noise, sr, self.sr)
        t = noise.shape[-1]
        if t < num_samples:                     # tile short clips
            reps = num_samples // t + 1
            noise = noise.repeat(1, reps)
        start = random.randint(0, noise.shape[-1] - num_samples)
        return noise[..., start:start + num_samples]

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [1, T] clean -> [1, T] noisy (or unchanged with prob 1-p)."""
        if not self.enable or random.random() > self.prob:
            return wav
        noise = self._load_noise(wav.shape[-1])
        snr_db = random.uniform(self.snr_low, self.snr_high)
        sig_p = wav.pow(2).mean().clamp_min(1e-12)
        noise_p = noise.pow(2).mean().clamp_min(1e-12)
        scale = torch.sqrt(sig_p / (noise_p * (10 ** (snr_db / 10.0))))
        mixed = wav + scale * noise
        peak = mixed.abs().max().clamp_min(1e-8)
        return mixed / peak if peak > 1.0 else mixed
