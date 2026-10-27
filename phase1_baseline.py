#!/usr/bin/env python3
"""STEP 1 - Baseline 3-class lung-sound classifier (Normal / Crackle / Wheeze).

A single, self-contained file you can run TODAY to get your first baseline number.
No robustness tricks (no noise mixing, MixStyle, contrastive, or distillation) -
just a clean classifier so you can see whether the data, front end, splits and
metrics are all sane before adding complexity.

It depends only on: torch, torchaudio, numpy, pandas  (no need to `pip install -e .`).

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
# 1) Try it with zero real data (auto-generates a tiny synthetic set):
python phase1_baseline.py --synthetic --epochs 5

# 2) Real data - point it at your manifest (see data/README.md for columns):
python phase1_baseline.py --manifest data/manifest.csv --epochs 60 --out runs/baseline

The manifest is a CSV with columns:
    filepath, patient_id, label(normal|crackle|wheeze), hospital(seoul|sungbook), device(stethoscope|smartphone)

WHAT TO LOOK AT FIRST: the printed confusion matrix and per-class sensitivity on
`external_test` (Sungbook). Accuracy alone is misleading when 'normal' dominates.
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

# Class set is chosen at runtime via --num-classes (3 = normal/crackle/wheeze;
# 4 = + 'both', the ICBHI-standard scheme). set_classes() reassigns these globals.
LABELS = ["normal", "crackle", "wheeze"]
L2I = {l: i for i, l in enumerate(LABELS)}


def set_classes(num_classes: int):
    global LABELS, L2I
    if num_classes == 2:
        LABELS = ["normal", "abnormal"]
    elif num_classes == 4:
        LABELS = ["normal", "crackle", "wheeze", "both"]
    else:
        LABELS = ["normal", "crackle", "wheeze"]
    L2I = {l: i for i, l in enumerate(LABELS)}
    return LABELS


# ── Multi-label mode ─────────────────────────────────────────────────────────
# Two INDEPENDENT binary heads: crackle and wheeze (sigmoid + BCE). A clip's raw
# label maps to a 2-vector; 'both' is exactly [1,1] with NO duplication.
MULTILABEL = False
ML_LABELS = ["crackle", "wheeze"]
RAW2MULTI = {"normal": (0.0, 0.0), "crackle": (1.0, 0.0),
             "wheeze": (0.0, 1.0), "both": (1.0, 1.0)}


def set_multilabel(flag: bool = True):
    """Switch to independent crackle/wheeze binary heads."""
    global MULTILABEL, LABELS, L2I
    MULTILABEL = flag
    if flag:
        LABELS = list(ML_LABELS)          # ["crackle", "wheeze"] -> 2 output logits
        L2I = {l: i for i, l in enumerate(LABELS)}
    return MULTILABEL


def encode_target(label: str):
    """Row label -> training target. Multi-label: float [crackle, wheeze]. Else: int."""
    if MULTILABEL:
        return torch.tensor(RAW2MULTI[str(label).lower().strip()], dtype=torch.float32)
    return L2I[label]


SR = 16000
CLIP_S = 4.0            # 4 s clips (Hanil: consistently beat 2 s)


# ─────────────────────────── front end (log-Mel) ────────────────────────────
class LogMel(nn.Module):
    """64-bin log-Mel front end, driven by feature_config.FeatureConfig (single source of
    truth). Firmware must reproduce feature_config.logmel_numpy() to match this exactly."""

    def __init__(self, cfg=None):
        super().__init__()
        import feature_config as fcfg
        self.cfg = cfg or fcfg.DEFAULT
        c = self.cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=c.sample_rate, n_fft=c.n_fft, win_length=c.win_length,
            hop_length=c.hop_length, f_min=c.f_min, f_max=c.f_max, n_mels=c.n_mels,
            power=c.power, norm=c.mel_norm, mel_scale=c.mel_scale,
            center=c.center, pad_mode=c.pad_mode)

    def forward(self, wav):                       # wav [B, T] -> [B, 1, n_mels, frames]
        logmel = torch.log(self.mel(wav) + self.cfg.log_offset)
        mu = logmel.mean(dim=(1, 2), keepdim=True)
        sd = logmel.std(dim=(1, 2), keepdim=True) + self.cfg.norm_eps
        return ((logmel - mu) / sd).unsqueeze(1)


# ─────────────────────── noise augmentation (Exp 1) ─────────────────────────
class NoiseMixer:
    """Mix a random noise clip into the waveform at an SNR drawn uniformly from
    snr_db, with probability `prob`. Accepts ONE or SEVERAL noise directories, so you
    can combine MUSAN + DESED (domestic events) + your own real handling/contact/room
    recordings (Codex: synthetic must be combined with real artifacts). Files are
    pooled across all dirs and drawn source-balanced-ish by pooling."""

    def __init__(self, noise_dir, prob=0.5, snr_db=(5, 20)):
        import glob
        dirs = noise_dir if isinstance(noise_dir, (list, tuple)) else [noise_dir]
        self.files, by_dir = [], {}
        for d in dirs:
            fs = glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
            by_dir[d] = len(fs); self.files += fs
        self.prob = prob
        self.snr_low, self.snr_high = snr_db
        if not self.files:
            raise FileNotFoundError(f"no .wav noise files under {dirs}")
        print(f"[aug] noise mixer: {len(self.files)} clips from {len(dirs)} source(s) "
              f"{by_dir}, p={prob}, SNR {snr_db} dB")

    def __call__(self, wav):                        # wav [T] float tensor
        import random
        if random.random() > self.prob:
            return wav
        try:
            noise, _ = load_audio(random.choice(self.files))
        except Exception:
            return wav
        noise = noise.mean(0) if noise.dim() > 1 else noise
        T = wav.shape[-1]
        if noise.shape[-1] < T:
            reps = T // noise.shape[-1] + 1
            noise = noise.repeat(reps)
        start = random.randint(0, noise.shape[-1] - T)
        noise = noise[start:start + T]
        sig_p = wav.pow(2).mean().clamp_min(1e-12)
        noise_p = noise.pow(2).mean().clamp_min(1e-12)
        snr = random.uniform(self.snr_low, self.snr_high)
        scale = torch.sqrt(sig_p / (noise_p * (10 ** (snr / 10.0))))
        mixed = wav + scale * noise
        peak = mixed.abs().max().clamp_min(1e-8)
        return mixed / peak if peak > 1.0 else mixed


def spec_augment(spec, freq_masks=2, freq_w=8, time_masks=2, time_w=16):
    """Frequency/time masking on a log-Mel batch [B,1,F,T]. Training only."""
    import random
    spec = spec.clone()
    B, _, F, T = spec.shape
    for i in range(B):
        for _ in range(freq_masks):
            f = random.randint(0, freq_w); f0 = random.randint(0, max(0, F - f))
            spec[i, :, f0:f0 + f, :] = 0.0
        for _ in range(time_masks):
            t = random.randint(0, time_w); t0 = random.randint(0, max(0, T - t))
            spec[i, :, :, t0:t0 + t] = 0.0
    return spec


# ─────────────────── MixStyle (training-only domain aug) ─────────────────────
class MixStyle(nn.Module):
    """Mixes per-sample channel statistics (mean/std) across the batch so the model
    relies less on device/site-specific feature statistics. Identity at eval, so it adds
    NOTHING to the exported model — deployment-neutral."""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__(); self.p = p; self.alpha = alpha; self.eps = eps

    def forward(self, x):
        if not self.training or x.size(0) < 2 or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        x_n = (x - mu) / sig
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        return x_n * (lam * sig + (1 - lam) * sig[perm]) + (lam * mu + (1 - lam) * mu[perm])


# ─────────────────────────── model (small CNN) ──────────────────────────────
class DWSep(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class BaselineCNN(nn.Module):
    """Depthwise-separable CNN. ESP32-friendly (ReLU6, no exotic ops).
    width scales channels: 1.0 ~65k params, 2.0 ~0.25M, 4.0 ~1M."""

    def __init__(self, num_classes=3, dropout=0.1, width=1.0, mixstyle=False):
        super().__init__()
        c = [max(8, int(round(w * width))) for w in (32, 64, 128, 128, 256)]
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(c[0]), nn.ReLU6(True))
        self.ms = MixStyle() if mixstyle else nn.Identity()
        self.body = nn.Sequential(
            DWSep(c[0], c[1], 2), DWSep(c[1], c[2], 2),
            DWSep(c[2], c[3], 1), DWSep(c[3], c[4], 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c[4], num_classes)

    def forward(self, x):
        x = self.body(self.ms(self.stem(x)))
        return self.head(self.drop(self.pool(x).flatten(1)))


# ─────────────────── DSCNN-TCN (MCU deployment target) ──────────────────────
class TCNBlock1d(nn.Module):
    """Causal dilated depthwise-separable Conv1d block with residual (INT8-friendly)."""
    def __init__(self, c, k=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        # Explicit left zeros are mathematically identical to symmetric Conv1d
        # padding followed by a right crop. Concatenation exports more reliably
        # than axis-sensitive Pad/Slice nodes in ONNX-to-TFLite converters.
        self.dw = nn.Conv1d(c, c, k, padding=0, dilation=dilation, groups=c, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.pw = nn.Conv1d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(c)
        self.act = nn.ReLU6(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        left = x.new_zeros((x.shape[0], x.shape[1], self.pad))
        y = self.dw(torch.cat((left, x), dim=-1))    # causal left padding
        y = self.act(self.bn1(y))
        y = self.act(self.bn2(self.pw(y)))
        return x + self.drop(y)                      # residual


class DSCNN_TCN(nn.Module):
    """DS-CNN backbone (2D depthwise-separable) + TCN head — the MCU model family.
    width=2.0 -> ~220K params (≈ hanil dscnn_tcn_tiny 208K); width=1.0 -> ~56K (<100K MCU).
    Same interface as BaselineCNN: log-Mel [B,1,n_mels,T] -> logits [B,num_classes]."""
    def __init__(self, num_classes=3, width=1.0, dropout=0.1, mixstyle=False):
        super().__init__()
        c = [max(8, _round(w * width)) for w in (32, 64, 96, 128)]
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(c[0]), nn.ReLU6(True))
        self.ms = MixStyle() if mixstyle else nn.Identity()
        self.b1 = DWSep(c[0], c[1], stride=2)
        self.b2 = DWSep(c[1], c[2], stride=2)
        self.b3 = DWSep(c[2], c[3], stride=2)
        self.tcn1 = TCNBlock1d(c[3], k=3, dilation=1, dropout=dropout)
        self.tcn2 = TCNBlock1d(c[3], k=3, dilation=2, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c[3], num_classes)

    def forward(self, x):
        x = self.b3(self.b2(self.b1(self.ms(self.stem(x)))))  # [B,C,F',T']
        x = x.mean(dim=2)                            # collapse frequency -> [B,C,T']
        x = self.tcn2(self.tcn1(x))                  # temporal conv net
        x = x.mean(dim=-1)                           # global time pool -> [B,C]
        return self.head(self.drop(x))


def _round(ch, divisor=8):
    new = max(divisor, int(ch + divisor / 2) // divisor * divisor)
    return int(new + divisor) if new < 0.9 * ch else int(new)


class SRAMValidDWSep(nn.Module):
    """VALID-padded depthwise-separable block for predictable TFLM memory.

    The existing deployment graph uses SAME padding.  The ONNX-to-TFLite path
    materialises some of those pads as full-sized tensors; the first one alone
    is about 442 KiB.  VALID padding avoids that hidden activation while later
    low-resolution layers recover model capacity with inexpensive 1x1 kernels.
    """

    def __init__(self, cin, cout, kernel_size=3, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(
            cin, cin, kernel_size, stride=stride, padding=0,
            groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class SRAMDSCNN(nn.Module):
    """DS-CNN whose exported peak activation is below 100 KiB at width=1.

    The high-resolution stem is deliberately narrow and downsamples time by
    four.  Capacity is moved into later layers after frequency/time dimensions
    are small, so weights can remain in flash without consuming internal SRAM.
    The graph intentionally avoids TCN transposes, residual concatenations and
    explicit padding, leaving only ESP-NN-friendly Conv2D/DepthwiseConv2D,
    mean-pool and FullyConnected operations.

    For the deployment input [1,1,64,401], width=1 produces a largest layer
    output [1,24,31,100] = 74,400 INT8 bytes.  The actual TFLM arena must still
    be measured after conversion because kernels also request scratch memory.
    """

    def __init__(self, num_classes=3, width=1.0, dropout=0.1, mixstyle=False):
        super().__init__()
        c = [_round(v * width) for v in (24, 48, 96, 160, 192, 256, 256)]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c[0], kernel_size=(3, 5), stride=(2, 4),
                      padding=0, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU6(inplace=True),
        )
        self.ms = MixStyle() if mixstyle else nn.Identity()
        self.body = nn.Sequential(
            SRAMValidDWSep(c[0], c[1], kernel_size=3, stride=2),
            SRAMValidDWSep(c[1], c[2], kernel_size=3, stride=2),
            SRAMValidDWSep(c[2], c[3], kernel_size=3, stride=2),
            SRAMValidDWSep(c[3], c[4], kernel_size=3, stride=2),
            SRAMValidDWSep(c[4], c[5], kernel_size=(1, 3), stride=1),
            SRAMValidDWSep(c[5], c[6], kernel_size=(1, 3), stride=1),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c[-1], num_classes)

    def forward(self, x):
        x = self.body(self.ms(self.stem(x)))
        x = x.mean(dim=(2, 3))
        return self.head(self.drop(x))


def int8_activation_profile(model, shape=(1, 1, 64, 401)):
    """Return Conv2D output sizes; a static pre-export guard, not arena usage."""
    records, hooks = [], []

    def record(name):
        def hook(_module, _inputs, output):
            records.append({
                "name": name,
                "shape": tuple(int(v) for v in output.shape),
                "int8_bytes": int(output.numel()),
            })
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(record(name)))
    was_training = model.training
    try:
        model.eval()
        parameter = next(model.parameters())
        with torch.no_grad():
            model(torch.zeros(shape, device=parameter.device))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return sorted(records, key=lambda item: item["int8_bytes"], reverse=True)


def build_model(name, num_classes, width=1.0, dropout=0.1, mixstyle=False):
    if name == "cnn":
        return BaselineCNN(num_classes=num_classes, width=width, dropout=dropout, mixstyle=mixstyle)
    if name == "dscnn_tcn":
        return DSCNN_TCN(num_classes=num_classes, width=width, dropout=dropout, mixstyle=mixstyle)
    if name == "sram_dscnn":
        return SRAMDSCNN(num_classes=num_classes, width=width, dropout=dropout,
                         mixstyle=mixstyle)
    raise ValueError(f"unknown --model {name}")


# ─────────────────────────── loss (focal) ───────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def subtype_discrimination_loss(logits, targets):
    """Teach pure crackle and pure wheeze clips to rank the correct head higher.

    Normal [0,0] and both [1,1] remain governed only by the multi-label BCE.
    This auxiliary loss changes training only; the exported model still has the
    same two logits and requires no additional ESP32 operators or memory.
    """
    pure = torch.isclose(targets.sum(dim=1), targets.new_tensor(1.0))
    if not torch.any(pure):
        return logits.sum() * 0.0
    subtype = targets[pure].argmax(dim=1)
    return F.cross_entropy(logits[pure], subtype)


# ─────────────────────────── data ───────────────────────────────────────────
def load_audio(path):
    """Load an audio file to (wav[C, T] float32, sr).

    Uses soundfile first — reliable for WAV/FLAC and avoids the torchcodec backend
    that newer torchaudio requires for torchaudio.load(). Falls back to torchaudio.
    """
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)       # [T, C]
    return torch.from_numpy(data.T.copy()), sr                      # [C, T]


def _fix_len_1d(x, n):
    t = x.shape[-1]
    if t == n:
        return x
    if t > n:
        s = (t - n) // 2
        return x[..., s:s + n]
    reps = n // t + 1
    return x.repeat(reps)[..., :n]


class EventBank:
    """Crackle/wheeze 'events' extracted from labelled abnormal TRAIN clips. High-passed to
    isolate the abnormal texture (crackle transients / wheeze tonality) over the breath
    baseline, so we can overlay them onto NORMAL breaths — including scarce-domain normals
    (e.g. Sungbook smartphone) — to synthesise abnormals where we lack data."""
    def __init__(self, frame, n, sr=SR, hp_hz=120.0, per_class_cap=400, seed=0):
        from scipy.signal import butter, sosfilt
        from collections import defaultdict
        sos = butter(4, hp_hz / (sr / 2.0), btype="high", output="sos")
        self.events, counts = [], defaultdict(int)
        rows = [r for _, r in frame.iterrows()
                if str(r["label"]).lower().strip() in ("crackle", "wheeze", "both")]
        random.Random(seed).shuffle(rows)
        for r in rows:
            lab = str(r["label"]).lower().strip()
            if counts[lab] >= per_class_cap:
                continue
            try:
                wav, s = load_audio(r["filepath"])
            except Exception:
                continue
            wav = wav.mean(0) if wav.dim() > 1 else wav.squeeze(0)
            if s != sr:
                wav = torchaudio.functional.resample(wav, s, sr)
            x = sosfilt(sos, wav.numpy()).astype("float32")
            xt = _fix_len_1d(torch.from_numpy(x), n)
            p = xt.abs().max()
            if p > 0:
                xt = xt / p
            self.events.append((xt.half(), torch.tensor(RAW2MULTI[lab], dtype=torch.float32)))
            counts[lab] += 1
        print(f"[event-overlay] bank built: {dict(counts)} events (high-pass {hp_hz:.0f} Hz)")

    def __len__(self):
        return len(self.events)

    def sample(self):
        ev, tgt = random.choice(self.events)
        return ev.float(), tgt


class ClipDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, training=False, mixer=None, bandlimit_prob=0.0,
                 force_mix=False, event_bank=None, overlay_prob=0.0, event_gain=(0.5, 1.5)):
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.mixer = mixer                          # NoiseMixer (train aug, or eval noise)
        self.force_mix = force_mix                  # apply mixer even at eval (noisy test)
        self.bandlimit_prob = bandlimit_prob        # simulate low-rate device (e.g. 4 kHz)
        self.event_bank = event_bank                # EventBank for crackle/wheeze overlay
        self.overlay_prob = overlay_prob            # P(overlay an event onto a NORMAL clip)
        self.event_gain = event_gain
        self.n = int(CLIP_S * SR)

    def __len__(self):
        return len(self.frame)

    def _fix_len(self, wav):
        t = wav.shape[-1]
        if t == self.n:
            return wav
        if t > self.n:
            s = int(torch.randint(0, t - self.n + 1, (1,))) if self.training else (t - self.n) // 2
            return wav[..., s:s + self.n]
        # repeat-pad short clips (Hanil recipe) instead of zero-padding
        reps = self.n // t + 1
        return wav.repeat(1, reps)[..., :self.n]

    def __getitem__(self, i, _depth=0):
        row = self.frame.iloc[i]
        try:
            wav, sr = load_audio(row["filepath"])
        except Exception as e:
            # survive a rare unreadable/corrupt file: warn once, use a neighbour.
            if _depth < 8:
                if _depth == 0:
                    print(f"[warn] could not read {row['filepath']} ({e}); substituting")
                return self.__getitem__((i + 1) % len(self.frame), _depth + 1)
            raise RuntimeError(f"too many unreadable files near index {i}") from e
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        # simulate a low-bandwidth device (e.g. Sungbook's 4 kHz): down- then up-sample
        if self.training and self.bandlimit_prob and random.random() < self.bandlimit_prob:
            target = random.choice([4000, 6000, 8000])
            wav = torchaudio.functional.resample(wav, SR, target)
            wav = torchaudio.functional.resample(wav, target, SR)
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak
        wav = self._fix_len(wav).squeeze(0)
        target = encode_target(row["label"])
        # event-overlay: turn a NORMAL breath into a synthetic abnormal by adding a real
        # crackle/wheeze event, and relabel it to the event's class (multi-label only).
        if (self.training and self.event_bank is not None and MULTILABEL
                and str(row["label"]).lower().strip() == "normal"
                and random.random() < self.overlay_prob):
            ev, ev_tgt = self.event_bank.sample()
            wav = wav + random.uniform(*self.event_gain) * ev
            peak = wav.abs().max()
            if peak > 0:
                wav = wav / peak
            target = ev_tgt
        if self.mixer is not None and (self.training or self.force_mix):
            wav = self.mixer(wav)                   # noise mixing: train aug OR noisy test
        return wav, target


def filter_existing(splits: dict) -> dict:
    """Drop manifest rows whose audio file is missing, with a per-split report.
    Catches path mistakes up front instead of crashing mid-epoch."""
    clean, total_missing, examples = {}, 0, []
    for name, df in splits.items():
        if len(df) == 0:
            clean[name] = df
            continue
        exists = df["filepath"].map(os.path.exists)
        miss = int((~exists).sum())
        total_missing += miss
        if miss:
            print(f"[warn] {name}: {miss}/{len(df)} files missing on disk (dropped)")
            examples.extend(df.loc[~exists, "filepath"].head(3).tolist())
        clean[name] = df[exists].reset_index(drop=True)
    if examples:
        print("[warn] example missing paths:")
        for p in examples[:5]:
            print(f"        {p}")
        print("[hint] if a whole folder is wrong, fix it when building the manifest, e.g.\n"
              '       tools/convert_manifests.py ... --path-replace "/wrong/prefix=/right/prefix"')
    return clean


def split_by_patient(df, ratios=(0.7, 0.15, 0.15), seed=1337):
    rng = np.random.default_rng(seed)
    pats = sorted(df.patient_id.unique())
    rng.shuffle(pats)
    n = len(pats)
    n_tr, n_va = int(round(ratios[0] * n)), int(round(ratios[1] * n))
    tr, va, te = set(pats[:n_tr]), set(pats[n_tr:n_tr + n_va]), set(pats[n_tr + n_va:])
    return (df[df.patient_id.isin(tr)], df[df.patient_id.isin(va)], df[df.patient_id.isin(te)])


def reserve_calibration_patients(df, fraction=0.0, seed=1337):
    """Move a deterministic patient-disjoint fraction into threshold calibration.

    Calibration patients are never returned in the test frame. This lets the
    operating thresholds see the target recording domain without leaking clips
    from the same patient into the published final test result.
    """
    if df is None or len(df) == 0 or fraction <= 0:
        empty = df.iloc[0:0].copy() if df is not None else pd.DataFrame()
        return empty, df
    if not 0 < fraction < 1:
        raise ValueError("calibration fraction must be between 0 and 1")
    patients = np.asarray(sorted(df.patient_id.unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_cal = max(1, min(len(patients) - 1, int(round(fraction * len(patients)))))
    calibration_ids = set(patients[:n_cal])
    calibration = df[df.patient_id.isin(calibration_ids)].reset_index(drop=True)
    test = df[~df.patient_id.isin(calibration_ids)].reset_index(drop=True)
    assert set(calibration.patient_id).isdisjoint(set(test.patient_id))
    return calibration, test


def balance_normal(df, ratio=0.5, seed=1337):
    """Hanil bal50: undersample Normal *per patient* to hit the target normal:abnormal
    ratio. All abnormal clips preserved; no patient dropped. Applied to TRAIN only."""
    normal = df[df.label == "normal"]
    abnormal = df[df.label != "normal"]
    if len(normal) == 0 or len(abnormal) == 0:
        return df
    target = int(len(abnormal) * ratio / (1 - ratio))      # ratio=0.5 -> target=|abnormal|
    if target >= len(normal):
        return df
    frac = target / len(normal)
    rng = np.random.default_rng(seed)
    kept = []
    for _pid, g in normal.groupby("patient_id"):
        k = max(1, int(round(len(g) * frac))) if len(g) else 0
        if k:
            kept.append(g.sample(n=min(k, len(g)), random_state=int(rng.integers(1 << 31))))
    normal_kept = pd.concat(kept) if kept else normal.iloc[0:0]
    out = pd.concat([normal_kept, abnormal]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[bal50] normal {len(normal)}->{len(normal_kept)}, abnormal {len(abnormal)} "
          f"(train now {len(out)} clips)")
    return out


def separate_both(df):
    """A 'both' clip contains crackle AND wheeze. Duplicate it into two labelled rows —
    one 'crackle', one 'wheeze' — so the SAME audio trains BOTH classes (not just one).
    Applied to TRAIN only (both copies share the patient_id, so no leakage)."""
    both = df[df.label == "both"]
    if len(both) == 0:
        return df
    c = both.copy(); c["label"] = "crackle"
    w = both.copy(); w["label"] = "wheeze"
    print(f"[both] separated {len(both)} 'both' clips into crackle AND wheeze "
          f"(+{len(both)} crackle, +{len(both)} wheeze in train)")
    return pd.concat([df[df.label != "both"], c, w], ignore_index=True)


def build_splits(manifest_csv, seed=1337, sungbook_train_frac=0.0,
                 smartphone_train_frac=0.0, balance_normal_ratio=0.5):
    """Seoul stethoscope -> train/internal_val/internal_test (patient-level).
    Sungbook stethoscope -> external_test.

    Label handling is baked in (no CLI needed):
      * 2-class : crackle/wheeze/both -> abnormal
      * 3-class : each 'both' clip is DUPLICATED into a crackle row AND a wheeze row in
                  TRAIN, so it trains both classes. 'both' is dropped from val/test so the
                  3-way evaluation stays unambiguous.
      * 4-class : 'both' kept as its own class.
    bal50 (default ON): undersample Normal per-patient in TRAIN to balance_normal_ratio.
    Domain adaptation: sungbook_train_frac>0 moves that fraction of Sungbook patients
    into training (disjoint from external_test).
    """
    n_cls = len(LABELS)
    df = pd.read_csv(manifest_csv)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    if MULTILABEL:
        df = df[df.label.isin(RAW2MULTI)].reset_index(drop=True)   # keep normal/crackle/wheeze/both
    else:
        if n_cls == 2:
            df["label"] = df["label"].map(lambda l: "normal" if l == "normal" else "abnormal")
        valid = set(L2I) | ({"both"} if n_cls == 3 else set())     # keep 'both' for 3-class
        df = df[df.label.isin(valid)].reset_index(drop=True)

    seoul = df[(df.hospital == "seoul") & (df.device == "stethoscope")]
    sung = df[(df.hospital == "sungbook") & (df.device == "stethoscope")]
    phone_seoul = df[(df.hospital == "seoul") & (df.device == "smartphone")]
    phone_sung = df[(df.hospital == "sungbook") & (df.device == "smartphone")]

    tr, va, te = split_by_patient(seoul, seed=seed)
    seoul_train_pat = set(tr.patient_id.unique())        # patients the model trains on
    rng = np.random.default_rng(seed)

    adapt = set()
    if sungbook_train_frac > 0 and len(sung):
        pats = sorted(sung.patient_id.unique())
        rng.shuffle(pats)
        n_adapt = int(round(sungbook_train_frac * len(pats)))
        adapt = set(pats[:n_adapt])
        tr = pd.concat([tr, sung[sung.patient_id.isin(adapt)]], ignore_index=True)
        sung = sung[~sung.patient_id.isin(adapt)]
        print(f"[domain-adapt] moved {len(adapt)}/{len(pats)} Sungbook patients into training; "
              f"{sung.patient_id.nunique()} remain as external_test")

    # Device adaptation: fold a fraction of SMARTPHONE patients into training so the model
    # actually sees phone audio. Remaining phone patients stay as a leakage-free device test.
    smart = set()
    if smartphone_train_frac > 0:
        for name, pdf in [("seoul", phone_seoul), ("sungbook", phone_sung)]:
            if not len(pdf):
                continue
            pats = sorted(pdf.patient_id.unique())
            rng.shuffle(pats)
            k = int(round(smartphone_train_frac * len(pats)))
            chosen = set(pats[:k]); smart |= chosen
            tr = pd.concat([tr, pdf[pdf.patient_id.isin(chosen)]], ignore_index=True)
            print(f"[device-adapt] moved {len(chosen)}/{len(pats)} {name} smartphone patients into training")

    # Final leakage sweep: any patient that ended up in TRAIN (via ANY device or adaptation)
    # is removed from EVERY test split. This is the single source of truth for leakage-freedom
    # and covers patients recorded on both stethoscope AND smartphone.
    train_pat = set(tr.patient_id.unique())
    va = va[~va.patient_id.isin(train_pat)]
    te = te[~te.patient_id.isin(train_pat)]
    sung = sung[~sung.patient_id.isin(train_pat)]
    sp_seoul = phone_seoul[~phone_seoul.patient_id.isin(train_pat)]
    sp_sung = phone_sung[~phone_sung.patient_id.isin(train_pat)]
    print(f"[device] smartphone_seoul={len(sp_seoul)} clips "
          f"({sp_seoul.patient_id.nunique()} held-out patients)  "
          f"smartphone_sungbook={len(sp_sung)} clips ({sp_sung.patient_id.nunique()} patients)")

    if not MULTILABEL and n_cls == 3:      # single-label: dup 'both' in train, drop from eval
        tr = separate_both(tr)
        va = va[va.label != "both"]; te = te[te.label != "both"]
        sung = sung[sung.label != "both"]
        sp_seoul = sp_seoul[sp_seoul.label != "both"]; sp_sung = sp_sung[sp_sung.label != "both"]

    if balance_normal_ratio is not None:   # bal50 (default; 'normal' vs everything else)
        tr = balance_normal(tr, ratio=balance_normal_ratio, seed=seed)

    return {"train": tr.reset_index(drop=True), "internal_val": va.reset_index(drop=True),
            "internal_test": te.reset_index(drop=True),
            "external_test": sung.reset_index(drop=True),
            "smartphone_seoul": sp_seoul.reset_index(drop=True),
            "smartphone_sungbook": sp_sung.reset_index(drop=True)}


# ─────────────────────────── metrics ────────────────────────────────────────
def report(y_true, y_pred, tag=""):
    k = len(LABELS)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    total = cm.sum()
    acc = np.trace(cm) / total if total else 0.0
    se, sp, f1 = [], [], []
    for i in range(k):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        se.append(rec)
        sp.append(tn / (tn + fp) if (tn + fp) else 0.0)
        f1.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    score = 0.5 * (np.mean(se) + np.mean(sp))
    macro_f1 = float(np.mean(f1))
    print(f"\n=== {tag} ===")
    print(f"accuracy={acc:.3f}  Se={np.mean(se):.3f}  Sp={np.mean(sp):.3f}  "
          f"Score={score:.3f}  macroF1={macro_f1:.3f}")
    for i, l in enumerate(LABELS):
        print(f"  {l:<8} Se={se[i]:.3f} Sp={sp[i]:.3f} F1={f1[i]:.3f}")
    print("  confusion (rows=true, cols=pred):")
    print("           " + " ".join(f"{l[:6]:>7}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"  {l:>8} " + " ".join(f"{v:>7d}" for v in cm[i]))
    return {"accuracy": float(acc), "score": float(score), "macro_f1": macro_f1}


@torch.no_grad()
def tune_thresholds(targets, probs, metric="youden"):
    """Pick a per-head decision threshold on VALIDATION data. Default maximises
    Youden's J (Se+Sp-1) = the balanced Se/Sp that the macro Score rewards, which
    fixes the over-flagging from a fixed 0.5 + aggressive pos_weight. metric='f1'
    optimises F1 instead (favours precision)."""
    T = np.asarray(targets); P = np.asarray(probs)
    grid = np.linspace(0.05, 0.95, 19)
    thrs = []
    for i in range(P.shape[1]):
        t, p = T[:, i], P[:, i]
        best_thr, best_val = 0.5, -1.0
        for th in grid:
            pred = (p >= th).astype(int)
            tp = ((t == 1) & (pred == 1)).sum(); fn = ((t == 1) & (pred == 0)).sum()
            fp = ((t == 0) & (pred == 1)).sum(); tn = ((t == 0) & (pred == 0)).sum()
            se = tp / (tp + fn) if (tp + fn) else 0.0
            sp = tn / (tn + fp) if (tn + fp) else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            val = (2 * prec * se / (prec + se) if (prec + se) else 0.0) if metric == "f1" else se + sp - 1
            if val > best_val:
                best_val, best_thr = val, th
        thrs.append(round(float(best_thr), 3))
    return np.array(thrs)


def abnormal_operating_metrics(targets, probs, thresholds):
    """Two-class metrics for the product rule: crackle OR wheeze is abnormal."""
    targets = np.asarray(targets, dtype=float)
    probs = np.asarray(probs, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    truth = np.any(targets >= 0.5, axis=1)
    pred = np.any(probs >= thresholds, axis=1)
    tp = int(np.sum(truth & pred)); fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred)); fp = int(np.sum(~truth & pred))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(truth) if len(truth) else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def product_three_class_metrics(targets, probs, thresholds):
    """Metrics for the exact deployed normal/crackle/wheeze decision rule.

    Ground-truth ``both`` clips are excluded because the product exposes three
    classes only.  If both heads fire, the head with the larger probability is
    emitted (a tie is resolved as crackle), matching the firmware evaluator.
    """
    targets = np.asarray(targets, dtype=float)
    probs = np.asarray(probs, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if targets.ndim != 2 or probs.shape != targets.shape or targets.shape[1] != 2:
        raise ValueError("three-class metrics expect [N,2] targets and probabilities")

    keep = ~np.all(targets >= 0.5, axis=1)
    targets, probs = targets[keep], probs[keep]
    truth = np.zeros(len(targets), dtype=int)
    truth[targets[:, 0] >= 0.5] = 1
    truth[targets[:, 1] >= 0.5] = 2

    crackle = probs[:, 0] >= thresholds[0]
    wheeze = probs[:, 1] >= thresholds[1]
    pred = np.zeros(len(probs), dtype=int)
    pred[crackle & ~wheeze] = 1
    pred[wheeze & ~crackle] = 2
    both = crackle & wheeze
    pred[both & (probs[:, 0] >= probs[:, 1])] = 1
    pred[both & (probs[:, 1] > probs[:, 0])] = 2

    names = ["normal", "crackle", "wheeze"]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    per_class, recalls, f1s = {}, [], []
    for i, name in enumerate(names):
        tp = int(cm[i, i]); fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(tp + fn),
        }
        if tp + fn:
            recalls.append(recall); f1s.append(f1)
    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "excluded_both": int(np.sum(~keep)),
        "per_class": per_class,
        "confusion_classes": names,
        "confusion": cm.tolist(),
    }


def tune_joint_thresholds(targets, probs, target_sensitivity=None,
                          grid_step=0.01):
    """Tune both heads for the actual two-class OR decision on calibration data.

    With no sensitivity constraint this maximises balanced accuracy. With a
    constraint it selects the highest specificity among operating points that
    retain at least ``target_sensitivity`` abnormal recall. Test patients must
    never be passed to this function.
    """
    targets = np.asarray(targets, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if targets.ndim != 2 or probs.shape != targets.shape or targets.shape[1] != 2:
        raise ValueError("joint threshold tuning expects [N,2] targets and probabilities")
    if target_sensitivity is not None and not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0,1]")
    grid = np.arange(0.05, 0.950001, grid_step)
    candidates = []
    for crackle_threshold in grid:
        for wheeze_threshold in grid:
            thresholds = np.array([crackle_threshold, wheeze_threshold])
            metrics = abnormal_operating_metrics(targets, probs, thresholds)
            candidates.append((thresholds, metrics))
    eligible = candidates
    if target_sensitivity is not None:
        eligible = [item for item in candidates
                    if item[1]["sensitivity"] >= target_sensitivity]
        if not eligible:
            eligible = candidates
    if target_sensitivity is None:
        best = max(eligible, key=lambda item: (
            item[1]["balanced_accuracy"], item[1]["specificity"],
            item[1]["sensitivity"]))
    else:
        best = max(eligible, key=lambda item: (
            item[1]["specificity"], item[1]["balanced_accuracy"],
            item[1]["sensitivity"]))
    thresholds = np.round(best[0], 3)
    return thresholds, abnormal_operating_metrics(targets, probs, thresholds)


def report_multilabel(targets, probs, tag="", thr=0.5):
    """Independent crackle/wheeze metrics at threshold `thr` (scalar or per-head array),
    plus AUROC and a combined normal/crackle/wheeze/both state confusion."""
    T = np.asarray(targets); P = np.asarray(probs)
    thr_arr = np.asarray(thr if np.ndim(thr) else [thr] * P.shape[1], dtype=float)
    pred = (P >= thr_arr).astype(int)
    names = ML_LABELS
    per, scores = [], []
    for i, nm in enumerate(names):
        t, p = T[:, i], pred[:, i]
        tp = int(((t == 1) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum()); tn = int(((t == 0) & (p == 0)).sum())
        se = tp / (tp + fn) if (tp + fn) else 0.0
        sp = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * se / (prec + se) if (prec + se) else 0.0
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(t, P[:, i]) if len(np.unique(t)) > 1 else float("nan")
        except Exception:
            auroc = float("nan")
        per.append((nm, se, sp, f1, auroc)); scores.append(0.5 * (se + sp))
    macro = float(np.mean(scores))
    print(f"\n=== {tag} (multi-label) ===")
    print(f"macro Score={macro:.3f}  thresholds(crackle,wheeze)={thr_arr.round(2).tolist()}")
    for nm, se, sp, f1, au in per:
        print(f"  {nm:<8} Se={se:.3f} Sp={sp:.3f} F1={f1:.3f} AUROC={au:.3f}")
    states = ["normal", "crackle", "wheeze", "both"]
    key = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    cm = np.zeros((4, 4), dtype=int)
    for tv, pv in zip(T.astype(int), pred):
        cm[key[(tv[0], tv[1])], key[(pv[0], pv[1])]] += 1

    # per-STATE metrics (incl. NORMAL) derived from the 4-state confusion
    total = int(cm.sum())
    state_metrics, f1s = {}, []
    for i, s in enumerate(states):
        tp = int(cm[i, i]); fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp); tn = total - tp - fn - fp
        se = tp / (tp + fn) if (tp + fn) else 0.0
        sp = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * se / (prec + se) if (prec + se) else 0.0
        state_metrics[s] = {"se": se, "sp": sp, "f1": f1, "support": int(tp + fn)}
        if (tp + fn) > 0:
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    # safety metrics: false alarm (true-normal called abnormal) & false-normal (missed abnormal)
    n_norm = int(cm[0].sum()); n_abn = int(cm[1:].sum())
    false_alarm = (n_norm - int(cm[0, 0])) / n_norm if n_norm else 0.0        # 1 - normal_Se
    false_normal = int(cm[1:, 0].sum()) / n_abn if n_abn else 0.0            # abnormal -> normal (dangerous)

    nm_ = state_metrics["normal"]
    print(f"  {'normal':<8} Se={nm_['se']:.3f} Sp={nm_['sp']:.3f} F1={nm_['f1']:.3f}  "
          f"(derived state)")
    print(f"  macroF1(normal,crackle,wheeze,both)={macro_f1:.3f}  "
          f"false_alarm={false_alarm:.3f}  false_normal(missed abnormal)={false_normal:.3f}")
    print("  combined state confusion (rows=true, cols=pred): " + " ".join(f"{s[:6]:>7}" for s in states))
    for i, s in enumerate(states):
        print(f"  {s:>8} " + " ".join(f"{v:>7d}" for v in cm[i]))
    two_class = abnormal_operating_metrics(T, P, thr_arr)
    print(f"  two-class abnormal Se={two_class['sensitivity']:.3f} "
          f"Sp={two_class['specificity']:.3f} "
          f"balanced={two_class['balanced_accuracy']:.3f}")
    three_class = product_three_class_metrics(T, P, thr_arr)
    print(f"  product three-class accuracy={three_class['accuracy']:.3f} "
          f"balanced={three_class['balanced_accuracy']:.3f} "
          f"macroF1={three_class['macro_f1']:.3f} "
          f"(excluded both={three_class['excluded_both']})")
    return {
        "score": macro,
        "macro_f1": macro_f1,
        "false_alarm": false_alarm,
        "false_normal": false_normal,
        "two_class": two_class,
        "three_class": three_class,
        "thresholds": thr_arr.round(4).tolist(),
        "per_label": {p[0]: {"se": p[1], "sp": p[2], "f1": p[3], "auroc": p[4]} for p in per},
        "per_state": state_metrics,
        "confusion_states": states,
        "confusion": cm.tolist(),
    }


def _ece(probs, labels, n_bins=15):
    probs = np.asarray(probs, float); labels = np.asarray(labels, float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if m.any():
            ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def calibration_report(val_targets, val_probs, n_bins=15):
    """Fit per-head temperature scaling on validation (logits recovered from probs)
    and report ECE before/after. Returns temperatures to apply as sigmoid(logit/T).
    Only meaningful in multi-label mode."""
    if not MULTILABEL:
        return None
    Y = np.asarray(val_targets, float); P = np.clip(np.asarray(val_probs, float), 1e-6, 1 - 1e-6)
    logits = np.log(P / (1 - P))
    temps, grid = [], np.concatenate([np.linspace(0.5, 1.0, 6), np.linspace(1.1, 4.0, 30)])
    print("\n[calibration] per-head temperature scaling (fit on val):")
    for i, nm in enumerate(ML_LABELS):
        best_T, best_nll = 1.0, np.inf
        for T in grid:
            p = np.clip(1 / (1 + np.exp(-logits[:, i] / T)), 1e-6, 1 - 1e-6)
            nll = -np.mean(Y[:, i] * np.log(p) + (1 - Y[:, i]) * np.log(1 - p))
            if nll < best_nll:
                best_nll, best_T = nll, float(T)
        temps.append(best_T)
        raw = 1 / (1 + np.exp(-logits[:, i]))
        cal = 1 / (1 + np.exp(-logits[:, i] / best_T))
        print(f"  {nm:<8} T={best_T:.2f}  ECE {_ece(raw, Y[:, i], n_bins):.3f} -> "
              f"{_ece(cal, Y[:, i], n_bins):.3f}")
    return np.array(temps)


def make_criterion(train_frame, device, focal_gamma=2.0, weight_power=0.5,
                   multilabel_pos_weight_power=0.5,
                   multilabel_pos_weight_cap=4.0):
    """Build a loss without accidentally correcting class imbalance twice.

    Multi-label weights use ``(negative / positive) ** power``. A power of 0
    is unweighted BCE, 0.5 is the recommended square-root correction, and 1.0
    reproduces the previous aggressive full correction. The optional cap keeps
    a very rare head from dominating every batch.
    """
    if MULTILABEL:
        pos = np.zeros(len(ML_LABELS))
        for lab in train_frame["label"]:
            v = RAW2MULTI[str(lab).lower().strip()]; pos += np.array(v)
        neg = len(train_frame) - pos
        ratio = neg / np.clip(pos, 1, None)
        weights = ratio ** float(multilabel_pos_weight_power)
        if multilabel_pos_weight_cap and multilabel_pos_weight_cap > 0:
            weights = np.minimum(weights, float(multilabel_pos_weight_cap))
        pw = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[loss] BCE multi-label positives={pos.astype(int).tolist()} "
              f"negatives={neg.astype(int).tolist()} ratio={ratio.round(2).tolist()} "
              f"power={multilabel_pos_weight_power:.2f} cap={multilabel_pos_weight_cap} "
              f"pos_weight={weights.round(2).tolist()}")
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return FocalLoss(gamma=focal_gamma, weight=class_weights(train_frame, device, power=weight_power))


@torch.no_grad()
def evaluate(model, frontend, frame, device, tag, mixer=None, thr=0.5, return_probs=False):
    if len(frame) == 0:
        print(f"[skip] {tag}: no clips in manifest")
        return (None, [], []) if return_probs else None
    loader = DataLoader(ClipDataset(frame, mixer=mixer, force_mix=mixer is not None),
                        batch_size=32)
    model.eval()
    yt, yo = [], []
    for wav, y in loader:
        logits = model(frontend(wav.to(device)))
        if MULTILABEL:
            yo.extend(torch.sigmoid(logits).cpu().tolist()); yt.extend(y.tolist())
        else:
            yo.extend(logits.argmax(1).cpu().tolist()); yt.extend(y.tolist())
    res = report_multilabel(yt, yo, tag, thr=thr) if MULTILABEL else report(yt, yo, tag)
    return (res, yt, yo) if return_probs else res


# ─────────────────────────── train ──────────────────────────────────────────
def class_weights(frame, device, power=0.5):
    """Inverse-frequency class weights raised to `power`.
      power=1.0 -> full inverse (aggressive; over-predicts minority classes)
      power=0.5 -> sqrt-inverse (gentler; the default — balances recall vs false positives)
      power=0.0 -> uniform (no weighting)
    """
    counts = np.array([(frame.label == l).sum() for l in LABELS], dtype=float)
    inv = (1.0 / np.clip(counts, 1, None)) ** power
    w = inv / inv.sum() * len(LABELS)
    print(f"[data] class counts {dict(zip(LABELS, counts.astype(int)))}  "
          f"weight_power={power}  weights={w.round(2)}")
    return torch.tensor(w, dtype=torch.float32, device=device)


def train(args):
    global CLIP_S
    CLIP_S = args.clip_seconds
    if args.target_abnormal_sensitivity == 0:
        args.target_abnormal_sensitivity = None
    elif not 0 < args.target_abnormal_sensitivity <= 1:
        raise SystemExit("--target-abnormal-sensitivity must be 0 or in (0,1]")
    if not 0 < args.threshold_grid_step <= 0.1:
        raise SystemExit("--threshold-grid-step must be in (0,0.1]")
    if not 0 <= args.threshold_external_frac < 1:
        raise SystemExit("--threshold-external-frac must be in [0,1)")
    if args.subtype_loss_weight < 0:
        raise SystemExit("--subtype-loss-weight must be >= 0")
    if args.subtype_loss_weight > 0 and not getattr(args, "multilabel", False):
        raise SystemExit("--subtype-loss-weight requires --multilabel")
    if args.init_checkpoint and not os.path.exists(args.init_checkpoint):
        raise SystemExit(f"--init-checkpoint not found: {args.init_checkpoint}")
    if not 0 <= args.distill_weight < 1:
        raise SystemExit("--distill-weight must be in [0,1)")
    if args.distill_weight > 0 and not getattr(args, "multilabel", False):
        raise SystemExit("self-distillation currently requires --multilabel")
    if args.distill_weight > 0 and not args.teacher_checkpoint:
        raise SystemExit("--distill-weight > 0 requires --teacher-checkpoint")
    if args.teacher_checkpoint and not os.path.exists(args.teacher_checkpoint):
        raise SystemExit(f"--teacher-checkpoint not found: {args.teacher_checkpoint}")
    if args.distill_temperature <= 0:
        raise SystemExit("--distill-temperature must be > 0")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if getattr(args, "multilabel", False):
        set_multilabel(True)
    else:
        set_classes(args.num_classes)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[info] device={device}  outputs={LABELS}  multilabel={MULTILABEL}  clip={CLIP_S}s")

    if args.synthetic:
        manifest = make_synthetic(Path(args.out) / "synth")
    else:
        manifest = args.manifest
        if not manifest or not os.path.exists(manifest):
            raise SystemExit("Provide --manifest <csv> or use --synthetic")

    imbalance_strategy = (args.imbalance_strategy if MULTILABEL else "single_label")
    if MULTILABEL and imbalance_strategy == "natural_sqrt":
        balance_ratio = None
        pos_weight_power = args.multilabel_pos_weight_power
        pos_weight_cap = args.multilabel_pos_weight_cap
    elif MULTILABEL and imbalance_strategy == "balanced_unweighted":
        balance_ratio = args.balance_ratio
        pos_weight_power = 0.0
        pos_weight_cap = 0.0
    elif MULTILABEL and imbalance_strategy == "legacy_double":
        balance_ratio = args.balance_ratio
        pos_weight_power = 1.0
        pos_weight_cap = 0.0
    else:
        balance_ratio = None if args.no_balance else args.balance_ratio
        pos_weight_power = args.multilabel_pos_weight_power
        pos_weight_cap = args.multilabel_pos_weight_cap
    print(f"[imbalance] strategy={imbalance_strategy} "
          f"normal_undersampling={'off' if balance_ratio is None else balance_ratio} "
          f"multilabel_pos_weight_power={pos_weight_power} cap={pos_weight_cap}")

    splits = build_splits(manifest, sungbook_train_frac=args.sungbook_train_frac,
                          smartphone_train_frac=args.smartphone_train_frac,
                          balance_normal_ratio=balance_ratio)
    if not args.skip_file_check:
        n_before = sum(len(v) for v in splits.values())
        splits = filter_existing(splits)
        n_after = sum(len(v) for v in splits.values())
        if n_before and n_after < 0.5 * n_before:
            raise SystemExit(
                f"\n{n_before - n_after}/{n_before} audio files are missing on disk - "
                "the manifest paths look wrong. Fix the paths (see the hint above) and "
                "re-run, or pass --skip-file-check to bypass this guard.")

    threshold_frames = [splits["internal_val"]]
    external_calibration, splits["external_test"] = reserve_calibration_patients(
        splits["external_test"], args.threshold_external_frac, seed=args.seed + 101)
    if len(external_calibration):
        threshold_frames.append(external_calibration)
        print(f"[threshold-calibration] reserved {len(external_calibration)} external "
              f"clips from {external_calibration.patient_id.nunique()} patients; "
              f"final external test keeps {len(splits['external_test'])} clips from "
              f"{splits['external_test'].patient_id.nunique()} disjoint patients")
    threshold_frame = pd.concat(threshold_frames, ignore_index=True)
    print(f"[data] train={len(splits['train'])} val={len(splits['internal_val'])} "
          f"internal_test={len(splits['internal_test'])} "
          f"external_test={len(splits['external_test'])} "
          f"phone_seoul={len(splits['smartphone_seoul'])} "
          f"phone_sungbook={len(splits['smartphone_sungbook'])}")

    frontend = LogMel().to(device)
    model = build_model(args.model, num_classes=len(LABELS), width=args.width,
                        dropout=args.dropout, mixstyle=args.mixstyle).to(device)
    if args.init_checkpoint:
        init_state = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(init_state["model"] if "model" in init_state else init_state)
        print(f"[model] initialized from {args.init_checkpoint}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} width={args.width} params={n_params:,}"
          f"{'  +mixstyle' if args.mixstyle else ''}")
    activation_profile = int8_activation_profile(model)
    largest_activation = activation_profile[0] if activation_profile else None
    if largest_activation:
        print(f"[memory-static] largest_conv={largest_activation['name']} "
              f"shape={largest_activation['shape']} "
              f"int8_kib={largest_activation['int8_bytes'] / 1024.0:.1f}")
        if (args.max_activation_kb > 0 and
                largest_activation["int8_bytes"] > args.max_activation_kb * 1024):
            raise SystemExit(
                f"static activation budget exceeded: "
                f"{largest_activation['int8_bytes'] / 1024.0:.1f} KiB > "
                f"{args.max_activation_kb:.1f} KiB. The post-quantization TFLM "
                f"arena must also be measured separately.")

    teacher = None
    if args.distill_weight > 0:
        teacher = build_model(
            args.teacher_model, num_classes=len(LABELS),
            width=args.teacher_width, dropout=0.0, mixstyle=False).to(device)
        teacher_state = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(
            teacher_state["model"] if "model" in teacher_state else teacher_state)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(f"[distill] teacher={args.teacher_model} width={args.teacher_width} "
              f"checkpoint={args.teacher_checkpoint} weight={args.distill_weight} "
              f"temperature={args.distill_temperature}")

    crit = make_criterion(
        splits["train"], device, focal_gamma=args.focal_gamma,
        weight_power=args.weight_power,
        multilabel_pos_weight_power=pos_weight_power,
        multilabel_pos_weight_cap=pos_weight_cap)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = max(0, args.warmup_epochs)         # linear warmup then cosine decay
    def _lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(1, warmup))
        progress = float(ep - warmup) / float(max(1, args.epochs - warmup))
        return float(0.5 * (1.0 + np.cos(np.pi * min(1.0, progress))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    amp_enabled = bool(args.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"[sched] warmup {warmup} ep -> cosine over {args.epochs} ep  base_lr={args.lr}  seed={args.seed}")
    print(f"[precision] amp={'fp16' if amp_enabled else 'off'}")
    print(f"[objective] subtype_loss_weight={args.subtype_loss_weight} "
          f"selection_metric={args.selection_metric} "
          f"distill_weight={args.distill_weight}")

    mixer = None
    if args.noise_dir:
        mixer = NoiseMixer(args.noise_dir, prob=args.noise_prob,
                           snr_db=tuple(args.snr_db))

    event_bank = None
    if args.event_overlay > 0 and MULTILABEL:
        event_bank = EventBank(splits["train"], int(CLIP_S * SR),
                               per_class_cap=args.event_cap)
        if len(event_bank) == 0:
            print("[event-overlay] no abnormal clips found -> disabled"); event_bank = None

    # only drop the last partial batch when there's enough data for a full one,
    # otherwise a small set (e.g. the synthetic smoke test) yields zero batches
    drop_last = len(splits["train"]) > args.batch_size
    train_ds = ClipDataset(splits["train"], training=True, mixer=mixer,
                           bandlimit_prob=args.bandlimit_prob,
                           event_bank=event_bank, overlay_prob=args.event_overlay)
    if args.bandlimit_prob:
        print(f"[aug] band-limit sim: p={args.bandlimit_prob} -> down/up-sample to 4-8 kHz")
    if event_bank:
        print(f"[aug] event-overlay: p={args.event_overlay} onto NORMAL clips (gain 0.5-1.5)")
    if args.mixup_alpha > 0:
        print(f"[aug] mixup: alpha={args.mixup_alpha}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=drop_last, num_workers=args.workers,
                              pin_memory=(device == "cuda"),
                              persistent_workers=(args.workers > 0),
                              prefetch_factor=(4 if args.workers > 0 else None))
    if len(splits["train"]) == 0:
        raise SystemExit("No training clips - check the manifest (hospital=seoul, device=stethoscope).")
    best, patience = -1.0, 0
    Path(args.out).mkdir(parents=True, exist_ok=True)

    import time
    n_steps = len(train_loader)
    print(f"[train] {len(splits['train'])} clips -> {n_steps} steps/epoch  "
          f"batch={args.batch_size}  workers={args.workers}  device={device}", flush=True)
    for epoch in range(args.epochs):
        model.train(); tot = 0.0; processed = 0; t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, (wav, y) in enumerate(train_loader):
            wav, y = wav.to(device), y.to(device)
            spec = frontend(wav)
            if args.specaugment:
                spec = spec_augment(spec)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                def objective(batch_logits, batch_targets):
                    value = crit(batch_logits, batch_targets)
                    if MULTILABEL and args.subtype_loss_weight > 0:
                        value = value + args.subtype_loss_weight * \
                            subtype_discrimination_loss(batch_logits, batch_targets)
                    return value

                if args.mixup_alpha > 0 and spec.size(0) > 1:
                    lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                    perm = torch.randperm(spec.size(0), device=device)
                    spec = lam * spec + (1 - lam) * spec[perm]
                    logits = model(spec)
                    task_loss = (lam * objective(logits, y)
                                 + (1 - lam) * objective(logits, y[perm]))
                else:
                    logits = model(spec)
                    task_loss = objective(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(spec)
                    temperature = args.distill_temperature
                    soft_targets = torch.sigmoid(teacher_logits / temperature)
                    distill_loss = F.binary_cross_entropy_with_logits(
                        logits / temperature, soft_targets) * temperature ** 2
                    loss = ((1.0 - args.distill_weight) * task_loss
                            + args.distill_weight * distill_loss)
                else:
                    loss = task_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * wav.size(0)
            processed += wav.size(0)
            if step % args.log_every == 0:
                rate = (step + 1) / (time.time() - t0)
                memory = (f" peak_vram={torch.cuda.max_memory_allocated()/2**20:.0f}MB"
                          if device == "cuda" else "")
                print(f"  e{epoch} step {step}/{n_steps} loss={loss.item():.3f} "
                      f"{rate:.1f} it/s  ETA {(n_steps-step-1)/max(rate,1e-6)/60:.1f}m"
                      f"{memory}", flush=True)
        sched.step()
        val = evaluate(model, frontend, splits["internal_val"], device, f"val (epoch {epoch})") \
            if (epoch % max(1, args.eval_every) == 0 or epoch == args.epochs - 1) else None
        if val and args.selection_metric == "product_composite":
            vscore = 0.5 * (val["two_class"]["balanced_accuracy"]
                            + val["three_class"]["macro_f1"])
        else:
            vscore = val["score"] if val else None
        line = f"epoch {epoch:03d} loss={tot/max(processed, 1):.4f}"
        if vscore is not None:
            line += f" val_score={vscore:.3f}"
        print(line)
        if vscore is not None and vscore > best:
            best, patience = vscore, 0
            torch.save({
                "model": model.state_dict(),
                "labels": LABELS,
                "training_config": {
                    "imbalance_strategy": imbalance_strategy,
                    "multilabel_pos_weight_power": pos_weight_power,
                    "multilabel_pos_weight_cap": pos_weight_cap,
                    "normal_balance_ratio": balance_ratio,
                    "subtype_loss_weight": args.subtype_loss_weight,
                    "selection_metric": args.selection_metric,
                    "init_checkpoint": args.init_checkpoint,
                    "largest_conv_activation_int8_bytes": (
                        largest_activation["int8_bytes"] if largest_activation else None),
                    "max_activation_kb": args.max_activation_kb,
                    "teacher_checkpoint": args.teacher_checkpoint,
                    "teacher_model": args.teacher_model,
                    "teacher_width": args.teacher_width,
                    "distill_weight": args.distill_weight,
                    "distill_temperature": args.distill_temperature,
                    "seed": args.seed,
                },
            }, Path(args.out) / "best.pt")
        elif vscore is not None:
            patience += 1
            if patience >= args.patience and (epoch + 1) >= args.min_epochs:
                print(f"[early-stop] no val improvement for {patience} evals "
                      f"(past min_epochs={args.min_epochs})"); break

    # final test on all held-out splits
    ckpt = Path(args.out) / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    test_splits = ["internal_test", "external_test", "smartphone_seoul", "smartphone_sungbook"]

    # Tune on patient-disjoint calibration data only. The external calibration
    # patients were removed from external_test above before any threshold search.
    thr = 0.5
    threshold_metrics = None
    temperature = None
    if MULTILABEL and not args.no_threshold_tune:
        _, vt, vp = evaluate(model, frontend, threshold_frame, device,
                             "patient-disjoint threshold calibration",
                             thr=0.5, return_probs=True)
        if args.threshold_mode == "joint":
            thr, threshold_metrics = tune_joint_thresholds(
                vt, vp, target_sensitivity=args.target_abnormal_sensitivity,
                grid_step=args.threshold_grid_step)
            objective = ("balanced accuracy" if args.target_abnormal_sensitivity is None
                         else f"maximum specificity at abnormal sensitivity "
                              f">={args.target_abnormal_sensitivity:.3f}")
        else:
            thr = tune_thresholds(vt, vp, metric=args.threshold_metric)
            threshold_metrics = abnormal_operating_metrics(vt, vp, thr)
            objective = f"per-head {args.threshold_metric}"
        print(f"\n[threshold] mode={args.threshold_mode} objective={objective} "
              f"thresholds(crackle,wheeze)={thr.tolist()} "
              f"two_class={threshold_metrics}")
        temperature = calibration_report(vt, vp)

    import json as _json
    results = {
        "best_val": best,
        "thresholds": (thr.tolist() if hasattr(thr, "tolist") else thr),
        "threshold_calibration": {
            "mode": args.threshold_mode,
            "n_clips": len(threshold_frame),
            "n_patients": int(threshold_frame.patient_id.nunique()),
            "external_fraction": args.threshold_external_frac,
            "target_abnormal_sensitivity": args.target_abnormal_sensitivity,
            "metrics": threshold_metrics,
            "temperature_diagnostic_only": (
                temperature.tolist() if temperature is not None else None),
        },
        "training_config": {
            "model": args.model, "width": args.width, "seed": args.seed,
            "imbalance_strategy": imbalance_strategy,
            "normal_balance_ratio": balance_ratio,
            "multilabel_pos_weight_power": pos_weight_power,
            "multilabel_pos_weight_cap": pos_weight_cap,
            "sungbook_train_frac": args.sungbook_train_frac,
            "smartphone_train_frac": args.smartphone_train_frac,
            "subtype_loss_weight": args.subtype_loss_weight,
            "selection_metric": args.selection_metric,
            "init_checkpoint": args.init_checkpoint,
            "largest_conv_activation_int8_bytes": (
                largest_activation["int8_bytes"] if largest_activation else None),
            "max_activation_kb": args.max_activation_kb,
            "teacher_checkpoint": args.teacher_checkpoint,
            "teacher_model": args.teacher_model,
            "teacher_width": args.teacher_width,
            "distill_weight": args.distill_weight,
            "distill_temperature": args.distill_temperature,
        },
        "splits": {},
    }
    print("\n" + "=" * 60 + "\nFINAL TEST - CLEAN\n" + "=" * 60)
    for split in test_splits:
        r = evaluate(model, frontend, splits[split], device, split, thr=thr)
        if r: results["splits"][split] = r
    if args.eval_noise_dir:                 # step 1.1: inject noise into the TEST set
        for snr in args.eval_snr:
            m = NoiseMixer(args.eval_noise_dir, prob=1.0, snr_db=(snr, snr))
            print("\n" + "=" * 60 + f"\nFINAL TEST - NOISY @ {snr} dB SNR\n" + "=" * 60)
            for split in test_splits:
                r = evaluate(model, frontend, splits[split], device, f"{split}@{snr}dB", mixer=m, thr=thr)
                if r: results["splits"][f"{split}@{snr}dB"] = r
    with open(Path(args.out) / "results.json", "w") as f:
        _json.dump(results, f, indent=2)
    print(f"\n[done] best val Score={best:.3f}  checkpoint -> {ckpt}")
    print(f"[saved] full per-class/per-state metrics -> {Path(args.out) / 'results.json'}")


# ─────────────────────── synthetic data (smoke test) ─────────────────────────
def make_synthetic(root: Path, n_patients=15):
    import soundfile as sf
    root = Path(root); rng = np.random.default_rng(0); rows = []
    gens = {
        "normal": lambda t: (0.2 * (1 + np.sin(2 * np.pi * 0.3 * t)) * np.sin(2 * np.pi * 250 * t)
                             + rng.normal(0, 0.03, len(t))),
        "wheeze": lambda t: 0.3 * np.sin(2 * np.pi * 400 * t) * ((t > 1) & (t < 4)),
        "crackle": lambda t: _crackle(t, rng)}
    for hosp in ["seoul", "sungbook"]:
        for pid in range(n_patients):
            patient = f"{hosp}_p{pid:03d}"; label = ["normal", "crackle", "wheeze"][pid % 3]
            devices = ["stethoscope", "smartphone"] if hosp == "seoul" else ["stethoscope"]
            for dev in devices:
                d = root / hosp / dev; d.mkdir(parents=True, exist_ok=True)
                t = np.arange(int(CLIP_S * SR)) / SR
                sig = gens[label](t).astype(np.float32)
                if dev == "smartphone":
                    sig = np.convolve(sig, [0.6, 0.3, 0.1], mode="same") + rng.normal(0, 0.02, len(sig))
                path = d / f"{patient}_{dev}.wav"
                sf.write(path, sig.astype(np.float32), SR)
                rows.append(dict(filepath=str(path), patient_id=patient, label=label,
                                 hospital=hosp, device=dev))
    man = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(man, index=False)
    print(f"[synth] wrote {len(rows)} clips -> {man}")
    return str(man)


def _crackle(t, rng):
    x = np.zeros(len(t))
    for c in rng.integers(0, len(t), size=40):
        x[c:c + 30] += np.hanning(30) * rng.uniform(0.3, 0.8)
    return x


def main():
    ap = argparse.ArgumentParser(description="Baseline 3-class lung-sound classifier")
    ap.add_argument("--manifest", default=None, help="path to manifest.csv")
    ap.add_argument("--num-classes", type=int, choices=[2, 3, 4], default=3,
                    help="3 = normal/crackle/wheeze ('both' duplicated into both). Default.")
    ap.add_argument("--multilabel", action="store_true",
                    help="independent crackle+wheeze binary heads (sigmoid+BCE); "
                         "'both'=[1,1], no duplication. Outputs two probabilities.")
    ap.add_argument("--subtype-loss-weight", type=float, default=0.0,
                    help="extra crackle-vs-wheeze cross-entropy on pure subtype clips; "
                         "training-only and deployment-neutral (try 0.25)")
    ap.add_argument("--selection-metric",
                    choices=["head_score", "product_composite"], default="head_score",
                    help="checkpoint objective; product_composite averages two-class "
                         "balanced accuracy and deployed three-class macro-F1")
    ap.add_argument("--init-checkpoint", default=None,
                    help="optional checkpoint to fine-tune instead of random initialization")
    ap.add_argument("--teacher-checkpoint", default=None,
                    help="optional frozen multi-label teacher checkpoint for compact-model "
                         "self-distillation")
    ap.add_argument("--teacher-model",
                    choices=["cnn", "dscnn_tcn", "sram_dscnn"],
                    default="dscnn_tcn")
    ap.add_argument("--teacher-width", type=float, default=2.0)
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="weight assigned to soft teacher targets; 0 disables distillation, "
                         "try 0.30-0.50 for the SRAM student")
    ap.add_argument("--distill-temperature", type=float, default=2.0)
    ap.add_argument("--threshold-metric", choices=["youden", "f1"], default="youden",
                    help="per-head objective used only with --threshold-mode per_head")
    ap.add_argument("--threshold-mode", choices=["joint", "per_head"], default="joint",
                    help="joint tunes crackle/wheeze together for the product's abnormal "
                         "OR rule; per_head retains the older independent tuning")
    ap.add_argument("--target-abnormal-sensitivity", type=float, default=0.80,
                    help="with joint tuning, maximise specificity while retaining this "
                         "two-class sensitivity; pass 0 to maximise balanced accuracy")
    ap.add_argument("--threshold-grid-step", type=float, default=0.01,
                    help="threshold-search resolution between 0.05 and 0.95")
    ap.add_argument("--threshold-external-frac", type=float, default=0.30,
                    help="fraction of external-test patients reserved for threshold "
                         "calibration and removed from the final external test")
    ap.add_argument("--no-threshold-tune", action="store_true",
                    help="keep fixed 0.5 threshold instead of tuning per head")
    ap.add_argument("--imbalance-strategy",
                    choices=["natural_sqrt", "balanced_unweighted", "legacy_double"],
                    default="natural_sqrt",
                    help="multi-label imbalance handling. Recommended natural_sqrt keeps "
                         "all normal clips and uses capped sqrt positive weights; "
                         "legacy_double reproduces the old over-flagging setup")
    ap.add_argument("--multilabel-pos-weight-power", type=float, default=0.5,
                    help="positive-weight exponent for natural_sqrt (0=unweighted, "
                         "0.5=sqrt correction, 1=full neg/pos)")
    ap.add_argument("--multilabel-pos-weight-cap", type=float, default=4.0,
                    help="maximum positive weight; 0 disables the cap")
    ap.add_argument("--no-balance", action="store_true",
                    help="disable Normal undersampling for single-label/legacy runs")
    ap.add_argument("--balance-ratio", type=float, default=0.5,
                    help="target normal:abnormal ratio for bal50 (default 0.5 = 50:50)")
    ap.add_argument("--clip-seconds", type=float, default=4.0,
                    help="clip length in seconds (Hanil: 4 s beat 2 s)")
    ap.add_argument("--weight-power", type=float, default=0.5,
                    help="class-weight strength: 1.0=full inverse (aggressive), "
                         "0.5=sqrt-inverse (gentler, default), 0.0=uniform")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="focal loss gamma (0 = plain weighted cross-entropy)")
    ap.add_argument("--model", choices=["cnn", "dscnn_tcn", "sram_dscnn"],
                    default="cnn",
                    help="cnn = depthwise CNN; dscnn_tcn = DS-CNN+TCN; "
                         "sram_dscnn = VALID-padded, early-downsampled internal-SRAM target")
    ap.add_argument("--width", type=float, default=1.0,
                    help="width multiplier: cnn 1.0~65k/4.0~1M; dscnn_tcn "
                         "2.0~220k; sram_dscnn 1.0 keeps its largest INT8 layer <100KiB")
    ap.add_argument("--max-activation-kb", type=float, default=0.0,
                    help="fail before training when the largest Conv2D output exceeds this "
                         "static INT8 budget; 0 disables the guard. TFLM arena validation "
                         "after quantization is still mandatory")
    ap.add_argument("--noise-dir", nargs="+", default=None,
                    help="one or more dirs of noise .wav to mix in during training. "
                         "Combine MUSAN + DESED + your own real handling/room noise, e.g. "
                         "--noise-dir /musan/noise /noise/train /noise/self_recorded")
    ap.add_argument("--noise-prob", type=float, default=0.5,
                    help="fraction of training clips that get noise mixed in")
    ap.add_argument("--snr-db", type=float, nargs=2, default=[5, 20],
                    help="SNR range in dB for TRAIN noise augmentation")
    ap.add_argument("--eval-noise-dir", nargs="+", default=None,
                    help="step 1.1: inject noise from these dir(s) into the TEST set to "
                         "measure the clean-vs-noisy drop (e.g. MUSAN [+ DESED + real])")
    ap.add_argument("--eval-snr", type=float, nargs="+", default=[20, 10, 5],
                    help="SNR levels (dB) to evaluate the noisy test at")
    ap.add_argument("--specaugment", action="store_true",
                    help="apply time/frequency masking on the spectrogram (training)")
    ap.add_argument("--bandlimit-prob", type=float, default=0.0,
                    help="prob. of band-limiting a training clip to ~4-8 kHz, to simulate "
                         "the low-rate external (Sungbook) domain. Try 0.5.")
    ap.add_argument("--event-overlay", type=float, default=0.0,
                    help="P(overlay a real crackle/wheeze event onto a NORMAL clip and relabel "
                         "it) — synthesises abnormals in scarce domains. Multi-label only. Try 0.3.")
    ap.add_argument("--event-cap", type=int, default=400,
                    help="max events stored per class in the event bank")
    ap.add_argument("--mixup-alpha", type=float, default=0.0,
                    help="Beta(alpha,alpha) mixup on spectrogram+labels (0=off). Try 0.2.")
    ap.add_argument("--mixstyle", action="store_true",
                    help="insert a training-only MixStyle layer (cross-device/site invariance; "
                         "identity at eval, deployment-neutral)")
    ap.add_argument("--synthetic", action="store_true", help="auto-generate a tiny test set")
    ap.add_argument("--sungbook-train-frac", type=float, default=0.0,
                    help="domain adaptation: fraction of Sungbook PATIENTS to move into "
                         "training (0.0=strict external test; try 0.3-0.5). Held-out "
                         "Sungbook patients still form external_test.")
    ap.add_argument("--smartphone-train-frac", type=float, default=0.0,
                    help="device adaptation: fraction of SMARTPHONE patients (Seoul+Sungbook) "
                         "to move into training so the model sees phone audio (try 0.3-0.5). "
                         "Held-out phone patients still form the smartphone_* device tests.")
    ap.add_argument("--skip-file-check", action="store_true",
                    help="skip the upfront file-existence validation")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="use CUDA fp16 automatic mixed precision (default on; ignored on CPU)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="training device (auto selects CUDA when available)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2,
                    help="DataLoader workers. Use 0 if training hangs at step 0 (fork deadlock).")
    ap.add_argument("--log-every", type=int, default=20,
                    help="print a train-step progress line every N steps")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337,
                    help="random seed; run 2-3 seeds and report mean +/- std for the product")
    ap.add_argument("--warmup-epochs", type=int, default=3,
                    help="linear LR warmup epochs before cosine decay")
    ap.add_argument("--min-epochs", type=int, default=0,
                    help="do not allow early-stop before this many epochs (avoids premature stop)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
