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
    """64-bin log-Mel, 25 ms / 10 ms, 50-2000 Hz. Identical to common/features.py
    so a model trained here is compatible with the rest of the pipeline."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=400, win_length=400, hop_length=160,
            f_min=50.0, f_max=2000.0, n_mels=64, power=2.0, norm="slaney",
            mel_scale="slaney")

    def forward(self, wav):                       # wav [B, T] -> [B, 1, 64, frames]
        mel = self.mel(wav)
        logmel = torch.log(mel + 1e-6)
        mu = logmel.mean(dim=(1, 2), keepdim=True)
        sd = logmel.std(dim=(1, 2), keepdim=True) + 1e-5
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

    def __init__(self, num_classes=3, dropout=0.1, width=1.0):
        super().__init__()
        c = [max(8, int(round(w * width))) for w in (32, 64, 128, 128, 256)]
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(c[0]), nn.ReLU6(True))
        self.body = nn.Sequential(
            DWSep(c[0], c[1], 2), DWSep(c[1], c[2], 2),
            DWSep(c[2], c[3], 1), DWSep(c[3], c[4], 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c[4], num_classes)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.head(self.drop(self.pool(x).flatten(1)))


# ─────────────────── DSCNN-TCN (MCU deployment target) ──────────────────────
class TCNBlock1d(nn.Module):
    """Causal dilated depthwise-separable Conv1d block with residual (INT8-friendly)."""
    def __init__(self, c, k=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.dw = nn.Conv1d(c, c, k, padding=self.pad, dilation=dilation, groups=c, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.pw = nn.Conv1d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(c)
        self.act = nn.ReLU6(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dw(x)[..., :x.shape[-1]]           # causal crop (drop right padding)
        y = self.act(self.bn1(y))
        y = self.act(self.bn2(self.pw(y)))
        return x + self.drop(y)                      # residual


class DSCNN_TCN(nn.Module):
    """DS-CNN backbone (2D depthwise-separable) + TCN head — the MCU model family.
    width=2.0 -> ~220K params (≈ hanil dscnn_tcn_tiny 208K); width=1.0 -> ~56K (<100K MCU).
    Same interface as BaselineCNN: log-Mel [B,1,n_mels,T] -> logits [B,num_classes]."""
    def __init__(self, num_classes=3, width=1.0, dropout=0.1):
        super().__init__()
        c = [max(8, _round(w * width)) for w in (32, 64, 96, 128)]
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(c[0]), nn.ReLU6(True))
        self.b1 = DWSep(c[0], c[1], stride=2)
        self.b2 = DWSep(c[1], c[2], stride=2)
        self.b3 = DWSep(c[2], c[3], stride=2)
        self.tcn1 = TCNBlock1d(c[3], k=3, dilation=1, dropout=dropout)
        self.tcn2 = TCNBlock1d(c[3], k=3, dilation=2, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c[3], num_classes)

    def forward(self, x):
        x = self.b3(self.b2(self.b1(self.stem(x))))  # [B,C,F',T']
        x = x.mean(dim=2)                            # collapse frequency -> [B,C,T']
        x = self.tcn2(self.tcn1(x))                  # temporal conv net
        x = x.mean(dim=-1)                           # global time pool -> [B,C]
        return self.head(self.drop(x))


def _round(ch, divisor=8):
    new = max(divisor, int(ch + divisor / 2) // divisor * divisor)
    return int(new + divisor) if new < 0.9 * ch else int(new)


def build_model(name, num_classes, width=1.0, dropout=0.1):
    if name == "cnn":
        return BaselineCNN(num_classes=num_classes, width=width, dropout=dropout)
    if name == "dscnn_tcn":
        return DSCNN_TCN(num_classes=num_classes, width=width, dropout=dropout)
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


# ─────────────────────────── data ───────────────────────────────────────────
def load_audio(path):
    """Load an audio file to (wav[C, T] float32, sr).

    Uses soundfile first — reliable for WAV/FLAC and avoids the torchcodec backend
    that newer torchaudio requires for torchaudio.load(). Falls back to torchaudio.
    """
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)       # [T, C]
    return torch.from_numpy(data.T.copy()), sr                      # [C, T]


class ClipDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, training=False, mixer=None, bandlimit_prob=0.0,
                 force_mix=False):
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.mixer = mixer                          # NoiseMixer (train aug, or eval noise)
        self.force_mix = force_mix                  # apply mixer even at eval (noisy test)
        self.bandlimit_prob = bandlimit_prob        # simulate low-rate device (e.g. 4 kHz)
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
        if self.mixer is not None and (self.training or self.force_mix):
            wav = self.mixer(wav)                   # noise mixing: train aug OR noisy test
        return wav, encode_target(row["label"])


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
                 balance_normal_ratio=0.5):
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

    adapt = set()
    if sungbook_train_frac > 0 and len(sung):
        rng = np.random.default_rng(seed)
        pats = sorted(sung.patient_id.unique())
        rng.shuffle(pats)
        n_adapt = int(round(sungbook_train_frac * len(pats)))
        adapt = set(pats[:n_adapt])
        tr = pd.concat([tr, sung[sung.patient_id.isin(adapt)]], ignore_index=True)
        sung = sung[~sung.patient_id.isin(adapt)]
        print(f"[domain-adapt] moved {len(adapt)}/{len(pats)} Sungbook patients into training; "
              f"{sung.patient_id.nunique()} remain as external_test")

    # Device-robustness test sets, LEAKAGE-FREE (patients disjoint from training):
    #   smartphone_seoul   = Seoul phone from patients NOT in training -> pure device shift
    #   smartphone_sungbook= Sungbook phone (minus any adapted patients) -> device + site shift
    sp_seoul = phone_seoul[~phone_seoul.patient_id.isin(seoul_train_pat)]
    sp_sung = phone_sung[~phone_sung.patient_id.isin(adapt)]
    n_leak = phone_seoul.patient_id.isin(seoul_train_pat).sum()
    print(f"[device] smartphone_seoul={len(sp_seoul)} clips "
          f"({sp_seoul.patient_id.nunique()} held-out patients; excluded {int(n_leak)} "
          f"train-patient clips)  smartphone_sungbook={len(sp_sung)} clips")

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
    return {
        "score": macro,
        "macro_f1": macro_f1,
        "false_alarm": false_alarm,
        "false_normal": false_normal,
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


def make_criterion(train_frame, device, focal_gamma=2.0, weight_power=0.5):
    """BCEWithLogits (per-label pos_weight) for multi-label; else focal loss."""
    if MULTILABEL:
        pos = np.zeros(len(ML_LABELS))
        for lab in train_frame["label"]:
            v = RAW2MULTI[str(lab).lower().strip()]; pos += np.array(v)
        neg = len(train_frame) - pos
        pw = torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32, device=device)
        print(f"[loss] BCE multi-label  pos_weight(crackle,wheeze)={pw.cpu().numpy().round(2)}")
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
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if getattr(args, "multilabel", False):
        set_multilabel(True)
    else:
        set_classes(args.num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  outputs={LABELS}  multilabel={MULTILABEL}  clip={CLIP_S}s")

    if args.synthetic:
        manifest = make_synthetic(Path(args.out) / "synth")
    else:
        manifest = args.manifest
        if not manifest or not os.path.exists(manifest):
            raise SystemExit("Provide --manifest <csv> or use --synthetic")

    splits = build_splits(manifest, sungbook_train_frac=args.sungbook_train_frac,
                          balance_normal_ratio=(None if args.no_balance else args.balance_ratio))
    if not args.skip_file_check:
        n_before = sum(len(v) for v in splits.values())
        splits = filter_existing(splits)
        n_after = sum(len(v) for v in splits.values())
        if n_before and n_after < 0.5 * n_before:
            raise SystemExit(
                f"\n{n_before - n_after}/{n_before} audio files are missing on disk — "
                "the manifest paths look wrong. Fix the paths (see the hint above) and "
                "re-run, or pass --skip-file-check to bypass this guard.")
    print(f"[data] train={len(splits['train'])} val={len(splits['internal_val'])} "
          f"internal_test={len(splits['internal_test'])} "
          f"external_test={len(splits['external_test'])} "
          f"phone_seoul={len(splits['smartphone_seoul'])} "
          f"phone_sungbook={len(splits['smartphone_sungbook'])}")

    frontend = LogMel().to(device)
    model = build_model(args.model, num_classes=len(LABELS), width=args.width,
                        dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} width={args.width} params={n_params:,}")

    crit = make_criterion(splits["train"], device, focal_gamma=args.focal_gamma,
                          weight_power=args.weight_power)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = max(0, args.warmup_epochs)         # linear warmup then cosine decay
    def _lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(1, warmup))
        progress = float(ep - warmup) / float(max(1, args.epochs - warmup))
        return float(0.5 * (1.0 + np.cos(np.pi * min(1.0, progress))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    print(f"[sched] warmup {warmup} ep -> cosine over {args.epochs} ep  base_lr={args.lr}  seed={args.seed}")

    mixer = None
    if args.noise_dir:
        mixer = NoiseMixer(args.noise_dir, prob=args.noise_prob,
                           snr_db=tuple(args.snr_db))

    # only drop the last partial batch when there's enough data for a full one,
    # otherwise a small set (e.g. the synthetic smoke test) yields zero batches
    drop_last = len(splits["train"]) > args.batch_size
    train_ds = ClipDataset(splits["train"], training=True, mixer=mixer,
                           bandlimit_prob=args.bandlimit_prob)
    if args.bandlimit_prob:
        print(f"[aug] band-limit sim: p={args.bandlimit_prob} -> down/up-sample to 4-8 kHz")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=drop_last, num_workers=args.workers)
    if len(splits["train"]) == 0:
        raise SystemExit("No training clips — check the manifest (hospital=seoul, device=stethoscope).")
    best, patience = -1.0, 0
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); tot = 0.0
        for wav, y in train_loader:
            wav, y = wav.to(device), y.to(device)
            spec = frontend(wav)
            if args.specaugment:
                spec = spec_augment(spec)
            logits = model(spec)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * wav.size(0)
        sched.step()
        val = evaluate(model, frontend, splits["internal_val"], device, f"val (epoch {epoch})") \
            if (epoch % max(1, args.eval_every) == 0 or epoch == args.epochs - 1) else None
        vscore = val["score"] if val else None
        line = f"epoch {epoch:03d} loss={tot/len(splits['train']):.4f}"
        if vscore is not None:
            line += f" val_score={vscore:.3f}"
        print(line)
        if vscore is not None and vscore > best:
            best, patience = vscore, 0
            torch.save({"model": model.state_dict(), "labels": LABELS}, Path(args.out) / "best.pt")
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

    # per-head thresholds tuned on VALIDATION (fixes over-flagging from 0.5 + pos_weight)
    thr = 0.5
    if MULTILABEL and not args.no_threshold_tune:
        _, vt, vp = evaluate(model, frontend, splits["internal_val"], device,
                             "val (threshold tuning)", thr=0.5, return_probs=True)
        thr = tune_thresholds(vt, vp, metric=args.threshold_metric)
        print(f"\n[threshold] tuned per-head thresholds (crackle,wheeze)={thr.tolist()} "
              f"on validation via '{args.threshold_metric}' (replaces 0.5)")
        calibration_report(vt, vp)          # ECE before/after temperature scaling

    import json as _json
    results = {"best_val": best, "thresholds": (thr.tolist() if hasattr(thr, "tolist") else thr),
               "splits": {}}
    print("\n" + "=" * 60 + "\nFINAL TEST — CLEAN\n" + "=" * 60)
    for split in test_splits:
        r = evaluate(model, frontend, splits[split], device, split, thr=thr)
        if r: results["splits"][split] = r
    if args.eval_noise_dir:                 # step 1.1: inject noise into the TEST set
        for snr in args.eval_snr:
            m = NoiseMixer(args.eval_noise_dir, prob=1.0, snr_db=(snr, snr))
            print("\n" + "=" * 60 + f"\nFINAL TEST — NOISY @ {snr} dB SNR\n" + "=" * 60)
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
            patient = f"{hosp}_p{pid:03d}"; label = LABELS[pid % 3]
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
    ap.add_argument("--threshold-metric", choices=["youden", "f1"], default="youden",
                    help="per-head threshold objective tuned on val (youden=balance Se/Sp)")
    ap.add_argument("--no-threshold-tune", action="store_true",
                    help="keep fixed 0.5 threshold instead of tuning per head")
    ap.add_argument("--no-balance", action="store_true",
                    help="disable bal50 Normal-undersampling (ON by default)")
    ap.add_argument("--balance-ratio", type=float, default=0.5,
                    help="target normal:abnormal ratio for bal50 (default 0.5 = 50:50)")
    ap.add_argument("--clip-seconds", type=float, default=4.0,
                    help="clip length in seconds (Hanil: 4 s beat 2 s)")
    ap.add_argument("--weight-power", type=float, default=0.5,
                    help="class-weight strength: 1.0=full inverse (aggressive), "
                         "0.5=sqrt-inverse (gentler, default), 0.0=uniform")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="focal loss gamma (0 = plain weighted cross-entropy)")
    ap.add_argument("--model", choices=["cnn", "dscnn_tcn"], default="cnn",
                    help="cnn = depthwise CNN; dscnn_tcn = DS-CNN+TCN (MCU deploy target)")
    ap.add_argument("--width", type=float, default=1.0,
                    help="width multiplier: cnn 1.0~65k/4.0~1M; dscnn_tcn 2.0~220k(tiny)/1.0~56k(<100K MCU)")
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
    ap.add_argument("--synthetic", action="store_true", help="auto-generate a tiny test set")
    ap.add_argument("--sungbook-train-frac", type=float, default=0.0,
                    help="domain adaptation: fraction of Sungbook PATIENTS to move into "
                         "training (0.0=strict external test; try 0.3-0.5). Held-out "
                         "Sungbook patients still form external_test.")
    ap.add_argument("--skip-file-check", action="store_true",
                    help="skip the upfront file-existence validation")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
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