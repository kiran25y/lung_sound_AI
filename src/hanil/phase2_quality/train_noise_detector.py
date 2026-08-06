"""Train the DESED_synthetic noise-event detector (Phase 2 second-opinion gate).

    python -m hanil.phase2_quality.train_noise_detector \
        --desed_dir data/noise/desed_synthetic --epochs 30

Expects DESED_synthetic soundscapes (wav) + a strong-label .tsv. Uses weak
(clip-level presence) targets with BCE. Keeps the model tiny so it also fits the
ESP32 if you later decide to ship an on-device quality gate.
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..common.audio import load_wav, fix_length
from ..common.config import Config
from ..common.features import LogMel
from .noise_detector import (NoiseEventCNN, parse_desed_tsv,
                             labels_to_multihot, DESED_CLASSES)


def _default_feature_cfg(sr=16000):
    return Config({"audio": {"sample_rate": sr, "clip_seconds": 10.0},
                   "features": {"type": "logmel", "n_fft": 400, "hop_length": 160,
                                "win_length": 400, "n_mels": 64, "f_min": 50,
                                "f_max": 8000, "log_offset": 1e-6}})


class DESEDDataset(Dataset):
    def __init__(self, desed_dir: str, cfg):
        tsvs = glob.glob(os.path.join(desed_dir, "**", "*.tsv"), recursive=True)
        self.labels = {}
        for t in tsvs:
            self.labels.update(parse_desed_tsv(t))
        wavs = glob.glob(os.path.join(desed_dir, "**", "*.wav"), recursive=True)
        self.items = [(w, self.labels.get(os.path.basename(w), [])) for w in wavs]
        self.cfg = cfg
        self.sr = cfg.audio["sample_rate"]
        self.n = int(cfg.audio["clip_seconds"] * self.sr)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, labels = self.items[i]
        wav = fix_length(load_wav(path, self.sr), self.n, training=True)
        return wav, labels_to_multihot(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desed_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="runs/noise_detector/best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _default_feature_cfg()
    frontend = LogMel(cfg).to(device)
    ds = DESEDDataset(args.desed_dir, cfg)
    if len(ds) == 0:
        raise FileNotFoundError(f"no DESED wav/tsv found under {args.desed_dir}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = NoiseEventCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train(); tot = 0.0
        for wav, y in loader:
            wav, y = wav.to(device), y.to(device)
            logits = model(frontend(wav))
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * wav.size(0)
        print(f"epoch {epoch:03d}  bce={tot/len(ds):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": DESED_CLASSES}, args.out)
    print(f"[done] saved -> {args.out}")


if __name__ == "__main__":
    main()
