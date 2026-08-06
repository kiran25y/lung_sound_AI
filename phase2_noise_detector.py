#!/usr/bin/env python3
"""Phase 2 — Ambient-noise detector trained on DESED (folder-per-class layout).

A small CNN that recognises the 10 DESED domestic sound classes (Alarm_bell_ringing,
Blender, Cat, Dishes, Dog, Electric_shaver_toothbrush, Frying, Running_water, Speech,
Vacuum_cleaner). Used as a learned second-opinion to the DSP quality gate: if a lung
recording lights up an ambient-noise class strongly, flag it.

Your DESED is folders-per-class (each subdir = one class of .wav), so we train a plain
multi-class classifier on log-Mel. Tiny enough to also run on-device later if wanted.

  # train
  python phase2_noise_detector.py --train --desed-dir /workspace/dataset/noise/train \
      --epochs 25 --out runs/noise_detector

  # inference on lung clips: max ambient-class probability per clip -> flag noisy
  python phase2_noise_detector.py --inference --ckpt runs/noise_detector/best.pt \
      --manifest data/manifest.csv --out runs/noise_flags.json --flag-threshold 0.5
"""
from __future__ import annotations
import argparse, glob, json, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

SR = 16000
CLIP_S = 4.0


def load_wav(path):
    try:
        import soundfile as sf
        d, s = sf.read(path, dtype="float32", always_2d=True)
        w = torch.from_numpy(d.T.copy())
    except Exception:
        w, s = torchaudio.load(path)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    if s != SR:
        w = torchaudio.functional.resample(w, s, SR)
    return w.squeeze(0)


def fix_len(w, n):
    if w.shape[-1] == n:
        return w
    if w.shape[-1] > n:
        return w[:n]
    reps = n // w.shape[-1] + 1
    return w.repeat(reps)[:n]


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=400, win_length=400, hop_length=160,
            f_min=50, f_max=8000, n_mels=64, power=2.0)

    def forward(self, wav):
        m = torch.log(self.mel(wav) + 1e-6)
        mu = m.mean((1, 2), keepdim=True); sd = m.std((1, 2), keepdim=True) + 1e-5
        return ((m - mu) / sd).unsqueeze(1)


class NoiseCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU6(True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU6(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU6(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, n_classes)

    def forward(self, spec):
        return self.head(self.net(spec))


class DesedFolders(Dataset):
    """Each immediate subdir of desed_dir is a class; its .wav files are examples."""
    def __init__(self, desed_dir, classes, n, limit_per_class=0):
        self.items, self.classes, self.n = [], classes, n
        c2i = {c: i for i, c in enumerate(classes)}
        for c in classes:
            fs = glob.glob(os.path.join(desed_dir, c, "**", "*.wav"), recursive=True)
            if limit_per_class:
                random.Random(0).shuffle(fs); fs = fs[:limit_per_class]
            print(f"  [scan] {c}: {len(fs)} wavs")
            for f in fs:
                self.items.append((f, c2i[c]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i, _depth=0):
        f, y = self.items[i]
        try:
            return fix_len(load_wav(f), int(CLIP_S * SR)), y
        except Exception:                       # survive a rare unreadable file
            if _depth < 8:
                return self.__getitem__((i + 1) % len(self.items), _depth + 1)
            raise


def discover_classes(desed_dir):
    return sorted(d for d in os.listdir(desed_dir)
                  if os.path.isdir(os.path.join(desed_dir, d)))


def train(args):
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = discover_classes(args.desed_dir)
    print(f"[desed] {len(classes)} classes: {classes}")
    print(f"[scan] indexing wavs (limit_per_class={args.limit_per_class or 'all'}) ...")
    ds = DesedFolders(args.desed_dir, classes, int(CLIP_S * SR), args.limit_per_class)
    print(f"[data] total clips: {len(ds)}")
    if len(ds) == 0:
        raise SystemExit(f"no wavs under class folders of {args.desed_dir}")
    n_val = max(1, int(0.15 * len(ds)))
    idx = list(range(len(ds))); random.Random(0).shuffle(idx)
    val_idx = set(idx[:n_val])
    tr = torch.utils.data.Subset(ds, [i for i in idx if i not in val_idx])
    va = torch.utils.data.Subset(ds, idx[:n_val])
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                     drop_last=True, pin_memory=(device == "cuda"))
    val = DataLoader(va, batch_size=args.batch_size, num_workers=max(0, args.workers // 2))
    n_steps = len(trl)
    print(f"[train] {len(tr)} train / {len(va)} val  ->  {n_steps} steps/epoch "
          f"batch={args.batch_size} workers={args.workers} device={device}")
    fe = LogMel().to(device); model = NoiseCNN(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    best = -1.0; Path(args.out).mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train(); tot = 0.0; t0 = time.time()
        for step, (wav, y) in enumerate(trl):
            wav, y = wav.to(device), y.to(device)
            loss = crit(model(fe(wav)), y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(y)
            if step % args.log_every == 0:
                rate = (step + 1) / (time.time() - t0)
                print(f"  e{ep} step {step}/{n_steps} loss={loss.item():.3f} "
                      f"{rate:.1f} it/s ETA {(n_steps-step-1)/max(rate,1e-6)/60:.1f}m", flush=True)
        model.eval(); correct = n = 0
        with torch.no_grad():
            for wav, y in val:
                p = model(fe(wav.to(device))).argmax(1).cpu()
                correct += (p == y).sum().item(); n += len(y)
        acc = correct / max(1, n)
        print(f"epoch {ep:02d} loss={tot/len(tr):.3f} val_acc={acc:.3f} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": classes}, Path(args.out) / "best.pt")
    print(f"[done] best val_acc={best:.3f} -> {args.out}/best.pt")


@torch.no_grad()
def inference(args):
    import csv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device); classes = ck["classes"]
    fe = LogMel().to(device); model = NoiseCNN(len(classes)).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    files = ([r["filepath"] for r in csv.DictReader(open(args.manifest))] if args.manifest
             else sorted(glob.glob(os.path.join(args.wav_dir, "**", "*.wav"), recursive=True)))
    out, flagged = [], 0
    for fp in files:
        if not os.path.exists(fp):
            continue
        wav = fix_len(load_wav(fp), int(CLIP_S * SR)).unsqueeze(0).to(device)
        p = torch.softmax(model(fe(wav)), 1)[0].cpu().numpy()
        top = int(p.argmax()); conf = float(p[top])
        noisy = conf >= args.flag_threshold
        flagged += noisy
        out.append({"filepath": fp, "top_noise_class": classes[top],
                    "confidence": round(conf, 3), "flag_noisy": bool(noisy)})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"n": len(out), "flagged_noisy": flagged, "classes": classes, "clips": out},
              open(args.out, "w"), indent=2)
    print(f"[noise-detector] {len(out)} clips, {flagged} flagged noisy (>= {args.flag_threshold}) -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description="DESED ambient-noise detector")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--inference", action="store_true")
    ap.add_argument("--desed-dir", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="cap wavs per class (0=all). Use e.g. 400 for a fast first run.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt", default="runs/noise_detector/best.pt")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--flag-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="runs/noise_detector")
    args = ap.parse_args()
    if args.train:
        if not args.desed_dir:
            raise SystemExit("--desed-dir required for --train")
        train(args)
    elif args.inference:
        if args.out.endswith("noise_detector"):
            args.out = "runs/noise_flags.json"
        inference(args)
    else:
        raise SystemExit("use --train or --inference")


if __name__ == "__main__":
    main()