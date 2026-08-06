#!/usr/bin/env python3
"""STEP 2 - Transfer learning: fine-tune an AudioSet-pretrained AST on your data.

Reuses phase1_baseline for splits/metrics/loss so results are directly comparable.
Feature extraction runs in the DataLoader workers (parallel) and training uses mixed
precision, with per-step progress so you can see it moving.

  pip install transformers
  python phase1_transfer.py --manifest data/manifest.csv --epochs 15 --batch-size 8

  --batch-size 4     if you hit GPU OOM
  --freeze-backbone  quick sanity check (train only the new head)
  --workers 8        more parallel feature extraction if you have CPU cores
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import phase1_baseline as pb          # reuse splits, dataset, metrics, loss


class ASTFeatureDataset(Dataset):
    """Wraps pb.ClipDataset and returns AST input features [1024,128] + label.
    Runs the AST feature extractor inside DataLoader workers (parallelised)."""

    def __init__(self, frame, extractor, training=False, mixer=None):
        self.base = pb.ClipDataset(frame, training=training, mixer=mixer,
                                   force_mix=(mixer is not None and not training))
        self.extractor = extractor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        wav, y = self.base[i]                       # wav [T] @ 16 kHz
        feats = self.extractor(wav.numpy(), sampling_rate=pb.SR,
                               return_tensors="pt")["input_values"][0]
        return feats, y                             # [1024,128], int


def build_extractor_and_model(model_name, num_labels, device):
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    extractor = ASTFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    return extractor, model.to(device)


@torch.no_grad()
def evaluate(model, extractor, frame, device, tag, batch_size, workers, mixer=None,
             thr=0.5, return_probs=False):
    if len(frame) == 0:
        print(f"[skip] {tag}: no clips")
        return (None, [], []) if return_probs else None
    loader = DataLoader(ASTFeatureDataset(frame, extractor, mixer=mixer),
                        batch_size=batch_size, num_workers=workers)
    model.eval(); yt, yo = [], []
    for feats, y in loader:
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(input_values=feats.to(device)).logits.float()
        if pb.MULTILABEL:
            yo.extend(torch.sigmoid(logits).cpu().tolist())
        else:
            yo.extend(logits.argmax(1).cpu().tolist())
        yt.extend(y.tolist())
    res = pb.report_multilabel(yt, yo, tag, thr=thr) if pb.MULTILABEL else pb.report(yt, yo, tag)
    return (res, yt, yo) if return_probs else res


def main():
    ap = argparse.ArgumentParser(description="Fine-tune AudioSet-pretrained AST")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--num-classes", type=int, choices=[2, 3, 4], default=3)
    ap.add_argument("--multilabel", action="store_true",
                    help="independent crackle+wheeze heads (sigmoid+BCE); 'both'=[1,1]")
    ap.add_argument("--no-balance", action="store_true",
                    help="disable bal50 Normal-undersampling (ON by default)")
    ap.add_argument("--balance-ratio", type=float, default=0.5)
    ap.add_argument("--clip-seconds", type=float, default=4.0)
    ap.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--out", default="runs/ast")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-power", type=float, default=0.5)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sungbook-train-frac", type=float, default=0.0,
                    help="domain adaptation: move this fraction of Sungbook patients "
                         "into training (try 0.3-0.5); rest stays external_test")
    ap.add_argument("--noise-dir", default=None,
                    help="step 1.1: mix noise from this dir into TRAINING (noise-robust "
                         "training). Pair with --eval-noise-dir to measure the improvement.")
    ap.add_argument("--noise-prob", type=float, default=0.5,
                    help="fraction of training clips that get noise mixed in")
    ap.add_argument("--snr-db", type=float, nargs=2, default=[5, 20],
                    help="SNR range (dB) for TRAIN noise augmentation")
    ap.add_argument("--threshold-metric", choices=["youden", "f1"], default="youden",
                    help="per-head threshold objective tuned on val (youden=balance Se/Sp)")
    ap.add_argument("--no-threshold-tune", action="store_true",
                    help="keep fixed 0.5 threshold instead of tuning per head")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load runs/<out>/best and just run the final test "
                         "(fast — use to measure noise on an already-trained model)")
    ap.add_argument("--eval-noise-dir", default=None,
                    help="step 1.1: inject noise from this dir into the TEST set (e.g. MUSAN)")
    ap.add_argument("--eval-snr", type=float, nargs="+", default=[20, 10, 5],
                    help="SNR levels (dB) to evaluate the noisy test at")
    ap.add_argument("--seed", type=int, default=1337,
                    help="random seed; run 2-3 seeds and report mean +/- std for the product")
    ap.add_argument("--warmup-epochs", type=int, default=2,
                    help="linear LR warmup epochs before cosine decay")
    ap.add_argument("--min-epochs", type=int, default=0,
                    help="do not allow early-stop before this many epochs")
    args = ap.parse_args()

    import random as _random
    torch.manual_seed(args.seed); np.random.seed(args.seed); _random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    pb.CLIP_S = args.clip_seconds
    pb.set_multilabel(True) if args.multilabel else pb.set_classes(args.num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  classes={pb.LABELS}  clip={pb.CLIP_S}s  model={args.model_name}")

    splits = pb.filter_existing(
        pb.build_splits(args.manifest, sungbook_train_frac=args.sungbook_train_frac,
                        balance_normal_ratio=(None if args.no_balance else args.balance_ratio)))
    print(f"[data] train={len(splits['train'])} val={len(splits['internal_val'])} "
          f"internal_test={len(splits['internal_test'])} external_test={len(splits['external_test'])} "
          f"phone_seoul={len(splits['smartphone_seoul'])} phone_sungbook={len(splits['smartphone_sungbook'])}")

    extractor, model = build_extractor_and_model(args.model_name, len(pb.LABELS), device)
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] AST  trainable params={n_train:,}")

    crit = pb.make_criterion(splits["train"], device, focal_gamma=args.focal_gamma,
                             weight_power=args.weight_power)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    warmup = max(0, args.warmup_epochs)         # linear warmup then cosine decay
    def _lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(1, warmup))
        progress = float(ep - warmup) / float(max(1, args.epochs - warmup))
        return float(0.5 * (1.0 + np.cos(np.pi * min(1.0, progress))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    print(f"[sched] warmup {warmup} ep -> cosine over {args.epochs} ep  base_lr={args.lr}  seed={args.seed}")
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    train_mixer = None
    if args.noise_dir:                      # step 1.1: noise-robust TRAINING augmentation
        from phase1_baseline import NoiseMixer
        train_mixer = NoiseMixer(args.noise_dir, prob=args.noise_prob, snr_db=tuple(args.snr_db))
    drop_last = len(splits["train"]) > args.batch_size
    loader = DataLoader(ASTFeatureDataset(splits["train"], extractor, training=True, mixer=train_mixer),
                        batch_size=args.batch_size, shuffle=True, drop_last=drop_last,
                        num_workers=args.workers, pin_memory=(device == "cuda"))
    n_steps = len(loader)
    print(f"[train] {n_steps} steps/epoch  batch={args.batch_size}  workers={args.workers}  amp={device=='cuda'}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best, patience = -1.0, 0

    if args.eval_only:                      # skip training, just evaluate an existing checkpoint
        if not (Path(args.out) / "best").exists():
            raise SystemExit(f"--eval-only needs a trained checkpoint at {args.out}/best")
        print("[eval-only] skipping training; using existing checkpoint")
        args.epochs = 0

    for epoch in range(args.epochs):
        model.train(); tot = 0.0; t0 = time.time()
        for step, (feats, y) in enumerate(loader):
            feats = feats.to(device, non_blocking=True); y = y.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(input_values=feats).logits
                loss = crit(logits, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * y.size(0)
            if step % args.log_every == 0:
                rate = (step + 1) / (time.time() - t0)
                eta = (n_steps - step - 1) / max(rate, 1e-6)
                print(f"  e{epoch} step {step}/{n_steps} loss={loss.item():.3f} "
                      f"{rate:.1f} it/s ETA {eta/60:.1f}m", flush=True)
        sched.step()
        val = evaluate(model, extractor, splits["internal_val"], device,
                       f"val (epoch {epoch})", args.batch_size, args.workers)
        vs = val["score"] if isinstance(val, dict) else 0.0
        print(f"epoch {epoch:03d}  loss={tot/len(splits['train']):.4f}  val_score={vs:.3f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  ({(time.time()-t0)/60:.1f}m)", flush=True)
        if vs > best:
            best, patience = vs, 0
            model.save_pretrained(Path(args.out) / "best")
        else:
            patience += 1
            if patience >= args.patience and (epoch + 1) >= args.min_epochs:
                print(f"[early-stop] no improvement for {patience} evals "
                      f"(past min_epochs={args.min_epochs})"); break

    from transformers import ASTForAudioClassification
    model = ASTForAudioClassification.from_pretrained(Path(args.out) / "best").to(device)
    test_splits = ["internal_test", "external_test", "smartphone_seoul", "smartphone_sungbook"]
    results = {}

    # per-head thresholds tuned on VALIDATION (fixes over-flagging from 0.5 + pos_weight)
    thr = 0.5
    if pb.MULTILABEL and not args.no_threshold_tune:
        _, vt, vp = evaluate(model, extractor, splits["internal_val"], device,
                             "val (threshold tuning)", args.batch_size, args.workers,
                             thr=0.5, return_probs=True)
        thr = pb.tune_thresholds(vt, vp, metric=args.threshold_metric)
        print(f"\n[threshold] tuned per-head thresholds (crackle,wheeze)={thr.tolist()} "
              f"on validation via '{args.threshold_metric}' (replaces 0.5)")

    print("\n" + "=" * 60 + "\nFINAL TEST — CLEAN\n" + "=" * 60)
    for split in test_splits:
        r = evaluate(model, extractor, splits[split], device, split, args.batch_size, args.workers, thr=thr)
        if r:
            results[split] = r
    if args.eval_noise_dir:                 # step 1.1: inject noise into the TEST set
        from phase1_baseline import NoiseMixer
        for snr in args.eval_snr:
            m = NoiseMixer(args.eval_noise_dir, prob=1.0, snr_db=(snr, snr))
            print("\n" + "=" * 60 + f"\nFINAL TEST — NOISY @ {snr} dB SNR\n" + "=" * 60)
            for split in test_splits:
                r = evaluate(model, extractor, splits[split], device, f"{split}@{snr}dB",
                             args.batch_size, args.workers, mixer=m, thr=thr)
                if r:
                    results[f"{split}@{snr}dB"] = r
    json.dump({"best_val": best, "splits": results},
              open(Path(args.out) / "results.json", "w"), indent=2)
    print(f"\n[done] best val Score={best:.3f}  -> {args.out}")


if __name__ == "__main__":
    main()