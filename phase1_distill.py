#!/usr/bin/env python3
"""Knowledge distillation: AST teacher -> small deployable student.

Recovers most of the ~5-8 point gap we measured between the from-scratch student and
the AST teacher, in a model that still fits the device (mobile 1-5M / MCU <100K).

Teacher: your fine-tuned AST checkpoint (runs/ast_noise/best), frozen.
Student: BaselineCNN (width sets size). Loss = alpha * distillation + (1-alpha) * hard.
  * Multi-label: distillation = BCE(student_logits, sigmoid(teacher_logits / T))  (soft targets)
  * plus hard BCE against the true [crackle,wheeze] labels.

Reuses phase1_baseline (pb) for data/splits/metrics/loss/threshold/calibration, so the
recipe (bal50, keep-both->[1,1], domain-adapt, noise aug, eval) is identical.

  python phase1_distill.py --manifest data/manifest.csv --multilabel \
      --teacher-ckpt runs/ast_noise/best --sungbook-train-frac 0.3 \
      --noise-dir /workspace/dataset/noise/musan /workspace/dataset/noise/train \
      --width 4.0 --epochs 80 --warmup-epochs 5 --min-epochs 40 \
      --eval-noise-dir /workspace/dataset/noise/musan --eval-snr 20 10 5 --out runs/student_distill

For MCU <100K use --width 1.0 (~65k params). QAT note: after this trains, run
phase4_deploy/quantize_tflite.py; if INT8 hurts crackle/wheeze sensitivity, retrain
with quantization-aware training (torch.ao.quantization) — the architecture is INT8-safe.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import phase1_baseline as pb


@torch.no_grad()
def teacher_logits_for(model, extractor, wav_batch, device):
    arrays = [w.numpy() for w in wav_batch]
    feats = extractor(arrays, sampling_rate=pb.SR, return_tensors="pt")["input_values"].to(device)
    with torch.autocast(device_type="cuda", enabled=device == "cuda"):
        return model(input_values=feats).logits.float()


def main():
    ap = argparse.ArgumentParser(description="Distill AST teacher into a small student")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--multilabel", action="store_true")
    ap.add_argument("--teacher-ckpt", default="runs/ast_noise/best")
    ap.add_argument("--teacher-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--width", type=float, default=4.0, help="student width (4.0~1M, 1.0~65k MCU)")
    ap.add_argument("--kd-temperature", type=float, default=3.0)
    ap.add_argument("--kd-alpha", type=float, default=0.7, help="weight on soft (teacher) loss")
    ap.add_argument("--sungbook-train-frac", type=float, default=0.0)
    ap.add_argument("--noise-dir", nargs="+", default=None)
    ap.add_argument("--noise-prob", type=float, default=0.5)
    ap.add_argument("--snr-db", type=float, nargs=2, default=[5, 20])
    ap.add_argument("--specaugment", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clip-seconds", type=float, default=4.0)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--balance-ratio", type=float, default=0.5)
    ap.add_argument("--eval-noise-dir", nargs="+", default=None)
    ap.add_argument("--eval-snr", type=float, nargs="+", default=[20, 10, 5])
    ap.add_argument("--out", default="runs/student_distill")
    args = ap.parse_args()

    pb.CLIP_S = args.clip_seconds
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.multilabel:
        pb.set_multilabel(True)
    else:
        raise SystemExit("distillation script currently targets --multilabel")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  outputs={pb.LABELS}  student_width={args.width}")

    splits = pb.filter_existing(
        pb.build_splits(args.manifest, sungbook_train_frac=args.sungbook_train_frac,
                        balance_normal_ratio=(None if args.no_balance else args.balance_ratio)))
    print(f"[data] train={len(splits['train'])} val={len(splits['internal_val'])}")

    # teacher (frozen)
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    extractor = ASTFeatureExtractor.from_pretrained(args.teacher_name)
    teacher = ASTForAudioClassification.from_pretrained(args.teacher_ckpt).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[teacher] loaded {args.teacher_ckpt} (frozen)")

    # student
    frontend = pb.LogMel().to(device)
    student = pb.BaselineCNN(num_classes=len(pb.LABELS), width=args.width).to(device)
    print(f"[student] BaselineCNN width={args.width} params={sum(p.numel() for p in student.parameters()):,}")

    hard_loss = pb.make_criterion(splits["train"], device)      # BCE pos_weight
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = max(0, args.warmup_epochs)
    def _lr(ep):
        if ep < warmup: return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, args.epochs - warmup)
        return float(0.5 * (1 + np.cos(np.pi * min(1.0, prog))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)

    mixer = pb.NoiseMixer(args.noise_dir, prob=args.noise_prob, snr_db=tuple(args.snr_db)) \
        if args.noise_dir else None
    train_ds = pb.ClipDataset(splits["train"], training=True, mixer=mixer)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    T, alpha = args.kd_temperature, args.kd_alpha
    best, patience = -1.0, 0

    for epoch in range(args.epochs):
        student.train(); tot = 0.0; t0 = time.time()
        for wav, y in loader:
            wav_d, y = wav.to(device), y.to(device)
            spec = frontend(wav_d)
            if args.specaugment:
                spec = pb.spec_augment(spec)
            s_logits = student(spec)
            with torch.no_grad():
                t_logits = teacher_logits_for(teacher, extractor, wav, device)
                soft_targets = torch.sigmoid(t_logits / T)
            soft = nn.functional.binary_cross_entropy_with_logits(s_logits, soft_targets)
            hard = hard_loss(s_logits, y)
            loss = alpha * soft + (1 - alpha) * hard
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * wav_d.size(0)
        sched.step()
        val = pb.evaluate(student, frontend, splits["internal_val"], device, f"val (epoch {epoch})")
        vs = val["score"] if isinstance(val, dict) else 0.0
        print(f"epoch {epoch:03d} loss={tot/len(train_ds):.4f} val_score={vs:.3f} "
              f"lr={opt.param_groups[0]['lr']:.2e} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if vs > best:
            best, patience = vs, 0
            torch.save({"model": student.state_dict(), "labels": pb.LABELS}, Path(args.out) / "best.pt")
        else:
            patience += 1
            if patience >= args.patience and (epoch + 1) >= args.min_epochs:
                print(f"[early-stop] no improvement for {patience} evals"); break

    student.load_state_dict(torch.load(Path(args.out) / "best.pt", map_location=device)["model"])
    thr = 0.5
    _, vt, vp = pb.evaluate(student, frontend, splits["internal_val"], device,
                            "val (threshold tuning)", return_probs=True)
    thr = pb.tune_thresholds(vt, vp)
    print(f"\n[threshold] tuned (crackle,wheeze)={thr.tolist()}")
    pb.calibration_report(vt, vp)
    tests = ["internal_test", "external_test", "smartphone_seoul", "smartphone_sungbook"]
    print("\n" + "=" * 60 + "\nSTUDENT FINAL TEST — CLEAN\n" + "=" * 60)
    for s in tests:
        pb.evaluate(student, frontend, splits[s], device, s, thr=thr)
    if args.eval_noise_dir:
        for snr in args.eval_snr:
            m = pb.NoiseMixer(args.eval_noise_dir, prob=1.0, snr_db=(snr, snr))
            print("\n" + "=" * 60 + f"\nSTUDENT FINAL TEST — NOISY @ {snr} dB\n" + "=" * 60)
            for s in tests:
                pb.evaluate(student, frontend, splits[s], device, f"{s}@{snr}dB", mixer=m, thr=thr)
    print(f"\n[done] distilled student best val={best:.3f} -> {args.out}/best.pt")
    print("Next: quantize with phase4_deploy/quantize_tflite.py (QAT if INT8 hurts sensitivity).")


if __name__ == "__main__":
    main()