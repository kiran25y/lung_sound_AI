"""Phase 1 trainer — runs Experiments 1/2/3, the teacher, and the final model.

    python -m hanil.phase1_classification.train --config configs/exp3_final.yaml

Experiment selection is driven entirely by the config:
  * noise.enable                      -> Exp 1 (noise-robust augmentation, MUSAN)
  * device_robust.mixstyle/contrastive-> Exp 2 (device-robust)
  * distill.enable                    -> distil from the AST teacher
Any combination is valid; exp3_final.yaml turns everything on.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common.config import load_config
from ..common.seed import set_seed
from ..common.features import LogMel
from .datasets import LungSoundDataset, build_splits
from .augment.noise_mixer import NoiseMixer
from .augment.spec_augment import SpecAugment
from .contrastive import StethoGuidedSupConLoss
from .distill import DistillationLoss
from .losses import build_loss
from .models import build_model
from .evaluate import evaluate_all


def make_collate(mixer: NoiseMixer | None):
    def collate(batch):
        wavs, labels, domains = zip(*batch)
        if mixer is not None:
            wavs = [mixer(w) for w in wavs]
        wav = torch.stack(wavs)
        return wav, torch.tensor(labels), torch.tensor(domains)
    return collate


def build_teacher(cfg, device):
    if not cfg.distill.get("enable"):
        return None
    ckpt = cfg.distill.get("teacher_ckpt")
    if not ckpt or not os.path.exists(ckpt):
        print(f"[distill] teacher ckpt '{ckpt}' not found — training without distillation")
        return None
    import copy
    tcfg = type(cfg)(copy.deepcopy(cfg)); tcfg["model"] = {"arch": "teacher_ast"}
    teacher = build_model(tcfg).to(device)
    teacher.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 1337))
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.paths["out_dir"]) / cfg.get("experiment", "run")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits(cfg.paths["manifest"], seed=cfg.get("seed", 1337))
    train_ds = LungSoundDataset(splits["train"], cfg, training=True)
    val_ds = LungSoundDataset(splits["internal_val"], cfg, training=False)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  "
          f"class_counts={train_ds.class_counts().tolist()}")

    mixer = NoiseMixer(cfg) if cfg.get_path("noise.enable") else None
    train_loader = DataLoader(train_ds, batch_size=cfg.train["batch_size"], shuffle=True,
                              num_workers=cfg.train.get("num_workers", 4),
                              collate_fn=make_collate(mixer), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train["batch_size"], shuffle=False,
                            num_workers=cfg.train.get("num_workers", 4),
                            collate_fn=make_collate(None))

    frontend = LogMel(cfg).to(device)
    specaug = SpecAugment(cfg)
    model = build_model(cfg).to(device)
    teacher = build_teacher(cfg, device)

    base_loss = build_loss(cfg, train_ds.class_counts(), device)
    distill_loss = DistillationLoss(base_loss,
                                    temperature=float(cfg.get_path("distill.temperature", 4.0)),
                                    alpha=float(cfg.get_path("distill.alpha", 0.7)))
    contrastive = None
    if cfg.get_path("device_robust.contrastive.enable"):
        contrastive = StethoGuidedSupConLoss(
            temperature=float(cfg.get_path("device_robust.contrastive.temperature", 0.1)))
        con_w = float(cfg.get_path("device_robust.contrastive.weight", 0.5))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train["lr"]),
                            weight_decay=float(cfg.train["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.get("amp", True) and device == "cuda")

    best_score, patience = -1.0, 0
    for epoch in range(cfg.train["epochs"]):
        model.train()
        running = 0.0
        for wav, labels, domains in train_loader:
            wav, labels, domains = wav.to(device), labels.to(device), domains.to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                spec = frontend(wav)
                spec = specaug(spec)
                want_emb = contrastive is not None
                out = model(spec, return_embedding=True) if want_emb else model(spec)
                logits, emb = out if want_emb else (out, None)

                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(spec)
                loss = distill_loss(logits, labels, teacher_logits)
                if contrastive is not None:
                    loss = loss + con_w * contrastive(emb, labels, domains)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * wav.size(0)
        sched.step()

        val_rep = _quick_val(model, val_loader, frontend, device, cfg)
        score = val_rep.icbhi_style_score
        print(f"epoch {epoch:03d}  train_loss={running/len(train_ds):.4f}  "
              f"val_acc={val_rep.accuracy:.4f}  val_score={score:.4f}")

        if score > best_score:
            best_score, patience = score, 0
            torch.save({"model": model.state_dict(), "cfg": dict(cfg), "score": score},
                       out_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg.train.get("early_stop_patience", 12):
                print(f"[early-stop] no val improvement for {patience} epochs")
                break

    # Final evaluation on all requested splits
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    results = evaluate_all(model, splits, cfg, device)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"val_score": best_score, "splits": results}, f, indent=2)
    print(f"\n[done] best val score={best_score:.4f}  -> {out_dir/'results.json'}")


@torch.no_grad()
def _quick_val(model, loader, frontend, device, cfg):
    from ..common.metrics import evaluate as eval_metrics
    model.eval()
    yt, yp = [], []
    for wav, labels, _ in loader:
        wav = wav.to(device)
        out = model(frontend(wav))
        logits = out[0] if isinstance(out, tuple) else out
        yp.extend(logits.argmax(1).cpu().tolist())
        yt.extend(labels.tolist())
    return eval_metrics(yt, yp, cfg.classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    train(ap.parse_args().config)


if __name__ == "__main__":
    main()
