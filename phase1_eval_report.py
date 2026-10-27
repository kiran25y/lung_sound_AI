#!/usr/bin/env python3
"""phase1_eval_report.py — auditable, read-only evaluation of a FROZEN classifier.

Produces the numbers a product review actually needs, without touching training or
peeking thresholds on the test set. Four things, all from one frozen checkpoint:

  1.3 HEADLINE  train Seoul steth -> test smartphone + HELD-OUT noise, Score=(Sp+Se)/2
                and false_normal at a FROZEN threshold. This is the real product condition
                (device shift + noise + optionally cross-site), reported as one worst-case
                number instead of buried across splits.

  BOOTSTRAP CIs 95% confidence intervals on Score and false_normal for every condition
                (resampling clips), so each number is a range, not a point estimate.

  GATE-COND     classifier performance on ALL clips vs GATE-ACCEPTED clips, plus the
                rejection rate. Answers the circularity worry: "is the model only good
                because the quality gate hid the hard clips?"

  UNSEEN NOISE  point --eval-noise-dir at a noise set NOT used in training (e.g. DEMAND)
                to measure robustness to noise the model never saw.

Thresholds are FROZEN: pass --thresholds (a results.json from training, or a plain JSON
list). We never re-tune on the test data here.

Example:
  python phase1_eval_report.py \
      --manifest data/manifest.csv --ckpt runs/cnn/best.pt --model cnn --width 1.0 \
      --thresholds runs/cnn/results.json \
      --eval-noise-dir /workspace/dataset/noise/demand --eval-snr 20 15 10 5 \
      --gate-config runs/gate_thresholds.json --bootstrap 1000 \
      --out runs/eval_report.json
"""
from __future__ import annotations
import argparse, json, os
from math import gcd
import numpy as np
import torch

import phase1_baseline as pb           # reuse splits / model / eval / noise mixer
from phase2_quality_gate import QualityGate, QualityConfig

TEST_SPLITS = ["internal_test", "external_test", "smartphone_seoul", "smartphone_sungbook"]
# the 1.3 product condition = trained on Seoul steth, tested on a smartphone under noise.
# smartphone_sungbook is the worst case (device + cross-site + noise).
HEADLINE_SPLITS = ["smartphone_seoul", "smartphone_sungbook"]


# ───────────────────────── metrics (mirror report_multilabel math) ───────────
def _states(T, pred):
    key = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    ts = np.array([key[(int(a), int(b))] for a, b in T[:, :2].astype(int)])
    ps = np.array([key[(int(a), int(b))] for a, b in pred[:, :2].astype(int)])
    return ts, ps


def compute_metrics(T, P, thr):
    """macro Score = mean over heads of 0.5*(Se+Sp); false_normal = missed abnormal / abnormal."""
    T = np.asarray(T, float); P = np.asarray(P, float)
    thr = np.asarray(thr if np.ndim(thr) else [thr] * P.shape[1], float)
    pred = (P >= thr).astype(int)
    scores = []
    for i in range(P.shape[1]):
        t, p = T[:, i], pred[:, i]
        tp = ((t == 1) & (p == 1)).sum(); fn = ((t == 1) & (p == 0)).sum()
        fp = ((t == 0) & (p == 1)).sum(); tn = ((t == 0) & (p == 0)).sum()
        se = tp / (tp + fn) if (tp + fn) else 0.0
        sp = tn / (tn + fp) if (tn + fp) else 0.0
        scores.append(0.5 * (se + sp))
    ts, ps = _states(T, pred)
    n_abn = int((ts != 0).sum())
    missed = int(((ts != 0) & (ps == 0)).sum())
    false_normal = missed / n_abn if n_abn else 0.0
    return float(np.mean(scores)), float(false_normal)


def split_frame_by_patient(frame, frac, seed=7):
    """Carve `frac` of a frame's PATIENTS into a validation slice, rest stays as test.
    Patient-disjoint, so thresholds tuned on the val slice never touch the reported test."""
    if frac <= 0 or len(frame) == 0:
        return frame.iloc[0:0], frame
    pats = sorted(frame.patient_id.unique())
    rng = np.random.default_rng(seed); rng.shuffle(pats)
    k = int(round(frac * len(pats))); vp = set(pats[:k])
    return frame[frame.patient_id.isin(vp)], frame[~frame.patient_id.isin(vp)]


def pick_thresholds(T, P, target_se=None):
    """Per-head threshold selection.
      target_se=None -> Youden J (balanced Se+Sp-1).
      target_se=x    -> highest threshold (best specificity) that still keeps Se >= x
                        (the screening operating point: bound the missed-abnormal rate).
    """
    T = np.asarray(T); P = np.asarray(P); grid = np.linspace(0.05, 0.95, 19); out = []
    for i in range(P.shape[1]):
        t, p = T[:, i], P[:, i]; best = 0.5
        if target_se is None:
            bestv = -1.0
            for th in grid:
                pr = (p >= th).astype(int)
                tp = ((t == 1) & (pr == 1)).sum(); fn = ((t == 1) & (pr == 0)).sum()
                fp = ((t == 0) & (pr == 1)).sum(); tn = ((t == 0) & (pr == 0)).sum()
                se = tp / (tp + fn) if (tp + fn) else 0.0
                sp = tn / (tn + fp) if (tn + fp) else 0.0
                if se + sp - 1 > bestv:
                    bestv, best = se + sp - 1, th
        else:
            chosen = None
            for th in grid:                       # ascending: keep the highest th that still meets target
                pr = (p >= th).astype(int)
                tp = ((t == 1) & (pr == 1)).sum(); fn = ((t == 1) & (pr == 0)).sum()
                se = tp / (tp + fn) if (tp + fn) else 0.0
                if se >= target_se:
                    chosen = th
            best = chosen if chosen is not None else grid[0]   # unattainable -> lowest th (max Se)
        out.append(round(float(best), 3))
    return np.array(out)


def bootstrap_ci(T, P, thr, B=1000, seed=0):
    T = np.asarray(T, float); P = np.asarray(P, float)
    if len(T) == 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(T); ms, fns = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        m, fn = compute_metrics(T[idx], P[idx], thr)
        ms.append(m); fns.append(fn)
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {"score_ci95": ci(ms), "false_normal_ci95": ci(fns)}


# ───────────────────────── audio for the gate ───────────────────────────────
def load16k(path, sr=16000):
    import soundfile as sf
    d, s = sf.read(path, dtype="float32", always_2d=True)
    x = d.mean(1)
    if s != sr:
        from scipy.signal import resample_poly
        g = gcd(int(s), sr)
        x = resample_poly(x, sr // g, int(s) // g)
    return x.astype(np.float64)


def gated_condition(model, frontend, gate, frame, device, mixer, thr):
    """Product-real evaluation: mix noise -> run the quality GATE -> classify ONLY what passes.
    Returns classifier metrics on gate-accepted clips + how many the gate rejected."""
    from torch.utils.data import DataLoader
    if len(frame) == 0:
        return None
    ds = pb.ClipDataset(frame, mixer=mixer, force_mix=(mixer is not None))
    loader = DataLoader(ds, batch_size=1)
    Tt, Pp, npass, ntot = [], [], 0, 0
    model.eval()
    for wav, y in loader:
        ntot += 1
        if not gate.check(wav[0].cpu().numpy()).passed:
            continue                                   # gate rejects -> RETRY, never classified
        npass += 1
        with torch.no_grad():
            p = torch.sigmoid(model(frontend(wav.to(device))))[0].cpu().tolist()
        Pp.append(p); Tt.append(y[0].tolist())
    rej = 1.0 - npass / max(ntot, 1)
    if npass == 0:
        return {"n_total": ntot, "n_pass": 0, "rejection_rate": round(rej, 3)}
    sc, fn = compute_metrics(Tt, Pp, thr)
    return {"n_total": ntot, "n_pass": npass, "rejection_rate": round(rej, 3),
            "score": round(sc, 3), "false_normal": round(fn, 3)}


def gate_pass_mask(frame, cfg):
    """Boolean mask (frame order) — True where the quality gate would PASS the clip."""
    gate = QualityGate(cfg)
    mask = []
    for fp in frame["filepath"]:
        try:
            mask.append(bool(gate.check(load16k(fp)).passed))
        except Exception:
            mask.append(False)          # unreadable clip = not accepted
    return np.asarray(mask, bool)


# ───────────────────────── checkpoint loader (auto-width) ───────────────────
def load_model(args, device):
    """Load a checkpoint, auto-detecting the width if --width doesn't match its shapes.
    Avoids the common 'size mismatch' error when you forget the trained width."""
    state = torch.load(args.ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    tried = [args.width] + [w for w in (0.5, 1.0, 1.5, 2.0, 3.0) if w != args.width]
    last = None
    for w in tried:
        m = pb.build_model(args.model, num_classes=len(pb.LABELS), width=w,
                           dropout=args.dropout).to(device)
        try:
            m.load_state_dict(sd); m.eval()
            if w != args.width:
                print(f"[model] --width {args.width} did not match the checkpoint; "
                      f"auto-detected width={w}")
            return m, w
        except Exception as e:
            last = e
    raise SystemExit(f"[error] could not load {args.ckpt} as '{args.model}' at any width "
                     f"{tried}. Wrong --model? Last error: {last}")


# ───────────────────────── thresholds (frozen) ──────────────────────────────
def load_thresholds(path):
    if not path:
        print("[thr] no --thresholds given -> using 0.5 (NOT recommended for the product number)")
        return 0.5
    with open(path) as f:
        obj = json.load(f)
    thr = obj.get("thresholds", obj) if isinstance(obj, dict) else obj
    thr = np.asarray(thr, float)
    print(f"[thr] frozen per-head thresholds (crackle,wheeze) = {thr.tolist()} (from {path})")
    return thr


# ───────────────────────── main ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Auditable frozen-model evaluation (1.3 + CIs + gate).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="cnn",
                    choices=["cnn", "dscnn_tcn", "sram_dscnn"])
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--thresholds", default=None, help="results.json or JSON list of per-head thresholds")
    ap.add_argument("--sungbook-train-frac", type=float, default=0.0,
                    help="reproduce the domain-adapt split used at training time")
    ap.add_argument("--smartphone-train-frac", type=float, default=0.0,
                    help="reproduce the device-adapt split used at training time")
    ap.add_argument("--eval-noise-dir", default=None,
                    help="noise dir NOT used in training (e.g. DEMAND) -> unseen-noise robustness")
    ap.add_argument("--eval-snr", type=float, nargs="*", default=[20, 15, 10, 5])
    ap.add_argument("--gate-config", default=None, help="calibrated QualityConfig JSON (else defaults)")
    ap.add_argument("--gate-splits", nargs="*", default=["internal_test", "external_test"],
                    help="splits to run gate-conditioned reporting on (clean audio)")
    ap.add_argument("--tune-thresholds", action="store_true",
                    help="ignore --thresholds; pick per-head thresholds on a mixed-device val slice")
    ap.add_argument("--target-se", type=float, default=None,
                    help="with --tune-thresholds: choose thresholds giving >= this abnormal "
                         "sensitivity per head (screening op-point). Omit -> Youden J.")
    ap.add_argument("--val-phone-frac", type=float, default=0.3,
                    help="with --tune-thresholds: fraction of phone/external TEST patients to "
                         "reserve for threshold tuning (disjoint from the reported test).")
    ap.add_argument("--gate-noisy", action="store_true",
                    help="product-real: run the quality gate on the noisy clips and score only "
                         "what passes (low-SNR clips become RETRY, not misclassified)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for CIs (0=off)")
    ap.add_argument("--out", default="runs/eval_report.json")
    args = ap.parse_args()

    pb.set_multilabel(True)             # independent crackle/wheeze heads
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thr = load_thresholds(args.thresholds)

    splits = pb.build_splits(args.manifest, sungbook_train_frac=args.sungbook_train_frac,
                             smartphone_train_frac=args.smartphone_train_frac,
                             balance_normal_ratio=None)   # no balancing on TEST sets
    splits = pb.filter_existing(splits)                    # drop missing files (keeps gate/probs aligned)

    frontend = pb.LogMel().to(device)
    model, used_width = load_model(args, device)
    print(f"[model] {args.model} width={used_width} "
          f"params={sum(p.numel() for p in model.parameters()):,}  ckpt={args.ckpt}")

    thr_source = f"frozen:{args.thresholds}"
    if args.tune_thresholds:
        # build a mixed-device validation slice (steth val + a patient-disjoint slice of the
        # phone/external tests) and pick thresholds there. The reported tests shrink accordingly.
        val_frames = []
        iv = splits.get("internal_val")
        if iv is not None and len(iv):
            val_frames.append(iv)
        for s in ["smartphone_seoul", "smartphone_sungbook", "external_test"]:
            vf, tf = split_frame_by_patient(splits[s], args.val_phone_frac)
            splits[s] = tf
            if len(vf):
                val_frames.append(vf)
        Tv, Pv = [], []
        for vf in val_frames:
            _, yt, yo = pb.evaluate(model, frontend, vf, device, "thr-val",
                                    mixer=None, thr=0.5, return_probs=True)
            Tv += yt; Pv += yo
        thr = pick_thresholds(Tv, Pv, target_se=args.target_se)
        thr_source = (f"tuned on mixed-device val (n={len(Tv)} clips), "
                      f"objective={'Se>=%.2f' % args.target_se if args.target_se else 'youden'}")
        print(f"[thr] {thr_source} -> per-head (crackle,wheeze)={thr.tolist()}")

    report = {"provenance": {
        "checkpoint": args.ckpt, "manifest": args.manifest, "model": args.model,
        "width": args.width, "thresholds": (thr.tolist() if hasattr(thr, "tolist") else thr),
        "threshold_source": thr_source,
        "split_rule": "patient-level, leakage-free; smartphone-test patients disjoint from train",
        "sungbook_train_frac": args.sungbook_train_frac,
        "smartphone_train_frac": args.smartphone_train_frac,
        "metric": "Score = 0.5*(Se+Sp) macro over crackle/wheeze heads; false_normal = missed abnormal / abnormal",
    }, "clip_counts": {k: int(len(v)) for k, v in splits.items()},
        "conditions": {}}

    def run(tag, frame, mixer, seed):
        res, yt, yo = pb.evaluate(model, frontend, frame, device, tag, mixer=mixer,
                                  thr=thr, return_probs=True)
        if res is None:
            return None
        entry = {"score": res["score"], "false_normal": res["false_normal"],
                 "macro_f1": res.get("macro_f1"), "n_clips": len(yt),
                 "per_label": res.get("per_label")}
        if args.bootstrap:
            ci = bootstrap_ci(yt, yo, thr, B=args.bootstrap, seed=seed)
            if ci: entry.update(ci)
        report["conditions"][tag] = entry
        return entry

    # ── clean baselines (all splits) ─────────────────────────────────────────
    print("\n" + "=" * 64 + "\nCLEAN — all test splits\n" + "=" * 64)
    for i, s in enumerate(TEST_SPLITS):
        run(s, splits[s], None, seed=i)

    # ── 1.3 headline: smartphone + unseen noise ──────────────────────────────
    if args.eval_noise_dir:
        noise_name = os.path.basename(args.eval_noise_dir.rstrip("/")) or "noise"
        print("\n" + "=" * 64 + f"\n1.3 STRESS — smartphone + UNSEEN noise ({noise_name})\n" + "=" * 64)
        for snr in args.eval_snr:
            m = pb.NoiseMixer(args.eval_noise_dir, prob=1.0, snr_db=(snr, snr))
            for j, s in enumerate(TEST_SPLITS):
                run(f"{s}@{noise_name}{int(snr)}dB", splits[s], m, seed=100 + int(snr) * 10 + j)

    # ── gate-conditioned (clean): all vs accepted + rejection rate ───────────
    cfg = QualityConfig.from_json(args.gate_config) if args.gate_config else QualityConfig()
    print("\n" + "=" * 64 + "\nGATE-CONDITIONED — classifier on ALL vs GATE-ACCEPTED clips\n" + "=" * 64)
    report["gate_conditioned"] = {}
    for s in args.gate_splits:
        frame = splits.get(s)
        if frame is None or len(frame) == 0:
            continue
        _, yt, yo = pb.evaluate(model, frontend, frame, device, f"{s} (for gate-cond)",
                                mixer=None, thr=thr, return_probs=True)
        yt = np.asarray(yt, float); yo = np.asarray(yo, float)
        mask = gate_pass_mask(frame, cfg)                  # frame order == eval order (no shuffle)
        rej = float(1.0 - mask.mean()) if len(mask) else 0.0
        all_score, all_fn = compute_metrics(yt, yo, thr)
        if mask.any():
            acc_score, acc_fn = compute_metrics(yt[mask], yo[mask], thr)
        else:
            acc_score = acc_fn = float("nan")
        entry = {"n_clips": int(len(yt)), "rejection_rate": round(rej, 3),
                 "all": {"score": round(all_score, 3), "false_normal": round(all_fn, 3)},
                 "accepted": {"score": round(acc_score, 3), "false_normal": round(acc_fn, 3),
                              "n_clips": int(mask.sum())}}
        report["gate_conditioned"][s] = entry
        print(f"  {s:<20} reject={rej:5.1%}  ALL Score={all_score:.3f} fn={all_fn:.3f}  "
              f"ACCEPTED Score={acc_score:.3f} fn={acc_fn:.3f}  (n {int(mask.sum())}/{len(yt)})")

    # ── product-real: gate the NOISY clips, score only what passes ───────────
    if args.gate_noisy and args.eval_noise_dir:
        gate = QualityGate(cfg)
        noise_name = os.path.basename(args.eval_noise_dir.rstrip("/")) or "noise"
        print("\n" + "=" * 64 + "\nGATED-NOISY — gate rejects low-SNR clips; classify only PASS\n" + "=" * 64)
        report["gated_noisy"] = {}
        gsplits = ["smartphone_seoul", "smartphone_sungbook", "internal_test", "external_test"]
        for snr in args.eval_snr:
            m = pb.NoiseMixer(args.eval_noise_dir, prob=1.0, snr_db=(snr, snr))
            for s in gsplits:
                e = gated_condition(model, frontend, gate, splits[s], device, m, thr)
                if not e:
                    continue
                report["gated_noisy"][f"{s}@{int(snr)}dB"] = e
                sc = e.get("score"); fn = e.get("false_normal")
                extra = (f"Score={sc:.3f} fn={fn:.3f}" if sc is not None else "no clips passed")
                print(f"  {s:<20}@{int(snr):>2}dB  gate_reject={e['rejection_rate']:5.1%}  "
                      f"classified {e['n_pass']}/{e['n_total']}  {extra}")

    # ── 1.3 headline summary ─────────────────────────────────────────────────
    print("\n" + "=" * 64 + "\n1.3 PRODUCT HEADLINE (worst-case operating point)\n" + "=" * 64)
    headline = {}
    for s in HEADLINE_SPLITS:
        # worst SNR available for this split, else clean
        keys = [k for k in report["conditions"] if k.startswith(s + "@")]
        pick = min(keys, key=lambda k: report["conditions"][k]["score"]) if keys else s
        e = report["conditions"].get(pick)
        if e:
            headline[s] = {"condition": pick, **{k: e[k] for k in
                           ("score", "false_normal", "score_ci95", "false_normal_ci95")
                           if k in e}}
            ci = e.get("score_ci95")
            ci_s = f" 95%CI[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
            print(f"  {s:<20} {pick:<28} Score={e['score']:.3f}{ci_s}  "
                  f"false_normal={e['false_normal']:.3f}")
    report["headline_1_3"] = headline

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] auditable report -> {args.out}")


if __name__ == "__main__":
    main()
