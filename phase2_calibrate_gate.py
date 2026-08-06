#!/usr/bin/env python3
"""Phase 2 — Calibrate the quality-gate thresholds on YOUR data (Codex: "quality gate
with its own labels and thresholds"). Turns the default cutoffs into data-driven ones.

Two modes:
  1) LABELED  — you provide good vs bad clips:
       python phase2_calibrate_gate.py --good good.csv --bad bad.csv --out runs/gate_thresholds.json
     (csv with a `filepath` column, or a folder path)
  2) AUTO     — no labels: synthesize BAD clips from your GOOD clips (clip, add noise,
     drop breath) and calibrate against them:
       python phase2_calibrate_gate.py --good good_clips.csv --auto --out runs/gate_thresholds.json

For each gate feature it finds the cut that best separates good/bad (Youden's J on the
ROC), then writes a thresholds JSON that phase2_quality_gate.py loads with --thresholds.
Review the report before trusting it — a clinician should sanity-check the labels.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, random
import numpy as np
from phase2_quality_gate import QualityGate, QualityConfig, _load_wav


def _files(spec):
    if os.path.isdir(spec):
        return sorted(glob.glob(os.path.join(spec, "**", "*.wav"), recursive=True))
    return [r["filepath"] for r in csv.DictReader(open(spec))]


def _degrade(x, sr, rng):
    """Synthesize a BAD version of a good clip (one random corruption)."""
    mode = rng.choice(["clip", "noise", "silence", "rustle"])
    if mode == "clip":
        return np.clip(x * rng.uniform(4, 12), -1, 1)
    if mode == "noise":                       # very low SNR
        n = rng.normal(0, 1, len(x)); snr = rng.uniform(-5, 5)
        sp = np.mean(x ** 2) + 1e-9; sc = np.sqrt(sp / (np.mean(n ** 2) * 10 ** (snr / 10)))
        return x + sc * n
    if mode == "silence":                     # no breath
        return rng.normal(0, 1e-3, len(x))
    hf = rng.normal(0, 0.3, len(x))           # rustle: broadband hiss bursts
    mask = (rng.random(len(x)) < 0.02).astype(float)
    return x * 0.3 + hf * np.convolve(mask, np.ones(200), "same").clip(0, 1)


def _collect(files, gate, sr, degrade=False, rng=None, cap=600):
    feats = []
    for fp in files[:cap]:
        if not os.path.exists(fp):
            continue
        try:
            x = _load_wav(fp, sr)
            if degrade:
                x = _degrade(x, sr, rng)
            x = x / (np.max(np.abs(x)) + 1e-9)
            feats.append(gate._feat(x))
        except Exception:
            continue
    return feats


def _best_cut(good, bad, higher_is_bad):
    """Youden-J threshold separating good vs bad for one feature."""
    g = np.array(good); b = np.array(bad)
    grid = np.percentile(np.concatenate([g, b]), np.linspace(2, 98, 49))
    best_t, best_j = grid[len(grid) // 2], -1
    for t in grid:
        if higher_is_bad:                      # bad has HIGHER values (flag if > t)
            tp = np.mean(b > t); fp = np.mean(g > t)
        else:                                  # bad has LOWER values (flag if < t)
            tp = np.mean(b < t); fp = np.mean(g < t)
        j = tp - fp
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t), float(best_j)


def main():
    ap = argparse.ArgumentParser(description="Calibrate quality-gate thresholds on your data")
    ap.add_argument("--good", required=True, help="good-quality clips (csv or folder)")
    ap.add_argument("--bad", default=None, help="bad-quality clips (csv or folder)")
    ap.add_argument("--auto", action="store_true", help="synthesize bad clips from good")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--out", default="runs/gate_thresholds.json")
    args = ap.parse_args()

    gate = QualityGate(QualityConfig(sample_rate=args.sample_rate))
    rng = np.random.default_rng(0)
    good_files = _files(args.good)
    gfe = _collect(good_files, gate, args.sample_rate)
    if args.auto:
        bfe = _collect(good_files, gate, args.sample_rate, degrade=True, rng=rng)
        print(f"[calib] {len(gfe)} good, {len(bfe)} synthetic-bad clips")
    else:
        if not args.bad:
            raise SystemExit("provide --bad or use --auto")
        bfe = _collect(_files(args.bad), gate, args.sample_rate)
        print(f"[calib] {len(gfe)} good, {len(bfe)} labeled-bad clips")
    if not gfe or not bfe:
        raise SystemExit("not enough clips to calibrate")

    def col(feats, k): return [f[k] for f in feats]
    thr = {}
    # SNR: bad has LOWER SNR -> set reject where good/bad separate; warn=+5, green=+10
    snr_reject, j = _best_cut(col(gfe, "snr_db"), col(bfe, "snr_db"), higher_is_bad=False)
    thr["snr_reject_db"] = round(snr_reject, 1)
    thr["snr_warn_db"] = round(snr_reject + 5, 1)
    thr["snr_green_db"] = round(snr_reject + 10, 1)
    print(f"  snr_reject_db = {thr['snr_reject_db']} (J={j:.2f})  warn={thr['snr_warn_db']} green={thr['snr_green_db']}")
    # clipping fraction: bad higher
    thr["clip_frac_max"], j = _best_cut(col(gfe, "clip_frac"), col(bfe, "clip_frac"), True)
    thr["clip_frac_max"] = round(thr["clip_frac_max"], 4)
    # contact cv: bad higher (unstable)
    thr["contact_stability_max"], _ = _best_cut(col(gfe, "contact_cv"), col(bfe, "contact_cv"), True)
    thr["contact_stability_max"] = round(thr["contact_stability_max"], 3)
    # breath active frac: bad lower
    thr["breath_min_active_frac"], _ = _best_cut(col(gfe, "breath_active_frac"),
                                                 col(bfe, "breath_active_frac"), False)
    thr["breath_min_active_frac"] = round(thr["breath_min_active_frac"], 3)
    # flatness / zcr: bad higher
    thr["flatness_max"], _ = _best_cut(col(gfe, "flatness"), col(bfe, "flatness"), True)
    thr["zcr_max"], _ = _best_cut(col(gfe, "zcr"), col(bfe, "zcr"), True)
    thr["flatness_max"] = round(thr["flatness_max"], 3); thr["zcr_max"] = round(thr["zcr_max"], 3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(thr, open(args.out, "w"), indent=2)
    print(f"\n[calibrated thresholds]\n{json.dumps(thr, indent=2)}")
    print(f"[saved] -> {args.out}\nUse: phase2_quality_gate.py --report --manifest ... --thresholds {args.out}")


if __name__ == "__main__":
    main()