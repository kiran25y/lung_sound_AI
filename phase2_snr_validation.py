# #!/usr/bin/env python3
# """phase2_snr_validation.py — validate the quality gate's SNR estimate and reject behaviour.

# Turns "the gate seems to work" into numbers:

#   ORACLE-SNR CHECK   Take clean clips, mix noise at KNOWN SNRs (5/10/15/20 dB), and compare the
#                      gate's ESTIMATED snr_db to the true (mix) SNR. Reports the calibration curve,
#                      the bias, and whether the estimate is monotonic with true SNR. If the estimate
#                      is biased, it recommends a corrected reject threshold.

#   REJECT-REASON      For every clip at every SNR, records the gate decision AND which check fired
#                      (SNR vs noise/rustle vs clipping vs contact/breath). Tells you WHY clips are
#                      rejected — e.g. whether the 20 dB over-rejection is the SNR check or the
#                      noise/rustle check.

# No training, no torch — just the gate. Uses your own clean clips as the signal and any noise dir.

# Example:
#   python phase2_snr_validation.py --manifest data/manifest.csv \
#       --noise-dir /workspace/dataset/noise/musan --snrs 20 15 10 5 \
#       --n-clips 400 --true-floor 10 --gate-config runs/gate_thresholds.json \
#       --out runs/snr_validation.json

# Self-test (no data needed):
#   python phase2_snr_validation.py --synthetic --out /tmp/snr_selftest.json
# """
# from __future__ import annotations
# import argparse, glob, json, os, random
# from collections import defaultdict
# from math import gcd
# import numpy as np

# from phase2_quality_gate import QualityGate, QualityConfig, Decision

# SR = 16000
# CLIP_S = 4.0
# N = int(SR * CLIP_S)

# REASON_CAT = [("low_snr", "snr"), ("noise/rustle", "noise_rustle"),
#               ("clipping", "clipping"), ("poor_contact", "contact"),
#               ("no_breathing", "breath"), ("out_of_distribution", "ood")]


# def _categorize(reasons):
#     cats = set()
#     for r in reasons:
#         for pfx, cat in REASON_CAT:
#             if r.startswith(pfx):
#                 cats.add(cat)
#     return cats


# def load_wav(path):
#     import soundfile as sf
#     d, s = sf.read(path, dtype="float32", always_2d=True)
#     x = d.mean(1)
#     if s != SR:
#         from scipy.signal import resample_poly
#         g = gcd(int(s), SR); x = resample_poly(x, SR // g, int(s) // g)
#     return x.astype(np.float64)


# def fit_len(x, n=N):
#     if len(x) == n:
#         return x
#     if len(x) > n:
#         return x[:n]
#     reps = n // max(1, len(x)) + 1
#     return np.tile(x, reps)[:n]


# def mix_at_snr(clean, noise, snr_db):
#     """Return clean + alpha*noise scaled so the signal-to-added-noise ratio is exactly snr_db."""
#     n = fit_len(noise)
#     pc = float(np.mean(clean ** 2)); pn = float(np.mean(n ** 2))
#     if pn <= 1e-12 or pc <= 1e-12:
#         return clean.copy()
#     alpha = np.sqrt(pc / (pn * (10 ** (snr_db / 10.0))))
#     return clean + alpha * n


# # ── data sampling ────────────────────────────────────────────────────────────
# def sample_clean_clips(manifest, gate, n_clips, prefer_normal, seed=0):
#     import pandas as pd
#     df = pd.read_csv(manifest)
#     df["label"] = df["label"].astype(str).str.lower().str.strip()
#     if prefer_normal and (df.label == "normal").any():
#         df = df[df.label == "normal"]
#     df = df.sample(frac=1.0, random_state=seed)
#     clean = []
#     for fp in df["filepath"]:
#         if len(clean) >= n_clips:
#             break
#         if not os.path.exists(fp):
#             continue
#         try:
#             x = fit_len(load_wav(fp))
#         except Exception:
#             continue
#         if gate.check(x).passed:                 # only keep clips that start out CLEAN (PASS)
#             clean.append(x)
#     return clean


# def sample_noise(noise_dir, k, seed=0):
#     fs = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
#     if not fs:
#         raise SystemExit(f"no .wav noise files under {noise_dir}")
#     rng = random.Random(seed); rng.shuffle(fs)
#     out = []
#     for f in fs[:max(k, 1) * 3]:
#         try:
#             out.append(fit_len(load_wav(f)))
#         except Exception:
#             continue
#         if len(out) >= k:
#             break
#     return out or [np.zeros(N)]


# def synth_clean(n_clips, seed=0):
#     rng = np.random.default_rng(seed); t = np.arange(N) / SR; out = []
#     for i in range(n_clips):
#         breath = (1 + np.sin(2 * np.pi * 0.3 * t)) * np.sin(2 * np.pi * (200 + 20 * (i % 5)) * t)
#         out.append(0.2 * breath + rng.normal(0, 0.01, N))
#     return out


# def synth_noise(k, seed=1):
#     rng = np.random.default_rng(seed)
#     return [rng.normal(0, 1, N) for _ in range(k)]


# # ── analysis ─────────────────────────────────────────────────────────────────
# def spearman(a, b):
#     a = np.asarray(a, float); b = np.asarray(b, float)
#     ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
#     if ra.std() == 0 or rb.std() == 0:
#         return float("nan")
#     return float(np.corrcoef(ra, rb)[0, 1])


# def recommend_threshold(rows, true_floor):
#     """Pick the gate SNR cut that best separates truly-bad (true<floor) from good clips (Youden)."""
#     est = np.array([r["est_snr"] for r in rows if np.isfinite(r["est_snr"])])
#     bad = np.array([(r["true_snr"] < true_floor) for r in rows if np.isfinite(r["est_snr"])])
#     if bad.sum() == 0 or (~bad).sum() == 0:
#         return None
#     best_cut, best_j = None, -1
#     for cut in np.arange(2, 30, 0.5):
#         pred_bad = est < cut
#         tp = (pred_bad & bad).sum(); fn = (~pred_bad & bad).sum()
#         fp = (pred_bad & ~bad).sum(); tn = (~pred_bad & ~bad).sum()
#         tpr = tp / (tp + fn) if (tp + fn) else 0.0
#         fpr = fp / (fp + tn) if (fp + tn) else 0.0
#         j = tpr - fpr
#         if j > best_j:
#             best_j, best_cut = j, float(cut)
#             best = dict(cut=float(cut), tpr=float(tpr), fpr=float(fpr),
#                         far=float(fp / (fp + tn) if (fp + tn) else 0.0))
#     return best


# def main():
#     ap = argparse.ArgumentParser(description="Validate the quality gate's SNR estimate & reject reasons.")
#     ap.add_argument("--manifest", default=None)
#     ap.add_argument("--noise-dir", default=None)
#     ap.add_argument("--snrs", type=float, nargs="+", default=[20, 15, 10, 5])
#     ap.add_argument("--n-clips", type=int, default=400)
#     ap.add_argument("--prefer-normal", action="store_true", default=True,
#                     help="use normal clips as the clean signal (cleanest base)")
#     ap.add_argument("--true-floor", type=float, default=10.0,
#                     help="true SNR (dB) below which a clip is considered genuinely BAD, for the "
#                          "reject-threshold recommendation")
#     ap.add_argument("--gate-config", default=None)
#     ap.add_argument("--synthetic", action="store_true", help="self-test with synthetic audio")
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--out", default="runs/snr_validation.json")
#     args = ap.parse_args()

#     cfg = QualityConfig.from_json(args.gate_config) if args.gate_config else QualityConfig()
#     gate = QualityGate(cfg)

#     if args.synthetic:
#         clean = synth_clean(args.n_clips, args.seed); noise = synth_noise(64, args.seed + 1)
#         print(f"[synthetic] {len(clean)} clean clips, {len(noise)} noise clips")
#     else:
#         if not args.manifest or not args.noise_dir:
#             raise SystemExit("need --manifest and --noise-dir (or use --synthetic)")
#         clean = sample_clean_clips(args.manifest, gate, args.n_clips, args.prefer_normal, args.seed)
#         noise = sample_noise(args.noise_dir, 128, args.seed + 1)
#         print(f"[data] {len(clean)} clean (PASS) clips, {len(noise)} noise clips")
#     if not clean:
#         raise SystemExit("no clean clips found — check the manifest / gate config")

#     rng = random.Random(args.seed)
#     conditions = [("clean", float("inf"))] + [(f"{int(s)}dB", float(s)) for s in args.snrs]
#     rows = []
#     for name, true_snr in conditions:
#         for x in clean:
#             y = x if not np.isfinite(true_snr) else mix_at_snr(x, rng.choice(noise), true_snr)
#             res = gate.check(y)
#             rows.append(dict(condition=name, true_snr=true_snr,
#                              est_snr=float(res.metrics.get("snr_db", float("nan"))),
#                              band=res.metrics.get("snr_band", "?"),
#                              decision=res.decision.value, reasons=list(res.reasons)))

#     # per-condition aggregation
#     per = {}
#     for name, true_snr in conditions:
#         sub = [r for r in rows if r["condition"] == name]
#         est = np.array([r["est_snr"] for r in sub if np.isfinite(r["est_snr"])])
#         rej = np.mean([r["decision"] != "pass" for r in sub])
#         reasons = defaultdict(int)
#         for r in sub:
#             for c in _categorize(r["reasons"]):
#                 reasons[c] += 1
#         per[name] = dict(
#             true_snr=(None if not np.isfinite(true_snr) else true_snr), n=len(sub),
#             est_snr_mean=round(float(est.mean()), 2) if len(est) else None,
#             est_snr_median=round(float(np.median(est)), 2) if len(est) else None,
#             est_snr_std=round(float(est.std()), 2) if len(est) else None,
#             reject_rate=round(float(rej), 3),
#             reject_reasons={k: int(v) for k, v in reasons.items()})

#     noisy = [r for r in rows if np.isfinite(r["true_snr"])]
#     rho = spearman([r["true_snr"] for r in noisy], [r["est_snr"] for r in noisy])
#     rec = recommend_threshold(rows, args.true_floor)

#     report = dict(config=dict(snr_reject_db=cfg.snr_reject_db, snr_warn_db=cfg.snr_warn_db,
#                               snr_green_db=cfg.snr_green_db, disabled_checks=list(cfg.disabled_checks)),
#                   n_clean=len(clean), spearman_true_vs_est=round(rho, 3),
#                   per_condition=per, recommended_reject=rec, true_floor=args.true_floor)

#     # ── print ────────────────────────────────────────────────────────────────
#     print("\n" + "=" * 70)
#     print("ORACLE-SNR CALIBRATION  (does the gate's estimate track the true SNR?)")
#     print("=" * 70)
#     print(f"{'condition':>9} | {'est_snr mean±std':>18} | {'median':>7} | {'reject%':>7} | reasons")
#     for name, _ in conditions:
#         p = per[name]
#         est = f"{p['est_snr_mean']}±{p['est_snr_std']}" if p['est_snr_mean'] is not None else "-"
#         rs = ", ".join(f"{k}:{v}" for k, v in sorted(p["reject_reasons"].items())) or "-"
#         print(f"{name:>9} | {est:>18} | {str(p['est_snr_median']):>7} | "
#               f"{p['reject_rate']*100:6.1f}% | {rs}")
#     print(f"\nSpearman(true SNR, estimated SNR) = {rho:.3f}   "
#           f"(want > 0.8: estimate is monotonic with true SNR)")
#     if rec:
#         print(f"\nCurrent reject threshold: snr_reject_db = {cfg.snr_reject_db}")
#         print(f"Recommended (separates true<{args.true_floor:.0f}dB from good, Youden): "
#               f"snr_reject_db ≈ {rec['cut']:.1f}  "
#               f"(catches {rec['tpr']*100:.0f}% of bad, wrongly rejects {rec['fpr']*100:.0f}% of good)")
#     os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
#     json.dump(report, open(args.out, "w"), indent=2)
#     print(f"\n[saved] {args.out}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""phase2_snr_validation.py — validate the quality gate's SNR estimate and reject behaviour.

Turns "the gate seems to work" into numbers:

  ORACLE-SNR CHECK   Take clean clips, mix noise at KNOWN SNRs (5/10/15/20 dB), and compare the
                     gate's ESTIMATED snr_db to the true (mix) SNR. Reports the calibration curve,
                     the bias, and whether the estimate is monotonic with true SNR. If the estimate
                     is biased, it recommends a corrected reject threshold.

  REJECT-REASON      For every clip at every SNR, records the gate decision AND which check fired
                     (SNR vs noise/rustle vs clipping vs contact/breath). Tells you WHY clips are
                     rejected — e.g. whether the 20 dB over-rejection is the SNR check or the
                     noise/rustle check.

No training, no torch — just the gate. Uses your own clean clips as the signal and any noise dir.

Example:
  python phase2_snr_validation.py --manifest data/manifest.csv \
      --noise-dir /workspace/dataset/noise/musan --snrs 20 15 10 5 \
      --n-clips 400 --true-floor 10 --gate-config runs/gate_thresholds.json \
      --out runs/snr_validation.json

Self-test (no data needed):
  python phase2_snr_validation.py --synthetic --out /tmp/snr_selftest.json
"""
from __future__ import annotations
import argparse, glob, json, os, random
from collections import defaultdict
from math import gcd
import numpy as np

from phase2_quality_gate import QualityGate, QualityConfig, Decision

SR = 16000
CLIP_S = 4.0
N = int(SR * CLIP_S)

REASON_CAT = [("low_snr", "snr"), ("noise/rustle", "noise_rustle"),
              ("clipping", "clipping"), ("poor_contact", "contact"),
              ("no_breathing", "breath"), ("out_of_distribution", "ood")]


def _categorize(reasons):
    cats = set()
    for r in reasons:
        for pfx, cat in REASON_CAT:
            if r.startswith(pfx):
                cats.add(cat)
    return cats


def load_wav(path):
    import soundfile as sf
    d, s = sf.read(path, dtype="float32", always_2d=True)
    x = d.mean(1)
    if s != SR:
        from scipy.signal import resample_poly
        g = gcd(int(s), SR); x = resample_poly(x, SR // g, int(s) // g)
    return x.astype(np.float64)


def fit_len(x, n=N):
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    reps = n // max(1, len(x)) + 1
    return np.tile(x, reps)[:n]


def m2m4_snr_db(y, ka=1.5, kw=3.0):
    """Blind M2M4 SNR estimate (Pauluzzi & Beaulieu) from the noisy signal alone.
    Assumes y = s + w, signal kurtosis ka, noise kurtosis kw (3=Gaussian). Mean-invariant.
    Returns NaN when the moment system has no valid root (signal/noise kurtosis too similar)."""
    y = np.asarray(y, float); y = y - y.mean()
    M2 = float(np.mean(y ** 2)); M4 = float(np.mean(y ** 4))
    if M2 <= 1e-12:
        return float("nan")
    a = ka + kw - 6.0
    b = M2 * (6.0 - 2.0 * kw)
    c = kw * M2 ** 2 - M4
    if abs(a) < 1e-12:                      # degenerate quadratic -> linear
        if abs(b) < 1e-12:
            return float("nan")
        S = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return float("nan")
        r = np.sqrt(disc)
        cand = [s for s in ((-b + r) / (2 * a), (-b - r) / (2 * a)) if 1e-9 < s < M2]
        if not cand:
            return float("nan")
        S = max(cand)
    N = M2 - S
    if N <= 1e-12 or S <= 1e-12:
        return float("nan")
    return 10.0 * np.log10(S / N)


def mix_at_snr(clean, noise, snr_db):
    """Return clean + alpha*noise scaled so the signal-to-added-noise ratio is exactly snr_db."""
    n = fit_len(noise)
    pc = float(np.mean(clean ** 2)); pn = float(np.mean(n ** 2))
    if pn <= 1e-12 or pc <= 1e-12:
        return clean.copy()
    alpha = np.sqrt(pc / (pn * (10 ** (snr_db / 10.0))))
    return clean + alpha * n


# ── data sampling ────────────────────────────────────────────────────────────
def sample_clean_clips(manifest, gate, n_clips, prefer_normal, seed=0):
    import pandas as pd
    df = pd.read_csv(manifest)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    if prefer_normal and (df.label == "normal").any():
        df = df[df.label == "normal"]
    df = df.sample(frac=1.0, random_state=seed)
    clean = []
    for fp in df["filepath"]:
        if len(clean) >= n_clips:
            break
        if not os.path.exists(fp):
            continue
        try:
            x = fit_len(load_wav(fp))
        except Exception:
            continue
        if gate.check(x).passed:                 # only keep clips that start out CLEAN (PASS)
            clean.append(x)
    return clean


def sample_noise(noise_dir, k, seed=0):
    fs = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
    if not fs:
        raise SystemExit(f"no .wav noise files under {noise_dir}")
    rng = random.Random(seed); rng.shuffle(fs)
    out = []
    for f in fs[:max(k, 1) * 3]:
        try:
            out.append(fit_len(load_wav(f)))
        except Exception:
            continue
        if len(out) >= k:
            break
    return out or [np.zeros(N)]


def synth_clean(n_clips, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(N) / SR; out = []
    for i in range(n_clips):
        breath = (1 + np.sin(2 * np.pi * 0.3 * t)) * np.sin(2 * np.pi * (200 + 20 * (i % 5)) * t)
        out.append(0.2 * breath + rng.normal(0, 0.01, N))
    return out


def synth_noise(k, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 1, N) for _ in range(k)]


# ── analysis ─────────────────────────────────────────────────────────────────
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def recommend_threshold(rows, true_floor):
    """Pick the gate SNR cut that best separates truly-bad (true<floor) from good clips (Youden)."""
    est = np.array([r["est_snr"] for r in rows if np.isfinite(r["est_snr"])])
    bad = np.array([(r["true_snr"] < true_floor) for r in rows if np.isfinite(r["est_snr"])])
    if bad.sum() == 0 or (~bad).sum() == 0:
        return None
    best_cut, best_j = None, -1
    for cut in np.arange(2, 30, 0.5):
        pred_bad = est < cut
        tp = (pred_bad & bad).sum(); fn = (~pred_bad & bad).sum()
        fp = (pred_bad & ~bad).sum(); tn = (~pred_bad & ~bad).sum()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_cut = j, float(cut)
            best = dict(cut=float(cut), tpr=float(tpr), fpr=float(fpr),
                        far=float(fp / (fp + tn) if (fp + tn) else 0.0))
    return best


def main():
    ap = argparse.ArgumentParser(description="Validate the quality gate's SNR estimate & reject reasons.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--snrs", type=float, nargs="+", default=[20, 15, 10, 5])
    ap.add_argument("--n-clips", type=int, default=400)
    ap.add_argument("--prefer-normal", action="store_true", default=True,
                    help="use normal clips as the clean signal (cleanest base)")
    ap.add_argument("--true-floor", type=float, default=10.0,
                    help="true SNR (dB) below which a clip is considered genuinely BAD, for the "
                         "reject-threshold recommendation")
    ap.add_argument("--gate-config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="self-test with synthetic audio")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/snr_validation.json")
    args = ap.parse_args()

    cfg = QualityConfig.from_json(args.gate_config) if args.gate_config else QualityConfig()
    gate = QualityGate(cfg)

    if args.synthetic:
        clean = synth_clean(args.n_clips, args.seed); noise = synth_noise(64, args.seed + 1)
        print(f"[synthetic] {len(clean)} clean clips, {len(noise)} noise clips")
    else:
        if not args.manifest or not args.noise_dir:
            raise SystemExit("need --manifest and --noise-dir (or use --synthetic)")
        clean = sample_clean_clips(args.manifest, gate, args.n_clips, args.prefer_normal, args.seed)
        noise = sample_noise(args.noise_dir, 128, args.seed + 1)
        print(f"[data] {len(clean)} clean (PASS) clips, {len(noise)} noise clips")
    if not clean:
        raise SystemExit("no clean clips found — check the manifest / gate config")

    rng = random.Random(args.seed)
    conditions = [("clean", float("inf"))] + [(f"{int(s)}dB", float(s)) for s in args.snrs]
    rows = []
    for name, true_snr in conditions:
        for x in clean:
            y = x if not np.isfinite(true_snr) else mix_at_snr(x, rng.choice(noise), true_snr)
            res = gate.check(y)
            rows.append(dict(condition=name, true_snr=true_snr,
                             est_snr=float(res.metrics.get("snr_db", float("nan"))),
                             est_m2m4=float(m2m4_snr_db(y)),
                             band=res.metrics.get("snr_band", "?"),
                             decision=res.decision.value, reasons=list(res.reasons)))

    # per-condition aggregation
    per = {}
    for name, true_snr in conditions:
        sub = [r for r in rows if r["condition"] == name]
        est = np.array([r["est_snr"] for r in sub if np.isfinite(r["est_snr"])])
        m24 = np.array([r["est_m2m4"] for r in sub if np.isfinite(r["est_m2m4"])])
        rej = np.mean([r["decision"] != "pass" for r in sub])
        reasons = defaultdict(int)
        for r in sub:
            for c in _categorize(r["reasons"]):
                reasons[c] += 1
        per[name] = dict(
            true_snr=(None if not np.isfinite(true_snr) else true_snr), n=len(sub),
            est_snr_mean=round(float(est.mean()), 2) if len(est) else None,
            est_snr_median=round(float(np.median(est)), 2) if len(est) else None,
            est_snr_std=round(float(est.std()), 2) if len(est) else None,
            m2m4_mean=round(float(m24.mean()), 2) if len(m24) else None,
            m2m4_median=round(float(np.median(m24)), 2) if len(m24) else None,
            m2m4_nan_frac=round(float(1 - len(m24) / max(1, len(sub))), 3),
            reject_rate=round(float(rej), 3),
            reject_reasons={k: int(v) for k, v in reasons.items()})

    noisy = [r for r in rows if np.isfinite(r["true_snr"])]
    rho = spearman([r["true_snr"] for r in noisy], [r["est_snr"] for r in noisy])
    m24_noisy = [(r["true_snr"], r["est_m2m4"]) for r in noisy if np.isfinite(r["est_m2m4"])]
    rho_m24 = spearman([a for a, _ in m24_noisy], [b for _, b in m24_noisy]) if m24_noisy else float("nan")
    rec = recommend_threshold(rows, args.true_floor)

    report = dict(config=dict(snr_reject_db=cfg.snr_reject_db, snr_warn_db=cfg.snr_warn_db,
                              snr_green_db=cfg.snr_green_db, disabled_checks=list(cfg.disabled_checks)),
                  n_clean=len(clean), spearman_true_vs_gate_snr=round(rho, 3),
                  spearman_true_vs_m2m4=round(rho_m24, 3),
                  per_condition=per, recommended_reject=rec, true_floor=args.true_floor)

    # ── print ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ORACLE-SNR CALIBRATION  (does the gate's estimate track the true SNR?)")
    print("=" * 70)
    print(f"{'condition':>9} | {'gate_snr med':>12} | {'M2M4 med':>9} | {'M2M4 NaN%':>9} | "
          f"{'reject%':>7} | reasons")
    for name, _ in conditions:
        p = per[name]
        gs = str(p['est_snr_median']) if p['est_snr_median'] is not None else "-"
        mm = str(p['m2m4_median']) if p['m2m4_median'] is not None else "-"
        rs = ", ".join(f"{k}:{v}" for k, v in sorted(p["reject_reasons"].items())) or "-"
        print(f"{name:>9} | {gs:>12} | {mm:>9} | {p['m2m4_nan_frac']*100:8.0f}% | "
              f"{p['reject_rate']*100:6.1f}% | {rs}")
    print(f"\nSpearman(true SNR, GATE snr)  = {rho:.3f}   (current gate estimator)")
    print(f"Spearman(true SNR, M2M4 snr)  = {rho_m24:.3f}   "
          f"(want > 0.8: M2M4 would be a real SNR estimator to swap in)")
    if rec:
        print(f"\nCurrent reject threshold: snr_reject_db = {cfg.snr_reject_db}")
        print(f"Recommended (separates true<{args.true_floor:.0f}dB from good, Youden): "
              f"snr_reject_db ≈ {rec['cut']:.1f}  "
              f"(catches {rec['tpr']*100:.0f}% of bad, wrongly rejects {rec['fpr']*100:.0f}% of good)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()