#!/usr/bin/env python3
"""phase3_device_benchmark.py — compare devices on PAIRED recordings of the same sound.

Devices: StethoMe / Skeeper / your reference stethoscope / your own PCB. The point of a
PAIRED design is that the same underlying respiratory event is captured by every device, so
differences are the DEVICE, not the patient.

Input: a pairs manifest CSV with columns
    pair_id, device, filepath   [, patient_id, site, label]
where pair_id groups recordings of the SAME event across devices.

Reports, per device:
    signal quality : quality-gate score, clipping %, noise floor (dB), dynamic range (crest, dB)
    frequency      : 95% spectral rolloff (Hz), centroid (Hz), high-band roll-off slope
    temporal       : crackle-transient strength (Teager-Kaiser p95), envelope modulation depth
    AI usefulness  : mean crackle/wheeze prob, quality-gate PASS rate      (needs --ckpt)
And cross-device: acoustic deltas vs a --reference device + classifier agreement.

Acoustic metrics need only numpy/scipy; the AI section runs only if --ckpt (torch) is given.

  python phase3_device_benchmark.py --pairs pairs.csv --reference 3m \
      --ckpt runs/aug_all/best.pt --model dscnn_tcn --width 2.0 --out runs/phase3.json
  python phase3_device_benchmark.py --protocol      # print the recording protocol
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from math import gcd
import numpy as np

SR = 16000
CLIP_S = 4.0
N = int(SR * CLIP_S)

PROTOCOL = """PAIRED MULTI-DEVICE RECORDING PROTOCOL (Phase 3)
------------------------------------------------------------------
Goal: capture the SAME respiratory event with every device so differences
are attributable to the device, not the patient.

Per capture session, for each patient AND each anatomical site:
  * Record with ALL devices as close to back-to-back as possible (ideally
    simultaneously; if sequential, keep the patient breathing steadily and
    note the order).
  * Same duration (>= 10 s recommended; we analyse 4 s windows).
  * Same/played-back sample rate where possible; document any conversion.
  * Multiple anatomical sites (e.g. LUL, RUL, LLL, RLL, trachea).
  * A range of conditions: normal, crackle, wheeze (and quiet room for a noise-floor baseline).
  * PRESERVE the raw, unnormalized signal (no AGC/denoise on export).
  * Log metadata: device, sensor pressure, gain, site, operator, patient id.

Devices to include: StethoMe, Skeeper, your reference electronic stethoscope,
your own PCB (once the Phase-4 model is deployed on it).

Deliver as a pairs manifest CSV: pair_id, device, filepath [, patient_id, site, label]
where one pair_id = one event captured across devices.
"""


def load_clip(path):
    import soundfile as sf
    d, s = sf.read(path, dtype="float32", always_2d=True)
    x = d.mean(1)
    if s != SR:
        from scipy.signal import resample_poly
        g = gcd(int(s), SR); x = resample_poly(x, SR // g, int(s) // g)
    if len(x) < N:
        x = np.tile(x, N // max(1, len(x)) + 1)
    return x[:N].astype(np.float64)


def acoustic_metrics(x):
    """Device fingerprint from the RAW (peak-normalized) waveform."""
    xr = x / (np.abs(x).max() + 1e-9)
    clip_frac = float(np.mean(np.abs(xr) >= 0.99))
    rms = float(np.sqrt(np.mean(xr ** 2)) + 1e-12)
    crest_db = float(20 * np.log10((np.abs(xr).max() + 1e-9) / rms))
    # framed power spectrum
    frame, hop = 400, 160
    nf = 1 + (len(xr) - frame) // hop
    win = np.hanning(frame)
    P = np.zeros(frame // 2 + 1)
    fe = []
    for t in range(nf):
        seg = xr[t * hop:t * hop + frame] * win
        sp = np.abs(np.fft.rfft(seg)) ** 2
        P += sp; fe.append(seg @ seg)
    P /= max(nf, 1)
    freqs = np.linspace(0, SR / 2, len(P))
    csum = np.cumsum(P) / (P.sum() + 1e-12)
    rolloff = float(freqs[np.searchsorted(csum, 0.95)])
    centroid = float((freqs * P).sum() / (P.sum() + 1e-12))
    # high-band roll-off slope (dB/kHz) fit over 500-2000 Hz
    band = (freqs >= 500) & (freqs <= 2000)
    slope = float(np.polyfit(freqs[band] / 1000.0, 10 * np.log10(P[band] + 1e-12), 1)[0]) \
        if band.sum() > 2 else float("nan")
    fe = np.array(fe)
    noise_floor_db = float(10 * np.log10(np.percentile(fe, 10) + 1e-12))
    # crackle transient (Teager-Kaiser) + envelope modulation depth
    tk = np.abs(xr[1:-1] ** 2 - xr[:-2] * xr[2:])
    teager_p95 = float(np.percentile(tk, 95))
    mod_depth = float(np.std(np.sqrt(fe)) / (np.mean(np.sqrt(fe)) + 1e-9))
    return dict(clip_pct=round(clip_frac * 100, 2), crest_db=round(crest_db, 1),
                rolloff_hz=round(rolloff), centroid_hz=round(centroid),
                hf_slope_db_per_khz=round(slope, 1), noise_floor_db=round(noise_floor_db, 1),
                teager_p95=round(teager_p95, 5), mod_depth=round(mod_depth, 3))


def load_model(args):
    import torch, phase1_baseline as pb
    pb.set_multilabel(True)
    st = torch.load(args.ckpt, map_location="cpu")
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    for w in [args.width, 2.0, 1.0, 1.5, 3.0, 0.5]:
        m = pb.build_model(args.model, num_classes=len(pb.LABELS), width=w)
        try:
            m.load_state_dict(sd); m.eval(); return m, pb
        except Exception:
            continue
    raise SystemExit("could not load --ckpt")


def main():
    ap = argparse.ArgumentParser(description="Phase 3 paired multi-device benchmark")
    ap.add_argument("--pairs", default=None, help="CSV: pair_id, device, filepath[, patient_id, site, label]")
    ap.add_argument("--reference", default=None, help="device name to use as the acoustic reference")
    ap.add_argument("--ckpt", default=None, help="classifier for the AI-usefulness section (optional)")
    ap.add_argument("--model", default="dscnn_tcn")
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--gate-config", default=None)
    ap.add_argument("--protocol", action="store_true", help="print the recording protocol and exit")
    ap.add_argument("--out", default="runs/phase3_benchmark.json")
    args = ap.parse_args()

    if args.protocol or not args.pairs:
        print(PROTOCOL)
        if not args.pairs:
            return

    import pandas as pd
    df = pd.read_csv(args.pairs)
    df["device"] = df["device"].astype(str).str.lower().str.strip()

    from phase2_quality_gate import QualityGate, QualityConfig
    cfg = QualityConfig.from_json(args.gate_config) if args.gate_config else QualityConfig()
    gate = QualityGate(cfg)

    model = pbmod = None
    if args.ckpt:
        model, pbmod = load_model(args)
        import torch

    rows = []
    for _, r in df.iterrows():
        if not os.path.exists(r["filepath"]):
            continue
        x = load_clip(r["filepath"])
        rec = {"pair_id": r.get("pair_id"), "device": r["device"], **acoustic_metrics(x)}
        gr = gate.check(x)
        rec["gate_pass"] = bool(gr.passed); rec["gate_quality"] = round(float(gr.metrics.get("snr_db", np.nan)), 1)
        if model is not None:
            import torch
            fe = pbmod.LogMel()
            with torch.no_grad():
                p = torch.sigmoid(model(fe(torch.tensor(x[None], dtype=torch.float32))))[0].numpy()
            rec["p_crackle"], rec["p_wheeze"] = round(float(p[0]), 3), round(float(p[1]), 3)
        rows.append(rec)

    if not rows:
        raise SystemExit("no readable recordings in --pairs")
    rdf = pd.DataFrame(rows)

    # per-device aggregates
    metric_cols = [c for c in rdf.columns if c not in ("pair_id", "device", "gate_pass")]
    per_device = {}
    for dev, g in rdf.groupby("device"):
        d = {c: round(float(g[c].mean()), 3) for c in metric_cols if np.issubdtype(g[c].dtype, np.number)}
        d["n"] = int(len(g)); d["gate_pass_rate"] = round(float(g["gate_pass"].mean()), 3)
        per_device[dev] = d

    print("\n=== PER-DEVICE (mean over recordings) ===")
    show = ["n", "gate_pass_rate", "gate_quality", "rolloff_hz", "centroid_hz",
            "hf_slope_db_per_khz", "noise_floor_db", "crest_db", "clip_pct", "teager_p95"]
    hdr = "device".ljust(12) + "".join(s.rjust(12) for s in show)
    print(hdr)
    for dev, d in per_device.items():
        print(dev.ljust(12) + "".join(str(d.get(s, "-")).rjust(12) for s in show))

    # cross-device: deltas vs reference + classifier agreement per pair
    report = {"per_device": per_device}
    if args.reference and args.reference.lower() in per_device:
        ref = per_device[args.reference.lower()]
        deltas = {}
        for dev, d in per_device.items():
            if dev == args.reference.lower():
                continue
            deltas[dev] = {k: round(d[k] - ref[k], 3) for k in ("rolloff_hz", "centroid_hz",
                           "hf_slope_db_per_khz", "noise_floor_db", "crest_db") if k in d and k in ref}
        report["delta_vs_reference"] = {"reference": args.reference.lower(), "deltas": deltas}
        print(f"\n=== ACOUSTIC DELTA vs '{args.reference}' (device - reference) ===")
        for dev, dd in deltas.items():
            print(f"  {dev:<12} " + "  ".join(f"{k}={v:+}" for k, v in dd.items()))

    if model is not None and "p_crackle" in rdf.columns:
        agree = []
        for pid, g in rdf.groupby("pair_id"):
            if len(g) < 2:
                continue
            for lab in ("p_crackle", "p_wheeze"):
                preds = (g[lab] >= 0.5).astype(int)
                agree.append(int(preds.nunique() == 1))
        report["classifier_pair_agreement"] = round(float(np.mean(agree)), 3) if agree else None
        print(f"\n[AI] cross-device prediction agreement per pair: {report['classifier_pair_agreement']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()