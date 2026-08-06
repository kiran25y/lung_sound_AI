#!/usr/bin/env python3
"""Phase 2 — Signal-quality gate (standalone, runs in /workspace).

Every recording passes this gate BEFORE the classifier. A market device prefers a
reliable RETRY / NO_DECISION over a confident wrong answer.

Checks (pure DSP + optional OOD hook, all portable to the ESP32):
  clipping/saturation · contact quality · breath presence · noise/rustle ·
  band-limited SNR (FIXED: breath envelope + crackle transient + wheeze tonality +
  noise floor — NOT '100-600 Hz = signal') · OOD (embedding Mahalanobis).

4-band SNR scheme (single source of truth; CALIBRATE on your data):
  >=20 dB GREEN clean · 15-20 YELLOW usable · 10-15 ORANGE warn · <10 RED re-record.

Usage:
  # report over a manifest (uses the filepath column) -> detailed JSON
  python phase2_quality_gate.py --report --manifest data/manifest.csv --out runs/quality_report.json
  # or over a folder of wavs
  python phase2_quality_gate.py --report --wav-dir /path/to/wavs --out runs/quality_report.json
  # load calibrated thresholds (from phase2_calibrate_gate.py)
  python phase2_quality_gate.py --report --manifest data/manifest.csv --thresholds runs/gate_thresholds.json
"""
from __future__ import annotations
import argparse, glob, json, os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict
import numpy as np
from scipy.signal import butter, sosfilt, hilbert


class Decision(Enum):
    PASS = "pass"
    RETRY = "retry"
    NO_DECISION = "no_decision"


@dataclass
class QualityConfig:
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    clip_level: float = 0.99
    clip_frac_max: float = 0.01
    flat_run_ms: float = 5.0
    contact_lowband: tuple = (20.0, 200.0)
    contact_min_energy: float = 1e-4
    contact_stability_max: float = 1.5
    breath_band: tuple = (100.0, 1000.0)
    breath_min_active_frac: float = 0.15
    flatness_max: float = 0.5
    zcr_max: float = 0.15
    snr_green_db: float = 20.0
    snr_warn_db: float = 15.0
    snr_reject_db: float = 10.0
    ood_max_zscore: float = 4.0
    # which checks are active. breath/contact are LIVE-CAPTURE checks — disable them on
    # pre-segmented, confirmed lung-sound datasets (breathing is already guaranteed).
    disabled_checks: tuple = ("breath", "contact")

    @classmethod
    def from_json(cls, path):
        cfg = cls()
        with open(path) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
        print(f"[gate] loaded calibrated thresholds from {path}")
        return cfg


@dataclass
class QualityResult:
    decision: Decision
    reasons: list = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self):
        return self.decision == Decision.PASS

    def to_dict(self):
        d = asdict(self); d["decision"] = self.decision.value; return d


def _sos(sr, band, btype="band", order=4):
    nyq = sr / 2.0
    if btype == "band":
        return butter(order, [band[0] / nyq, band[1] / nyq], btype="band", output="sos")
    return butter(order, band / nyq, btype=btype, output="sos")


def _frames(x, frame, hop):
    if len(x) < frame:
        return x[None, :]
    n = 1 + (len(x) - frame) // hop
    return np.stack([x[i * hop:i * hop + frame] for i in range(n)])


class QualityGate:
    def __init__(self, cfg: QualityConfig = QualityConfig(), ood_reference=None):
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.frame = int(self.sr * cfg.frame_ms / 1000)
        self.hop = int(self.sr * cfg.hop_ms / 1000)
        self.sos_contact = _sos(self.sr, cfg.contact_lowband)
        self.sos_breath = _sos(self.sr, cfg.breath_band)
        self.sos_hp100 = _sos(self.sr, 100.0, "high", 2)
        self.ood_reference = ood_reference

    def _feat(self, x):
        """Return all raw features (used by the gate AND by the calibration tool)."""
        c = self.cfg
        # clipping
        frac = float(np.mean(np.abs(x) >= c.clip_level))
        dxflat = (np.abs(np.diff(x)) < 1e-5); longest = flat = 0
        for v in dxflat:
            flat = flat + 1 if v else 0; longest = max(longest, flat)
        # contact
        low = sosfilt(self.sos_contact, x); env_l = np.abs(hilbert(low))
        contact_energy = float(np.mean(low ** 2))
        contact_cv = float(np.std(env_l) / (np.mean(env_l) + 1e-9))
        # breath presence
        b = sosfilt(self.sos_breath, x); env_b = np.abs(hilbert(b))
        breath_active = float(np.mean(env_b > 0.3 * (np.max(env_b) + 1e-9)))
        # noise/rustle (wideband)
        w = sosfilt(self.sos_hp100, x); fw = _frames(w, self.frame, self.hop)
        flat_vals = [np.exp(np.mean(np.log(np.abs(np.fft.rfft(f)) + 1e-10))) /
                     (np.mean(np.abs(np.fft.rfft(f)) + 1e-10)) for f in fw]
        zcr_vals = [np.mean(np.abs(np.diff(np.sign(f))) > 0) for f in fw]
        flatness = float(np.mean(flat_vals)); zcr = float(np.mean(zcr_vals))
        # SNR (multi-feature)
        fb = _frames(b, self.frame, self.hop); fe = np.mean(fb ** 2, axis=1) + 1e-12
        active = fe >= np.percentile(fe, 60); quiet = fe <= np.percentile(fe, 20)
        teager = np.abs(w[1:-1] ** 2 - w[:-2] * w[2:])
        crackle_transient = float(np.percentile(teager, 95))
        spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) + 1e-10
        wheeze_tonality = float(np.max(spec) / (np.mean(spec) + 1e-9))
        sig = float(np.mean(fe[active])) if active.any() else float(np.mean(fe))
        noise = float(np.mean(fe[quiet])) if quiet.any() else 1e-9
        snr_db = 10.0 * np.log10(sig / max(noise, 1e-12))
        rms_var = float(np.var(np.sqrt(fe)))
        return dict(clip_frac=frac, flat_run=longest, contact_energy=contact_energy,
                    contact_cv=contact_cv, breath_active_frac=breath_active,
                    flatness=flatness, zcr=zcr, snr_db=snr_db,
                    crackle_transient=crackle_transient, wheeze_tonality=wheeze_tonality,
                    rms_var=rms_var)

    def check(self, wav, embedding=None) -> QualityResult:
        x = np.asarray(wav, np.float64).flatten()
        peak = np.max(np.abs(x)) + 1e-9
        m = self._feat(x / peak)
        c = self.cfg
        dis = set(c.disabled_checks)
        retry, nod = [], []
        run = int(c.flat_run_ms * self.sr / 1000)
        if "clipping" not in dis and (m["clip_frac"] > c.clip_frac_max or m["flat_run"] >= run):
            retry.append("clipping/saturation")
        if "contact" not in dis and (m["contact_energy"] < c.contact_min_energy
                                     or m["contact_cv"] > c.contact_stability_max):
            retry.append("poor_contact")
        if "breath" not in dis and m["breath_active_frac"] < c.breath_min_active_frac:
            retry.append("no_breathing")
        if "noise" not in dis and (m["flatness"] > c.flatness_max and m["zcr"] > c.zcr_max):
            retry.append("noise/rustle")
        snr = m["snr_db"]
        m["snr_band"] = ("green" if snr >= c.snr_green_db else "yellow" if snr >= c.snr_warn_db
                         else "orange" if snr >= c.snr_reject_db else "red")
        if "snr" not in dis and snr < c.snr_reject_db:
            retry.append(f"low_snr({snr:.0f}<{c.snr_reject_db:.0f})")
        if self.ood_reference is not None and embedding is not None:
            z = self.ood_reference.zscore(embedding); m["ood_zscore"] = z
            if z > c.ood_max_zscore:
                nod.append("out_of_distribution")
        if nod:
            return QualityResult(Decision.NO_DECISION, nod + retry, m)
        if retry:
            return QualityResult(Decision.RETRY, retry, m)
        return QualityResult(Decision.PASS, [], m)


class OODReference:
    def __init__(self): self.mean = self.inv_cov = self.d = None
    def fit(self, E):
        E = np.asarray(E, float); self.mean = E.mean(0)
        cov = np.cov(E, rowvar=False) + 1e-4 * np.eye(E.shape[1])
        self.inv_cov = np.linalg.inv(cov); self.d = E.shape[1]; return self
    def zscore(self, e):
        if self.mean is None: return float("nan")
        v = np.asarray(e, float).flatten() - self.mean
        return float(np.sqrt(max(0.0, v @ self.inv_cov @ v)) / np.sqrt(self.d))


def _load_wav(path, sr):
    import soundfile as sf
    data, s = sf.read(path, dtype="float32", always_2d=True); x = data.mean(1)
    if s != sr:
        x = np.interp(np.linspace(0, len(x) - 1, int(round(len(x) * sr / s))),
                      np.arange(len(x)), x)
    return x


def run_report(args):
    cfg = QualityConfig.from_json(args.thresholds) if args.thresholds else QualityConfig(sample_rate=args.sample_rate)
    if args.disable is not None:                 # override which checks run
        cfg.disabled_checks = tuple(args.disable)
    print(f"[gate] active checks disabled: {list(cfg.disabled_checks) or 'none'}")
    gate = QualityGate(cfg)
    if args.manifest:
        import csv
        files = [r["filepath"] for r in csv.DictReader(open(args.manifest))]
    else:
        files = sorted(glob.glob(os.path.join(args.wav_dir, "**", "*.wav"), recursive=True))
    if args.limit:
        files = files[:args.limit]
    clips, summ = [], {"decision": {}, "snr_band": {}, "reasons": {}}
    snrs, miss = [], 0
    for fp in files:
        if not os.path.exists(fp):
            miss += 1; continue
        try:
            r = gate.check(_load_wav(fp, cfg.sample_rate))
        except Exception as e:
            clips.append({"filepath": fp, "error": str(e)}); continue
        rec = {"filepath": fp, "decision": r.decision.value, "reasons": r.reasons,
               "snr_db": round(r.metrics.get("snr_db", float("nan")), 2),
               "snr_band": r.metrics.get("snr_band", "na"),
               "metrics": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                           for k, v in r.metrics.items()}}
        clips.append(rec); snrs.append(rec["snr_db"])
        summ["decision"][r.decision.value] = summ["decision"].get(r.decision.value, 0) + 1
        summ["snr_band"][rec["snr_band"]] = summ["snr_band"].get(rec["snr_band"], 0) + 1
        for rs in r.reasons:
            k = rs.split("(")[0]; summ["reasons"][k] = summ["reasons"].get(k, 0) + 1
    snrs = np.array([s for s in snrs if np.isfinite(s)])
    summ.update({"n_clips": len(files), "n_evaluated": int(len(snrs)), "n_missing": miss,
                 "snr_mean": float(np.mean(snrs)) if len(snrs) else None,
                 "snr_median": float(np.median(snrs)) if len(snrs) else None,
                 "pass_rate": summ["decision"].get("pass", 0) / max(1, len(snrs))})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "clips": clips}, open(args.out, "w"), indent=2)
    print(f"[quality] {len(snrs)}/{len(files)} evaluated (missing {miss})")
    print(f"  decisions={summ['decision']}  bands={summ['snr_band']}")
    print(f"  reasons={dict(sorted(summ['reasons'].items(), key=lambda x:-x[1])[:5])}")
    if len(snrs):
        print(f"  SNR mean={summ['snr_mean']:.1f} median={summ['snr_median']:.1f}  "
              f"pass_rate={summ['pass_rate']:.2f}")
    print(f"[saved] -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Phase 2 signal-quality gate")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--thresholds", default=None, help="calibrated thresholds JSON")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--disable", nargs="*", default=None,
                    choices=["clipping", "contact", "breath", "noise", "snr"],
                    help="checks to turn off (default disables breath+contact for datasets). "
                         "Pass '--disable' with nothing to enable ALL checks.")
    ap.add_argument("--out", default="runs/quality_report.json")
    args = ap.parse_args()
    if not (args.manifest or args.wav_dir):
        raise SystemExit("provide --manifest or --wav-dir")
    run_report(args)


if __name__ == "__main__":
    main()