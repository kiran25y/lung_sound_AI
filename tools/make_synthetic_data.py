"""Generate a tiny synthetic dataset + manifest so you can smoke-test the whole
pipeline without real recordings.

    python tools/make_synthetic_data.py --root data_synth --n_patients 12

Creates wavs for Seoul(steth), Sungbook(steth), smartphone, a few MUSAN-style noise
clips, and a manifest.csv. Signals are crude (tonal wheeze, click-train crackle,
band-limited noise for normal) — enough to exercise shapes, splits, training loop,
metrics and the export chain, NOT to produce a real model.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import soundfile as sf
import pandas as pd

SR = 16000


def _normal(dur=5.0):
    n = int(SR * dur); rng = np.random.default_rng()
    x = rng.normal(0, 0.05, n)
    # gentle in-band breathing envelope
    t = np.arange(n) / SR
    env = 0.2 * (1 + np.sin(2 * np.pi * 0.3 * t))
    return (x + env * np.sin(2 * np.pi * 250 * t) * 0.1).astype(np.float32)


def _wheeze(dur=5.0):
    n = int(SR * dur); t = np.arange(n) / SR
    return (0.3 * np.sin(2 * np.pi * 400 * t) * (t > 1) * (t < 4)).astype(np.float32)


def _crackle(dur=5.0):
    n = int(SR * dur); x = np.zeros(n)
    rng = np.random.default_rng()
    for c in rng.integers(0, n, size=40):
        x[c:c + 30] += np.hanning(30) * rng.uniform(0.3, 0.8)
    return x.astype(np.float32)


GEN = {"normal": _normal, "wheeze": _wheeze, "crackle": _crackle}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_synth")
    ap.add_argument("--n_patients", type=int, default=12)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for hosp in ["seoul", "sungbook"]:
        for pid in range(args.n_patients):
            patient = f"{hosp}_p{pid:03d}"
            label = ["normal", "crackle", "wheeze"][pid % 3]
            devices = ["stethoscope", "smartphone"] if hosp == "seoul" else ["stethoscope"]
            for dev in devices:
                d = os.path.join(args.root, hosp, dev)
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"{patient}_{dev}.wav")
                sig = GEN[label]()
                if dev == "smartphone":                 # simulate device shift
                    sig = np.convolve(sig, [0.6, 0.3, 0.1], mode="same") + \
                          rng.normal(0, 0.02, len(sig))
                sf.write(path, sig.astype(np.float32), SR)
                rows.append(dict(filepath=path, patient_id=patient, label=label,
                                 hospital=hosp, device=dev, split=""))

    # a few noise clips (stand in for MUSAN noise)
    nd = os.path.join(args.root, "noise", "musan_noise")
    os.makedirs(nd, exist_ok=True)
    for i in range(6):
        sf.write(os.path.join(nd, f"noise_{i}.wav"),
                 rng.normal(0, 0.2, SR * 5).astype(np.float32), SR)

    man = os.path.join(args.root, "manifest.csv")
    pd.DataFrame(rows).to_csv(man, index=False)
    print(f"[synth] {len(rows)} clips + noise -> {args.root}\n[synth] manifest -> {man}")
    print("Point configs at it:  paths.manifest, paths.noise_dir")


if __name__ == "__main__":
    main()
