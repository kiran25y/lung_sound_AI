#!/usr/bin/env python3
"""Calibrate the pragmatic firmware quality gate against its exact C math."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from send_wav import load_pcm16  # noqa: E402


def firmware_metrics(pcm: np.ndarray) -> dict[str, float]:
    raw = pcm.astype(np.float64)
    peak = max(1.0, float(np.max(np.abs(raw))))
    normalized = raw / peak
    clip_fraction = float(np.mean(np.abs(normalized) >= 0.99))
    rms = float(np.sqrt(np.mean(normalized**2)))

    padded = np.pad(raw / 32768.0, 200, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160][:401]
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(400) / 400)
    windowed = frames * window
    power = np.abs(np.fft.rfft(windowed, n=400, axis=1)) ** 2
    wide = power[:, 2:] + 1e-10
    flatness = float(np.mean(np.exp(np.mean(np.log(wide), axis=1)) / np.mean(wide, axis=1)))
    zcr = float(np.mean(np.mean((windowed[:, 1:] >= 0) != (windowed[:, :-1] >= 0), axis=1)))
    band_energy = np.sum(power[:, 2:26], axis=1)
    ordered = np.sort(band_energy)
    q20 = len(ordered) // 5
    q60 = len(ordered) * 6 // 10
    quiet = float(np.mean(ordered[: q20 + 1]))
    active = float(np.mean(ordered[q60:]))
    quality_score = float(10.0 * np.log10(active / (quiet + 1e-12) + 1e-9))
    return {
        "clip_frac": clip_fraction,
        "rms": rms,
        "flatness": flatness,
        "zcr": zcr,
        "quality_score": quality_score,
    }


def synthetic_cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260825)
    n = np.arange(64000)
    white = np.clip(rng.normal(0.0, 0.20, n.size), -1.0, 1.0)
    clipped = np.clip(3.0 * np.sin(2.0 * np.pi * 250.0 * n / 16000.0), -1.0, 1.0)
    tone = 0.25 * np.sin(2.0 * np.pi * 500.0 * n / 16000.0)
    return {
        "silence": np.zeros(n.size, dtype="<i2"),
        "white_noise": np.rint(white * 32767.0).astype("<i2"),
        "hard_clipped": np.rint(clipped * 32767.0).astype("<i2"),
        "steady_tone": np.rint(tone * 32767.0).astype("<i2"),
    }


def percentiles(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in ("clip_frac", "rms", "flatness", "zcr", "quality_score"):
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[key] = {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="ICBHI-compatible CSV containing filepath and label4 columns",
    )
    parser.add_argument(
        "--audio-root", type=Path,
        help="directory containing WAV files (default: <manifest>/audio/icbhi)",
    )
    parser.add_argument("--per-state", type=int, default=12)
    parser.add_argument("--out", type=Path, default=ROOT / "quality_gate_calibration.json")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    audio_root = (args.audio_root.resolve() if args.audio_root is not None
                  else manifest.parent / "audio" / "icbhi")
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    randomizer = random.Random(20260825)
    selected = []
    for label in ("0", "1", "2", "3"):
        candidates = [row for row in rows if row["label4"] == label]
        selected.extend(randomizer.sample(candidates, min(args.per_state, len(candidates))))

    real_records = []
    for index, row in enumerate(selected, 1):
        path = audio_root / Path(row["filepath"]).name
        pcm = load_pcm16(path)
        metrics = firmware_metrics(pcm)
        real_records.append({"file": path.name, "label4": int(row["label4"]), **metrics})
        print(f"[{index:02d}/{len(selected)}] {path.name} label={row['label4']} "
              f"flat={metrics['flatness']:.4f} zcr={metrics['zcr']:.4f} "
              f"q={metrics['quality_score']:.2f}")

    bad_records = {name: firmware_metrics(pcm) for name, pcm in synthetic_cases().items()}
    report = {
        "real_clip_count": len(real_records),
        "real_percentiles": percentiles(real_records),
        "real_records": real_records,
        "synthetic_cases": bad_records,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"real_percentiles": report["real_percentiles"],
                      "synthetic_cases": bad_records}, indent=2))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
