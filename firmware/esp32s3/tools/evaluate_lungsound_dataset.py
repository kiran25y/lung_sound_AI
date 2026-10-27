#!/usr/bin/env python3
"""Evaluate the deployed ESP32 model and quality gate on the lung-sound dataset.

This script deliberately mirrors the firmware, rather than an older training
checkpoint:

* 16 kHz mono PCM, repeated if shorter than four seconds, first four seconds used
* 400-point periodic-Hann STFT with 160-sample hop and reflect padding
* the exported 64-band Slaney Mel filterbank used by the ESP32
* the exact lightweight firmware quality-gate formulas and thresholds
* the exact INT8 TFLite model embedded in ``main/model/model_int8.tflite``

It writes a per-clip CSV, a JSON summary, and a Markdown report. Training and
held-out results remain separate because training-set accuracy is not an honest
generalization estimate.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


SAMPLE_RATE = 16_000
CLIP_SAMPLES = 64_000
N_FFT = 400
HOP = 160
N_FRAMES = 401
CRACKLE_THRESHOLD = 0.45
WHEEZE_THRESHOLD = 0.50

GATE_THRESHOLDS = {
    "clip_frac_max": 0.0263,
    "rms_min": 0.004,
    "flatness_max": 0.05,
    "zcr_max": 0.10,
    "quality_min": 3.0,
}

LABEL_CODE = {"N": "normal", "C": "crackle", "W": "wheeze", "B": "both"}
FOUR_STATES = ("normal", "crackle", "wheeze", "both")
THREE_CLASSES = ("normal", "crackle", "wheeze")


def apply_test_condition(pcm: np.ndarray, condition: str, key: str) -> np.ndarray:
    """Apply a deterministic stress condition without changing the source WAV."""
    if condition == "clean":
        return pcm
    x = pcm.astype(np.float64) / 32768.0
    if condition.startswith("white_noise_"):
        snr_db = float(condition.rsplit("_", 1)[1].replace("db", ""))
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(x.size)
        signal = x - float(np.mean(x))
        signal_rms = max(float(np.sqrt(np.mean(signal ** 2))), 1.0 / 32768.0)
        noise_rms = max(float(np.sqrt(np.mean(noise ** 2))), 1.0e-12)
        noise *= signal_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
        x = x + noise
    elif condition == "silence":
        x = np.zeros_like(x)
    elif condition == "hard_clip":
        peak = max(float(np.max(np.abs(x))), 1.0 / 32768.0)
        x = np.clip(x * (4.0 / peak), -1.0, 32767.0 / 32768.0)
    elif condition == "steady_tone":
        t = np.arange(x.size, dtype=np.float64) / SAMPLE_RATE
        x = 0.8 * np.sin(2.0 * np.pi * 520.0 * t)
    elif condition == "low_gain":
        # The live-capture raw-level check sees this. The C quality gate itself
        # peak-normalizes PCM and therefore cannot detect gain loss alone.
        x *= 0.001
    else:
        raise ValueError(f"unknown test condition: {condition}")
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype("<i2")


def resample_poly_kaiser(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """NumPy equivalent of SciPy's default resample_poly FIR path."""
    divisor = math.gcd(int(up), int(down))
    up //= divisor
    down //= divisor
    if up == down == 1:
        return x.copy()
    max_rate = max(up, down)
    half_len = 10 * max_rate
    positions = np.arange(2 * half_len + 1, dtype=np.float64) - half_len
    cutoff = 1.0 / max_rate
    h = cutoff * np.sinc(cutoff * positions)
    h *= np.kaiser(h.size, 5.0)
    h /= h.sum()
    h *= up
    n_pre_pad = down - half_len % down
    h = np.concatenate((np.zeros(n_pre_pad, dtype=np.float64), h))
    n_pre_remove = (half_len + n_pre_pad) // down
    n_out = (x.size * up + down - 1) // down
    output_positions = np.arange(n_out, dtype=np.int64) + n_pre_remove
    time_positions = output_positions * down
    phase = time_positions % up
    base = time_positions // up
    y = np.zeros(n_out, dtype=np.float64)
    for offset in range((h.size + up - 1) // up):
        h_index = phase + offset * up
        x_index = base - offset
        valid = (h_index < h.size) & (x_index >= 0) & (x_index < x.size)
        y[valid] += h[h_index[valid]] * x[x_index[valid]]
    return y


def decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (b[:, 0].astype(np.int32) |
                  (b[:, 1].astype(np.int32) << 8) |
                  (b[:, 2].astype(np.int32) << 16))
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2_147_483_648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width * 8} bits")


def load_clip(path: Path) -> tuple[np.ndarray, dict[str, float | int]]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        compression = wav.getcomptype()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)
    if compression != "NONE":
        raise ValueError(f"compressed WAV is unsupported: {compression}")
    waveform = decode_pcm(raw, sample_width)
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    if waveform.size == 0:
        raise ValueError("empty WAV")
    source_duration = waveform.size / sample_rate
    if sample_rate != SAMPLE_RATE:
        waveform = resample_poly_kaiser(waveform, SAMPLE_RATE, sample_rate)
    if waveform.size < CLIP_SAMPLES:
        waveform = np.tile(waveform, CLIP_SAMPLES // waveform.size + 1)
    waveform = np.clip(waveform[:CLIP_SAMPLES], -1.0, 32767.0 / 32768.0)
    pcm = np.clip(np.rint(waveform * 32768.0), -32768, 32767).astype("<i2")
    return pcm, {
        "source_sample_rate": sample_rate,
        "source_channels": channels,
        "source_sample_bits": sample_width * 8,
        "source_duration_sec": source_duration,
    }


def label_from_filename(path: Path) -> str:
    # One real training clip is named ``304)2-1-C_7.wav``. Keep the label
    # encoded before that duplicate/version suffix instead of silently omitting it.
    match = re.search(r"-([NCWB])(?:_\d+)?$", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError("filename does not end in -N, -C, -W, or -B")
    return LABEL_CODE[match.group(1).upper()]


def discover_dataset(root: Path, include_smartphone: bool) -> list[dict[str, str]]:
    sources = [
        ("training", "hospital_stethoscope", root / "hospital_dataset" / "audio_data"),
        ("heldout", "sungbook_stethoscope", root / "stethoscope_dataset_sungbook"),
    ]
    if include_smartphone:
        sources.extend([
            ("external", "smartphone_seoul", root / "smartphone" / "iphone_dataset_seoul"),
            ("external", "smartphone_sungbook", root / "smartphone" / "iphone_dataset_sungbook"),
        ])
    records: list[dict[str, str]] = []
    for split, source, directory in sources:
        if not directory.is_dir():
            raise FileNotFoundError(f"dataset source is missing: {directory}")
        for path in sorted(directory.glob("*.wav"), key=lambda p: p.name.lower()):
            records.append({
                "split": split,
                "source": source,
                "filepath": str(path.resolve()),
                "filename": path.name,
                "truth_4state": label_from_filename(path),
            })
    return records


class DeployedEvaluator:
    def __init__(
        self,
        model_path: Path,
        mel_path: Path,
        window_path: Path,
        crackle_threshold: float = CRACKLE_THRESHOLD,
        wheeze_threshold: float = WHEEZE_THRESHOLD,
    ):
        try:
            from ai_edge_litert.interpreter import Interpreter, OpResolverType
        except ImportError as exc:
            raise RuntimeError(
                "LiteRT is missing. Install it with: "
                "python -m pip install --target tools/.eval_deps ai-edge-litert==2.2.0"
            ) from exc
        self.mel_fb = np.load(mel_path).astype(np.float64)
        self.window = np.load(window_path).astype(np.float64)
        self.crackle_threshold = float(crackle_threshold)
        self.wheeze_threshold = float(wheeze_threshold)
        if self.mel_fb.shape != (64, 201) or self.window.shape != (400,):
            raise ValueError(
                f"unexpected frontend tables: mel={self.mel_fb.shape}, window={self.window.shape}"
            )
        # The reference resolver is deterministic and avoids desktop acceleration
        # delegates. ESP-TFLite-Micro/ESP-NN parity is reported separately.
        self.interpreter = Interpreter(
            model_path=str(model_path), num_threads=1,
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if tuple(self.input["shape"]) != (1, 64, 401, 1) or self.input["dtype"] != np.int8:
            raise ValueError(f"unexpected model input: {self.input}")
        if tuple(self.output["shape"]) != (1, 2) or self.output["dtype"] != np.int8:
            raise ValueError(f"unexpected model output: {self.output}")

    def evaluate(
        self, pcm: np.ndarray, classify_retries: bool = False
    ) -> dict[str, float | str | bool]:
        preprocess_started = time.perf_counter()
        raw = pcm.astype(np.float64)
        normalized_peak = raw / max(1.0, float(np.max(np.abs(raw))))
        gate_clip_frac = float(np.mean(np.abs(normalized_peak) >= 0.99))
        gate_rms = float(np.sqrt(np.mean(normalized_peak ** 2)))
        waveform = raw / 32768.0
        raw_peak = float(np.max(np.abs(waveform)))
        raw_rms = float(np.sqrt(np.mean(waveform ** 2)))
        peak_dbfs = 20.0 * math.log10(max(raw_peak, 1.0 / 32768.0))
        rms_dbfs = 20.0 * math.log10(max(raw_rms, 1.0 / 32768.0))

        padded = np.pad(waveform, N_FFT // 2, mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:N_FRAMES]
        windowed = frames * self.window
        power = np.abs(np.fft.rfft(windowed, n=N_FFT, axis=1)) ** 2
        wide = power[:, 2:] + 1.0e-10
        flatness = float(np.mean(
            np.exp(np.mean(np.log(wide), axis=1)) / np.mean(wide, axis=1)
        ))
        zcr = float(np.mean(
            np.mean((windowed[:, 1:] >= 0.0) != (windowed[:, :-1] >= 0.0), axis=1)
        ))
        energies = np.sort(np.sum(power[:, 2:26], axis=1))
        q20 = energies.size // 5
        q60 = energies.size * 6 // 10
        quiet = float(np.mean(energies[:q20 + 1]))
        active = float(np.mean(energies[q60:]))
        quality_score = float(10.0 * math.log10(active / (quiet + 1.0e-12) + 1.0e-9))

        threshold = GATE_THRESHOLDS
        if gate_clip_frac > threshold["clip_frac_max"]:
            reason = "clipping_saturation"
        elif gate_rms < threshold["rms_min"]:
            reason = "silence_no_contact"
        elif flatness > threshold["flatness_max"] and zcr > threshold["zcr_max"]:
            reason = "noise_rustle"
        elif quality_score < threshold["quality_min"]:
            reason = "low_signal_quality"
        else:
            reason = "pass"

        result: dict[str, float | str | bool] = {
            "gate_decision": "pass" if reason == "pass" else "retry",
            "gate_reason": reason,
            "gate_clip_frac": gate_clip_frac,
            "gate_rms": gate_rms,
            "flatness": flatness,
            "zcr": zcr,
            "quality_score_db": quality_score,
            "peak_dbfs": peak_dbfs,
            "rms_dbfs": rms_dbfs,
        }
        if reason != "pass" and not classify_retries:
            result["preprocess_ms"] = (time.perf_counter() - preprocess_started) * 1000.0
            return result

        mel = self.mel_fb @ power.T
        feature = np.log(mel + 1.0e-6)
        feature = (feature - feature.mean()) / (feature.std() + 1.0e-5)
        input_scale, input_zero = self.input["quantization"]
        quantized = np.clip(
            np.rint(feature / input_scale + input_zero), -128, 127
        ).astype(np.int8)[None, :, :, None]
        result["preprocess_ms"] = (time.perf_counter() - preprocess_started) * 1000.0
        invoke_started = time.perf_counter()
        self.interpreter.set_tensor(self.input["index"], quantized)
        self.interpreter.invoke()
        result["inference_ms"] = (time.perf_counter() - invoke_started) * 1000.0
        logits = self.interpreter.get_tensor(self.output["index"]).astype(np.float64).reshape(-1)
        output_scale, output_zero = self.output["quantization"]
        logits = (logits - output_zero) * output_scale
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        pc, pw = float(probabilities[0]), float(probabilities[1])
        crackle = pc >= self.crackle_threshold
        wheeze = pw >= self.wheeze_threshold
        if crackle and wheeze:
            state_4 = "both"
            class_3 = "crackle" if pc >= pw else "wheeze"
        elif crackle:
            state_4 = class_3 = "crackle"
        elif wheeze:
            state_4 = class_3 = "wheeze"
        else:
            state_4 = class_3 = "normal"
        result.update({
            "p_crackle": pc,
            "p_wheeze": pw,
            "p_abnormal": 1.0 - (1.0 - pc) * (1.0 - pw),
            "predicted_2class": "abnormal" if crackle or wheeze else "normal",
            "predicted_3class": class_3,
            "predicted_4state_internal": state_4,
            "predicted_crackle": crackle,
            "predicted_wheeze": wheeze,
            "classified_despite_retry": reason != "pass",
        })
        return result


_WORKER_EVALUATOR: DeployedEvaluator | None = None
_WORKER_CLASSIFY_RETRIES = False
_WORKER_CONDITION = "clean"


def init_worker(
    model_path: str,
    mel_path: str,
    window_path: str,
    crackle_threshold: float = CRACKLE_THRESHOLD,
    wheeze_threshold: float = WHEEZE_THRESHOLD,
    classify_retries: bool = False,
    condition: str = "clean",
) -> None:
    global _WORKER_EVALUATOR, _WORKER_CLASSIFY_RETRIES, _WORKER_CONDITION
    _WORKER_EVALUATOR = DeployedEvaluator(
        Path(model_path), Path(mel_path), Path(window_path),
        crackle_threshold, wheeze_threshold,
    )
    _WORKER_CLASSIFY_RETRIES = classify_retries
    _WORKER_CONDITION = condition


def evaluate_row(row: dict[str, object]) -> dict[str, object]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("dataset evaluator worker was not initialized")
    result = dict(row)
    try:
        load_started = time.perf_counter()
        pcm, audio_info = load_clip(Path(str(result["filepath"])))
        result["audio_load_ms"] = (time.perf_counter() - load_started) * 1000.0
        result.update(audio_info)
        condition = str(result.get("test_condition", _WORKER_CONDITION))
        pcm = apply_test_condition(
            pcm, condition,
            f"{result['source']}::{result['filename']}::{condition}",
        )
        result["test_condition"] = condition
        result.update(_WORKER_EVALUATOR.evaluate(pcm, _WORKER_CLASSIFY_RETRIES))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def binary_metrics(truth: list[bool], predicted: list[bool]) -> dict[str, float | int | None]:
    tp = sum(t and p for t, p in zip(truth, predicted))
    tn = sum((not t) and (not p) for t, p in zip(truth, predicted))
    fp = sum((not t) and p for t, p in zip(truth, predicted))
    fn = sum(t and (not p) for t, p in zip(truth, predicted))
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    recall = sensitivity
    if recall is None:
        f1 = None
    else:
        precision_for_f1 = precision if precision is not None else 0.0
        f1 = (0.0 if precision_for_f1 + recall == 0
              else 2.0 * precision_for_f1 * recall / (precision_for_f1 + recall))
    balanced = (None if sensitivity is None or specificity is None
                else (sensitivity + specificity) / 2.0)
    return {
        "n": len(truth), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": safe_div(tp + tn, len(truth)), "sensitivity": sensitivity,
        "specificity": specificity, "balanced_accuracy": balanced,
        "precision": precision, "recall": recall, "f1": f1,
    }


def multiclass_metrics(
    truth: list[str], predicted: list[str], classes: Iterable[str]
) -> dict[str, object]:
    classes = tuple(classes)
    confusion = {
        actual: {guess: 0 for guess in classes}
        for actual in classes
    }
    for actual, guess in zip(truth, predicted):
        if actual in confusion and guess in confusion[actual]:
            confusion[actual][guess] += 1
    correct = sum(confusion[c][c] for c in classes)
    per_class = {}
    f1_values = []
    recall_values = []
    for cls in classes:
        tp = confusion[cls][cls]
        fn = sum(confusion[cls][guess] for guess in classes if guess != cls)
        fp = sum(confusion[actual][cls] for actual in classes if actual != cls)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if recall is None:
            f1 = None
        else:
            precision_for_f1 = precision if precision is not None else 0.0
            f1 = (0.0 if precision_for_f1 + recall == 0
                  else 2.0 * precision_for_f1 * recall / (precision_for_f1 + recall))
        per_class[cls] = {
            "support": tp + fn, "precision": precision, "recall": recall, "f1": f1,
        }
        if f1 is not None:
            f1_values.append(f1)
        if recall is not None:
            recall_values.append(recall)
    return {
        "n": len(truth), "accuracy": safe_div(correct, len(truth)),
        "balanced_accuracy": (sum(recall_values) / len(recall_values) if recall_values else None),
        "macro_f1": (sum(f1_values) / len(f1_values) if f1_values else None),
        "confusion_matrix": confusion, "per_class": per_class,
    }


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0, "mean": None, "std": None, "min": None, "p05": None,
                "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    return {
        "n": int(array.size), "mean": float(np.mean(array)), "std": float(np.std(array)),
        "min": float(np.min(array)), "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)), "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)), "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def classification_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    """Classification metrics independent of the quality-gate decision."""
    truth_4 = [str(row["truth_4state"]) for row in rows]
    pred_4 = [str(row["predicted_4state_internal"]) for row in rows]
    truth_2 = [label != "normal" for label in truth_4]
    pred_2 = [str(row["predicted_2class"]) == "abnormal" for row in rows]
    no_both = [row for row in rows if row["truth_4state"] != "both"]
    truth_3 = [str(row["truth_4state"]) for row in no_both]
    pred_3 = [str(row["predicted_3class"]) for row in no_both]
    crackle_truth = [label in ("crackle", "both") for label in truth_4]
    wheeze_truth = [label in ("wheeze", "both") for label in truth_4]
    crackle_pred = [bool(row["predicted_crackle"]) for row in rows]
    wheeze_pred = [bool(row["predicted_wheeze"]) for row in rows]
    return {
        "two_class_normal_vs_abnormal": binary_metrics(truth_2, pred_2),
        "three_class_excluding_true_both": multiclass_metrics(
            truth_3, pred_3, THREE_CLASSES
        ),
        "four_state_internal_heads": multiclass_metrics(truth_4, pred_4, FOUR_STATES),
        "crackle_head": binary_metrics(crackle_truth, crackle_pred),
        "wheeze_head": binary_metrics(wheeze_truth, wheeze_pred),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    valid = [row for row in rows if not row.get("error")]
    passed = [row for row in valid if row.get("gate_decision") == "pass"]
    retry = [row for row in valid if row.get("gate_decision") == "retry"]
    classified = [row for row in valid if "predicted_2class" in row]
    truth_4 = [str(row["truth_4state"]) for row in passed]
    pred_4 = [str(row["predicted_4state_internal"]) for row in passed]
    truth_2 = [label != "normal" for label in truth_4]
    pred_2 = [str(row["predicted_2class"]) == "abnormal" for row in passed]
    no_both = [row for row in passed if row["truth_4state"] != "both"]
    truth_3 = [str(row["truth_4state"]) for row in no_both]
    pred_3 = [str(row["predicted_3class"]) for row in no_both]
    crackle_truth = [label in ("crackle", "both") for label in truth_4]
    wheeze_truth = [label in ("wheeze", "both") for label in truth_4]
    crackle_pred = [bool(row["predicted_crackle"]) for row in passed]
    wheeze_pred = [bool(row["predicted_wheeze"]) for row in passed]
    quality_keys = (
        "gate_clip_frac", "gate_rms", "flatness", "zcr", "quality_score_db",
        "peak_dbfs", "rms_dbfs",
    )
    four_state = multiclass_metrics(truth_4, pred_4, FOUR_STATES)
    two_class = binary_metrics(truth_2, pred_2)
    three_class = multiclass_metrics(truth_3, pred_3, THREE_CLASSES)
    return {
        "n_total": total,
        "n_evaluated": len(valid),
        "n_errors": total - len(valid),
        "n_pass": len(passed),
        "n_retry": len(retry),
        "n_classified": len(classified),
        "pass_rate": safe_div(len(passed), len(valid)),
        "retry_rate": safe_div(len(retry), len(valid)),
        "retry_reasons": dict(Counter(str(row["gate_reason"]) for row in retry)),
        "retry_clips": [
            {
                key: row.get(key)
                for key in (
                    "source", "filename", "truth_4state", "gate_reason", "flatness",
                    "zcr", "quality_score_db", "rms_dbfs",
                )
            }
            for row in retry
        ],
        "ground_truth_counts": dict(Counter(str(row["truth_4state"]) for row in valid)),
        "pass_by_ground_truth": dict(Counter(str(row["truth_4state"]) for row in passed)),
        "predicted_3class_counts": dict(Counter(str(row["predicted_3class"]) for row in passed)),
        "predicted_4state_internal_counts": dict(
            Counter(str(row["predicted_4state_internal"]) for row in passed)
        ),
        "quality_distributions": {
            key: distribution(float(row[key]) for row in valid if key in row)
            for key in quality_keys
        },
        "accepted_clip_metrics": {
            "two_class_normal_vs_abnormal": two_class,
            "three_class_excluding_true_both": three_class,
            "four_state_internal_heads": four_state,
            "crackle_head": binary_metrics(crackle_truth, crackle_pred),
            "wheeze_head": binary_metrics(wheeze_truth, wheeze_pred),
        },
        "classification_only_metrics": classification_metrics(classified),
        "latency_ms": {
            key: distribution(float(row[key]) for row in valid if key in row)
            for key in ("audio_load_ms", "preprocess_ms", "inference_ms")
        },
        "correct_yield_over_all_evaluated": {
            "two_class": safe_div(
                sum(t == p for t, p in zip(truth_2, pred_2)), len(valid)
            ),
            "four_state_internal": safe_div(
                sum(t == p for t, p in zip(truth_4, pred_4)), len(valid)
            ),
        },
    }


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def report_markdown(summary: dict[str, object], model_path: Path, dataset: Path) -> str:
    lines = [
        "# Lung-sound dataset evaluation",
        "",
        f"Dataset: `{dataset}`  ",
        f"Model: `{model_path}`  ",
        "Model input: INT8 `[1,64,401,1]`, 16 kHz, four seconds  ",
        f"Thresholds: crackle `{summary['decision_thresholds']['crackle']}`, "
        f"wheeze `{summary['decision_thresholds']['wheeze']}`",
        "",
        "The PASS/RETRY decision uses the exact lightweight gate deployed in the ESP32 firmware.",
        "Classification uses the exact INT8 model in the standard desktop LiteRT reference runtime.",
        "Accuracy is measured only on PASS clips. Correct-yield counts RETRY clips in the",
        "denominator and therefore combines gate coverage with classification accuracy.",
        "",
        "**Runtime parity:** the current deployed model passed host-versus-board checks",
        "with the portable TFLite Micro reference kernels. ESP-NN remains disabled because",
        "its optimized output did not match. Repeat hardware parity whenever `--model`",
        "selects a different artifact.",
        "",
        "## Coverage and accuracy",
        "",
        "| Group | Clips | PASS | RETRY | PASS rate | 2-class accuracy | 3-class accuracy* | 4-state head accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in summary["groups"].items():
        metrics = data["accepted_clip_metrics"]
        lines.append(
            f"| {name} | {data['n_evaluated']} | {data['n_pass']} | {data['n_retry']} | "
            f"{fmt_pct(data['pass_rate'])} | "
            f"{fmt_pct(metrics['two_class_normal_vs_abnormal']['accuracy'])} | "
            f"{fmt_pct(metrics['three_class_excluding_true_both']['accuracy'])} | "
            f"{fmt_pct(metrics['four_state_internal_heads']['accuracy'])} |"
        )
    lines.extend([
        "",
        "\\* Three-class accuracy excludes ground-truth `both`, because the product's",
        "three-class output intentionally resolves simultaneous heads to crackle or wheeze.",
        "",
        "## Class-imbalance-aware performance",
        "",
        "| Group | 2-class balanced accuracy | Abnormal sensitivity | Normal specificity | 2-class F1 | 3-class balanced accuracy | 3-class macro-F1 | 4-state balanced accuracy | 4-state macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, data in summary["groups"].items():
        metrics = data["accepted_clip_metrics"]
        two = metrics["two_class_normal_vs_abnormal"]
        three = metrics["three_class_excluding_true_both"]
        four = metrics["four_state_internal_heads"]
        lines.append(
            f"| {name} | {fmt_pct(two['balanced_accuracy'])} | "
            f"{fmt_pct(two['sensitivity'])} | {fmt_pct(two['specificity'])} | "
            f"{fmt_pct(two['f1'])} | {fmt_pct(three['balanced_accuracy'])} | "
            f"{fmt_pct(three['macro_f1'])} | {fmt_pct(four['balanced_accuracy'])} | "
            f"{fmt_pct(four['macro_f1'])} |"
        )
    lines.extend([
        "",
        "## Retry reasons",
        "",
        "| Group | Clipping | Silence/no contact | Noise/rustle | Low signal quality |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, data in summary["groups"].items():
        reasons = data["retry_reasons"]
        lines.append(
            f"| {name} | {reasons.get('clipping_saturation', 0)} | "
            f"{reasons.get('silence_no_contact', 0)} | {reasons.get('noise_rustle', 0)} | "
            f"{reasons.get('low_signal_quality', 0)} |"
        )
    retry_clips = summary["groups"]["all_real_sources"].get("retry_clips", [])
    if retry_clips:
        lines.extend([
            "",
            "### Rejected recordings",
            "",
            "| Source | File | Truth | Reason | Flatness | ZCR | Quality (dB) | RMS (dBFS) |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ])
        for clip in retry_clips:
            lines.append(
                f"| {clip['source']} | `{clip['filename']}` | {clip['truth_4state']} | "
                f"{clip['gate_reason']} | {clip['flatness']:.4f} | {clip['zcr']:.4f} | "
                f"{clip['quality_score_db']:.2f} | {clip['rms_dbfs']:.2f} |"
            )
    lines.extend([
        "",
        "## Signal and noise distributions",
        "",
        "| Group | Quality median / p05 / p95 (dB) | Flatness median / p95 | ZCR median / p95 | RMS dBFS median / p05 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, data in summary["groups"].items():
        q = data["quality_distributions"]["quality_score_db"]
        f = data["quality_distributions"]["flatness"]
        z = data["quality_distributions"]["zcr"]
        r = data["quality_distributions"]["rms_dbfs"]
        lines.append(
            f"| {name} | {q['median']:.2f} / {q['p05']:.2f} / {q['p95']:.2f} | "
            f"{f['median']:.4f} / {f['p95']:.4f} | {z['median']:.4f} / {z['p95']:.4f} | "
            f"{r['median']:.2f} / {r['p05']:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Held-out stethoscope performance is the main generalization result.",
        "- Training accuracy is descriptive and may be optimistic because those clips were used to fit the model.",
        "- Smartphone recordings are an external-domain stress test, not part of the supplied training manifest.",
        "- Near-100% gate coverage means this clean, pre-segmented dataset does not by itself validate rejection of bad live captures; calibrate with deliberately noisy, weak, clipped, and poor-contact recordings.",
        "- `four_state_internal_heads` is an analysis view; the product-facing three-class output never emits `both`.",
        "- These figures measure acoustic classification, not clinical diagnostic performance.",
        "",
    ])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    preferred = [
        "split", "source", "filename", "filepath", "truth_4state", "gate_decision",
        "gate_reason", "source_sample_rate", "source_channels", "source_sample_bits",
        "source_duration_sec", "gate_clip_frac", "gate_rms", "flatness", "zcr",
        "quality_score_db", "peak_dbfs", "rms_dbfs", "p_crackle", "p_wheeze",
        "p_abnormal", "predicted_2class", "predicted_3class",
        "predicted_4state_internal", "predicted_crackle", "predicted_wheeze",
        "test_condition", "classified_despite_retry", "audio_load_ms", "preprocess_ms",
        "inference_ms", "error",
    ]
    keys = preferred + sorted({key for row in rows for key in row} - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=root / "main" / "model" / "model_int8.tflite")
    parser.add_argument("--mel", type=Path, default=root / "main" / "frontend" / "mel_fb.npy")
    parser.add_argument("--window", type=Path, default=root / "main" / "frontend" / "window.npy")
    parser.add_argument("--out-dir", type=Path, default=root / "dataset_evaluation")
    parser.add_argument("--exclude-smartphone", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="cap each source for smoke tests")
    parser.add_argument(
        "--limit-per-source-class", type=int, default=0,
        help="balanced stress-test cap for every source and N/C/W/B class",
    )
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--crackle-threshold", type=float, default=CRACKLE_THRESHOLD)
    parser.add_argument("--wheeze-threshold", type=float, default=WHEEZE_THRESHOLD)
    parser.add_argument(
        "--classify-retries", action="store_true",
        help="run inference after a RETRY solely to measure classifier-only performance",
    )
    parser.add_argument(
        "--condition", default="clean",
        choices=("clean", "white_noise_20db", "white_noise_10db", "white_noise_5db",
                 "silence", "hard_clip", "steady_tone", "low_gain"),
        help="deterministic, non-destructive input stress condition",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        choices=("clean", "white_noise_20db", "white_noise_10db", "white_noise_5db",
                 "silence", "hard_clip", "steady_tone", "low_gain"),
        help="evaluate several conditions in one process (use with a balanced limit)",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    model_path = args.model.resolve()
    rows = discover_dataset(dataset, not args.exclude_smartphone)
    if args.limit:
        limited = []
        seen: Counter[str] = Counter()
        for row in rows:
            if seen[row["source"]] < args.limit:
                limited.append(row)
                seen[row["source"]] += 1
        rows = limited
    if args.limit_per_source_class:
        limited = []
        seen_source_class: Counter[tuple[str, str]] = Counter()
        for row in rows:
            key = (row["source"], row["truth_4state"])
            if seen_source_class[key] < args.limit_per_source_class:
                limited.append(row)
                seen_source_class[key] += 1
        rows = limited
    conditions = args.conditions or [args.condition]
    if len(conditions) > 1:
        rows = [dict(row, test_condition=condition) for condition in conditions for row in rows]
    print(f"[dataset] {dataset}")
    print(f"[model]   {model_path}")
    print(f"[clips]   {len(rows)}")
    print(f"[workers] {args.workers}")
    started = time.monotonic()
    completed_rows: list[dict[str, object]] = []
    passed = retry = 0
    worker_args = (
        str(model_path), str(args.mel.resolve()), str(args.window.resolve()),
        args.crackle_threshold, args.wheeze_threshold,
        args.classify_retries, args.condition,
    )
    if args.workers <= 1:
        init_worker(*worker_args)
        result_stream = map(evaluate_row, rows)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=worker_args,
        )
        result_stream = executor.map(evaluate_row, rows, chunksize=16)
    try:
        for index, row in enumerate(result_stream, 1):
            completed_rows.append(row)
            passed += row.get("gate_decision") == "pass"
            retry += row.get("gate_decision") == "retry"
            if index % args.progress_every == 0 or index == len(rows):
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0.0
                eta = (len(rows) - index) / rate if rate else 0.0
                print(
                    f"[progress] {index}/{len(rows)} pass={passed} retry={retry} "
                    f"rate={rate:.1f} clips/s eta={eta / 60.0:.1f} min",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    rows = completed_rows

    groups: dict[str, object] = {"all_real_sources": summarize(rows)}
    for source in sorted({str(row["source"]) for row in rows}):
        groups[source] = summarize([row for row in rows if row["source"] == source])
    for split in ("training", "heldout", "external"):
        subset = [row for row in rows if row["split"] == split]
        if subset:
            groups[f"split:{split}"] = summarize(subset)
    for condition in sorted({str(row.get("test_condition", "clean")) for row in rows}):
        subset = [row for row in rows if row.get("test_condition", "clean") == condition]
        groups[f"condition:{condition}"] = summarize(subset)
        for source in sorted({str(row["source"]) for row in subset}):
            groups[f"condition:{condition}/source:{source}"] = summarize(
                [row for row in subset if row["source"] == source]
            )
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    verified_sha256 = "38a9f22723847ffed658a0b6a549a728ba15061b11f0eeb9f34e4066177c9ce8"
    summary = {
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset_root": str(dataset),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "classification_runtime": "ai-edge-litert 2.2.0 BUILTIN_REF",
        "device_runtime_parity": {
            "status": ("verified_for_current_deployed_model"
                       if model_sha256 == verified_sha256 else "hardware_parity_required"),
            "verified_model_sha256": verified_sha256,
            "deployed_backend": "tflm_portable_int8_static_shapes",
            "esp_nn_status": "disabled_output_mismatch",
            "effect": (
                "The current deployed model matched host classification during physical-board "
                "testing. A different --model artifact requires a new hardware parity check."
            ),
        },
        "frontend": {
            "sample_rate_hz": SAMPLE_RATE, "clip_seconds": 4.0, "n_fft": N_FFT,
            "hop_length": HOP, "n_frames": N_FRAMES, "n_mels": 64,
        },
        "decision_thresholds": {
            "crackle": args.crackle_threshold,
            "wheeze": args.wheeze_threshold,
        },
        "test_condition": args.condition,
        "classify_retries": args.classify_retries,
        "gate_thresholds": GATE_THRESHOLDS,
        "groups": groups,
        "runtime_seconds": time.monotonic() - started,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_clip_results.csv"
    json_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(report_markdown(summary, model_path, dataset), encoding="utf-8")
    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
