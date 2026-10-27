#!/usr/bin/env python3
"""Send a real WAV to the ESP32-S3 lung-sound model and verify frontend parity.

The default ``both`` mode runs the model twice on the same recording:

1. reference: numpy preprocessing on the PC, then INT8 features to ESP32/TFLM;
2. device: raw PCM16 to the ESP32, where the complete frontend runs on-device.

Matching qhash values mean every quantized model-input byte matches. Matching
outputs with a small hash mismatch still quantify the end-to-end numerical gap.
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import time
import wave
from pathlib import Path

# The bundled runtime may inherit a very high BLAS thread count. SciPy startup
# and this small single-clip job are both more reliable with one worker.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import serial


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline_source" / "hanil_lung_ai"
sys.path.insert(0, str(PIPELINE))
import feature_config as fc  # noqa: E402


SAMPLES = 64000
FEATURE_BYTES = 64 * 401
MODEL_RE = re.compile(r"input_scale=([0-9.eE+-]+) input_zp=(-?\d+)")
RESULT_RE = re.compile(
    r"RESULT source=(\S+) p_crackle=([0-9.]+) p_wheeze=([0-9.]+) "
    r"p_abnormal=([0-9.]+) class2=(\S+) class3=(\S+).*qhash=([0-9a-fA-F]{8})"
)
GATE_RE = re.compile(
    r"GATE decision=(PASS|RETRY) reason=(\S+) clip_frac=([0-9.]+) "
    r"rms=([0-9.]+) flatness=([0-9.]+) zcr=([0-9.]+) quality=([-0-9.]+)"
)
PARITY_RE = re.compile(
    r"PARITY mismatched=(\d+)/(\d+) max_abs_q=(\d+) mean_abs_q=([0-9.]+)"
)


def resample_poly_kaiser(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """NumPy implementation of SciPy resample_poly's default FIR path.

    This reproduces scipy.signal.resample_poly(window=('kaiser', 5.0),
    padtype='constant') without importing SciPy's compiled signal module.
    """
    from math import gcd

    divisor = gcd(up, down)
    up //= divisor
    down //= divisor
    if up == down == 1:
        return x.copy()

    max_rate = max(up, down)
    half_len = 10 * max_rate
    taps = 2 * half_len + 1
    positions = np.arange(taps, dtype=np.float64) - half_len
    cutoff = 1.0 / max_rate
    h = cutoff * np.sinc(cutoff * positions)
    h *= np.kaiser(taps, 5.0)
    h /= h.sum()
    h *= up

    n_pre_pad = down - half_len % down
    h = np.concatenate((np.zeros(n_pre_pad, dtype=np.float64), h))
    n_pre_remove = (half_len + n_pre_pad) // down
    n_out = (x.size * up + down - 1) // down

    # Direct polyphase evaluation of y[j] = sum_l h[phase+l*up] * x[base-l].
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


def load_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        compression = wav.getcomptype()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if sample_width != 2 or compression != "NONE":
        raise SystemExit(
            f"WAV must be uncompressed PCM16; got sr={sample_rate}, "
            f"width={sample_width * 8} bits, compression={compression}"
        )
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if pcm.size == 0:
        raise SystemExit("WAV contains no samples")
    if sample_rate != 16000:
        pcm = resample_poly_kaiser(pcm / 32768.0, 16000, sample_rate)
        pcm = np.clip(np.rint(pcm * 32768.0), -32768, 32767)
        print(f"[host] SciPy-compatible polyphase FIR: {sample_rate} Hz -> 16000 Hz")
    pcm = pcm.astype("<i2")
    # Match inference.py exactly: repeat a short clip, then take the first 4 s.
    if pcm.size < SAMPLES:
        pcm = np.tile(pcm, SAMPLES // pcm.size + 1)
    return np.ascontiguousarray(pcm[:SAMPLES], dtype="<i2")


def reference_logmel(pcm: np.ndarray) -> np.ndarray:
    waveform = pcm.astype(np.float64) / 32768.0
    feature = fc.logmel_numpy(waveform, fc.DEFAULT)
    if feature.shape != (64, 401):
        raise RuntimeError(f"unexpected feature shape {feature.shape}")
    return feature


def quantize(feature: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    # np.rint and C lrintf both use round-to-nearest-even under their default modes.
    values = np.rint(feature / scale + zero_point)
    return np.ascontiguousarray(np.clip(values, -128, 127).astype(np.int8))


def fnv1a(data: bytes) -> str:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return f"{value:08x}"


def read_line(port: serial.Serial) -> str:
    raw = port.readline()
    if not raw:
        return ""
    line = raw.decode("utf-8", errors="replace").strip()
    if line:
        print(f"[esp32] {line}", flush=True)
    return line


def wait_ready(port: serial.Serial, timeout: float, model_info: dict[str, float | int]) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = read_line(port)
        if not line:
            continue
        match = MODEL_RE.search(line)
        if match:
            model_info["scale"] = float(match.group(1))
            model_info["zero_point"] = int(match.group(2))
        if "READY " in line and "PCM1=" in line:
            return
    raise TimeoutError("ESP32 did not become ready")


def send_command(port: serial.Serial, magic: bytes, payload: bytes,
                 timeout: float = 180.0) -> dict[str, float | str]:
    print(f"[host] sending {magic.decode()} ({len(payload)} bytes)", flush=True)
    port.write(struct.pack("<4sI", magic, len(payload)))
    port.write(payload)
    port.flush()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = read_line(port)
        if not line:
            continue
        gate_match = GATE_RE.search(line)
        if gate_match and gate_match.group(1) == "RETRY":
            return {
                "decision": "retry",
                "reason": gate_match.group(2),
                "clip_frac": float(gate_match.group(3)),
                "rms": float(gate_match.group(4)),
                "flatness": float(gate_match.group(5)),
                "zcr": float(gate_match.group(6)),
                "quality": float(gate_match.group(7)),
            }
        match = RESULT_RE.search(line)
        if match:
            return {
                "decision": "pass",
                "source": match.group(1),
                "p_crackle": float(match.group(2)),
                "p_wheeze": float(match.group(3)),
                "p_abnormal": float(match.group(4)),
                "class2": match.group(5),
                "class3": match.group(6),
                "qhash": match.group(7).lower(),
            }
    raise TimeoutError(f"no RESULT received for {magic.decode()}")


def send_compare(port: serial.Serial, payload: bytes) -> dict[str, float | int]:
    print(f"[host] sending CMP1 ({len(payload)} bytes)", flush=True)
    port.write(struct.pack("<4sI", b"CMP1", len(payload)))
    port.write(payload)
    port.flush()
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        line = read_line(port)
        if not line:
            continue
        match = PARITY_RE.search(line)
        if match:
            return {
                "mismatched": int(match.group(1)),
                "total": int(match.group(2)),
                "max_abs_q": int(match.group(3)),
                "mean_abs_q": float(match.group(4)),
            }
    raise TimeoutError("no PARITY result received for CMP1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument(
        "--synthetic", choices=("silence", "white_noise"),
        help="generate a deterministic gate-test signal instead of reading a WAV",
    )
    source.add_argument(
        "--mic", action="store_true",
        help="ask the ESP32-S3 to record six live I2S microphone seconds and classify the middle four",
    )
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--mode", choices=("both", "pcm", "features"), default="both")
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    if args.mic:
        pcm = None
        source_name = "live:I2S_microphone"
    elif args.wav is not None:
        pcm = load_pcm16(args.wav)
        source_name = str(args.wav.resolve())
    elif args.synthetic == "silence":
        pcm = np.zeros(SAMPLES, dtype="<i2")
        source_name = "synthetic:silence"
    else:
        rng = np.random.default_rng(20260825)
        pcm = np.ascontiguousarray(
            np.rint(rng.uniform(-0.5, 0.5, SAMPLES) * 32767.0).astype("<i2")
        )
        source_name = "synthetic:white_noise(seed=20260825)"
    feature = (reference_logmel(pcm)
               if pcm is not None and args.mode in ("both", "features") else None)
    print(f"[host] source={source_name}")
    if pcm is not None:
        rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768.0) ** 2)))
        peak = float(np.max(np.abs(pcm.astype(np.int32))) / 32768.0)
        print(f"[host] prepared 4.000 s: samples={pcm.size} rms={rms:.6f} peak={peak:.6f}")
    else:
        print("[host] the board will record 6.000 s and classify the middle 4.000 s")

    model_info: dict[str, float | int] = {}
    results: dict[str, dict[str, float | str]] = {}
    reference_payload: bytes | None = None
    parity: dict[str, float | int] | None = None
    with serial.Serial(args.port, 115200, timeout=0.25, write_timeout=30) as port:
        port.dtr = False
        port.rts = False
        port.reset_input_buffer()
        if not args.no_reset:
            port.rts = True
            time.sleep(0.15)
            port.rts = False
        wait_ready(port, 60.0, model_info)

        if args.mic:
            results["mic"] = send_command(port, b"MIC1", b"")
        elif args.mode in ("both", "features"):
            if "scale" not in model_info:
                raise RuntimeError("device did not report its model input quantization")
            quantized = quantize(feature, float(model_info["scale"]), int(model_info["zero_point"]))
            payload = quantized.tobytes(order="C")
            reference_payload = payload
            print(
                f"[host] reference log-Mel shape={feature.shape} "
                f"scale={model_info['scale']:.9g} zp={model_info['zero_point']} qhash={fnv1a(payload)}"
            )
            results["features"] = send_command(port, b"FEA1", payload)

        if not args.mic and args.mode in ("both", "pcm"):
            if args.mode == "both":
                wait_ready(port, 15.0, model_info)
            results["pcm"] = send_command(port, b"PCM1", pcm.tobytes(order="C"))
            if (args.mode == "both" and reference_payload is not None and
                    results["pcm"]["decision"] == "pass"):
                wait_ready(port, 15.0, model_info)
                parity = send_compare(port, reference_payload)

    print("\n[prediction]")
    for name, result in results.items():
        if result["decision"] == "retry":
            print(
                f"  {name:8s} RETRY reason={result['reason']} "
                f"clip={result['clip_frac']:.6f} rms={result['rms']:.6f} "
                f"flatness={result['flatness']:.6f} zcr={result['zcr']:.6f} "
                f"quality={result['quality']:.2f}"
            )
        else:
            print(
                f"  {name:8s} class2={result['class2']:8s} class3={result['class3']:7s} "
                f"crackle={result['p_crackle']:.6f} wheeze={result['p_wheeze']:.6f} "
                f"abnormal={result['p_abnormal']:.6f} qhash={result['qhash']}"
            )
    if ("features" in results and "pcm" in results and
            results["features"]["decision"] == "pass" and
            results["pcm"]["decision"] == "pass"):
        reference = results["features"]
        device = results["pcm"]
        print("\n[frontend parity]")
        print(f"  exact quantized input match: {reference['qhash'] == device['qhash']}")
        if parity is not None:
            fraction = int(parity["mismatched"]) / int(parity["total"])
            print(
                f"  differing bytes: {parity['mismatched']}/{parity['total']} "
                f"({fraction:.3%}); max |q gap|={parity['max_abs_q']}; "
                f"mean |q gap|={parity['mean_abs_q']:.8f}"
            )
        print(f"  |crackle gap|: {abs(float(reference['p_crackle']) - float(device['p_crackle'])):.6f}")
        print(f"  |wheeze gap|:  {abs(float(reference['p_wheeze']) - float(device['p_wheeze'])):.6f}")


if __name__ == "__main__":
    main()
