#!/usr/bin/env python3
"""Benchmark and parity-check classifier-only INT8 inference on an ESP32-S3."""
from __future__ import annotations

import argparse
import json
import re
import struct
import time
from pathlib import Path

import numpy as np
import serial

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ModuleNotFoundError:
    Interpreter = None
    OpResolverType = None

from send_wav import fnv1a, load_pcm16, quantize, reference_logmel


RESULT_RE = re.compile(
    r"RESULT source=host_features p_crackle=([0-9.]+) p_wheeze=([0-9.]+) "
    r"p_abnormal=([0-9.]+) class2=(\S+) class3=(\S+) "
    r"latency_ms=([0-9.]+) latency_us=(\d+) qhash=([0-9a-fA-F]{8}) "
    r"out_q=\[(-?\d+),(-?\d+)\]"
)
MODEL_RE = re.compile(r"input_scale=([0-9.eE+-]+) input_zp=(-?\d+)")
LABEL_RE = re.compile(r"-([NCWB])\.wav$", re.IGNORECASE)


def read_result(port: serial.Serial, timeout: float = 15.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        match = RESULT_RE.search(line)
        if match:
            return {
                "p_crackle": float(match.group(1)),
                "p_wheeze": float(match.group(2)),
                "p_abnormal": float(match.group(3)),
                "class2": match.group(4),
                "class3": match.group(5),
                "latency_ms": float(match.group(6)),
                "latency_us": int(match.group(7)),
                "qhash": match.group(8).lower(),
                "out_q": [int(match.group(9)), int(match.group(10))],
            }
    raise TimeoutError("ESP32 did not return a RESULT")


def wait_for_model(port: serial.Serial, timeout: float = 30.0) -> tuple[float, int]:
    deadline = time.monotonic() + timeout
    quantization: tuple[float, int] | None = None
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if "CLASSIFIER_MEMORY" in line or "SRAM stage=tensors_ready" in line:
            print(f"[esp32] {line}")
        match = MODEL_RE.search(line)
        if match:
            quantization = float(match.group(1)), int(match.group(2))
        if "READY " in line and quantization is not None:
            return quantization
    raise TimeoutError("ESP32 classifier did not become ready")


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wavs", nargs="+", type=Path)
    parser.add_argument("--port", default="COM4")
    parser.add_argument(
        "--model", type=Path,
        default=(Path(__file__).resolve().parents[1] / "main" / "model" /
                 "sram_student_int8.tflite"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--expected-json", type=Path,
        help=("reuse host output bytes from an earlier benchmark when desktop "
              "LiteRT is unavailable; WAV paths must occur in this JSON"),
    )
    parser.add_argument("--input-scale", type=float, default=0.0510409623)
    parser.add_argument("--input-zero-point", type=int, default=5)
    args = parser.parse_args()

    expected_rows: dict[str, dict[str, object]] | None = None
    if args.expected_json:
        expected_doc = json.loads(args.expected_json.read_text(encoding="utf-8"))
        expected_rows = {
            str(Path(row["wav"]).resolve()).lower(): row
            for row in expected_doc["rows"]
        }
        interpreter = None
        input_scale = args.input_scale
        input_zp = args.input_zero_point
    else:
        if Interpreter is None or OpResolverType is None:
            raise RuntimeError(
                "ai-edge-litert is unavailable; pass --expected-json to replay "
                "a previously validated golden set"
            )
        interpreter = Interpreter(
            model_path=str(args.model), num_threads=1,
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
        input_scale, input_zp = input_info["quantization"]

    prepared: list[tuple[Path, str, np.ndarray, list[int]]] = []
    for wav in args.wavs:
        label_match = LABEL_RE.search(wav.name)
        if not label_match:
            raise ValueError(f"cannot read N/C/W/B label from {wav.name}")
        input_q = quantize(
            reference_logmel(load_pcm16(wav)), float(input_scale), int(input_zp))
        if expected_rows is not None:
            expected = expected_rows.get(str(wav.resolve()).lower())
            if expected is None:
                raise ValueError(f"{wav} is not present in {args.expected_json}")
            actual_hash = fnv1a(input_q.tobytes(order="C"))
            if actual_hash != expected["input_qhash"]:
                raise ValueError(
                    f"feature hash changed for {wav.name}: {actual_hash} != "
                    f"{expected['input_qhash']}"
                )
            host_out = [int(value) for value in expected["host_out_q"]]
        else:
            assert interpreter is not None
            interpreter.set_tensor(input_info["index"], input_q[None, ..., None])
            interpreter.invoke()
            host_out = [int(value) for value in
                        interpreter.get_tensor(output_info["index"])[0]]
        prepared.append((wav, label_match.group(1).upper(), input_q, host_out))

    rows: list[dict[str, object]] = []
    with serial.Serial(args.port, 115200, timeout=0.25, write_timeout=30) as port:
        port.dtr = False
        port.rts = False
        port.reset_input_buffer()
        port.rts = True
        time.sleep(0.15)
        port.rts = False
        device_scale, device_zp = wait_for_model(port)
        if not np.isclose(device_scale, input_scale) or device_zp != input_zp:
            raise RuntimeError("host and ESP32 input quantization do not match")

        for index, (wav, label, input_q, host_out) in enumerate(prepared, 1):
            payload = input_q.tobytes(order="C")
            port.write(struct.pack("<4sI", b"FEA1", len(payload)))
            port.write(payload)
            port.flush()
            device = read_result(port)
            expected_hash = fnv1a(payload)
            parity = (device["qhash"] == expected_hash and
                      device["out_q"] == host_out)
            truth2 = "normal" if label == "N" else "abnormal"
            truth3 = {"N": "normal", "C": "crackle", "W": "wheeze"}.get(label)
            row = {
                "wav": str(wav.resolve()), "label": label,
                "truth2": truth2, "truth3": truth3,
                "input_qhash": expected_hash, "host_out_q": host_out,
                "exact_host_device_parity": parity, **device,
                "correct2": device["class2"] == truth2,
                "correct3": truth3 is not None and device["class3"] == truth3,
            }
            rows.append(row)
            print(
                f"[{index:02d}/{len(prepared):02d}] {wav.name} label={label} "
                f"pred={device['class2']}/{device['class3']} "
                f"latency={device['latency_ms']:.3f}ms parity={parity}"
            )

    three_class_rows = [row for row in rows if row["truth3"] is not None]
    result = {
        "board": "ESP32-S3 240 MHz",
        "runtime": "portable TFLM INT8, tensor arena in internal SRAM",
        "model": str(args.model.resolve()),
        "samples": len(rows),
        "exact_parity": sum(bool(row["exact_host_device_parity"]) for row in rows),
        "two_class_accuracy": sum(bool(row["correct2"]) for row in rows) / len(rows),
        "three_class_accuracy_excluding_both": (
            sum(bool(row["correct3"]) for row in three_class_rows) /
            len(three_class_rows)
        ),
        "latency_ms": stats([float(row["latency_ms"]) for row in rows]),
        "rows": rows,
    }
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
