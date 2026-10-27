#!/usr/bin/env python3
"""Benchmark the complete ESP32-S3 lung-sound pipeline on labeled WAVs."""
from __future__ import annotations

import json
import re
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

from send_wav import load_pcm16


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "comparison_evaluation" / "esp32_retest_payloads_2026-09-15" / "manifest.json"
OUTPUT = ROOT / "comparison_evaluation" / "esp32_student_full_pipeline_2026-09-22.json"
PORT = "COM4"
BAUD = 115200
SAMPLES = 64000
OUTPUT_SCALE = 0.0378384441
OUTPUT_ZERO_POINT = 10
CRACKLE_THRESHOLD = 0.49
WHEEZE_THRESHOLD = 0.28

KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def fields(line: str) -> dict[str, str]:
    return {key: value.rstrip(",") for key, value in KEY_VALUE_RE.findall(line)}


def read_line(port: serial.Serial) -> str:
    raw = port.readline()
    if not raw:
        return ""
    line = raw.decode("utf-8", errors="replace").strip()
    if line:
        print(f"[esp32] {line}", flush=True)
    return line


def wait_ready(port: serial.Serial, timeout: float = 60.0) -> dict:
    boot = {"memory": [], "lines": []}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = read_line(port)
        if not line:
            continue
        boot["lines"].append(line)
        if "MEMORY stage=" in line:
            boot["memory"].append(fields(line))
        if "MODEL input=" in line:
            boot["model"] = fields(line)
        if "GATE_MEMORY " in line:
            boot["gate_memory"] = fields(line)
        if "FRONTEND_MEMORY " in line:
            boot["frontend_memory"] = fields(line)
        if "READY build=student_pipeline_sram" in line:
            boot["ready"] = fields(line)
            return boot
    raise TimeoutError("ESP32 did not become ready")


def run_command(port: serial.Serial, magic: bytes, payload: bytes,
                timeout: float = 45.0) -> dict:
    port.write(struct.pack("<4sI", magic, len(payload)))
    if payload:
        port.write(payload)
    port.flush()
    record: dict = {"command": magic.decode(), "payload_bytes": len(payload)}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = read_line(port)
        if not line:
            continue
        if "GATE decision=" in line:
            data = fields(line)
            record["gate"] = data
            if data.get("decision") == "RETRY":
                record["decision"] = "retry"
        elif "RESULT source=" in line:
            record["result"] = fields(line)
            record["decision"] = "pass"
        elif "PIPELINE_TIMING source=" in line:
            record["timing"] = fields(line)
        elif "MEMORY stage=after_command" in line:
            record["memory_after"] = fields(line)
        elif "READY build=student_pipeline_sram" in line and "decision" in record:
            return record
    raise TimeoutError(f"no completed response for {magic.decode()}")


def truth(label: str) -> tuple[str, str | None]:
    if label == "N":
        return "normal", "normal"
    if label == "C":
        return "abnormal", "crackle"
    if label == "W":
        return "abnormal", "wheeze"
    return "abnormal", None


def numeric_dict(data: dict[str, str]) -> dict:
    converted = {}
    for key, value in data.items():
        # A hash containing only decimal digits is still an eight-character
        # hexadecimal identifier. Keep hashes and vector text lossless.
        if key in {"qhash", "out_q"}:
            converted[key] = value
            continue
        try:
            converted[key] = float(value) if "." in value else int(value)
        except ValueError:
            converted[key] = value
    return converted


def normalize_record(record: dict) -> dict:
    for key in ("gate", "result", "timing", "memory_after"):
        if key in record:
            record[key] = numeric_dict(record[key])
    return record


def parse_out_q(value: str) -> list[int]:
    return [int(number) for number in re.findall(r"-?\d+", value)]


def decision_from_out_q(out_q: list[int]) -> tuple[str, str]:
    logits = [(value - OUTPUT_ZERO_POINT) * OUTPUT_SCALE for value in out_q]
    probabilities = [1.0 / (1.0 + np.exp(-value)) for value in logits]
    crackle = probabilities[0] >= CRACKLE_THRESHOLD
    wheeze = probabilities[1] >= WHEEZE_THRESHOLD
    class2 = "abnormal" if crackle or wheeze else "normal"
    if crackle and wheeze:
        class3 = "crackle" if probabilities[0] >= probabilities[1] else "wheeze"
    elif crackle:
        class3 = "crackle"
    elif wheeze:
        class3 = "wheeze"
    else:
        class3 = "normal"
    return class2, class3


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["rows"] if isinstance(manifest, dict) else manifest
    report = {
        "tested_at": datetime.now().astimezone().isoformat(),
        "board": "ESP32-S3 240 MHz, 16 MB flash, 8 MB octal PSRAM",
        "firmware": "student_pipeline_sram",
        "runtime": "portable TFLM INT8",
        "labeled_samples": [],
        "synthetic_gate_tests": [],
    }

    with serial.Serial(PORT, BAUD, timeout=0.25, write_timeout=30) as port:
        port.dtr = False
        port.rts = False
        port.reset_input_buffer()
        port.rts = True
        time.sleep(0.15)
        port.rts = False
        report["boot"] = wait_ready(port)

        for index, item in enumerate(rows, start=1):
            wav_path = Path(item["wav"])
            label = item["label"]
            pcm = load_pcm16(wav_path)
            print(f"[host] labeled {index}/{len(rows)} label={label} wav={wav_path.name}", flush=True)
            record = run_command(port, b"PCM1", pcm.tobytes(order="C"))
            record.update({"wav": str(wav_path), "label": label})
            truth2, truth3 = truth(label)
            record["truth2"] = truth2
            record["truth3"] = truth3
            if record.get("decision") == "pass":
                result = record["result"]
                record["correct2"] = result.get("class2") == truth2
                record["correct3"] = truth3 is not None and result.get("class3") == truth3
                host_out_q = [int(value) for value in item["host_out_q"]]
                esp_out_q = parse_out_q(result.get("out_q", ""))
                host_class2, host_class3 = decision_from_out_q(host_out_q)
                record["parity"] = {
                    "host_input_qhash": str(item["input_qhash"]),
                    "esp_input_qhash": str(result.get("qhash", "")),
                    "frontend_qhash_exact": str(result.get("qhash", "")) == str(item["input_qhash"]),
                    "host_out_q": host_out_q,
                    "esp_out_q": esp_out_q,
                    "output_int8_exact": esp_out_q == host_out_q,
                    "host_class2": host_class2,
                    "host_class3": host_class3,
                    "decision_exact": (
                        result.get("class2") == host_class2
                        and result.get("class3") == host_class3
                    ),
                }
            else:
                record["correct2"] = False
                record["correct3"] = False
            report["labeled_samples"].append(normalize_record(record))

        rng = np.random.default_rng(20260922)
        synthetics = {
            "silence": np.zeros(SAMPLES, dtype="<i2"),
            "white_noise": np.rint(rng.uniform(-0.5, 0.5, SAMPLES) * 32767).astype("<i2"),
            "clipped_square": np.where(np.arange(SAMPLES) % 40 < 20, 32767, -32768).astype("<i2"),
        }
        for name, pcm in synthetics.items():
            print(f"[host] synthetic gate test: {name}", flush=True)
            record = run_command(port, b"PCM1", pcm.tobytes(order="C"))
            record["name"] = name
            report["synthetic_gate_tests"].append(normalize_record(record))

        print("[host] live INMP441 capture", flush=True)
        report["live_microphone"] = normalize_record(
            run_command(port, b"MIC1", b"", timeout=60.0)
        )

    labeled = report["labeled_samples"]
    passed = [row for row in labeled if row["decision"] == "pass"]
    three_class = [row for row in passed if row["truth3"] is not None]
    report["summary"] = {
        "samples": len(labeled),
        "passed": len(passed),
        "retried": len(labeled) - len(passed),
        "pass_rate": len(passed) / len(labeled),
        "two_class_accuracy_all": sum(row["correct2"] for row in labeled) / len(labeled),
        "two_class_accuracy_passed": sum(row["correct2"] for row in passed) / len(passed) if passed else 0,
        "three_class_accuracy_excluding_both": (
            sum(row["correct3"] for row in three_class) / len(three_class) if three_class else 0
        ),
        "parity": {
            "frontend_qhash_exact": sum(
                row.get("parity", {}).get("frontend_qhash_exact", False) for row in passed
            ),
            "output_int8_exact": sum(
                row.get("parity", {}).get("output_int8_exact", False) for row in passed
            ),
            "decision_exact": sum(
                row.get("parity", {}).get("decision_exact", False) for row in passed
            ),
            "samples": len(passed),
        },
    }
    timing_rows = [row["timing"] for row in passed if "timing" in row]
    if timing_rows:
        for key in ("frontend_ms", "gate_ms", "quantize_ms", "inference_ms", "compute_ms", "end_to_end_ms"):
            values = [float(row[key]) for row in timing_rows]
            report["summary"][key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    report["boot"].pop("lines", None)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"[saved] {OUTPUT}")


if __name__ == "__main__":
    main()
