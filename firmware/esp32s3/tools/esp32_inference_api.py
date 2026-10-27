#!/usr/bin/env python3
"""Local HTTP API for running WAV inference on the connected ESP32-S3.

The PC decodes, resamples, and length-normalizes the uploaded WAV. The ESP32
receives PCM16 through the existing PCM1 protocol and performs the deployed
log-Mel frontend, quality gate, and TFLite Micro inference itself.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
LOCAL_DEPS = TOOLS_DIR / ".api_deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

from flask import Flask, jsonify, request  # noqa: E402
from werkzeug.exceptions import RequestEntityTooLarge  # noqa: E402

import numpy as np  # noqa: E402
import serial  # noqa: E402
from serial import SerialException  # noqa: E402

import send_wav as board  # noqa: E402


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEVICE_LOCK = threading.Lock()


def _prepare_uploaded_wav(upload: Any) -> tuple[np.ndarray, dict[str, Any]]:
    filename = Path(upload.filename or "upload.wav").name
    suffix = Path(filename).suffix.lower()
    if suffix not in (".wav", ".wave"):
        raise ValueError("audio must be an uncompressed PCM16 WAV file")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            upload.save(temporary)
            temporary_path = Path(temporary.name)
        try:
            pcm = board.load_pcm16(temporary_path)
        except SystemExit as error:
            raise ValueError(str(error)) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    normalized = pcm.astype(np.float64) / 32768.0
    return pcm, {
        "filename": filename,
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_format": "PCM16LE",
        "samples": int(pcm.size),
        "duration_seconds": pcm.size / 16000.0,
        "rms": float(np.sqrt(np.mean(normalized * normalized))),
        "peak": float(np.max(np.abs(pcm.astype(np.int32))) / 32768.0),
    }


def _run_device_command(serial_port: str, baud: int, magic: bytes,
                        payload: bytes) -> tuple[dict[str, float | str], float]:
    started = time.monotonic()
    with DEVICE_LOCK:
        with serial.Serial(
            serial_port, baud, timeout=0.25, write_timeout=30
        ) as connection:
            connection.dtr = False
            connection.rts = False
            connection.reset_input_buffer()
            connection.rts = True
            time.sleep(0.15)
            connection.rts = False
            model_info: dict[str, float | int] = {}
            board.wait_ready(connection, 60.0, model_info)
            result = board.send_command(connection, magic, payload)
    return result, (time.monotonic() - started) * 1000.0


def create_app(serial_port: str = "COM4", baud: int = 115200) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["ESP32_SERIAL_PORT"] = serial_port
    app.config["ESP32_SERIAL_BAUD"] = baud

    @app.get("/")
    def index():
        return jsonify({
            "service": "ESP32-S3 lung-audio inference",
            "serial_port": app.config["ESP32_SERIAL_PORT"],
            "routes": {
                "health": "GET /health",
                "wav_inference": "POST /infer (multipart field: audio)",
                "live_microphone": "POST /infer/live",
            },
            "execution": "preprocessing, quality gate, and model inference run on ESP32",
            "disclaimer": "Model output is an engineering test, not a clinical diagnosis.",
        })

    @app.get("/health")
    def health():
        # This deliberately does not open/reset COM4. A real connection is
        # established for each inference request while DEVICE_LOCK is held.
        return jsonify({
            "status": "ready",
            "serial_port": app.config["ESP32_SERIAL_PORT"],
            "device_busy": DEVICE_LOCK.locked(),
        })

    @app.post("/infer")
    def infer_wav():
        upload = request.files.get("audio") or request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({
                "error": "missing multipart WAV upload",
                "expected_field": "audio",
            }), 400
        try:
            pcm, audio_info = _prepare_uploaded_wav(upload)
            result, elapsed_ms = _run_device_command(
                app.config["ESP32_SERIAL_PORT"],
                app.config["ESP32_SERIAL_BAUD"],
                b"PCM1",
                pcm.tobytes(order="C"),
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except TimeoutError as error:
            return jsonify({"error": str(error), "type": "device_timeout"}), 504
        except SerialException as error:
            return jsonify({"error": str(error), "type": "serial_error"}), 503

        return jsonify({
            "accepted": result.get("decision") == "pass",
            "audio": audio_info,
            "device_result": result,
            "elapsed_ms": round(elapsed_ms, 1),
            "execution": "ESP32 PCM preprocessing + quality gate + TFLite Micro",
            "disclaimer": "Model output is an engineering test, not a clinical diagnosis.",
        })

    @app.post("/infer/live")
    def infer_live():
        try:
            result, elapsed_ms = _run_device_command(
                app.config["ESP32_SERIAL_PORT"],
                app.config["ESP32_SERIAL_BAUD"],
                b"MIC1",
                b"",
            )
        except TimeoutError as error:
            return jsonify({"error": str(error), "type": "device_timeout"}), 504
        except SerialException as error:
            return jsonify({"error": str(error), "type": "serial_error"}), 503
        return jsonify({
            "accepted": result.get("decision") == "pass",
            "source": "live_i2s_microphone",
            "device_result": result,
            "elapsed_ms": round(elapsed_ms, 1),
            "disclaimer": "Model output is an engineering test, not a clinical diagnosis.",
        })

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error: RequestEntityTooLarge):
        return jsonify({
            "error": "upload is too large",
            "maximum_bytes": MAX_UPLOAD_BYTES,
        }), 413

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", default=os.environ.get("ESP32_PORT", "COM4"))
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app(args.serial_port, args.baud)
    app.run(
        host=args.host,
        port=args.http_port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
