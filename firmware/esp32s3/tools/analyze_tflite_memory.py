#!/usr/bin/env python3
"""Inspect INT8 TFLite topology and calculate an activation-memory lower bound.

The liveness estimate excludes TFLM allocator metadata, alignment, persistent
kernel state, and operator scratch buffers. It is intentionally labelled a
lower bound; ``MicroInterpreter::arena_used_bytes()`` on the target is the
deployment acceptance measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def tensor_bytes(detail: dict[str, Any]) -> int:
    shape = tuple(int(x) for x in detail["shape"])
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(detail["dtype"]).itemsize


def analyze(path: Path) -> dict[str, Any]:
    interpreter = Interpreter(
        model_path=str(path), num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    details = {int(t["index"]): t for t in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()
    inputs = {int(t["index"]) for t in interpreter.get_input_details()}
    outputs = {int(t["index"]) for t in interpreter.get_output_details()}
    produced = {int(index) for op in ops for index in op["outputs"] if int(index) >= 0}
    activations = produced | inputs

    births: dict[int, int] = {index: -1 for index in inputs}
    deaths: dict[int, int] = {index: -1 for index in activations}
    for op_index, op in enumerate(ops):
        for index in op["outputs"]:
            index = int(index)
            if index >= 0:
                births.setdefault(index, op_index)
        for index in op["inputs"]:
            index = int(index)
            if index in activations:
                deaths[index] = max(deaths.get(index, -1), op_index)
    for index in outputs:
        deaths[index] = max(deaths.get(index, -1), len(ops))

    live_steps = []
    for step in range(-1, len(ops) + 1):
        live = [
            index for index in activations
            if births.get(index, len(ops) + 1) <= step <= deaths.get(index, -1)
        ]
        live_steps.append((sum(tensor_bytes(details[i]) for i in live), step, live))
    peak_bytes, peak_step, peak_tensors = max(live_steps, default=(0, 0, []))

    macs = 0
    for op in ops:
        name = op["op_name"]
        indexes = [int(x) for x in op["inputs"] if int(x) >= 0]
        outs = [int(x) for x in op["outputs"] if int(x) >= 0]
        if not indexes or not outs:
            continue
        out_shape = tuple(int(x) for x in details[outs[0]]["shape"])
        out_elems = int(np.prod(out_shape, dtype=np.int64))
        if name == "CONV_2D" and len(indexes) >= 2:
            filt = tuple(int(x) for x in details[indexes[1]]["shape"])
            macs += out_elems * filt[1] * filt[2] * filt[3]
        elif name == "DEPTHWISE_CONV_2D" and len(indexes) >= 2:
            filt = tuple(int(x) for x in details[indexes[1]]["shape"])
            macs += out_elems * filt[1] * filt[2]
        elif name == "FULLY_CONNECTED" and len(indexes) >= 2:
            filt = tuple(int(x) for x in details[indexes[1]]["shape"])
            macs += out_elems * filt[-1]

    largest = sorted(
        ({"index": i, "name": details[i]["name"], "bytes": tensor_bytes(details[i]),
          "shape": [int(x) for x in details[i]["shape"]]}
         for i in activations), key=lambda x: x["bytes"], reverse=True,
    )[:12]
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_bytes_flash": path.stat().st_size,
        "operator_count": len(ops),
        "operator_histogram": dict(Counter(op["op_name"] for op in ops)),
        "estimated_macs": macs,
        "input": {"shape": input_detail["shape"].tolist(), "dtype": str(input_detail["dtype"]),
                  "quantization": list(input_detail["quantization"])},
        "output": {"shape": output_detail["shape"].tolist(), "dtype": str(output_detail["dtype"]),
                   "quantization": list(output_detail["quantization"])},
        "activation_tensor_count": len(activations),
        "activation_bytes_sum_not_arena": sum(tensor_bytes(details[i]) for i in activations),
        "activation_liveness_lower_bound_bytes": peak_bytes,
        "activation_liveness_peak_op_index": peak_step,
        "activation_liveness_peak_tensors": peak_tensors,
        "largest_activation_tensors": largest,
        "warning": (
            "Not a TFLM arena measurement: excludes alignment, allocator metadata, persistent "
            "kernel data, and operator scratch. Measure arena_used_bytes() on ESP32-S3."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {path.stem + "_" + hashlib.sha256(path.read_bytes()).hexdigest()[:8]: analyze(path)
              for path in args.models}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
