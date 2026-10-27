#!/usr/bin/env python3
"""Run isolated single-thread LiteRT invoke benchmarks for INT8 models."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def stats(values_ms: list[float]) -> dict[str, float | int]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "n": int(values.size), "mean": float(values.mean()),
        "median": float(np.median(values)), "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)), "min": float(values.min()),
        "max": float(values.max()),
    }


def benchmark(path: Path, warmup: int, iterations: int) -> dict[str, object]:
    interpreter = Interpreter(
        model_path=str(path), num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    data = np.zeros(tuple(int(x) for x in inp["shape"]), dtype=inp["dtype"])
    interpreter.set_tensor(inp["index"], data)
    for _ in range(warmup):
        interpreter.invoke()
    elapsed = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        interpreter.invoke()
        elapsed.append((time.perf_counter_ns() - started) / 1.0e6)
    return {
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runtime": "ai-edge-litert BUILTIN_REF, one interpreter thread, isolated process",
        "input": "constant zero INT8 tensor; invoke only", "warmup": warmup,
        "invoke_ms": stats(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for path in args.models:
        key = path.stem + "_" + hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        result[key] = benchmark(path, args.warmup, args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
