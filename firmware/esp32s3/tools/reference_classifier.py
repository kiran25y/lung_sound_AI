#!/usr/bin/env python3
"""Run the deployment INT8 classifier on WAVs with desktop LiteRT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from send_wav import fnv1a, load_pcm16, quantize, reference_logmel


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wavs", nargs="+", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "main" / "model" /
                 "sram_student_int8.tflite"),
    )
    args = parser.parse_args()

    interpreter = Interpreter(
        model_path=str(args.model), num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    input_scale, input_zp = input_info["quantization"]
    output_scale, output_zp = output_info["quantization"]

    for wav in args.wavs:
        feature = reference_logmel(load_pcm16(wav))
        input_q = quantize(feature, float(input_scale), int(input_zp))
        interpreter.set_tensor(input_info["index"], input_q[None, ..., None])
        interpreter.invoke()
        output_q = interpreter.get_tensor(output_info["index"])[0]
        probabilities = sigmoid(
            (output_q.astype(np.float64) - output_zp) * output_scale)
        crackle, wheeze = float(probabilities[0]), float(probabilities[1])
        has_crackle, has_wheeze = crackle >= 0.49, wheeze >= 0.28
        class2 = "abnormal" if has_crackle or has_wheeze else "normal"
        if has_crackle and has_wheeze:
            class3 = "crackle" if crackle >= wheeze else "wheeze"
        elif has_crackle:
            class3 = "crackle"
        elif has_wheeze:
            class3 = "wheeze"
        else:
            class3 = "normal"
        print(json.dumps({
            "wav": str(wav.resolve()),
            "qhash": fnv1a(input_q.tobytes()),
            "out_q": [int(value) for value in output_q],
            "p_crackle": crackle,
            "p_wheeze": wheeze,
            "class2": class2,
            "class3": class3,
        }))


if __name__ == "__main__":
    main()
