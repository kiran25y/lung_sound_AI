"""Create a TFLM-safe copy of the INT8 model with resolved static shapes.

LiteRT dynamically resolves SPACE_TO_BATCH_ND's batch dimension from 1 to 2.
TensorFlow Lite Micro does not resize tensors at runtime and instead allocates
the FlatBuffer's static shapes. The converted model has stale batch=1 shapes
for tensors 68..71, so TFLM executes only half of the dilated-convolution
block. This tool resolves those four static shapes without changing weights,
quantization, operators, or dynamic shape signatures.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from ai_edge_litert.tools import flatbuffer_utils


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "main" / "model" / "model_int8.tflite"
DEFAULT_OUTPUT = ROOT / "main" / "model" / "model_int8_tflm.tflite"

EXPECTED = {
    68: ([1, 17, 256], [2, 17, 256], b"SpaceToBatchND"),
    69: ([1, 1, 17, 256], [2, 1, 17, 256], b"ExpandDims"),
    70: ([1, 1, 15, 256], [2, 1, 15, 256], b"conv1d1"),
    71: ([1, 15, 256], [2, 15, 256], b"Squeeze1"),
}


def inference_output(path: Path) -> np.ndarray:
    interpreter = Interpreter(
        model_path=str(path),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    # A deterministic nonuniform tensor detects accidental model-data changes.
    values = ((np.arange(np.prod(input_info["shape"]), dtype=np.int32) * 37 + 11)
              % 256 - 128).astype(np.int8).reshape(input_info["shape"])
    interpreter.set_tensor(input_info["index"], values)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_output = inference_output(args.input)
    model = flatbuffer_utils.read_model(args.input)
    graph = model.subgraphs[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_contract = len(graph.tensors) > max(EXPECTED) and all(
        list(graph.tensors[index].shape) == old_shape
        and name_fragment in graph.tensors[index].name
        for index, (old_shape, _, name_fragment) in EXPECTED.items()
    )
    if old_contract:
        for index, (_, new_shape, _) in EXPECTED.items():
            graph.tensors[index].shape = new_shape
        flatbuffer_utils.write_model(model, args.output)
        shape_action = ("patched legacy shapes: 68=[2,17,256], 69=[2,1,17,256], "
                        "70=[2,1,15,256], 71=[2,15,256]")
    else:
        dynamic = []
        for index, tensor in enumerate(graph.tensors):
            dims = list(tensor.shape)
            signature = (list(tensor.shapeSignature)
                         if tensor.shapeSignature is not None else [])
            if any(int(dim) < 0 for dim in dims + signature):
                dynamic.append((index, tensor.name, dims, signature))
        if dynamic:
            raise RuntimeError(
                "model no longer matches the legacy patch contract and still has "
                f"dynamic tensors: {dynamic[:8]}"
            )
        args.output.write_bytes(args.input.read_bytes())
        shape_action = f"already fully static ({len(graph.tensors)} tensors); copied unchanged"

    deployed_output = inference_output(args.output)
    if not np.array_equal(source_output, deployed_output):
        raise RuntimeError(
            f"LiteRT output changed: source={source_output.tolist()} "
            f"deployment={deployed_output.tolist()}"
        )

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output}")
    print(shape_action)
    print(f"LiteRT invariant output={deployed_output.tolist()}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
