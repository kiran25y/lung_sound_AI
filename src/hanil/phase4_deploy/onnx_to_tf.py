"""Phase 5.5 (step 2) — ONNX -> TensorFlow SavedModel (via onnx-tf)."""
from __future__ import annotations
import argparse


def convert(onnx_path: str, saved_model_dir: str) -> None:
    import onnx
    from onnx_tf.backend import prepare
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    prepare(model).export_graph(saved_model_dir)
    print(f"[tf] SavedModel -> {saved_model_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--saved_model", default="runs/export/saved_model")
    a = ap.parse_args()
    convert(a.onnx, a.saved_model)
