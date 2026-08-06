"""Phase 5.5 (step 4) — Parity check: PyTorch FP32 vs INT8 TFLite.

Runs the same batch of log-Mel inputs through both and reports agreement and
per-class sensitivity change, so you can decide whether post-training INT8 is good
enough or whether you need quantization-aware training (QAT).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from ..common.config import load_config
from ..common.features import LogMel
from ..common.metrics import evaluate
from ..phase1_classification.models import build_model
from ..phase1_classification.datasets import build_splits, LungSoundDataset, LABEL2IDX


def _tflite_predict(interpreter, x_int8_input, spec_float):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zp = inp["quantization"]
    q = np.round(spec_float / scale + zp).astype(np.int8)
    interpreter.set_tensor(inp["index"], q)
    interpreter.invoke()
    o = interpreter.get_tensor(out["index"]).astype(np.float32)
    oscale, ozp = out["quantization"]
    return (o - ozp) * oscale


def verify(config_path: str, ckpt: str, tflite_path: str, split: str = "internal_test"):
    import tensorflow as tf
    cfg = load_config(config_path)
    frontend = LogMel(cfg).eval()
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval()

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    frame = build_splits(cfg.paths["manifest"], seed=cfg.get("seed", 1337))[split]
    ds = LungSoundDataset(frame, cfg, training=False)

    yt, yp_fp, yp_int = [], [], []
    agree = 0
    for i in range(len(ds)):
        wav, label, _ = ds[i]
        with torch.no_grad():
            spec = frontend(wav.unsqueeze(0))
            fp = model(spec)
            fp = fp[0] if isinstance(fp, tuple) else fp
            p_fp = int(fp.argmax(1))
        p_int = int(_tflite_predict(interp, None, spec.numpy().astype(np.float32)).argmax(1))
        yt.append(label); yp_fp.append(p_fp); yp_int.append(p_int)
        agree += (p_fp == p_int)

    n = len(ds)
    print(f"top-1 agreement FP32 vs INT8: {agree/n:.4f} on {n} clips")
    print("\n[FP32]\n", evaluate(yt, yp_fp, cfg.classes))
    print("\n[INT8]\n", evaluate(yt, yp_int, cfg.classes))
    print("\nIf INT8 crackle/wheeze sensitivity drops >2-3 pts, switch to QAT.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--split", default="internal_test")
    a = ap.parse_args()
    verify(a.config, a.ckpt, a.tflite, a.split)
