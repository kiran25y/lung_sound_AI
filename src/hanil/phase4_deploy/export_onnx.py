"""Phase 5.5 (step 1) — Export the trained PyTorch student to ONNX.

The exported graph takes a log-Mel spectrogram [1, 1, n_mels, frames] as input.
The front end itself runs on-device in fixed point (ESP-DSP), so it is NOT part of
the exported network — keeping the model integer-friendly for INT8.
"""
from __future__ import annotations
import argparse
import torch

from ..common.config import load_config
from ..phase1_classification.models import build_model


def frames_for(cfg) -> int:
    n = int(cfg.audio["clip_seconds"] * cfg.audio["sample_rate"])
    hop = int(cfg.features["hop_length"])
    return n // hop + 1


def export(config_path: str, ckpt_path: str, onnx_path: str) -> None:
    cfg = load_config(config_path)
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    n_mels = int(cfg.features["n_mels"])
    dummy = torch.randn(1, 1, n_mels, frames_for(cfg))
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["logmel"], output_names=["logits"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes=None)            # fixed shapes -> best for TFLite Micro
    print(f"[onnx] wrote {onnx_path}  (input 1x1x{n_mels}x{frames_for(cfg)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", default="runs/export/student.onnx")
    a = ap.parse_args()
    export(a.config, a.ckpt, a.onnx)
