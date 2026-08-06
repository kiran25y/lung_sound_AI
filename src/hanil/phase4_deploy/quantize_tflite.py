"""Phase 5.5 (step 3) — INT8 TFLite quantization for ESP32-S3.

Full-integer quantization with a representative dataset. The representative
samples MUST span both devices (stethoscope + smartphone), all three classes, and
your full noise range — otherwise activation ranges are mis-estimated and on-device
accuracy collapses. We draw them straight from the training manifest + noise mixer.

Outputs an INT8 model with int8 input/output, ready for `xxd -i` into the ESP-IDF
project (see esp32/README.md).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from ..common.config import load_config
from ..common.features import LogMel
from ..phase1_classification.datasets import build_splits, LungSoundDataset
from ..phase1_classification.augment.noise_mixer import NoiseMixer


def representative_dataset_gen(cfg, n_samples: int = 500):
    """Yield [1,1,n_mels,frames] float32 log-Mel tensors matching the ONNX input."""
    import torch
    splits = build_splits(cfg.paths["manifest"], seed=cfg.get("seed", 1337))
    # mix internal (steth) + smartphone so ranges cover both devices
    frames = splits["train"]
    if len(splits.get("smartphone_test", [])):
        frames = frames._append(splits["smartphone_test"]) if hasattr(frames, "_append") \
            else frames
    ds = LungSoundDataset(frames, cfg, training=False)
    frontend = LogMel(cfg).eval()
    mixer = NoiseMixer(_calib_noise_cfg(cfg))
    idx = np.random.default_rng(0).integers(0, len(ds), size=min(n_samples, len(ds)))

    def gen():
        for i in idx:
            wav, _, _ = ds[int(i)]
            wav = mixer(wav)                       # include noisy calibration points
            with torch.no_grad():
                spec = frontend(wav.unsqueeze(0))   # [1,1,n_mels,frames]
            yield [spec.numpy().astype(np.float32)]
    return gen


def _calib_noise_cfg(cfg):
    import copy
    c = type(cfg)(copy.deepcopy(cfg))
    c["noise"] = dict(c.get("noise", {}))
    c["noise"].update({"enable": True, "prob": 0.5})   # ~half calib samples noisy
    return c


def quantize(config_path: str, saved_model_dir: str, tflite_path: str,
             n_calib: int = 500) -> None:
    import tensorflow as tf
    cfg = load_config(config_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(cfg, n_calib)
    # full-integer INT8, int8 IO (required for ESP-NN / TFLite Micro)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tflite_path).write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"[tflite] INT8 model -> {tflite_path}  ({size_kb:.1f} KB)")
    if size_kb > 700:
        print("  ! larger than ~700 KB — check width/arch fits ESP32-S3 flash budget")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--saved_model", required=True)
    ap.add_argument("--tflite", default="runs/export/student_int8.tflite")
    ap.add_argument("--n_calib", type=int, default=500)
    a = ap.parse_args()
    quantize(a.config, a.saved_model, a.tflite, a.n_calib)
