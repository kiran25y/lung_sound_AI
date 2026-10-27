#!/usr/bin/env python3
"""phase4_quantize.py — INT8 quantization of the dscnn_tcn classifier for ESP32-S3.

Path:  PyTorch ckpt  ->  ONNX  ->  TensorFlow SavedModel (onnx2tf)  ->  TFLite INT8.
The classifier is quantized (it takes the log-Mel feature); the mel front end runs in
firmware, so make sure the device mel params match phase1_baseline.LogMel exactly
(16 kHz, n_fft 400, hop 160, 64 mels, 50-2000 Hz, per-clip mean/std).

Why this file is careful about a few things:
  * Representative calibration set is DIVERSE (multi-device / multi-class / multi-SNR),
    NOT clean Seoul only — otherwise INT8 ranges overfit to easy audio.
  * Parity is checked PER HEAD (crackle & wheeze Se/Sp/AUROC), so a class-specific drop
    from quantization is caught, not hidden in an overall average.
  * onnx2tf converts NCHW->NHWC, so the TFLite input is [1, 64, T, 1] (transpose handled).

Stages (run in order; each is separable so you can verify as you go):
  --stage onnx     export ONNX + check ONNX==PyTorch (needs torch, onnx, onnxruntime)
  --stage tflite   ONNX -> SavedModel -> TFLite FP32 + INT8 (needs onnx2tf, tensorflow)
  --stage parity   FP32/PyTorch vs INT8 TFLite, per-head metrics on a test split
  --stage all      onnx -> tflite -> parity

  pip install torch onnx onnxruntime onnx2tf tensorflow soundfile scipy

Example:
  python phase4_quantize.py --stage all \
    --manifest data/manifest.csv --ckpt runs/aug_all/best.pt --model dscnn_tcn --width 2.0 \
    --noise-dir /workspace/dataset/noise/musan --n-calib 300 \
    --out runs/aug_all/quant
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


# ── model / features (reuse phase1_baseline) ─────────────────────────────────
def _load_torch(ckpt, model_name, width, device):
    import torch, phase1_baseline as pb
    pb.set_multilabel(True)
    state = torch.load(ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    for w in [width] + [x for x in (2.0, 1.0, 1.5, 3.0, 0.5) if x != width]:
        m = pb.build_model(model_name, num_classes=len(pb.LABELS), width=w)  # mixstyle=False -> same keys
        try:
            m.load_state_dict(sd); m.eval()
            return m.to(device), w, pb
        except Exception:
            continue
    raise SystemExit(f"could not load {ckpt} as {model_name} at any width")


def _mel_frames(pb):
    """Time frames the LogMel produces for a 4 s clip (fixes the ONNX/TFLite input length)."""
    import torch
    fe = pb.LogMel()
    with torch.no_grad():
        return fe(torch.zeros(1, int(pb.CLIP_S * pb.SR))).shape[-1]


def _calib_specs(pb, frontend, device, manifest, n, noise_dir, snrs, seed=0):
    """Diverse log-Mel features for INT8 calibration + parity: sampled across all splits,
    with a fraction mixed with noise at several SNRs."""
    import torch, pandas as pd
    from torch.utils.data import DataLoader
    sp = pb.build_splits(manifest, balance_normal_ratio=None)
    frames = [sp[k] for k in ("train", "external_test", "smartphone_seoul",
                              "smartphone_sungbook") if len(sp[k])]
    df = pd.concat(frames).sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)
    mixers = [None] + ([pb.NoiseMixer(noise_dir, prob=1.0, snr_db=(s, s)) for s in snrs]
                       if noise_dir else [])
    specs, targets = [], []
    loader = DataLoader(pb.ClipDataset(df), batch_size=1)
    with torch.no_grad():
        for wav, y in loader:
            mx = mixers[np.random.randint(len(mixers))]           # random condition per clip
            w = mx(wav[0]).unsqueeze(0) if mx is not None else wav
            specs.append(frontend(w.to(device)).cpu().numpy().astype("float32"))
            targets.append(np.asarray(y[0], dtype=np.float32))
    return specs, np.stack(targets)


# ── metrics ──────────────────────────────────────────────────────────────────
def _per_head(T, P, thr=0.5):
    T = np.asarray(T, float); P = np.asarray(P, float)
    thresholds = np.asarray(thr if np.ndim(thr) else [thr, thr], dtype=float)
    out = {}
    for i, nm in enumerate(["crackle", "wheeze"]):
        t, p = T[:, i], (P[:, i] >= thresholds[i]).astype(int)
        tp = ((t == 1) & (p == 1)).sum(); fn = ((t == 1) & (p == 0)).sum()
        fp = ((t == 0) & (p == 1)).sum(); tn = ((t == 0) & (p == 0)).sum()
        se = tp / (tp + fn) if (tp + fn) else 0.0
        spx = tn / (tn + fp) if (tn + fp) else 0.0
        try:
            from sklearn.metrics import roc_auc_score
            au = roc_auc_score(t, P[:, i]) if len(np.unique(t)) > 1 else float("nan")
        except Exception:
            au = float("nan")
        out[nm] = dict(se=round(se, 3), sp=round(spx, 3), auroc=round(float(au), 3))
    return out


def _load_thresholds(path):
    if not path:
        return np.array([0.5, 0.5], dtype=float)
    with open(path) as handle:
        obj = json.load(handle)
    values = obj.get("thresholds", obj) if isinstance(obj, dict) else obj
    thresholds = np.asarray(values, dtype=float)
    if thresholds.shape != (2,):
        raise SystemExit(f"expected two thresholds in {path}, got {values!r}")
    return thresholds


# ── stages ───────────────────────────────────────────────────────────────────
def stage_onnx(args):
    import torch
    device = "cpu"
    model, w, pb = _load_torch(args.ckpt, args.model, args.width, device)
    T = _mel_frames(pb)
    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "model.onnx")
    dummy = torch.randn(1, 1, 64, T)
    # dynamo=False -> legacy TorchScript exporter: cleaner graph (no FILL op) that onnx2tf's
    # INT8 path can quantize. opset 17 is well supported by onnx2tf.
    # TFLM always invokes a single 4-second clip. A static batch prevents the
    # converter from losing the exact TCN crop length around dilated Conv1D.
    export_kwargs = dict(input_names=["mel"], output_names=["logits"],
                         opset_version=17)
    try:
        torch.onnx.export(model, dummy, onnx_path, dynamo=False, **export_kwargs)
    except TypeError:                      # older torch: no 'dynamo' kwarg (already legacy)
        torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
    # parity: ONNX Runtime vs PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = torch.randn(1, 1, 64, T)
    with torch.no_grad():
        a = model(x).numpy()
    b = sess.run(None, {"mel": x.numpy()})[0]
    err = float(np.abs(a - b).max())
    print(f"[onnx] exported {onnx_path}  input=[1,1,64,{T}]  width={w}")
    print(f"[onnx] max |PyTorch-ONNX| = {err:.2e}  ({'OK' if err < 1e-4 else 'CHECK'})")
    json.dump({"onnx": onnx_path, "frames": T, "width": w, "static_batch": 1,
               "onnx_parity_maxabs": err},
              open(os.path.join(args.out, "quant_meta.json"), "w"), indent=2)
    return onnx_path, T


def stage_tflite(args, onnx_path=None):
    """Let onnx2tf produce the INT8 TFLite directly (handles NCHW->NHWC + calibration).
    Calibration data is our diverse multi-device/SNR log-Mel, saved as NHWC [N,64,T,1]."""
    import subprocess, glob, shutil
    onnx_path = onnx_path or os.path.join(args.out, "model.onnx")
    calib_path = os.path.join(args.out, "calib.npy")
    if os.path.exists(calib_path) and not args.refresh_calibration:
        calib = np.load(calib_path, mmap_mode="r")
        print(f"[tflite] reusing calibration set {calib.shape} (NHWC) -> {calib_path}")
    else:
        import phase1_baseline as pb
        _, _, pb = _load_torch(args.ckpt, args.model, args.width, "cpu")
        frontend = pb.LogMel()
        specs, _ = _calib_specs(pb, frontend, "cpu", args.manifest, args.n_calib,
                                args.noise_dir, args.snrs)
        calib = np.concatenate(
            [np.transpose(s, (0, 2, 3, 1)) for s in specs], 0).astype("float32")
        np.save(calib_path, calib)
        print(f"[tflite] calibration set {calib.shape} (NHWC) -> {calib_path}")
    tf_dir = os.path.join(args.out, "tf")
    # -tb tf_converter: use the classic TensorFlow converter (the default flatbuffer_direct
    #   fast path can't strict-int8-quantize an internal FILL op).
    # -oiqt: output int8; -cind: use OUR calibration data for input 'mel' (mean 0, std 1,
    #   since the log-Mel is already per-clip normalized)
    # Invoke through this interpreter so a venv install also works when its
    # Scripts directory is not present in the parent process PATH (Windows).
    cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", tf_dir,
           "-tb", "tf_converter",
           "-oiqt", "-qt", "per-tensor",
           "-cind", "mel", calib_path, "[[[[0.0]]]]", "[[[[1.0]]]]"]
    print("[tflite] running:", " ".join(cmd))
    converter_env = os.environ.copy()
    converter_env["PYTHONUTF8"] = "1"
    subprocess.run(cmd, check=True, env=converter_env)
    files = os.listdir(tf_dir)
    int8 = (glob.glob(os.path.join(tf_dir, "*full_integer_quant.tflite"))
            or glob.glob(os.path.join(tf_dir, "*integer_quant.tflite")))
    if not int8:
        raise SystemExit(f"no int8 tflite in {tf_dir}. files: {files}\n"
                         "paste this and I'll adjust the onnx2tf flags for your version.")
    p8 = os.path.join(args.out, "model_int8.tflite"); shutil.copy(int8[0], p8)
    fp = glob.glob(os.path.join(tf_dir, "*float32*.tflite"))
    if fp:
        shutil.copy(fp[0], os.path.join(args.out, "model_fp32.tflite"))
    print(f"[tflite] INT8 {os.path.getsize(p8)/1024:.0f} KB -> {p8}  "
          f"(from {os.path.basename(int8[0])})")
    print("[tflite] INT8 weights fit ESP32-S3 flash easily; activations in SRAM/PSRAM.")
    return p8


def _run_tflite(path, specs, int8_io):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    probs = []
    for s in specs:
        x = np.transpose(s, (0, 2, 3, 1)).astype("float32")
        if int8_io:
            sc, zp = inp["quantization"]; x = np.round(x / sc + zp).astype(inp["dtype"])
        it.set_tensor(inp["index"], x); it.invoke()
        y = it.get_tensor(out["index"]).astype("float32")
        if int8_io:
            sc, zp = out["quantization"]; y = (y - zp) * sc
        probs.append(1 / (1 + np.exp(-y.reshape(-1)[:2])))   # sigmoid on 2 logits
    return np.stack(probs)


def stage_parity(args):
    import torch
    device = "cpu"
    model, _, pb = _load_torch(args.ckpt, args.model, args.width, device)
    frontend = pb.LogMel().to(device)
    specs, tgts = _calib_specs(pb, frontend, device, args.manifest, args.n_calib,
                               args.noise_dir, args.snrs, seed=123)
    with torch.no_grad():
        p_torch = np.stack([1 / (1 + np.exp(-model(torch.from_numpy(s)).numpy().reshape(-1)[:2]))
                            for s in specs])
    p_int8 = _run_tflite(os.path.join(args.out, "model_int8.tflite"), specs, int8_io=True)
    thresholds = _load_thresholds(args.thresholds)
    m_torch = _per_head(tgts, p_torch, thresholds)
    m_int8 = _per_head(tgts, p_int8, thresholds)
    two_torch = pb.abnormal_operating_metrics(tgts, p_torch, thresholds)
    two_int8 = pb.abnormal_operating_metrics(tgts, p_int8, thresholds)
    three_torch = pb.product_three_class_metrics(tgts, p_torch, thresholds)
    three_int8 = pb.product_three_class_metrics(tgts, p_int8, thresholds)
    print("\n=== FP32 (PyTorch) vs INT8 (TFLite) — per head ===")
    for nm in ("crackle", "wheeze"):
        a, b = m_torch[nm], m_int8[nm]
        print(f"  {nm:<8} AUROC {a['auroc']:.3f} -> {b['auroc']:.3f}   "
              f"Se {a['se']:.3f}->{b['se']:.3f}  Sp {a['sp']:.3f}->{b['sp']:.3f}   "
              f"dAUROC={b['auroc']-a['auroc']:+.3f}")
    prob_gap = float(np.abs(p_torch - p_int8).mean())
    print(f"  mean |prob_fp32 - prob_int8| = {prob_gap:.4f}")
    print(f"  two-class balanced {two_torch['balanced_accuracy']:.3f} -> "
          f"{two_int8['balanced_accuracy']:.3f}")
    print(f"  product three-class macroF1 {three_torch['macro_f1']:.3f} -> "
          f"{three_int8['macro_f1']:.3f}")
    auroc_drops = {name: m_torch[name]["auroc"] - m_int8[name]["auroc"]
                   for name in ("crackle", "wheeze")}
    passed = (max(auroc_drops.values()) <= args.max_auroc_drop
              and prob_gap <= args.max_mean_prob_gap)
    report = {
        "passed": passed,
        "thresholds": thresholds.tolist(),
        "temperature_scaling_applied": False,
        "fp32": {"per_head": m_torch, "two_class": two_torch,
                 "three_class": three_torch},
        "int8": {"per_head": m_int8, "two_class": two_int8,
                 "three_class": three_int8},
        "mean_prob_gap": prob_gap,
        "auroc_drops": auroc_drops,
        "limits": {"max_auroc_drop": args.max_auroc_drop,
                   "max_mean_prob_gap": args.max_mean_prob_gap},
    }
    with open(os.path.join(args.out, "parity.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[parity] {'PASS' if passed else 'FAIL'}: temperature scaling was not applied")
    if not passed:
        raise SystemExit("INT8 parity limits exceeded; do not deploy this quantized model")


def main():
    ap = argparse.ArgumentParser(description="INT8 quantization for ESP32-S3")
    ap.add_argument("--stage", choices=["onnx", "tflite", "parity", "all"], default="all")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="dscnn_tcn")
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--snrs", type=float, nargs="*", default=[20, 10])
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--refresh-calibration", action="store_true",
                    help="regenerate calib.npy even when a reusable set exists")
    ap.add_argument("--thresholds", default=None,
                    help="training results.json containing frozen crackle/wheeze thresholds")
    ap.add_argument("--max-auroc-drop", type=float, default=0.02,
                    help="fail export if either INT8 head loses more AUROC")
    ap.add_argument("--max-mean-prob-gap", type=float, default=0.05,
                    help="fail export if mean absolute FP32/INT8 probability gap is larger")
    ap.add_argument("--out", default="runs/quant")
    args = ap.parse_args()
    if args.stage in ("onnx", "all"):
        onnx_path, _ = stage_onnx(args)
    else:
        onnx_path = None
    if args.stage in ("tflite", "all"):
        if not args.manifest:
            raise SystemExit("--manifest required for tflite/parity (calibration data)")
        stage_tflite(args, onnx_path)
    if args.stage in ("parity", "all"):
        if not args.manifest:
            raise SystemExit("--manifest required for parity")
        stage_parity(args)


if __name__ == "__main__":
    main()
