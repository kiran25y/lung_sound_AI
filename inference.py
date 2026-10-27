#!/usr/bin/env python3
"""inference.py — full inference + decision layer: quality gate -> model -> ALL output views.

The model has two sigmoid heads (crackle, wheeze). From those two probabilities we derive
every view the product needs — no separate models:

  multi-label : crackle yes/no, wheeze yes/no
  2-class     : normal / abnormal          (+ p_abnormal)
  3-class     : normal / crackle / wheeze   ('both' resolves to the dominant head)
  4-state     : normal / crackle / wheeze / both

Gated: a recording that fails the quality check returns RETRY / NO_DECISION and is NOT
classified (a blank/no-lung clip is never called 'normal').

`decide()` is pure logic and is the spec firmware reproduces. Backends:
  --tflite model_int8.tflite   (deployment; numpy log-Mel, matches firmware)
  --ckpt   best.pt             (dev; torch)

  python inference.py --wav clip.wav --tflite runs/aug_all/quant/model_int8.tflite \
      --thresholds runs/aug_all/results.json --gate-config runs/gate_thresholds.json
  python inference.py --manifest data/manifest.csv --ckpt runs/aug_all/best.pt \
      --model dscnn_tcn --width 2.0 --thresholds runs/aug_all/results.json --out preds.json
"""
from __future__ import annotations
import argparse, json, os
from math import gcd
import numpy as np

SR = 16000
CLIP_S = 4.0
N = int(SR * CLIP_S)


# ── decision logic (firmware spec) ───────────────────────────────────────────
def decide(pc, pw, thr_c=0.5, thr_w=0.5):
    """Two head probabilities -> all output views. Pure, deterministic, firmware-replicable."""
    pc, pw = float(pc), float(pw)
    c, w = pc >= thr_c, pw >= thr_w
    p_abnormal = 1.0 - (1.0 - pc) * (1.0 - pw)          # P(at least one abnormality)
    if c and w:
        state = "both"
    elif c:
        state = "crackle"
    elif w:
        state = "wheeze"
    else:
        state = "normal"
    class_2 = "abnormal" if (c or w) else "normal"
    if not (c or w):
        class_3 = "normal"
    elif c and w:
        class_3 = "crackle" if pc >= pw else "wheeze"    # dominant head; 'both' kept in state_4
    elif c:
        class_3 = "crackle"
    else:
        class_3 = "wheeze"
    return {
        "p_crackle": round(pc, 3), "p_wheeze": round(pw, 3), "p_abnormal": round(p_abnormal, 3),
        "crackle": bool(c), "wheeze": bool(w),
        "class_2": class_2, "class_3": class_3, "state_4": state,
    }


# ── audio + thresholds ───────────────────────────────────────────────────────
def load_clip(path):
    import soundfile as sf
    d, s = sf.read(path, dtype="float32", always_2d=True); x = d.mean(1)
    if s != SR:
        from scipy.signal import resample_poly
        g = gcd(int(s), SR); x = resample_poly(x, SR // g, int(s) // g)
    if len(x) < N:
        x = np.tile(x, N // max(1, len(x)) + 1)
    return x[:N].astype(np.float64)


def load_thresholds(path):
    if not path:
        return (0.5, 0.5)
    obj = json.load(open(path))
    thr = obj.get("thresholds", obj) if isinstance(obj, dict) else obj
    thr = np.asarray(thr, float).reshape(-1)
    return (float(thr[0]), float(thr[1]))


# ── model backends -> (p_crackle, p_wheeze) ──────────────────────────────────
class TorchModel:
    def __init__(self, args):
        import torch, phase1_baseline as pb
        pb.set_multilabel(True)
        st = torch.load(args.ckpt, map_location="cpu")
        sd = st["model"] if isinstance(st, dict) and "model" in st else st
        for w in [args.width, 2.0, 1.0, 1.5, 3.0, 0.5]:
            m = pb.build_model(args.model, num_classes=len(pb.LABELS), width=w)
            try:
                m.load_state_dict(sd); m.eval(); self.m = m; break
            except Exception:
                self.m = None
        if self.m is None:
            raise SystemExit("could not load --ckpt")
        self.fe = pb.LogMel(); self.torch = torch

    def probs(self, x):
        with self.torch.no_grad():
            spec = self.fe(self.torch.tensor(x[None], dtype=self.torch.float32))
            p = self.torch.sigmoid(self.m(spec))[0].numpy()
        return float(p[0]), float(p[1])


class TFLiteModel:
    def __init__(self, path):
        import tensorflow as tf, feature_config as fc
        self.fc = fc
        self.it = tf.lite.Interpreter(model_path=path); self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]; self.out = self.it.get_output_details()[0]

    def probs(self, x):
        m = self.fc.logmel_numpy(x, self.fc.DEFAULT)              # [64, T], firmware reference
        xin = np.transpose(m[None, None], (0, 2, 3, 1)).astype("float32")  # NHWC [1,64,T,1]
        if self.inp["dtype"] == np.int8:
            sc, zp = self.inp["quantization"]; xin = np.round(xin / sc + zp).astype(np.int8)
        self.it.set_tensor(self.inp["index"], xin); self.it.invoke()
        y = self.it.get_tensor(self.out["index"]).astype("float32").reshape(-1)[:2]
        if self.out["dtype"] == np.int8:
            sc, zp = self.out["quantization"]; y = (y - zp) * sc
        p = 1.0 / (1.0 + np.exp(-y))
        return float(p[0]), float(p[1])


def main():
    ap = argparse.ArgumentParser(description="Gated inference with 2-class / 3-class / 4-state outputs")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--tflite", default=None, help="INT8 tflite (deployment backend)")
    ap.add_argument("--ckpt", default=None, help="torch checkpoint (dev backend)")
    ap.add_argument("--model", default="dscnn_tcn")
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--thresholds", default=None, help="results.json or [thr_c, thr_w]")
    ap.add_argument("--gate-config", default=None)
    ap.add_argument("--no-gate", action="store_true", help="skip the quality gate")
    ap.add_argument("--limit", type=int, default=0, help="manifest mode: cap number of clips (0=all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    thr_c, thr_w = load_thresholds(args.thresholds)
    from phase2_quality_gate import QualityGate, QualityConfig
    gate = None if args.no_gate else QualityGate(
        QualityConfig.from_json(args.gate_config) if args.gate_config else QualityConfig())
    if args.tflite:
        backend = TFLiteModel(args.tflite)
    elif args.ckpt:
        backend = TorchModel(args)
    else:
        raise SystemExit("give --tflite or --ckpt")

    files = [args.wav] if args.wav else None
    if not files:
        if not args.manifest:
            raise SystemExit("give --wav or --manifest")
        import pandas as pd
        files = list(pd.read_csv(args.manifest)["filepath"])

    print(f"[thresholds] crackle>={thr_c}  wheeze>={thr_w}   gate={'off' if args.no_gate else 'on'}")
    if args.limit:
        files = files[:args.limit]
    total = len(files); missing = 0
    results = []
    for i, fp in enumerate(files):
        if not os.path.exists(fp):
            missing += 1
            if args.wav:
                raise SystemExit(f"[error] file not found: {fp}")
            continue
        x = load_clip(fp)
        rec = {"filepath": fp}
        if gate is not None:
            gr = gate.check(x)
            if not gr.passed:
                rec.update({"decision": gr.decision.value, "reasons": list(gr.reasons)})
                results.append(rec)
                if args.wav:
                    print(json.dumps(rec, indent=2))
                continue
        pc, pw = backend.probs(x)
        rec.update({"decision": "classified", **decide(pc, pw, thr_c, thr_w)})
        results.append(rec)
        if args.wav:
            print(json.dumps(rec, indent=2))
        elif (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{total}", flush=True)

    if not args.wav:
        n = len(results)
        cl = [r for r in results if r.get("decision") == "classified"]
        gated = n - len(cl)
        from collections import Counter
        print(f"\n[summary] {n} clips (missing on disk: {missing})  |  "
              f"gated(RETRY/NO_DECISION)={gated}  classified={len(cl)}")
        print("  2-class:", dict(Counter(r["class_2"] for r in cl)))
        print("  3-class:", dict(Counter(r["class_3"] for r in cl)))
        print("  4-state:", dict(Counter(r["state_4"] for r in cl)))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()