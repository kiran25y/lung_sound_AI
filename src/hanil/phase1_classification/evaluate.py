"""Phase 1 evaluation — runs a model over named splits and prints clinical metrics.

Supports 'noisy' variants (e.g. internal_test_noisy, smartphone_test_noisy) by
applying the noise mixer at eval time to measure robustness degradation.
"""
from __future__ import annotations
from typing import Dict, List
import torch
from torch.utils.data import DataLoader

from ..common.features import LogMel
from ..common.metrics import evaluate as eval_metrics, ClassificationReport
from .datasets import LungSoundDataset
from .augment.noise_mixer import NoiseMixer


@torch.no_grad()
def run_split(model, frame, cfg, device, add_noise: bool = False) -> ClassificationReport:
    frontend = LogMel(cfg).to(device).eval()
    ds = LungSoundDataset(frame, cfg, training=False)
    loader = DataLoader(ds, batch_size=cfg.train["batch_size"], shuffle=False,
                        num_workers=cfg.train.get("num_workers", 4))
    mixer = None
    if add_noise:
        noisy_cfg = _force_noise(cfg)
        mixer = NoiseMixer(noisy_cfg)

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for wav, label, _domain in loader:
        if mixer is not None:
            wav = torch.stack([mixer(w) for w in wav])
        wav = wav.to(device)
        spec = frontend(wav)
        out = model(spec)
        logits = out[0] if isinstance(out, tuple) else out
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(label.tolist())
    return eval_metrics(y_true, y_pred, cfg.classes)


def _force_noise(cfg):
    import copy
    c = copy.deepcopy(cfg)
    c["noise"] = dict(c.get("noise", {}))
    c["noise"].update({"enable": True, "prob": 1.0, "tier": "research"})
    return type(cfg)(c)


def evaluate_all(model, splits: Dict[str, "pd.DataFrame"], cfg, device) -> Dict[str, dict]:
    results = {}
    for name in cfg.eval["report_splits"]:
        base = name.replace("_noisy", "")
        if base not in splits:
            print(f"[eval] split '{base}' not in manifest — skipping")
            continue
        rep = run_split(model, splits[base], cfg, device, add_noise=name.endswith("_noisy"))
        print(f"\n=== {name} ===\n{rep}")
        results[name] = rep.to_dict()
    return results
