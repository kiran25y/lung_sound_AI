"""Tests for the front end, metrics and model shapes (needs torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hanil.common.config import Config
from hanil.common.metrics import evaluate


def _cfg():
    return Config({
        "seed": 0, "classes": ["normal", "crackle", "wheeze"],
        "audio": {"sample_rate": 16000, "clip_seconds": 1.0},
        "features": {"type": "logmel", "n_fft": 400, "hop_length": 160,
                     "win_length": 400, "n_mels": 64, "f_min": 50, "f_max": 2000,
                     "log_offset": 1e-6},
        "model": {"arch": "student_cnn", "student_width": 0.75, "dropout": 0.1},
        "device_robust": {"mixstyle": {"enable": True, "p": 0.5, "alpha": 0.3},
                          "contrastive": {"enable": True, "temperature": 0.1}},
    })


def test_logmel_shape():
    from hanil.common.features import LogMel
    cfg = _cfg()
    fe = LogMel(cfg)
    wav = torch.randn(4, 16000)
    spec = fe(wav)
    assert spec.shape[0] == 4 and spec.shape[1] == 1 and spec.shape[2] == 64


def test_student_forward_and_param_budget():
    from hanil.phase1_classification.models import build_model
    from hanil.phase1_classification.models.student_cnn import count_params
    cfg = _cfg()
    model = build_model(cfg)
    n_mels, frames = 64, 101
    logits, emb = model(torch.randn(2, 1, n_mels, frames), return_embedding=True)
    assert logits.shape == (2, 3)
    assert emb.shape[0] == 2
    n = count_params(model)
    assert 300_000 < n < 2_000_000, f"param count {n} outside ESP32 budget"


def test_metrics_perfect_and_confusion():
    rep = evaluate([0, 1, 2, 0], [0, 1, 2, 0], ["normal", "crackle", "wheeze"])
    assert rep.accuracy == 1.0
    assert abs(rep.icbhi_style_score - 1.0) < 1e-9


def test_contrastive_loss_runs():
    from hanil.phase1_classification.contrastive import StethoGuidedSupConLoss
    loss = StethoGuidedSupConLoss()
    feats = torch.randn(8, 32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    val = loss(feats, labels, domains)
    assert torch.isfinite(val)
