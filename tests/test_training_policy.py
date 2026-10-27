"""Regression tests for the false-positive reduction training policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import phase1_baseline as training  # noqa: E402


def test_patient_calibration_split_is_disjoint_and_deterministic():
    rows = []
    for patient in range(10):
        for clip in range(3):
            rows.append({"patient_id": f"p{patient}", "clip": clip})
    frame = pd.DataFrame(rows)
    calibration_a, test_a = training.reserve_calibration_patients(
        frame, fraction=0.3, seed=7)
    calibration_b, test_b = training.reserve_calibration_patients(
        frame, fraction=0.3, seed=7)
    assert set(calibration_a.patient_id).isdisjoint(set(test_a.patient_id))
    assert calibration_a.patient_id.nunique() == 3
    assert calibration_a.equals(calibration_b)
    assert test_a.equals(test_b)


def test_joint_threshold_tuning_reduces_false_positives_at_sensitivity_floor():
    targets = np.array([
        [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
        [1, 0], [1, 0], [0, 1], [0, 1],
    ], dtype=float)
    probs = np.array([
        [0.20, 0.20], [0.30, 0.55], [0.25, 0.58],
        [0.48, 0.30], [0.49, 0.20], [0.15, 0.65],
        [0.80, 0.20], [0.75, 0.25], [0.20, 0.90], [0.30, 0.85],
    ])
    current = training.abnormal_operating_metrics(targets, probs, [0.45, 0.50])
    thresholds, tuned = training.tune_joint_thresholds(
        targets, probs, target_sensitivity=1.0, grid_step=0.01)
    assert tuned["sensitivity"] == 1.0
    assert tuned["specificity"] > current["specificity"]
    assert thresholds[1] > 0.50


def test_sqrt_positive_weight_is_less_aggressive_than_full_ratio():
    training.set_multilabel(True)
    frame = pd.DataFrame({"label": ["normal"] * 80 + ["crackle"] * 15
                                    + ["wheeze"] * 5})
    sqrt_loss = training.make_criterion(
        frame, "cpu", multilabel_pos_weight_power=0.5,
        multilabel_pos_weight_cap=4.0)
    full_loss = training.make_criterion(
        frame, "cpu", multilabel_pos_weight_power=1.0,
        multilabel_pos_weight_cap=0.0)
    assert np.all(sqrt_loss.pos_weight.numpy() < full_loss.pos_weight.numpy())
    assert np.all(sqrt_loss.pos_weight.numpy() <= 4.0)


def test_subtype_loss_rewards_the_correct_pure_head():
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0],
                            [0.0, 0.0], [1.0, 1.0]])
    correct = torch.tensor([[3.0, -1.0], [-1.0, 3.0],
                            [2.0, -2.0], [-2.0, 2.0]])
    reversed_heads = correct[:, [1, 0]]
    assert (training.subtype_discrimination_loss(correct, targets)
            < training.subtype_discrimination_loss(reversed_heads, targets))


def test_product_three_class_rule_excludes_both_and_resolves_dual_fire():
    targets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    probs = np.array([[0.1, 0.2], [0.8, 0.7], [0.6, 0.9], [0.9, 0.8]])
    metrics = training.product_three_class_metrics(targets, probs, [0.5, 0.5])
    assert metrics["excluded_both"] == 1
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_explicit_left_pad_matches_old_causal_crop():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 17)
    for dilation in (1, 2):
        pad = 2 * dilation
        old = torch.nn.Conv1d(3, 3, 3, padding=pad, dilation=dilation,
                              groups=3, bias=False)
        new = torch.nn.Conv1d(3, 3, 3, padding=0, dilation=dilation,
                              groups=3, bias=False)
        new.weight.data.copy_(old.weight.data)
        expected = old(x)[..., :x.shape[-1]]
        left = x.new_zeros((x.shape[0], x.shape[1], pad))
        actual = new(torch.cat((left, x), dim=-1))
        assert torch.allclose(actual, expected, atol=1e-6)


def test_sram_dscnn_stays_under_static_activation_budget():
    model = training.build_model(
        "sram_dscnn", num_classes=2, width=1.0, dropout=0.1)
    profile = training.int8_activation_profile(model)
    output = model(torch.zeros(1, 1, 64, 401))
    assert output.shape == (1, 2)
    assert profile[0]["shape"] == (1, 24, 31, 100)
    assert profile[0]["int8_bytes"] == 74_400
    assert profile[0]["int8_bytes"] <= 100 * 1024


def test_sram_dscnn_uses_no_explicit_convolution_padding():
    model = training.build_model("sram_dscnn", num_classes=2, width=1.0)
    convolutions = [module for module in model.modules()
                    if isinstance(module, torch.nn.Conv2d)]
    assert convolutions
    assert all(module.padding == (0, 0) for module in convolutions)
