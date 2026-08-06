"""Phase 2 — Learned noise-event detector, trained on DESED_synthetic.

A small CNN that classifies the *type* of ambient sound present (speech, dishes,
running water, vacuum, ...), acting as a second opinion to the DSP noise-check.
Used during development/benchmarking; DESED audio is research-tier (see
LICENSING.md) so verify licensing before shipping a model trained on it.

DESED_synthetic ships Scaper .jams / .tsv strong labels; here we use the 10-class
weak (clip-level) presence formulation, which is enough for a gate.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn

DESED_CLASSES = ["alarm_bell_ringing", "blender", "cat", "dishes", "dog",
                 "electric_shaver_toothbrush", "frying", "running_water",
                 "speech", "vacuum_cleaner"]


class NoiseEventCNN(nn.Module):
    """Tiny multi-label CNN over log-Mel; sigmoid outputs (sounds co-occur)."""

    def __init__(self, n_classes: int = len(DESED_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU6(True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU6(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU6(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, n_classes)

    def forward(self, spec):
        return self.head(self.net(spec))     # logits (apply sigmoid outside)


def parse_desed_tsv(tsv_path: str) -> "dict[str, list[str]]":
    """Map filename -> list of present event labels from a DESED strong-label TSV
    (columns: filename, onset, offset, event_label)."""
    import csv
    present: dict[str, set] = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fn = row["filename"]
            lbl = row.get("event_label", "").strip()
            if lbl:
                present.setdefault(fn, set()).add(lbl)
    return {k: sorted(v) for k, v in present.items()}


def labels_to_multihot(labels: List[str]) -> torch.Tensor:
    vec = torch.zeros(len(DESED_CLASSES))
    idx = {c: i for i, c in enumerate(DESED_CLASSES)}
    for l in labels:
        if l in idx:
            vec[idx[l]] = 1.0
    return vec
