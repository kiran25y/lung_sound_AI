"""Phase 2.1 — Dataset strategy.

Builds patient-level splits from a manifest CSV so no patient leaks across
train/test. Produces the named splits the evaluator expects:
    train, internal_val, internal_test  (Seoul stethoscope)
    external_test                        (Sungbook stethoscope — never trained on)
    smartphone_test                      (either hospital, smartphone)
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..common.audio import load_wav, fix_length

LABEL2IDX = {"normal": 0, "crackle": 1, "wheeze": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
DEVICE2IDX = {"stethoscope": 0, "smartphone": 1}


def _split_patients(patient_ids: List[str], ratios=(0.7, 0.15, 0.15),
                    seed: int = 1337) -> Dict[str, set]:
    rng = np.random.default_rng(seed)
    pats = sorted(set(patient_ids))
    rng.shuffle(pats)
    n = len(pats)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    return {"train": set(pats[:n_tr]),
            "val": set(pats[n_tr:n_tr + n_va]),
            "test": set(pats[n_tr + n_va:])}


def build_splits(manifest_csv: str, seed: int = 1337) -> Dict[str, pd.DataFrame]:
    """Return a dict of split-name -> DataFrame, all splits patient-disjoint."""
    df = pd.read_csv(manifest_csv)
    required = {"filepath", "patient_id", "label", "hospital", "device"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(LABEL2IDX)].reset_index(drop=True)

    seoul_steth = df[(df.hospital == "seoul") & (df.device == "stethoscope")]
    sungbook_steth = df[(df.hospital == "sungbook") & (df.device == "stethoscope")]
    smartphone = df[df.device == "smartphone"]

    # Patient-level split of the internal (Seoul) stethoscope set
    p = _split_patients(seoul_steth.patient_id.tolist(), seed=seed)
    splits = {
        "train": seoul_steth[seoul_steth.patient_id.isin(p["train"])],
        "internal_val": seoul_steth[seoul_steth.patient_id.isin(p["val"])],
        "internal_test": seoul_steth[seoul_steth.patient_id.isin(p["test"])],
        "external_test": sungbook_steth,               # headline generalisation
        "smartphone_test": smartphone,
    }
    for name, part in splits.items():
        splits[name] = part.reset_index(drop=True)
    return splits


class LungSoundDataset(Dataset):
    """Returns (waveform[1,T], label:int, domain:int). Augmentation happens in the
    collate/training loop so the same Dataset serves clean and augmented runs."""

    def __init__(self, frame: pd.DataFrame, cfg, training: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.training = training
        self.sr = cfg.audio["sample_rate"]
        self.num_samples = int(cfg.audio["clip_seconds"] * self.sr)

    def __len__(self) -> int:
        return len(self.frame)

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(LABEL2IDX), dtype=np.int64)
        for lbl in self.frame["label"]:
            counts[LABEL2IDX[lbl]] += 1
        return counts

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        wav = load_wav(row["filepath"], self.sr)
        wav = fix_length(wav, self.num_samples, training=self.training)
        label = LABEL2IDX[row["label"]]
        domain = DEVICE2IDX.get(row["device"], 0)
        return wav, label, domain
