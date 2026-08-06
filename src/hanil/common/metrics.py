"""Metrics for imbalanced 3-class respiratory classification.

Reports the numbers that matter clinically: per-class sensitivity/specificity,
macro Se/Sp and their mean (the ICBHI-style "Score"), confusion matrix, and
overall accuracy. Accuracy alone is misleading when 'normal' dominates.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Sequence
import numpy as np


@dataclass
class ClassificationReport:
    classes: Sequence[str]
    confusion: np.ndarray
    accuracy: float
    per_class_sensitivity: Dict[str, float] = field(default_factory=dict)
    per_class_specificity: Dict[str, float] = field(default_factory=dict)
    macro_sensitivity: float = 0.0
    macro_specificity: float = 0.0
    icbhi_style_score: float = 0.0     # mean(Se, Sp) — comparable to the literature

    def __str__(self) -> str:
        lines = [f"accuracy={self.accuracy:.4f}  "
                 f"Se={self.macro_sensitivity:.4f}  Sp={self.macro_specificity:.4f}  "
                 f"Score={self.icbhi_style_score:.4f}"]
        for c in self.classes:
            lines.append(f"  {c:<8} Se={self.per_class_sensitivity[c]:.3f} "
                         f"Sp={self.per_class_specificity[c]:.3f}")
        lines.append("  confusion (rows=true, cols=pred):")
        lines.append("           " + " ".join(f"{c[:6]:>7}" for c in self.classes))
        for i, c in enumerate(self.classes):
            lines.append(f"  {c[:8]:>8} " + " ".join(f"{v:>7d}" for v in self.confusion[i]))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_sensitivity": self.macro_sensitivity,
            "macro_specificity": self.macro_specificity,
            "icbhi_style_score": self.icbhi_style_score,
            "per_class_sensitivity": self.per_class_sensitivity,
            "per_class_specificity": self.per_class_specificity,
            "confusion": self.confusion.tolist(),
        }


def evaluate(y_true: Sequence[int], y_pred: Sequence[int],
             classes: Sequence[str]) -> ClassificationReport:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0

    se, sp = {}, {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        se[c] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        sp[c] = float(tn / (tn + fp)) if (tn + fp) else 0.0

    macro_se = float(np.mean(list(se.values())))
    macro_sp = float(np.mean(list(sp.values())))
    return ClassificationReport(
        classes=list(classes), confusion=cm, accuracy=acc,
        per_class_sensitivity=se, per_class_specificity=sp,
        macro_sensitivity=macro_se, macro_specificity=macro_sp,
        icbhi_style_score=0.5 * (macro_se + macro_sp),
    )
