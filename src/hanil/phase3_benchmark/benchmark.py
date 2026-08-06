"""Phase 4.1 — Sound Quality Analysis / device benchmarking.

Compares your PCB against commercial devices (StethoMe, Skeeper) and smartphone on
the SAME patients recorded simultaneously, scored by the SAME AI model. Produces:
  * per-device clinical metrics (Se/Sp/Score)
  * inter-rater agreement (Cohen's / Fleiss' kappa) for the pulmonologist labels
  * device-vs-reference agreement (Bland-Altman stats) for equivalence claims

Input: a CSV with columns
  patient_id, device, true_label, pred_label, rater1, rater2[, raterN...]
where 'device' is one of {stethome, skeeper, smartphone, hanil_esp32}.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd

from ..common.metrics import evaluate, ClassificationReport

LABELS = ["normal", "crackle", "wheeze"]
L2I = {l: i for i, l in enumerate(LABELS)}


def cohen_kappa(a: List[int], b: List[int], k: int) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cm = np.zeros((k, k))
    for x, y in zip(a, b):
        cm[x, y] += 1
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = (cm.sum(0) / n) @ (cm.sum(1) / n)
    return float((po - pe) / (1 - pe)) if pe != 1 else 1.0


def fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """rating_matrix [N_items, N_categories] of counts per item."""
    N, k = rating_matrix.shape
    n = rating_matrix.sum(axis=1)
    if not np.all(n == n[0]):
        # allow variable raters: normalise per item
        pass
    n_raters = rating_matrix.sum(axis=1)
    p_j = rating_matrix.sum(axis=0) / rating_matrix.sum()
    P_i = ((rating_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1) + 1e-12)
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e != 1 else 1.0


def bland_altman(reference: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    """Agreement between a continuous score from reference vs test device
    (e.g. model probability of 'abnormal'). Returns bias and limits of agreement."""
    reference, test = np.asarray(reference, float), np.asarray(test, float)
    diff = test - reference
    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return {"bias": mean_diff,
            "loa_lower": mean_diff - 1.96 * sd,
            "loa_upper": mean_diff + 1.96 * sd,
            "sd": sd}


@dataclass
class BenchmarkReport:
    per_device: Dict[str, ClassificationReport] = field(default_factory=dict)
    rater_kappa: Dict[str, float] = field(default_factory=dict)
    agreement_vs_reference: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=== Per-device clinical metrics ==="]
        for dev, rep in self.per_device.items():
            lines.append(f"\n[{dev}]  acc={rep.accuracy:.3f}  Score={rep.icbhi_style_score:.3f} "
                         f"Se={rep.macro_sensitivity:.3f} Sp={rep.macro_specificity:.3f}")
        lines.append("\n=== Inter-rater agreement (kappa) ===")
        for k, v in self.rater_kappa.items():
            lines.append(f"  {k}: {v:.3f}")
        if self.agreement_vs_reference:
            lines.append("\n=== Device vs reference agreement (Bland-Altman) ===")
            for dev, ba in self.agreement_vs_reference.items():
                lines.append(f"  {dev}: bias={ba['bias']:+.3f} "
                             f"LoA=[{ba['loa_lower']:+.3f}, {ba['loa_upper']:+.3f}]")
        return "\n".join(lines)


def run_benchmark(csv_path: str, reference_device: str = "stethome",
                  score_col: str | None = None) -> BenchmarkReport:
    df = pd.read_csv(csv_path)
    report = BenchmarkReport()

    for dev, sub in df.groupby("device"):
        yt = [L2I[l] for l in sub["true_label"].str.lower()]
        yp = [L2I[l] for l in sub["pred_label"].str.lower()]
        report.per_device[dev] = evaluate(yt, yp, LABELS)

    # inter-rater agreement (pairwise Cohen + overall Fleiss)
    rater_cols = [c for c in df.columns if c.startswith("rater")]
    if len(rater_cols) >= 2:
        for i in range(len(rater_cols)):
            for j in range(i + 1, len(rater_cols)):
                a = [L2I[l] for l in df[rater_cols[i]].str.lower()]
                b = [L2I[l] for l in df[rater_cols[j]].str.lower()]
                report.rater_kappa[f"cohen_{rater_cols[i]}_{rater_cols[j]}"] = \
                    cohen_kappa(a, b, len(LABELS))
        # Fleiss over all raters
        mat = np.zeros((len(df), len(LABELS)))
        for r, (_, row) in enumerate(df.iterrows()):
            for c in rater_cols:
                mat[r, L2I[str(row[c]).lower()]] += 1
        report.rater_kappa["fleiss_all"] = fleiss_kappa(mat)

    # device-vs-reference agreement on a continuous score, if provided
    if score_col and score_col in df.columns and reference_device in set(df["device"]):
        ref = df[df.device == reference_device].set_index("patient_id")[score_col]
        for dev, sub in df.groupby("device"):
            if dev == reference_device:
                continue
            s = sub.set_index("patient_id")[score_col]
            common = ref.index.intersection(s.index)
            if len(common):
                report.agreement_vs_reference[dev] = bland_altman(
                    ref.loc[common].values, s.loc[common].values)
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reference", default="stethome")
    ap.add_argument("--score_col", default=None)
    a = ap.parse_args()
    print(run_benchmark(a.csv, a.reference, a.score_col).summary())
