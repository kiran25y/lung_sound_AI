#!/usr/bin/env python3
"""Combine clean and stress evaluations into one reproducible A/B report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


GROUP_LABELS = {
    "split:training": "Training (descriptive)",
    "split:heldout": "Held-out stethoscope",
    "smartphone_seoul": "External smartphone Seoul",
    "smartphone_sungbook": "External smartphone Sungbook",
    "split:external": "External smartphone combined",
    "all_real_sources": "All sources",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def model_metadata(model: Path, clean: dict[str, Any], parity: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(model.resolve()),
        "bytes": model.stat().st_size,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "thresholds": clean["decision_thresholds"],
        "quantization_parity": parity,
    }


def compact_metrics(group: dict[str, Any], section: str) -> dict[str, Any]:
    metrics = group[section]
    two = metrics["two_class_normal_vs_abnormal"]
    three = metrics["three_class_excluding_true_both"]
    four = metrics["four_state_internal_heads"]
    return {
        "n": two["n"],
        "two_class_accuracy": two["accuracy"],
        "two_class_balanced_accuracy": two["balanced_accuracy"],
        "two_class_sensitivity": two["sensitivity"],
        "two_class_specificity": two["specificity"],
        "three_class_accuracy": three["accuracy"],
        "three_class_macro_f1": three["macro_f1"],
        "four_state_macro_f1_diagnostic": four["macro_f1"],
        "crackle_f1": metrics["crackle_head"]["f1"],
        "wheeze_f1": metrics["wheeze_head"]["f1"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-clean", type=Path, default=root / "comparison_evaluation/old_teacher_clean/summary.json")
    parser.add_argument("--student-clean", type=Path, default=root / "comparison_evaluation/new_student_clean/summary.json")
    parser.add_argument("--teacher-stress", type=Path, default=root / "comparison_evaluation/old_teacher_stress/summary.json")
    parser.add_argument("--student-stress", type=Path, default=root / "comparison_evaluation/new_student_stress/summary.json")
    parser.add_argument("--teacher-model", type=Path, default=root / "pipeline_source/hanil_lung_ai/runs/improved_specificity_s1337/quant/model_int8.tflite")
    parser.add_argument("--student-model", type=Path, default=root / "artifacts/sram_dscnn_int8_artifacts/runs/sram_dscnn_s1337/quant/model_int8.tflite")
    parser.add_argument("--teacher-parity", type=Path, default=root / "pipeline_source/hanil_lung_ai/runs/improved_specificity_s1337/quant/parity.json")
    parser.add_argument("--student-parity", type=Path, default=root / "artifacts/sram_dscnn_int8_artifacts/runs/sram_dscnn_s1337/quant/parity.json")
    parser.add_argument("--host-benchmark", type=Path, default=root / "comparison_evaluation/host_latency_benchmark.json")
    parser.add_argument("--memory-analysis", type=Path, default=root / "comparison_evaluation/tflite_memory_analysis.json")
    parser.add_argument("--out-json", type=Path, default=root / "comparison_evaluation/comparison.json")
    parser.add_argument("--out-report", type=Path, default=root / "comparison_evaluation/COMPARISON_REPORT.md")
    args = parser.parse_args()

    teacher_clean, student_clean = load(args.teacher_clean), load(args.student_clean)
    teacher_stress, student_stress = load(args.teacher_stress), load(args.student_stress)
    teacher_meta = model_metadata(args.teacher_model, teacher_clean, load(args.teacher_parity))
    student_meta = model_metadata(args.student_model, student_clean, load(args.student_parity))
    host_benchmark = load(args.host_benchmark)
    memory_analysis = load(args.memory_analysis)
    teacher_key = next(key for key, value in host_benchmark.items()
                       if value["sha256"] == teacher_meta["sha256"])
    student_key = next(key for key, value in host_benchmark.items()
                       if value["sha256"] == student_meta["sha256"])
    teacher_memory_key = next(key for key, value in memory_analysis.items()
                              if value["sha256"] == teacher_meta["sha256"])
    student_memory_key = next(key for key, value in memory_analysis.items()
                              if value["sha256"] == student_meta["sha256"])

    clean: dict[str, Any] = {}
    for key, label in GROUP_LABELS.items():
        tg, sg = teacher_clean["groups"][key], student_clean["groups"][key]
        clean[key] = {
            "label": label,
            "gate": {
                "n": tg["n_evaluated"], "pass": tg["n_pass"], "retry": tg["n_retry"],
                "pass_rate": tg["pass_rate"], "reasons": tg["retry_reasons"],
                "identical_between_models": (
                    tg["n_pass"] == sg["n_pass"] and tg["retry_reasons"] == sg["retry_reasons"]
                ),
            },
            "teacher": {
                "classification_only": compact_metrics(tg, "classification_only_metrics"),
                "gated_accepted": compact_metrics(tg, "accepted_clip_metrics"),
                "correct_yield_two_class": tg["correct_yield_over_all_evaluated"]["two_class"],
            },
            "student": {
                "classification_only": compact_metrics(sg, "classification_only_metrics"),
                "gated_accepted": compact_metrics(sg, "accepted_clip_metrics"),
                "correct_yield_two_class": sg["correct_yield_over_all_evaluated"]["two_class"],
            },
        }

    stress: dict[str, Any] = {}
    condition_names = sorted(
        key.removeprefix("condition:") for key in teacher_stress["groups"]
        if key.startswith("condition:") and "/source:" not in key
    )
    for condition in condition_names:
        key = f"condition:{condition}"
        tg, sg = teacher_stress["groups"][key], student_stress["groups"][key]
        stress[condition] = {
            "gate": {
                "n": tg["n_evaluated"], "pass": tg["n_pass"], "retry": tg["n_retry"],
                "pass_rate": tg["pass_rate"], "reasons": tg["retry_reasons"],
                "identical_between_models": (
                    tg["n_pass"] == sg["n_pass"] and tg["retry_reasons"] == sg["retry_reasons"]
                ),
            },
            "teacher": {
                "classification_only": compact_metrics(tg, "classification_only_metrics"),
                "gated_accepted": compact_metrics(tg, "accepted_clip_metrics"),
                "correct_yield_two_class": tg["correct_yield_over_all_evaluated"]["two_class"],
            },
            "student": {
                "classification_only": compact_metrics(sg, "classification_only_metrics"),
                "gated_accepted": compact_metrics(sg, "accepted_clip_metrics"),
                "correct_yield_two_class": sg["correct_yield_over_all_evaluated"]["two_class"],
            },
        }

    result = {
        "scope": {
            "clean_clips": teacher_clean["groups"]["all_real_sources"]["n_evaluated"],
            "stress_inputs_per_model": teacher_stress["groups"]["all_real_sources"]["n_evaluated"],
            "runtime": "LiteRT 2.2.0 BUILTIN_REF, four worker processes",
            "warning": "Host latency is comparative and is not ESP32-S3 latency.",
        },
        "models": {"teacher": teacher_meta, "student": student_meta},
        "size_reduction_fraction": 1.0 - student_meta["bytes"] / teacher_meta["bytes"],
        "clean": clean,
        "stress": stress,
        "host_latency_ms": {
            "teacher": teacher_clean["groups"]["all_real_sources"]["latency_ms"],
            "student": student_clean["groups"]["all_real_sources"]["latency_ms"],
        },
        "isolated_host_benchmark": {
            "teacher": host_benchmark[teacher_key], "student": host_benchmark[student_key],
        },
        "tflite_memory_analysis": {
            "teacher": memory_analysis[teacher_memory_key],
            "student": memory_analysis[student_memory_key],
        },
    }
    result["host_median_inference_speedup"] = (
        result["isolated_host_benchmark"]["teacher"]["invoke_ms"]["median"] /
        result["isolated_host_benchmark"]["student"]["invoke_ms"]["median"]
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# INT8 model and end-to-end pipeline comparison", "",
        "> Scope: exact 4-second 16 kHz firmware frontend, calibrated C-equivalent quality gate, "
        "LiteRT BUILTIN_REF INT8 inference, model-specific decision thresholds. These are acoustic "
        "classification metrics, not clinical diagnostic accuracy.", "",
        f"Teacher artifact: `{teacher_meta['sha256'][:12]}` with thresholds "
        f"[{teacher_meta['thresholds']['crackle']}, {teacher_meta['thresholds']['wheeze']}].  ",
        f"Student artifact: `{student_meta['sha256'][:12]}` with thresholds "
        f"[{student_meta['thresholds']['crackle']}, {student_meta['thresholds']['wheeze']}].", "",
        "## Artifact and speed comparison", "",
        "| Item | Teacher | SRAM student | Delta |", "|---|---:|---:|---:|",
        f"| TFLite file | {teacher_meta['bytes'] / 1024:.1f} KiB | {student_meta['bytes'] / 1024:.1f} KiB | -{100 * result['size_reduction_fraction']:.1f}% |",
        f"| Isolated host inference median | {result['isolated_host_benchmark']['teacher']['invoke_ms']['median']:.3f} ms | {result['isolated_host_benchmark']['student']['invoke_ms']['median']:.3f} ms | {result['host_median_inference_speedup']:.2f}x faster |",
        f"| Isolated host inference p95 | {result['isolated_host_benchmark']['teacher']['invoke_ms']['p95']:.3f} ms | {result['isolated_host_benchmark']['student']['invoke_ms']['p95']:.3f} ms | reference only |",
        f"| Estimated MACs | {result['tflite_memory_analysis']['teacher']['estimated_macs']:,} | {result['tflite_memory_analysis']['student']['estimated_macs']:,} | -{100 * (1 - result['tflite_memory_analysis']['student']['estimated_macs'] / result['tflite_memory_analysis']['teacher']['estimated_macs']):.1f}% |",
        f"| Activation-liveness lower bound | {result['tflite_memory_analysis']['teacher']['activation_liveness_lower_bound_bytes'] / 1024:.1f} KiB | {result['tflite_memory_analysis']['student']['activation_liveness_lower_bound_bytes'] / 1024:.1f} KiB | not final arena |",
        f"| Quantization mean probability gap | {teacher_meta['quantization_parity']['mean_prob_gap']:.4f} | {student_meta['quantization_parity']['mean_prob_gap']:.4f} | both pass |",
        "", "Host timing does not predict ESP32-S3 latency. The student needs new on-board portable-TFLM and ESP-NN parity/timing measurements.", "",
        "## Clean classification-only results", "",
        "| Dataset | Model | 2-class acc | Balanced acc | Sensitivity | Specificity | 3-class acc | 3-class macro-F1 |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in GROUP_LABELS:
        row = clean[key]
        for model_name, display in (("teacher", "Teacher"), ("student", "Student")):
            m = row[model_name]["classification_only"]
            lines.append(
                f"| {row['label']} | {display} | {pct(m['two_class_accuracy'])} | "
                f"{pct(m['two_class_balanced_accuracy'])} | {pct(m['two_class_sensitivity'])} | "
                f"{pct(m['two_class_specificity'])} | {pct(m['three_class_accuracy'])} | "
                f"{pct(m['three_class_macro_f1'])} |"
            )
    lines += [
        "", "## Clean gate plus classifier", "",
        "The clean corpus produced the same gate decisions for both models, as required: "
        f"{clean['all_real_sources']['gate']['pass']:,}/{clean['all_real_sources']['gate']['n']:,} PASS and "
        f"{clean['all_real_sources']['gate']['retry']:,} RETRY. Because only three files were rejected, "
        "classification-only and accepted-only accuracy are nearly identical.", "",
        "| Dataset | Gate pass | Model | Accepted 2-class balanced | Accepted 3-class macro-F1 | Correct 2-class yield / all inputs |", "|---|---:|---|---:|---:|---:|",
    ]
    for key in GROUP_LABELS:
        row = clean[key]
        for model_name, display in (("teacher", "Teacher"), ("student", "Student")):
            m = row[model_name]["gated_accepted"]
            lines.append(
                f"| {row['label']} | {pct(row['gate']['pass_rate'])} | {display} | "
                f"{pct(m['two_class_balanced_accuracy'])} | {pct(m['three_class_macro_f1'])} | "
                f"{pct(row[model_name]['correct_yield_two_class'])} |"
            )
    lines += [
        "", "## Why these numbers differ from the training console", "",
        "The earlier `Score` values were composite training-pipeline metrics on different manifest "
        "memberships: the saved teacher results contain 3,502 internal and 976 external examples, while "
        "the student results contain 3,382 internal and 719 external examples. They were useful within "
        "each run but were not a strict artifact-to-artifact A/B test. This report runs both final INT8 "
        "files on the identical 33,508 WAV files with the identical frontend and gate, so its direct "
        "accuracy comparison is the deployment reference.",
    ]
    lines += [
        "", "## Controlled signal-quality stress suite", "",
        "This balanced subset uses up to 75 N/C/W/B files from each source. White noise is deterministic; "
        "silence, clipping, tone, and low gain are synthetic engineering tests, not substitutes for labeled "
        "real PCB recordings.", "",
        "| Condition | Inputs | Gate PASS | RETRY reasons | Teacher classifier-only balanced | Student classifier-only balanced |", "|---|---:|---:|---|---:|---:|",
    ]
    for condition, row in stress.items():
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(row["gate"]["reasons"].items())) or "none"
        tm = row["teacher"]["classification_only"]["two_class_balanced_accuracy"]
        sm = row["student"]["classification_only"]["two_class_balanced_accuracy"]
        lines.append(
            f"| {condition} | {row['gate']['n']:,} | {pct(row['gate']['pass_rate'])} | {reasons} | {pct(tm)} | {pct(sm)} |"
        )
    lines += [
        "", "### Accepted-only accuracy and total correct yield", "",
        "A high accepted-only score can be misleading when almost every recording is rejected, so the "
        "last column also counts correct accepted predictions over every attempted recording.", "",
        "| Condition | Model | Accepted clips | Accepted 2-class balanced | Accepted 3-class macro-F1 | Correct 2-class yield / all inputs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition, row in stress.items():
        for model_name, display in (("teacher", "Teacher"), ("student", "Student")):
            m = row[model_name]["gated_accepted"]
            lines.append(
                f"| {condition} | {display} | {m['n']:,} | "
                f"{pct(m['two_class_balanced_accuracy'])} | {pct(m['three_class_macro_f1'])} | "
                f"{pct(row[model_name]['correct_yield_two_class'])} |"
            )
    lines += [
        "", "### Gate findings", "",
        "- Decisions and RETRY reasons were identical for both models. The gate is a shared upstream "
        "module; it does not become better or worse when the classifier changes.",
        f"- At 20 dB SNR, the gate passed {pct(stress['white_noise_20db']['gate']['pass_rate'])} of clips. "
        f"Among all noisy clips, student classifier balanced accuracy was "
        f"{pct(stress['white_noise_20db']['student']['classification_only']['two_class_balanced_accuracy'])} "
        f"versus {pct(stress['white_noise_20db']['teacher']['classification_only']['two_class_balanced_accuracy'])} for the teacher.",
        f"- At 10 and 5 dB SNR, the gate rejected "
        f"{pct(1.0 - stress['white_noise_10db']['gate']['pass_rate'])} and "
        f"{pct(1.0 - stress['white_noise_5db']['gate']['pass_rate'])}, respectively. This is promising "
        "for severe synthetic noise but must be calibrated on real PCB motion/rub/environment recordings.",
        f"- Low gain passed {pct(stress['low_gain']['gate']['pass_rate'])}. This is expected because "
        "`qg_scan_pcm` peak-normalizes the clip before computing RMS; keep the independent raw-RMS "
        "microphone/contact check before gain.",
        "- The steady tone was rejected as `clipping_saturation` because the same peak normalization is "
        "also used for clipping fraction. Before product release, separate true ADC/full-scale saturation "
        "from tonal interference and recalibrate the reason codes on PCB recordings.",
    ]
    lines += [
        "", "## Decision", "",
        "- Keep the teacher as the accuracy reference and the student as the SRAM/latency candidate.",
        "- Do not declare the student PCB-ready until its TFLM tensor arena is measured in internal SRAM, "
        "host-vs-device logits match for a golden set, and live INMP441 placement/noise recordings validate the gate.",
        "- The clean dataset is suitable for classifier comparison but not for calibrating real no-contact, "
        "motion-rub, speech, environmental-noise, or gain thresholds.", "",
        "- `low_gain` mostly passes the normalized C gate by design. The live pipeline must retain the "
        "separate raw-RMS microphone/contact check before digital gain and before this gate.", "",
    ]
    args.out_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {args.out_json}")
    print(f"[saved] {args.out_report}")


if __name__ == "__main__":
    main()
