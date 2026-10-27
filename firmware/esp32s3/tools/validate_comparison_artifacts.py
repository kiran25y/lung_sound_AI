#!/usr/bin/env python3
"""Fail closed if the teacher/student comparison evidence is incomplete."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLEAN_CLIPS = 33_508
EXPECTED_STRESS_PER_MODEL = 9_088
EXPECTED_STRESS_CONDITIONS = {
    "clean", "white_noise_20db", "white_noise_10db", "white_noise_5db",
    "silence", "hard_clip", "steady_tone", "low_gain",
}


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    teacher_clean = load(ROOT / "comparison_evaluation/old_teacher_clean/summary.json")
    student_clean = load(ROOT / "comparison_evaluation/new_student_clean/summary.json")
    teacher_stress = load(ROOT / "comparison_evaluation/old_teacher_stress/summary.json")
    student_stress = load(ROOT / "comparison_evaluation/new_student_stress/summary.json")
    comparison = load(ROOT / "comparison_evaluation/comparison.json")
    memory = load(ROOT / "comparison_evaluation/tflite_memory_analysis.json")
    latency = load(ROOT / "comparison_evaluation/host_latency_benchmark.json")

    teacher_model = ROOT / "pipeline_source/hanil_lung_ai/runs/improved_specificity_s1337/quant/model_int8.tflite"
    student_model = ROOT / "artifacts/sram_dscnn_int8_artifacts/runs/sram_dscnn_s1337/quant/model_int8.tflite"
    teacher_sha, student_sha = file_sha256(teacher_model), file_sha256(student_model)
    assert teacher_sha == teacher_clean["model_sha256"] == comparison["models"]["teacher"]["sha256"]
    assert student_sha == student_clean["model_sha256"] == comparison["models"]["student"]["sha256"]
    assert load(ROOT / "pipeline_source/hanil_lung_ai/runs/improved_specificity_s1337/quant/parity.json")["passed"] is True
    assert load(ROOT / "artifacts/sram_dscnn_int8_artifacts/runs/sram_dscnn_s1337/quant/parity.json")["passed"] is True

    for name, summary in (("teacher", teacher_clean), ("student", student_clean)):
        group = summary["groups"]["all_real_sources"]
        assert group["n_evaluated"] == EXPECTED_CLEAN_CLIPS, name
        assert group["n_errors"] == 0, name
        assert group["n_classified"] == EXPECTED_CLEAN_CLIPS, name
        assert group["classification_only_metrics"]["two_class_normal_vs_abnormal"]["n"] == EXPECTED_CLEAN_CLIPS
    teacher_gate = teacher_clean["groups"]["all_real_sources"]
    student_gate = student_clean["groups"]["all_real_sources"]
    assert teacher_gate["n_pass"] == student_gate["n_pass"] == 33_505
    assert teacher_gate["n_retry"] == student_gate["n_retry"] == 3
    assert teacher_gate["retry_reasons"] == student_gate["retry_reasons"]

    for name, summary in (("teacher", teacher_stress), ("student", student_stress)):
        group = summary["groups"]["all_real_sources"]
        assert group["n_evaluated"] == EXPECTED_STRESS_PER_MODEL, name
        assert group["n_errors"] == 0, name
        assert group["n_classified"] == EXPECTED_STRESS_PER_MODEL, name
        conditions = {
            key.removeprefix("condition:") for key in summary["groups"]
            if key.startswith("condition:") and "/source:" not in key
        }
        assert conditions == EXPECTED_STRESS_CONDITIONS, (name, conditions)
    for condition in EXPECTED_STRESS_CONDITIONS:
        key = f"condition:{condition}"
        tg, sg = teacher_stress["groups"][key], student_stress["groups"][key]
        assert tg["n_evaluated"] == sg["n_evaluated"] == 1_136, condition
        assert tg["n_pass"] == sg["n_pass"], condition
        assert tg["retry_reasons"] == sg["retry_reasons"], condition

    assert count_csv_rows(ROOT / "comparison_evaluation/old_teacher_clean/per_clip_results.csv") == EXPECTED_CLEAN_CLIPS
    assert count_csv_rows(ROOT / "comparison_evaluation/new_student_clean/per_clip_results.csv") == EXPECTED_CLEAN_CLIPS
    assert count_csv_rows(ROOT / "comparison_evaluation/old_teacher_stress/per_clip_results.csv") == EXPECTED_STRESS_PER_MODEL
    assert count_csv_rows(ROOT / "comparison_evaluation/new_student_stress/per_clip_results.csv") == EXPECTED_STRESS_PER_MODEL

    memory_by_sha = {value["sha256"]: value for value in memory.values()}
    assert memory_by_sha[teacher_sha]["activation_liveness_lower_bound_bytes"] > 512 * 1024
    assert memory_by_sha[student_sha]["activation_liveness_lower_bound_bytes"] < 512 * 1024
    latency_by_sha = {value["sha256"]: value for value in latency.values()}
    assert latency_by_sha[teacher_sha]["invoke_ms"]["n"] == 1_000
    assert latency_by_sha[student_sha]["invoke_ms"]["n"] == 1_000
    assert latency_by_sha[student_sha]["invoke_ms"]["median"] < latency_by_sha[teacher_sha]["invoke_ms"]["median"]

    firmware = (ROOT / "main/main.cc").read_text(encoding="utf-8")
    assert "psram_alloc(kArenaBytes)" in firmware
    assert "ws.pcm = static_cast<int16_t *>(psram_alloc" in firmware
    assert "ws.logmel = static_cast<float *>(psram_alloc" in firmware
    assert file_sha256(ROOT / "main/model/model_int8.tflite") == teacher_sha
    assert (ROOT / "comparison_evaluation/COMPARISON_REPORT.md").is_file()
    assert (ROOT / "docs/ESP32_S3_MEMORY_PLACEMENT_STUDY.md").is_file()
    assert (ROOT / "scripts/run_end_to_end_comparison.sh").is_file()
    assert (ROOT / "scripts/run_end_to_end_comparison.ps1").is_file()

    audit = {
        "status": "pass",
        "clean_clips_per_model": EXPECTED_CLEAN_CLIPS,
        "stress_inputs_per_model": EXPECTED_STRESS_PER_MODEL,
        "gate_results_identical": True,
        "quantization_parity_passed": True,
        "latency_iterations_per_model": 1_000,
        "teacher_sha256": teacher_sha,
        "student_sha256": student_sha,
        "firmware_memory_layout_changed": False,
        "student_hardware_arena_measurement": "pending_user_approval",
    }
    output = ROOT / "comparison_evaluation/validation.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
