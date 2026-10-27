#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-${HANIL_DATASET_ROOT:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-4}"
STRESS_PER_SOURCE_CLASS="${STRESS_PER_SOURCE_CLASS:-75}"

if [[ -z "$DATASET" ]]; then
  echo "usage: $0 DATASET_ROOT" >&2
  echo "or set HANIL_DATASET_ROOT" >&2
  exit 2
fi

TEACHER="$ROOT/pipeline_source/hanil_lung_ai/runs/improved_specificity_s1337/quant/model_int8.tflite"
STUDENT="$ROOT/artifacts/sram_dscnn_int8_artifacts/runs/sram_dscnn_s1337/quant/model_int8.tflite"
EVALUATOR="$ROOT/tools/evaluate_lungsound_dataset.py"
OUT="$ROOT/comparison_evaluation"

for required in "$DATASET" "$TEACHER" "$STUDENT" "$EVALUATOR"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

run_clean() {
  local model="$1" crackle="$2" wheeze="$3" destination="$4"
  "$PYTHON_BIN" -u "$EVALUATOR" \
    --dataset "$DATASET" --model "$model" \
    --crackle-threshold "$crackle" --wheeze-threshold "$wheeze" \
    --classify-retries --condition clean --workers "$WORKERS" \
    --progress-every 1000 --out-dir "$OUT/$destination"
}

run_stress() {
  local model="$1" crackle="$2" wheeze="$3" destination="$4"
  "$PYTHON_BIN" -u "$EVALUATOR" \
    --dataset "$DATASET" --model "$model" \
    --crackle-threshold "$crackle" --wheeze-threshold "$wheeze" \
    --classify-retries --limit-per-source-class "$STRESS_PER_SOURCE_CLASS" \
    --conditions clean white_noise_20db white_noise_10db white_noise_5db \
      silence hard_clip steady_tone low_gain \
    --workers "$WORKERS" --progress-every 1000 --out-dir "$OUT/$destination"
}

run_clean "$TEACHER" 0.49 0.52 old_teacher_clean
run_clean "$STUDENT" 0.49 0.28 new_student_clean
run_stress "$TEACHER" 0.49 0.52 old_teacher_stress
run_stress "$STUDENT" 0.49 0.28 new_student_stress

"$PYTHON_BIN" "$ROOT/tools/analyze_tflite_memory.py" \
  "$TEACHER" "$STUDENT" --out "$OUT/tflite_memory_analysis.json"
"$PYTHON_BIN" "$ROOT/tools/benchmark_tflite_host.py" \
  "$TEACHER" "$STUDENT" --warmup 50 --iterations 1000 \
  --out "$OUT/host_latency_benchmark.json"
"$PYTHON_BIN" "$ROOT/tools/build_model_comparison.py"
"$PYTHON_BIN" "$ROOT/tools/validate_comparison_artifacts.py"

echo "comparison complete: $OUT/COMPARISON_REPORT.md"
