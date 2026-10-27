#!/usr/bin/env bash
set -euo pipefail

# Remote-GPU launcher for the zero-PSRAM classifier candidate.  Override paths
# with environment variables instead of editing this file:
#   HANIL_MANIFEST=/path/manifest.csv \
#   HANIL_TEACHER=/path/improved_specificity/best.pt \
#   HANIL_NOISE_ROOT=/path/noise \
#   HANIL_OUTPUT=runs/sram_dscnn_s1337 \
#   bash scripts/run_sram_dscnn.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${HANIL_MANIFEST:-data/manifest.csv}"
TEACHER="${HANIL_TEACHER:-runs/improved_specificity_s1337/best.pt}"
NOISE_ROOT="${HANIL_NOISE_ROOT:-/workspace/dataset/noise}"
OUTPUT="${HANIL_OUTPUT:-runs/sram_dscnn_s1337}"
SEED="${HANIL_SEED:-1337}"
WORKERS="${HANIL_WORKERS:-8}"
BATCH_SIZE="${HANIL_BATCH_SIZE:-32}"
SUNGBOOK_TRAIN_FRAC="${HANIL_SUNGBOOK_TRAIN_FRAC:-0.30}"
SMARTPHONE_TRAIN_FRAC="${HANIL_SMARTPHONE_TRAIN_FRAC:-0.30}"

cd "$(dirname "$0")/.."

args=(
  phase1_baseline.py
  --manifest "$MANIFEST"
  --multilabel
  --model sram_dscnn
  --width 1.0
  --max-activation-kb 100
  --teacher-checkpoint "$TEACHER"
  --teacher-model dscnn_tcn
  --teacher-width 2.0
  --distill-weight 0.35
  --distill-temperature 2.0
  --selection-metric product_composite
  --imbalance-strategy natural_sqrt
  --multilabel-pos-weight-power 0.5
  --multilabel-pos-weight-cap 4.0
  --sungbook-train-frac "$SUNGBOOK_TRAIN_FRAC"
  --smartphone-train-frac "$SMARTPHONE_TRAIN_FRAC"
  --threshold-external-frac 0.30
  --threshold-mode joint
  --target-abnormal-sensitivity 0.80
  --threshold-grid-step 0.01
  --specaugment
  --bandlimit-prob 0.50
  --event-overlay 0.20
  --mixup-alpha 0.20
  --mixstyle
  --epochs 80
  --min-epochs 30
  --patience 15
  --batch_size "$BATCH_SIZE"
  --amp
  --workers "$WORKERS"
  --seed "$SEED"
  --out "$OUTPUT"
)

if [[ -d "$NOISE_ROOT" ]]; then
  args+=(--noise-dir "$NOISE_ROOT" --noise-prob 0.50 --snr-db 5 20)
  args+=(--eval-noise-dir "$NOISE_ROOT" --eval-snr 20 10 5)
else
  printf 'WARNING: noise directory not found: %s; continuing without noise mixing\n' "$NOISE_ROOT" >&2
fi

"$PYTHON_BIN" "${args[@]}"
