#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${HANIL_MANIFEST:-data/manifest.csv}"
OUTPUT="${HANIL_OUTPUT:-runs/improved_specificity_s1337}"
SEED="${HANIL_SEED:-1337}"

cd "$(dirname "$0")/.."
"$PYTHON_BIN" phase1_baseline.py \
  --manifest "$MANIFEST" \
  --multilabel \
  --model dscnn_tcn \
  --width 2.0 \
  --imbalance-strategy natural_sqrt \
  --multilabel-pos-weight-power 0.5 \
  --multilabel-pos-weight-cap 4.0 \
  --sungbook-train-frac 0.30 \
  --threshold-external-frac 0.30 \
  --threshold-mode joint \
  --target-abnormal-sensitivity 0.80 \
  --threshold-grid-step 0.01 \
  --specaugment \
  --bandlimit-prob 0.50 \
  --event-overlay 0.20 \
  --mixup-alpha 0.20 \
  --mixstyle \
  --epochs 80 \
  --min-epochs 30 \
  --patience 15 \
  --batch_size 16 \
  --amp \
  --workers 4 \
  --seed "$SEED" \
  --out "$OUTPUT"
