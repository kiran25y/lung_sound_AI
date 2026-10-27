#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${HANIL_MANIFEST:-data/manifest.csv}"
INITIAL_CHECKPOINT="${HANIL_INITIAL_CHECKPOINT:-runs/improved_specificity_s1337/best.pt}"
OUTPUT="${HANIL_OUTPUT:-runs/improved_subtype_s1337}"
SEED="${HANIL_SEED:-1337}"

cd "$(dirname "$0")/.."
"$PYTHON_BIN" phase1_baseline.py \
  --manifest "$MANIFEST" \
  --multilabel \
  --model dscnn_tcn \
  --width 2.0 \
  --init-checkpoint "$INITIAL_CHECKPOINT" \
  --subtype-loss-weight 0.25 \
  --selection-metric product_composite \
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
  --lr 0.0001 \
  --epochs 40 \
  --warmup-epochs 1 \
  --min-epochs 10 \
  --patience 10 \
  --batch_size 16 \
  --amp \
  --workers 4 \
  --seed "$SEED" \
  --out "$OUTPUT"
