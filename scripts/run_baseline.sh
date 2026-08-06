#!/usr/bin/env bash
# STEP 1 - baseline 3-class classifier. Either run the standalone script:
set -e
python phase1_baseline.py --manifest data/manifest.csv --epochs 60 --out runs/baseline
# ...or, once installed (pip install -e .), the packaged trainer with the same effect:
# python -m hanil.phase1_classification.train --config configs/baseline.yaml
