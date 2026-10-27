# Remote SRAM-model training

This update adds an internal-SRAM-oriented `sram_dscnn` candidate without
replacing the current checkpoint. Its width-1 model has about 175K parameters
and a largest PyTorch INT8-equivalent layer output of 74,400 bytes. Actual
TFLite Micro arena usage must be measured after quantization.

## 1. Extract and verify

Extract the ZIP at the remote `hanil_lung_ai` project root, the directory that
contains `phase1_baseline.py`. Back up remote modifications before allowing the
archive to overwrite files.

```bash
cd /path/to/hanil_lung_ai
unzip -o /path/to/sram_training_update_20260907.zip
sha256sum -c SHA256SUMS.txt
python3 -m py_compile phase1_baseline.py phase1_eval_report.py phase4_quantize.py
python3 -m pytest -q tests/test_training_policy.py -k sram
```

Use the existing CUDA environment if it already has the pipeline dependencies.
Otherwise create a virtual environment and install `requirements.txt`. The
export stage additionally needs the same working `onnx2tf` and TensorFlow
versions used for the previous model.

## 2. Set the actual remote paths

```bash
export HANIL_MANIFEST=/workspace/dataset/lungsound/manifest.csv
export HANIL_NOISE_ROOT=/workspace/dataset/noise
export HANIL_TEACHER="$PWD/runs/improved_specificity_s1337/best.pt"
export HANIL_OUTPUT="$PWD/runs/sram_dscnn_s1337"
export HANIL_SEED=1337
export HANIL_WORKERS=8
export HANIL_BATCH_SIZE=32
```

The archive includes the known teacher checkpoint under the default path. Set
`HANIL_TEACHER` differently only when the remote copy is known to be the same
stronger checkpoint.

The default launcher uses 30% patient-disjoint Sungbook and smartphone domain
adaptation. The remaining patients stay in their respective test groups. Set
either value to zero for a strict untouched-domain experiment:

```bash
export HANIL_SUNGBOOK_TRAIN_FRAC=0.30
export HANIL_SMARTPHONE_TRAIN_FRAC=0.30
```

## 3. Check the manifest and GPU

```bash
python3 check_split_config.py \
  --results runs/improved_specificity_s1337/results.json \
  --manifest "$HANIL_MANIFEST"
nvidia-smi
```

Confirm that the printed split counts include the intended stethoscope and
smartphone sources. Do not start a long run if file checking removes a large
part of the manifest.

## 4. Start training

For an interactive run with a persistent log:

```bash
mkdir -p "$HANIL_OUTPUT"
bash scripts/run_sram_dscnn.sh 2>&1 | tee "$HANIL_OUTPUT/train.log"
```

For a disconnect-safe run, open `tmux` first:

```bash
tmux new -s hani_sram
bash scripts/run_sram_dscnn.sh 2>&1 | tee "$HANIL_OUTPUT/train.log"
```

Detach from tmux with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t hani_sram
```

The launch command enforces a 100 KiB static layer-output limit and uses the
current width-2 model as a frozen teacher with distillation weight 0.35. It
writes only under `runs/sram_dscnn_s1337`.

## 5. Evaluate the frozen checkpoint

Training writes `best.pt` and `results.json`. Run the auditable evaluation with
the same domain fractions:

```bash
python3 phase1_eval_report.py \
  --manifest "$HANIL_MANIFEST" \
  --ckpt "$HANIL_OUTPUT/best.pt" \
  --model sram_dscnn --width 1.0 \
  --thresholds "$HANIL_OUTPUT/results.json" \
  --sungbook-train-frac "$HANIL_SUNGBOOK_TRAIN_FRAC" \
  --smartphone-train-frac "$HANIL_SMARTPHONE_TRAIN_FRAC" \
  --eval-noise-dir "$HANIL_NOISE_ROOT" \
  --gate-noisy \
  --out "$HANIL_OUTPUT/eval_report.json"
```

Do not promote based on training accuracy. Compare patient-disjoint
stethoscope and smartphone results, sensitivity, specificity and three-class
macro-F1 with the supplied baseline comparison.

## 6. Export INT8

```bash
python3 phase4_quantize.py --stage all \
  --manifest "$HANIL_MANIFEST" \
  --ckpt "$HANIL_OUTPUT/best.pt" \
  --thresholds "$HANIL_OUTPUT/results.json" \
  --model sram_dscnn --width 1.0 \
  --noise-dir "$HANIL_NOISE_ROOT" \
  --n-calib 300 --refresh-calibration \
  --out "$HANIL_OUTPUT/quant"
```

The export must pass FP32-to-INT8 parity. Do not overwrite the PCB firmware
model yet.

## 7. Return these artifacts for PCB integration

Copy back:

```text
runs/sram_dscnn_s1337/best.pt
runs/sram_dscnn_s1337/results.json
runs/sram_dscnn_s1337/eval_report.json
runs/sram_dscnn_s1337/quant/model_int8.tflite
runs/sram_dscnn_s1337/quant/parity.json
runs/sram_dscnn_s1337/train.log
```

The firmware promotion gates are: TFLM arena at most 180 KiB, at least 48 KiB
free internal heap after initialization, no PSRAM allocation, exact host/board
output parity, and acceptable held-out classification performance. The current
deployed model remains the fallback until every gate passes.
