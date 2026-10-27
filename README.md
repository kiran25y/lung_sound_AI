# Lung-Sound AI Pipeline

End-to-end pipeline for **Normal / Crackle / Wheeze** classification from digital-stethoscope
and smartphone recordings, robust to noise and device shift, and deployable to **ESP32-S3**.

> 📖 **Full process walkthrough: [`PIPELINE.md`](PIPELINE.md)** — detailed, stage-by-stage
> documentation of the whole pipeline (philosophy, every phase, runbook, config reference,
> targets, licensing, troubleshooting). This README is the quick version.

Trains in **PyTorch**; ships an INT8
**TFLite** student model to the ESP32-S3 via ONNX -> TensorFlow -> TFLite.

## Complete training and firmware repository

This repository now contains both sides of the product pipeline:

| Area | Location | Contents |
|---|---|---|
| Model training | repository root and `src/hanil/` | dataset preparation, robust training, evaluation, calibration, distillation, and quantization |
| Final ESP32-S3 firmware | `firmware/esp32s3/` | INMP441 capture, placement validation, signal/noise gate, log-Mel frontend, INT8 inference, and 2-class/3-class decisions |
| Reusable firmware component | `firmware/esp32s3/components/hanil_ai/` | source component with a public C header for every pipeline module |
| Prebuilt library package | `firmware/esp32s3/release/hanil_ai_esp32s3.zip` | ESP-IDF component containing `libhanil_ai.a`, headers, checksums, and example code |
| Final deployment release | `firmware/esp32s3/release/final_ai/` | checkpoint, INT8 TFLite model, flashable binaries, evaluation results, and engineering report |

The reusable library exposes separate APIs for microphone capture, signal/noise
quality gating, log-Mel preprocessing, INT8 classification, memory telemetry,
and the complete end-to-end pipeline. See
[`firmware/esp32s3/README.md`](firmware/esp32s3/README.md) for build, flash,
serial WAV testing, live microphone testing, memory placement, and measured
ESP32-S3 performance.

The final model outputs `normal/abnormal` and
`normal/crackle/wheeze`. It is an engineering research model and must not be
used as a clinical diagnosis without the required validation and regulatory
process.


## Phase map (matches the project spec)

| Phase | Module | Code |
|---|---|---|
| **1** AI Classification | 2.1 Dataset strategy | `phase1_classification/datasets.py` |
| | 2.2 Noise-robust training (Exp 1) — MUSAN | `augment/noise_mixer.py`, `train.py` |
| | 2.3 Device-robust training (Exp 2) | `mixstyle.py`, `contrastive.py` |
| | 2.4 Final combined model (Exp 3) | `train.py` (`--experiment final`) |
| | 2.5 Model architectures | `models/teacher_ast.py`, `models/student_cnn.py`, `distill.py` |
| **2** Quality Check | 3.1 Noise check | `phase2_quality/noise_check.py` |
| | 3.2 Real-time SNR check | `phase2_quality/snr_check.py` |
| | 3.3 Alert logic | `phase2_quality/alert.py` |
| | Noise detection (DESED_synthetic) | `phase2_quality/noise_detector.py` |
| **3** Sound Quality Analysis | 4.1 Benchmarking + agreement | `phase3_benchmark/benchmark.py` |
| **4** ESP32 Deployment | 5.2 ESP32-S3 target / 5.6 PCB | `phase4_deploy/esp32/README.md` |
| | 5.4 Deployment strategy | `phase4_deploy/README.md` |
| | 5.5 Quantization pipeline | `phase4_deploy/quantize_tflite.py` |

## Dataset roles (locked)

| Dataset | Role | License note |
|---|---|---|
| Seoul (internal, stethoscope) | Train + internal test | own data |
| Sungbook (external, stethoscope) | **External test only** (headline) | own data |
| Smartphone (both hospitals) | Device-robustness test | own data |
| **MUSAN (noise category only)** | Phase 1 noise-robust augmentation | CC BY 4.0 — commercial OK w/ attribution |
| **DESED_synthetic** | Phase 2 noise-detection training | Scaper=MIT; audio content mixed — verify before shipping |

## Step 1 — baseline 3-class classifier (start here)

Before any robustness work, get a clean baseline. A **single self-contained file**,
`phase1_baseline.py`, trains the Normal/Crackle/Wheeze classifier and prints per-class
sensitivity + a confusion matrix. It only needs `torch, torchaudio, numpy, pandas`.

```bash
# smoke-test with zero real data (auto-generates a tiny synthetic set)
python phase1_baseline.py --synthetic --epochs 5

# real data
python phase1_baseline.py --manifest data/manifest.csv --epochs 60 --out runs/baseline
```
Look at the confusion matrix on `external_test` (Sungbook) first. Once that's sane,
add robustness via `configs/exp1_noise.yaml` → `exp2_device.yaml` → `exp3_final.yaml`
(the packaged trainer below). The equivalent packaged baseline is `configs/baseline.yaml`.

### False-positive reduction run

The ESP32 deployment uses the two-head `phase1_baseline.py` path. Its improved
training policy keeps the natural normal/abnormal distribution and applies a
capped square-root positive weight instead of combining 50:50 normal
undersampling with full inverse-frequency weighting. It also tunes crackle and
wheeze thresholds jointly for the product's `crackle OR wheeze` abnormal rule.

Thirty percent of the remaining Sungbook patients are reserved for threshold
calibration and removed from the final external test. The launch scripts keep
all patient groups disjoint:

```powershell
scripts\run_improved_specificity.ps1 -Manifest data\manifest_local.csv
```

```bash
bash scripts/run_improved_specificity.sh
```

The default operating-point target retains at least 80% abnormal sensitivity
on calibration patients while maximising normal specificity. Change that target
only after agreeing on the clinical cost of a missed abnormal versus a repeat
recording or false alarm.

### Crackle/wheeze subtype fine-tuning and INT8 export

Fine-tune the frozen screening checkpoint with an auxiliary objective on pure
crackle and pure wheeze clips. The checkpoint objective combines two-class
balanced accuracy with the exact deployed three-class macro-F1. This auxiliary
loss is training-only, so it adds no ESP32-S3 parameters, operators, latency,
or memory:

```powershell
scripts\run_improved_subtype.ps1 -Manifest data\manifest_local.csv
```

Quantize only a candidate that beats the frozen baseline. The parity stage
uses the candidate's frozen thresholds, reports both product metrics, and fails
if INT8 degradation exceeds the configured limits:

```powershell
python phase4_quantize.py --stage all `
  --manifest data\manifest_local.csv `
  --ckpt runs\improved_subtype_s1337\best.pt `
  --thresholds runs\improved_subtype_s1337\results.json `
  --model dscnn_tcn --width 2.0 --n-calib 300 `
  --out runs\improved_subtype_s1337\quant
```

Temperature scaling is diagnostic-only and is never applied to exported logits
or firmware probabilities.

### Internal-SRAM candidate

`sram_dscnn` moves capacity into late, low-resolution layers and uses VALID
padding so conversion cannot materialise the current model's 442 KiB padded
activation. At width 1.0 its largest static INT8 layer output is 74,400 bytes;
the training command fails before epoch 1 if it exceeds the 100 KiB design
limit. This is only the architecture guard: the exported model must still pass
an on-device TFLM arena limit before promotion.

The remote launcher uses the current width-2 checkpoint as a frozen teacher,
includes patient-disjoint smartphone adaptation when phone rows exist, and
uses the configured noise tree for training plus clean/noisy evaluation:

```bash
HANIL_MANIFEST=/workspace/dataset/lungsound/manifest.csv \
HANIL_TEACHER=runs/improved_specificity_s1337/best.pt \
HANIL_NOISE_ROOT=/workspace/dataset/noise \
bash scripts/run_sram_dscnn.sh
```

Export without replacing the deployed model:

```bash
python phase4_quantize.py --stage all \
  --manifest "$HANIL_MANIFEST" \
  --ckpt runs/sram_dscnn_s1337/best.pt \
  --thresholds runs/sram_dscnn_s1337/results.json \
  --model sram_dscnn --width 1.0 --n-calib 300 \
  --out runs/sram_dscnn_s1337/quant
```

Promotion requires all of: TFLM arena at most 180 KiB, at least 48 KiB free
internal heap after initialization, host/board output parity, and an agreed
non-inferiority margin against the current held-out stethoscope metrics.

## Quickstart (full packaged pipeline)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 0. Prepare a metadata manifest (see data/README.md)
# 1. Train the cloud teacher (AST)
bash scripts/train_teacher.sh
# 2. Experiment 1 — noise-robust student
bash scripts/run_exp1_noise.sh
# 3. Experiment 2 — device-robust student
bash scripts/run_exp2_device.sh
# 4. Experiment 3 — final combined student (distilled)
bash scripts/run_exp3_final.sh
# 5. Export INT8 TFLite for ESP32-S3
bash scripts/export_esp32.sh
```

## Design rules baked into the code
- **Patient-level splits only** — no patient in both train and test (`datasets.py`).
- **Own-data metrics** — per-class sensitivity/specificity + confusion matrix (`common/metrics.py`).
- **One portable front end** — 64-bin log-Mel, 16 kHz, 25/10 ms (`common/features.py`); the exact
  same math is re-implemented in fixed point on the ESP32 (see esp32 README).
- **Teacher stays in the cloud; only the INT8 student ships.**

See `LICENSING.md` for the two-tier noise-license policy.
