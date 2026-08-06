# Hanil Lung-Sound AI Pipeline

End-to-end pipeline for **Normal / Crackle / Wheeze** classification from digital-stethoscope
and smartphone recordings, robust to noise and device shift, and deployable to **ESP32-S3**.

> 📖 **Full process walkthrough: [`PIPELINE.md`](PIPELINE.md)** — detailed, stage-by-stage
> documentation of the whole pipeline (philosophy, every phase, runbook, config reference,
> targets, licensing, troubleshooting). This README is the quick version.

Built for the Medisensing / BrAInLab research program. Trains in **PyTorch**; ships an INT8
**TFLite** student model to the ESP32-S3 via ONNX -> TensorFlow -> TFLite.

```
Seoul (internal steth)  ─┐
Sungbook (external steth)─┼─► Phase 1  ─► Phase 2 ─► Phase 3 ─► Phase 4
Smartphone (both)        ─┘   classify     quality    bench      ESP32
MUSAN noise (aug)             + robust     control     mark       INT8
DESED_synthetic (detect)
```

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
