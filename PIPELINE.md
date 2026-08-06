# Hanil Lung-Sound AI — Full Pipeline Documentation

> End-to-end process for building a **Normal / Crackle / Wheeze** classifier that is
> **noise-robust** and **device-robust**, validated on **your own hospital data only**,
> and deployed as an **INT8 model on the ESP32-S3**, aligned to a StethoMe-class product.

This document explains *what happens at every stage, why, and how to run it*. It is the
narrative companion to the code. For the one-screen version see `README.md`.

---

## 0. Philosophy — the five rules the whole pipeline obeys

1. **Own data decides.** Training and evaluation use Seoul (internal) + Sungbook (external)
   + smartphone recordings. No ICBHI data is used for training or as a benchmark. We still
   *initialise* from AudioSet-pretrained weights — that is generic pretraining, not clinical data.
2. **No patient leaks across a split.** Splits are made by `patient_id`. A random clip-level
   split inflates accuracy by 10–20 points and is the #1 reason medical audio models fail
   external validation.
3. **The external hospital is the headline.** Train on Seoul, report on Sungbook. That number,
   plus the smartphone number, is what proves the product generalises — not the peak in-house score.
4. **Catch the abnormal, not just the common.** "Normal" dominates, so we optimise and report
   per-class sensitivity/specificity, never accuracy alone. Focal loss + class balancing are on
   by default.
5. **The teacher stays in the cloud; only the small INT8 student ships.** Deployment is ESP32-S3
   only, so the shipped model is 0.6–1.2M params, distilled from a large AST teacher.

---

## 1. Data flow at a glance

```
                         ┌─────────────────────── Phase 1: CLASSIFICATION ───────────────────────┐
 Seoul  (steth)  ─┐      │  log-Mel front end → [MixStyle] → student CNN → logits                 │
 Sungbook(steth) ─┼─►  manifest.csv ─► patient-level splits ─►  train.py  ─────────────► best.pt  │
 Smartphone      ─┘      │      ▲                         ▲            │                            │
                         │  MUSAN noise (SNR 5–20 dB)   AST teacher   └─► contrastive + distill    │
                         └───────────────────────────────────────────────────────────────────────┘
                                                     │ best.pt
                    ┌──────────── Phase 2: QUALITY ───────────┐        ┌──── Phase 4: DEPLOY (ESP32) ────┐
   live audio ─►    │ noise_check → snr_check → alert (gate)  │        │ ONNX → TF → INT8 TFLite → verify │
                    │ DESED-trained noise_detector (2nd opinion)│      │ xxd -i → ESP-IDF + ESP-NN        │
                    └─────────────────────────────────────────┘        └─────────────────────────────────┘
                                                     │
                              ┌──────── Phase 3: BENCHMARK ────────┐
                              │ 4-device compare · κ · Bland–Altman │
                              └────────────────────────────────────┘
```

The **front-end definition is shared** by every stage (`common/features.py`) and must be
reproduced bit-for-bit on the ESP32 — this is the single most important consistency rule.

---

## 2. Repository map

| Path | Phase / role |
|---|---|
| `common/features.py` | The one portable log-Mel front end (training + device must match) |
| `common/metrics.py` | Per-class Se/Sp, macro Score, confusion matrix |
| `common/config.py` | YAML loader with `_base_` inheritance |
| `phase1_classification/datasets.py` | 2.1 patient-level splits |
| `phase1_classification/augment/noise_mixer.py` | 2.2 MUSAN SNR mixing + licence guard |
| `phase1_classification/augment/spec_augment.py` | 2.2 SpecAugment |
| `phase1_classification/mixstyle.py` | 2.3 device-robust (feature-stat mixing) |
| `phase1_classification/contrastive.py` | 2.3 stethoscope-guided supervised contrastive |
| `phase1_classification/distill.py` · `losses.py` | 2.5 distillation + focal/balanced loss |
| `phase1_classification/models/` | 2.5 AST teacher, depthwise student, factory |
| `phase1_classification/train.py` · `evaluate.py` | 2.4 trainer (Exp 1/2/3) + evaluator |
| `phase2_quality/noise_check.py` | 3.1 noise-check DSP |
| `phase2_quality/snr_check.py` | 3.2 real-time SNR |
| `phase2_quality/alert.py` | 3.3 alert state machine |
| `phase2_quality/noise_detector.py` · `train_noise_detector.py` | DESED noise detector |
| `phase3_benchmark/benchmark.py` | 4.1 device comparison + agreement stats |
| `phase4_deploy/*.py` | 5.5 export + INT8 quantization + parity |
| `phase4_deploy/esp32/README.md` | 5.2 / 5.6 firmware + PCB |
| `tools/make_synthetic_data.py` | synthetic dataset for smoke tests |

---

## 3. Phase 0 — Setup and the data manifest

### 3.1 Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3.2 Build `data/manifest.csv`
One row per clip. The loader derives every split from this file.

| column | meaning |
|---|---|
| `filepath` | path to a wav |
| `patient_id` | unique per patient — **drives the split** |
| `label` | `normal` / `crackle` / `wheeze` |
| `hospital` | `seoul` / `sungbook` |
| `device` | `stethoscope` / `smartphone` |

Splits produced automatically (`datasets.build_splits`):
`train`, `internal_val`, `internal_test` (Seoul steth, patient-disjoint) ·
`external_test` (Sungbook steth, never trained on) · `smartphone_test`.

### 3.3 Drop in the noise
```
data/noise/musan_noise/     # MUSAN, NOISE category only (delete music/ and speech/)
data/noise/self_recorded/   # 30–60 min of your own ward/clinic/home ambience (best match)
data/noise/desed_synthetic/ # DESED soundscapes + .tsv (Phase-2 detector only)
```

### 3.4 No data yet? Smoke-test on synthetic
```bash
python tools/make_synthetic_data.py --root data_synth
# then set paths.manifest=data_synth/manifest.csv, paths.noise_dir=data_synth/noise/musan_noise
```

---

## 4. Phase 1 — AI Classification Model

### 4.1 The shared front end (`common/features.py`)
Every clip becomes a **log-Mel spectrogram**: 16 kHz, `n_fft=400` (25 ms), `hop=160` (10 ms),
`n_mels=64`, band 50–2000 Hz, then `log(mel+1e-6)` and per-clip mean/variance standardisation.
Chosen because it is portable (identical in PyTorch, CMSIS-DSP, ESP-DSP) and keeps the spectral
detail crackles/wheezes live in — unlike MFCC, whose DCT step discards it.

### 4.2 The three experiments (all driven by config, one trainer)
`train.py` reads the config and turns capabilities on/off. This is how the same file runs all
three experiments plus the teacher.

**Experiment 1 — Noise-robust (`configs/exp1_noise.yaml`).**
Train on clean stethoscope audio; each clip has MUSAN noise mixed in at an SNR sampled uniformly
from 5–20 dB, re-randomised every epoch (`noise_mixer.py`). SpecAugment adds time/frequency
masking. Tested on clean vs injected-noise to measure the degradation (target: <5% drop at 10 dB).

**Experiment 2 — Device-robust (`configs/exp2_device.yaml`).**
Train on stethoscope, test on smartphone from the same patients. Two mechanisms:
- **MixStyle** (`mixstyle.py`) mixes channel mean/std between samples so the backbone ignores
  device "style".
- **Stethoscope-guided supervised contrastive loss** (`contrastive.py`) pulls same-class
  embeddings together across devices, using the clean stethoscope domain as the anchor.
Target: >90% of stethoscope accuracy retained on unseen smartphone audio.

**Experiment 3 — Final combined (`configs/exp3_final.yaml`).**
Everything on: MixStyle + contrastive + noise augmentation + **knowledge distillation** from the
AST teacher. Tested on the hardest case — smartphone + injected noise. This is the model you ship.

### 4.3 Models (`models/`)
- **Teacher — `teacher_ast.py`.** AudioSet-pretrained Audio Spectrogram Transformer (~87M). Trained
  only to teach; never deployed. Falls back to a tiny transformer if `transformers` can't download.
- **Student — `student_cnn.py`.** Depthwise-separable CNN on log-Mel. `width=1.0` ≈ 1.2M params,
  `width=0.75` ≈ 0.6M. ReLU6, no exotic ops → maps cleanly to ESP-NN / TFLite Micro INT8 kernels.
  Returns `(logits, embedding)`; the embedding feeds the contrastive loss.

### 4.4 Handling class imbalance (`losses.py`)
`normal` dominates, so plain cross-entropy learns to predict `normal`. Default is **focal loss**
with **auto inverse-frequency class weights**; class-aware sampling is available. We report and
optimise the macro Score, so the model is pushed to actually catch crackles and wheezes.

### 4.5 Distillation (`distill.py`)
`loss = α · KL(teacher‖student, T) · T² + (1−α) · focal(student, labels)`, defaults T=4, α=0.7.
Buys the small student 2–4 points it can't reach trained alone, at zero inference cost.

### 4.6 Training loop (`train.py`)
Per step: waveform → (noise mix in collate) → log-Mel → SpecAugment → student forward
(with embedding if contrastive on) → teacher forward (no grad, if distilling) → combined loss →
AMP backward. Cosine LR schedule, early stopping on the validation macro Score, best checkpoint
saved to `runs/<experiment>/best.pt`. Finishes by evaluating every split in `eval.report_splits`
and writing `results.json`.

Run:
```bash
bash scripts/train_teacher.sh      # optional, enables distillation
bash scripts/run_exp1_noise.sh
bash scripts/run_exp2_device.sh
bash scripts/run_exp3_final.sh
```

### 4.7 Evaluation (`evaluate.py`)
Runs a checkpoint over the named splits. `*_noisy` variants apply the noise mixer at eval time so
you can quantify robustness degradation. Every split prints accuracy, macro Se/Sp, the ICBHI-style
Score, and a confusion matrix.

---

## 5. Phase 2 — Quality Check Module

Runs before classification, on-device, to reject unusable recordings. Pure DSP so it ports to the
ESP32 unchanged; mirrored in Python here for development and calibration.

### 5.1 Noise check (`noise_check.py`, spec 3.1)
Per 10 ms frame, two representations: band-passed 100–2000 Hz → RMS energy and its variance
(is breathing present?); wideband DC-removed → spectral flatness and zero-crossing rate (is it
noise-like?). Decision: RMS variance too low → **no_breathing**; flatness AND ZCR both high →
**too_noisy**; otherwise **pass**. (Measuring flatness/ZCR wideband, not band-passed, is deliberate —
band-passing hides broadband hiss inside the band.) Thresholds are defaults; calibrate from labelled
good/bad clips with an ROC sweep and lock them.

### 5.2 Real-time SNR (`snr_check.py`, spec 3.2)
Streaming SNR over a 1 s window, updated every 100 ms. Signal = low-band 100–600 Hz energy where
breathing lives; noise floor = EMA of the minimum band energy in quiet gaps. Returns SNR in dB.

### 5.3 Alert logic (`alert.py`, spec 3.3)
State machine with hysteresis: GREEN >20 dB, YELLOW 10–20 dB, ORANGE first <10 dB window,
**RED only after 3 consecutive bad windows (~300 ms)** so brief noises don't false-alarm; plus a
NO_BREATHING state. Each state carries a vibration pattern and a user message. `gate_recording()`
combines the noise check and SNR readings into one accept/reject decision.

### 5.4 DESED noise detector (`noise_detector.py`, `train_noise_detector.py`)
A tiny multi-label CNN trained on **DESED_synthetic** to name the ambient sound present (speech,
dishes, running water, vacuum, …) as a learned second opinion to the DSP gate. DESED audio is
research-tier (mixed licences) — fine for development, verify before shipping a model trained on it.
```bash
python -m hanil.phase2_quality.train_noise_detector --desed_dir data/noise/desed_synthetic --epochs 30
```

---

## 6. Phase 3 — Sound Quality Analysis / Benchmarking (`benchmark.py`, spec 4.1)

Record the same patients simultaneously on StethoMe, Skeeper, smartphone, and your PCB; score all
with the same model. Input CSV: `patient_id, device, true_label, pred_label, rater1, rater2[...]`.
Outputs:
- **Per-device clinical metrics** (Se/Sp/Score) — is your PCB as good as the commercial devices?
- **Inter-rater agreement** — pairwise Cohen's κ and overall Fleiss' κ for the pulmonologist labels;
  this defines your effective ceiling and reviewers will ask for it.
- **Device-vs-reference agreement** — Bland–Altman bias and limits of agreement, because "our PCB
  equals StethoMe" is an *agreement* claim, not an accuracy one.
```bash
python -m hanil.phase3_benchmark.benchmark --csv bench.csv --reference stethome --score_col p_abnormal
```
Note: 20–30 patients is a pilot; power a larger cohort before any ≤3% equivalence claim.

---

## 7. Phase 4 — ESP32-S3 Deployment

### 7.1 Export chain (`phase4_deploy/`, spec 5.5)
```
best.pt ─ export_onnx.py ─► student.onnx
        ─ onnx_to_tf.py  ─► saved_model/
        ─ quantize_tflite.py ─► student_int8.tflite   (representative set = both devices + noise)
        ─ verify_tflite.py   ─► FP32-vs-INT8 parity + per-class sensitivity change
```
The front end is **not** part of the exported graph — it runs on-device in fixed point — which keeps
the network fully integer and small.

### 7.2 INT8 quantization notes
- Full-integer INT8 with int8 input/output (required for ESP-NN acceleration).
- The **representative dataset must span both devices, all three classes and the noise range**
  (`quantize_tflite.py` draws from the manifest + noise mixer, ~half samples noisy). Get this wrong
  and activation ranges are mis-estimated and real-world accuracy collapses.
- Budget 1–3% accuracy loss; **check crackle/wheeze sensitivity specifically** — minority recall
  degrades first. If it drops >2–3 points, switch to quantization-aware training (QAT).

### 7.3 On-device (`esp32/README.md`, spec 5.2 / 5.6)
ESP-IDF ≥5.1 + esp-tflite-micro + **ESP-NN** (the difference between ~40 ms and ~300 ms) + ESP-DSP.
Budgets for the 0.6–1.2M INT8 student: ~150–500 KB flash, ~60–300 KB tensor arena in PSRAM,
~40–120 ms inference, ~2–5 ms front end. Embed via `xxd -i model.tflite > model_data.cc`. Run the
Phase-2 gate before every inference and drive the LED/vibration from the alert state machine.

### 7.4 Front-end parity — the #1 on-device pitfall
The ESP-DSP log-Mel must reproduce `common/features.py` exactly (`sr, n_fft, hop, n_mels, f_min,
f_max`, log offset, standardisation). Verify a few frames against the Python reference before
trusting board results — front-end mismatch, not the network, is where most edge accuracy is lost.

### 7.5 Smartphone
The same INT8 student runs in a few ms via TFLite/LiteRT (Android) or Core ML (iOS). Reuse the same
quality gate in the app. This gives you the StethoMe-style app+device experience from one model.

---

## 8. End-to-end runbook

```bash
# 0. install + data
pip install -r requirements.txt && pip install -e .
#    build data/manifest.csv, populate data/noise/musan_noise/ and data/noise/self_recorded/

# 1. (optional) cloud teacher for distillation
bash scripts/train_teacher.sh

# 2. the three experiments
bash scripts/run_exp1_noise.sh      # noise robustness
bash scripts/run_exp2_device.sh     # device robustness
bash scripts/run_exp3_final.sh      # final shippable model -> runs/final/best.pt

# 3. Phase 2 detector (optional second-opinion gate)
python -m hanil.phase2_quality.train_noise_detector --desed_dir data/noise/desed_synthetic

# 4. export to ESP32
bash scripts/export_esp32.sh        # onnx -> tf -> int8 tflite -> parity report

# 5. benchmark against commercial devices
python -m hanil.phase3_benchmark.benchmark --csv bench.csv --reference stethome
```

Artifacts: `runs/<experiment>/best.pt`, `runs/<experiment>/results.json`,
`runs/export/student_int8.tflite`.

---

## 9. Configuration reference (key knobs)

| Key | Effect |
|---|---|
| `noise.tier` | `shippable` (blocks research-only corpora) / `research` |
| `noise.snr_db` | SNR mixing range, default `[5, 20]` |
| `device_robust.mixstyle.enable` | turn Exp-2 MixStyle on/off |
| `device_robust.contrastive.weight` | strength of the contrastive term |
| `distill.enable` / `distill.alpha` | distillation on/off and soft/hard balance |
| `model.arch` / `model.student_width` | `student_cnn`/`teacher_ast`; 0.75≈0.6M, 1.0≈1.2M |
| `train.loss` | `focal` / `ce_balanced` (imbalance handling) |
| `eval.report_splits` | which splits to score (add `_noisy` for robustness) |

Configs inherit via `_base_: default.yaml` and deep-merge, so each experiment file only lists its
differences.

---

## 10. What "good" looks like (targets)

| Measure | Target |
|---|---|
| Noise robustness (Exp 1) | <5% accuracy drop at 10 dB SNR |
| Device robustness (Exp 2) | >90% of stethoscope accuracy retained on smartphone |
| External generalisation | Report Sungbook Se/Sp/Score as the headline; small gap vs internal |
| INT8 vs FP32 | <1–3% overall; crackle/wheeze sensitivity drop <2–3 pts (else QAT) |
| ESP32 latency | 40–120 ms inference + 2–5 ms front end |
| Benchmark vs StethoMe | high per-device Score; report κ and Bland–Altman, not accuracy alone |

Always read per-class sensitivity and the confusion matrix, never accuracy alone.

---

## 11. Licensing (see `LICENSING.md`)

Two tiers. **Only the shippable tier may train the model you sell.**
- **Shippable:** MUSAN noise (CC BY 4.0), your own recordings, DEMAND (CC BY-SA).
- **Research only:** DESED_synthetic (mixed), ESC-50 / UrbanSound8K / WHAM! (CC BY-NC).
The noise mixer enforces this: with `noise.tier: shippable` it refuses a research-tier source.

---

## 12. Gotchas / troubleshooting

- **Great in the notebook, poor on the board** → front-end mismatch. Re-verify ESP-DSP log-Mel
  against `common/features.py` (Section 7.4).
- **High accuracy, misses abnormals** → class imbalance and/or clip-level split. Check per-class Se
  and confirm patient-level splitting.
- **INT8 tanks accuracy** → unrepresentative calibration set, or genuine sensitivity loss → use QAT.
- **Teacher won't download** → `teacher_ast.py` falls back to a tiny transformer so the repo still
  runs; for real distillation ensure `transformers` + network access, or skip distillation.
- **Noise aug errors on start** → empty `data/noise/<source>/`; populate it or set `noise.enable:false`.
