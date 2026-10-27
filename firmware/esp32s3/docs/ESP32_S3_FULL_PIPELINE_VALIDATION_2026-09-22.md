# ESP32-S3 Full Pipeline Validation — 2026-09-22

## Result

The compact INT8 student model, TensorFlow Lite Micro tensor arena, and quality-gate mutable state run in ESP32-S3 internal SRAM. Large audio and feature buffers use PSRAM. The firmware builds, flashes, boots, and completes the end-to-end path:

`6 s I2S capture -> central 4 s crop -> log-mel frontend -> quality gate -> INT8 classification -> 2-class and 3-class decision`

The canonical build is `build_student_pipeline_sram/model_test.bin` with `HANIL_STUDENT_PIPELINE_SRAM=ON` and `HANIL_PIPELINE_COPY_MODEL_TO_SRAM=ON`.

## Board and build

| Item | Measured value |
|---|---:|
| Board | ESP32-S3, 240 MHz |
| Flash | 16 MiB |
| Octal PSRAM | 8 MiB |
| Firmware image | 512,032 bytes (500.03 KiB) |
| Application partition | 4 MiB |
| Application partition free | 3,682,272 bytes (88%) |
| Model file | 188,128 bytes (183.72 KiB) |
| Runtime | Portable TFLM INT8, static shapes |

## Runtime memory placement

| Allocation | Memory | Bytes |
|---|---|---:|
| Model weights/copy | Internal SRAM | 188,128 |
| Tensor arena capacity | Internal SRAM | 120,832 |
| Tensor arena actually used | Internal SRAM | 117,132 |
| Quality-gate mutable state | Internal SRAM | 4,128 |
| Frontend frame + power + spectrum | Internal SRAM | 4,012 |
| Four-second PCM | PSRAM | 128,000 |
| Log-mel feature buffer | PSRAM | 102,656 |
| Six-second I2S capture buffer | PSRAM | 384,000 |

Steady-state memory after USB and I2S initialization:

| Heap | Free | Largest free block | Total reported heap |
|---|---:|---:|---:|
| Internal SRAM | 41,659 bytes (40.68 KiB) | 23,552 bytes | 416,459 bytes |
| PSRAM | 7,767,264 bytes (7.41 MiB) | 7,733,248 bytes | 8,388,608 bytes |

The reported internal heap is smaller than the ESP32-S3's nominal SRAM because the firmware image, static data, caches, RTOS, drivers, stacks, DMA allocations, and reserved regions consume memory before heap accounting. Memory remained unchanged across the 20-sample run and three rejected synthetic samples, so no runtime leak was observed.

## On-device accuracy sample

Twenty labeled Sungbook stethoscope recordings were processed from raw PCM through the on-device frontend, gate, and model: five normal, five crackle, five wheeze, and five both.

| Metric | Result |
|---|---:|
| Quality-gate pass | 20/20 (100%) |
| Normal/abnormal accuracy | 18/20 (90%) |
| Three-class accuracy, excluding `both` | 9/15 (60%) |

This 20-recording hardware check verifies execution and an initial accuracy sample; it is not a replacement for the full patient-disjoint evaluation. The full clean-set desktop result for the student remains internal Score 0.823 and external Score 0.756. The teacher result was internal 0.843 and external 0.734.

## Host/device parity

| Check | Result |
|---|---:|
| Exact quantized frontend hash | 16/20 |
| Exact raw INT8 model output | 19/20 |
| Exact final 2-class and 3-class decisions | 20/20 |

Four recordings had small frontend quantization differences. Only one of those propagated to the model output, where both head values differed from the host reference by one INT8 count. No final decision changed. This deployment therefore uses the parity-safe portable TFLM kernels; optimized ESP-NN remains disabled until it can independently pass the same test.

## Latency at 240 MHz

Mean of 20 passing recordings:

| Stage | Mean latency |
|---|---:|
| Log-mel frontend | 203.19 ms |
| Signal/noise quality gate | 971.30 ms |
| Input quantization | 21.46 ms |
| INT8 inference | 2,110.28 ms |
| Total device compute | 3,311.61 ms |
| Host-command end-to-end, excluding serial transfer and capture | 3,320.05 ms |

A passing live recording is expected to take about 9.7 seconds from the start of a six-second capture through classification. A rejected recording stops before inference, reducing post-capture processing.

## Quality-gate checks

| Input | Decision | Reason | Processing time |
|---|---|---|---:|
| Silence | RETRY | `silence_no_contact` | 1,143.19 ms compute |
| White noise | RETRY | `noise_rustle` | 1,169.40 ms compute |
| Fully clipped square wave | RETRY | `clipping_saturation` | 1,189.51 ms compute |

The gate correctly prevented classification for all three invalid inputs.

## Live INMP441 result

The microphone is no longer returning all-zero PCM. It captured 96,000 samples in approximately 6.30 seconds using WS GPIO 1, BCLK GPIO 2, and SD GPIO 4.

| Measurement | Result |
|---|---:|
| Raw peak | -44.6 dBFS |
| Raw RMS | -58.8 dBFS |
| Applied gain | 32x (30.1 dB) |
| Post-gain peak | -14.5 dBFS |
| Post-gain RMS | -28.7 dBFS |
| Gate decision | RETRY: `poor_chest_contact` |
| Placement instruction | Improve chest contact and avoid rubbing |

This proves the I2S capture path works. A mandatory raw-level placement guard now stops classification before digital gain can make a weakly coupled recording appear valid. Microphone placement/acoustic coupling and gain still require calibration on the assembled stethoscope before live predictions can be trusted.

## PCB readiness

The firmware is ready for controlled engineering tests on the PCB. Model execution, internal-SRAM placement, PSRAM buffering, quality-gate rejection, I2S capture, and serial reporting are working.

It is not yet ready for a clinical or final product claim. The next required validation is repeated chest recordings with known labels while tuning acoustic contact and gain, followed by a larger patient-disjoint PCB dataset evaluation. ESP-NN optimization may be revisited only after its outputs match the portable TFLM reference.

## Evidence

- Full raw benchmark: `comparison_evaluation/esp32_student_full_pipeline_2026-09-22.json`
- Hardware test driver: `tools/benchmark_esp32_pipeline.py`
- Firmware implementation: `main/main.cc`
- Quality gate: `main/quality_gate.c` and `main/quality_gate.h`
