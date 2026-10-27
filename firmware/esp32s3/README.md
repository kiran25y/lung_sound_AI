# Hanil lung-audio inference firmware

ESP32-S3 firmware for lung-audio preprocessing, signal-quality
gating, and INT8 TFLite Micro inference. It provides two output views:

- two-class: `normal` or `abnormal`
- three-class: `normal`, `crackle`, or `wheeze`

When both abnormal heads exceed their thresholds, the three-class result uses
the stronger head.

> This project is for engineering validation. Its outputs are not a clinical
> diagnosis or a substitute for review by a qualified medical professional.

## Pipeline

```text
4 s PCM16, 16 kHz mono
  -> 400-point STFT / 160-sample hop
  -> 64-band log-Mel frontend (50-2000 Hz)
  -> per-clip normalization and INT8 quantization
  -> clipping / silence / noise / signal-quality gate
  -> ESP-TFLite-Micro reference-integer model kernels
  -> crackle and wheeze probabilities
```

The deployed model input is `INT8 [1,64,401,1]`. The complete frontend and
quality gate run on the ESP32 for both uploaded PCM and live microphone input.

## Hardware

- ESP32-S3 DevKitC-compatible board with 8 MB PSRAM
- INMP441 I2S microphone
- USB Serial/JTAG connection to the host PC

Current microphone mapping, confirmed by the working microphone test firmware:

| INMP441 signal | ESP32-S3 |
|---|---:|
| SCK / BCLK | GPIO2 |
| WS / LRCLK | GPIO1 |
| SD / DIN | GPIO4 |
| L/R | GND (left slot) |
| VDD | 3V3 |
| GND | GND |

The I2S receiver uses Philips format, 16 kHz, and 32-bit mono-left slots. It
retains the microphone's signed 24-bit samples, removes measured DC, then uses
bounded 1x-32x digital auto-gain with a -12 dBFS peak target before PCM16
conversion. Live capture records six seconds and classifies the middle four
seconds. The serial log reports raw/post-gain dBFS, clipping, applied gain, and
placement status; `improve_chest_contact` means the unamplified signal remains
below -55 dBFS RMS and the sensor should be pressed firmly without rubbing.

## Repository layout

- `main/` — ESP-IDF application, deployed model, frontend constants, and gate
- `components/hanil_ai/` — reusable source component with one public header per module
- `release/hanil_ai_esp32s3/` — prebuilt `.a`, public headers, and integration example
- `tools/send_wav.py` — serial WAV/microphone test client and parity checker
- `tools/esp32_inference_api.py` — local Flask WAV-to-device inference API
- `pipeline_source/hanil_lung_ai/` — Python preprocessing/training/deployment source
- `release/final_ai/` — final checkpoint, INT8 model, flashable firmware, results, and report
- `API_TESTING.md` — Flask setup and request examples
- `quality_gate_calibration.json` — gate calibration evidence
- `real_prediction_results.json` — recorded device smoke-test results

Generated ESP-IDF build trees, downloaded managed components, local Python
packages, datasets, and intermediate training runs are intentionally excluded.
Only the validated final artifacts are retained under `release/final_ai/`.

## Reusable static-library package

The complete pipeline is also available as a reusable ESP-IDF component. Its
public C API is split by responsibility so PCB firmware can use either the
complete pipeline or an individual module.

| Module | Public header |
|---|---|
| Common status, timing, and heap telemetry | `hanil_ai_types.h` |
| INMP441 I2S capture and placement check | `hanil_microphone.h` |
| Noise and signal-quality gate | `hanil_quality_gate.h` |
| Exact log-Mel preprocessing | `hanil_frontend.h` |
| INT8 TFLite Micro classifier | `hanil_classifier.h` |
| End-to-end orchestration | `hanil_pipeline.h` |

The distributable package is `release/hanil_ai_esp32s3.zip`. Extract its
contents into `components/hanil_ai` in an ESP-IDF 6.0.x ESP32-S3 project, then
declare `REQUIRES hanil_ai` in the application component. The archive embeds
the INT8 model and frontend constants; no separate model file is required.

To rebuild the library and refresh the distributable package:

```powershell
idf.py -B build_hanil_ai_package `
  -D HANIL_AI_LIBRARY_EXAMPLE=ON `
  -D HANIL_PIPELINE_COPY_MODEL_TO_SRAM=ON build
powershell -ExecutionPolicy Bypass -File scripts/package_hanil_ai.ps1
```

`tests/prebuilt_package_smoke` is a separate firmware project that consumes
only the packaged archive and headers. It verifies that no private source file
or undeclared asset is needed by an integrating PCB application.

## Final release artifacts

| Artifact | Repository path |
|---|---|
| Final PyTorch checkpoint | `release/final_ai/checkpoint/sram_student_best.pt` |
| Deployable INT8 TFLite model | `release/final_ai/model/sram_student_int8.tflite` |
| Flashable ESP32-S3 application | `release/final_ai/firmware/model_test.bin` |
| Final training metrics | `release/final_ai/results/training_results.json` |
| FP32-to-INT8 parity results | `release/final_ai/results/int8_parity.json` |
| ESP32 full-pipeline results | `release/final_ai/results/esp32_full_pipeline.json` |
| Final engineering report | `release/final_ai/report/Pediatric_LungSound_AI_Final_Report.docx` |

The release contains one checkpoint and one deployable INT8 model. Intermediate
checkpoints, FP32 models, ONNX exports, calibration arrays, and complete build
directories are not committed.

## Build and flash

Install ESP-IDF 5.3.x, then build the final SRAM-student pipeline from an
ESP-IDF terminal:

```powershell
idf.py set-target esp32s3
idf.py -B build_student_pipeline_sram `
  -D HANIL_STUDENT_PIPELINE_SRAM=ON `
  -D HANIL_PIPELINE_COPY_MODEL_TO_SRAM=ON build
idf.py -B build_student_pipeline_sram -p COM4 flash monitor
```

To flash the committed final binaries without rebuilding, run from the
repository root with `esptool.py` available:

```powershell
esptool.py --chip esp32s3 -p COM4 -b 460800 write_flash `
  --flash-mode dio --flash-freq 80m --flash-size 16MB `
  0x0 release/final_ai/firmware/bootloader/bootloader.bin `
  0x8000 release/final_ai/firmware/partition_table/partition-table.bin `
  0x10000 release/final_ai/firmware/model_test.bin
```

The project CMake file explicitly disables the ESP-NN wrapper branches and
uses TFLM reference-integer kernels. This parity mode prioritizes matching the
desktop LiteRT INT8 output over ESP-NN inference speed.

The converter left stale batch-1 allocation shapes around the model's
`SPACE_TO_BATCH_ND` dilated-convolution block. LiteRT resizes those tensors to
batch 2 at runtime, while TFLM uses the FlatBuffer's static allocation. The
deployed `model_int8_tflm.tflite` resolves tensors 68-71 to batch 2; the
original `model_int8.tflite` remains unchanged. Regenerate and validate the
deployment copy with:

```powershell
python tools/fix_tflm_static_shapes.py
```

On this model, the locked ESP-NN kernels and a mixed ESP-NN/reference build did
not reproduce LiteRT's intermediate activations. They therefore remain
disabled until a future optimized kernel version passes the same parity test.

The partition table reserves a 4 MB factory application partition and the
firmware expects octal PSRAM.

## Test a WAV directly through the ESP32

```powershell
python -m pip install numpy pyserial
python tools/send_wav.py --port COM4 --wav C:\path\to\lung_sound.wav --mode pcm
```

This converts the WAV to four seconds of 16 kHz mono PCM16 and sends it using
the `PCM1` command. Preprocessing, gating, and inference then run on the board.

## Flask API

```powershell
python -m pip install -r tools/requirements-api.txt
python tools/esp32_inference_api.py --serial-port COM4
```

Upload a WAV:

```powershell
curl.exe -X POST http://127.0.0.1:5000/infer `
  -F "audio=@C:\path\to\lung_sound.wav"
```

Run the attached I2S microphone after its power and wiring are verified:

```powershell
curl.exe -X POST http://127.0.0.1:5000/infer/live
```

Only one program can own the serial port at a time, so close ESP-IDF monitor
before starting the API.

## Final ESP32-S3 validation

The final compact student pipeline was tested on an ESP32-S3 at 240 MHz with
16 MB flash and 8 MB octal PSRAM.

| Measurement | Final result |
|---|---:|
| INT8 model size | 188,128 bytes (183.72 KiB) |
| Tensor arena used | 117,132 bytes (114.39 KiB) |
| Internal SRAM free after initialization | 41,659 bytes (40.68 KiB) |
| Firmware application size | 512,032 bytes (500.03 KiB) |
| Mean INT8 inference latency | 2,110.28 ms |
| Mean total device compute | 3,311.61 ms |
| Two-class accuracy on 20 board clips | 90% |
| Three-class accuracy, excluding `both` | 60% |
| Final host/device decision parity | 20/20 |

The gate passed all 20 labeled test recordings and rejected the synthetic
silence, white-noise, and clipped inputs. The final clean-set student scores
were 0.823 on the internal test and 0.756 on the external test.

The live INMP441 path captured non-zero PCM, but the tested chest recording was
weak at approximately `-58.8 dBFS` raw RMS. The raw-level placement guard
correctly returned `RETRY poor_chest_contact` instead of issuing a prediction.
Acoustic coupling and gate calibration on the final chestpiece remain required
before live predictions are trusted.

## Reproducible teacher-versus-student evaluation

The comparison keeps classification-only metrics separate from the
quality-gated product pipeline and includes deterministic signal/noise stress
conditions. On Windows with Ubuntu WSL:

```powershell
.\scripts\run_end_to_end_comparison.ps1 `
  -DatasetRoot C:\Users\user\Downloads\lungsound
```

On Linux or the training server, install `tools/requirements-eval.txt`, set
`PYTHON_BIN` if needed, and run:

```bash
bash scripts/run_end_to_end_comparison.sh /path/to/lungsound
```

Results are written to `comparison_evaluation/COMPARISON_REPORT.md`. The SRAM
and PSRAM analysis is in `docs/ESP32_S3_MEMORY_PLACEMENT_STUDY.md`. The final
firmware implements the measured mixed-memory layout: the compact model,
tensor arena, and quality-gate state use internal SRAM, while large audio and
feature buffers use PSRAM.
