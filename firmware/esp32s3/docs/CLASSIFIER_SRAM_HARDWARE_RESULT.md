# ESP32-S3 classification-only SRAM result

## Outcome

The compact INT8 student classifier was built, flashed, and run on the connected ESP32-S3 at 240 MHz. In the current test firmware, both the model flatbuffer and the TensorFlow Lite Micro tensor arena are allocated from internal byte-addressable SRAM. The firmware reports zero PSRAM allocations.

This proves that the classifier's **runtime model data and inference workspace** fit and execute from internal SRAM. Normal ESP-IDF firmware instructions and embedded constants remain flash-mapped. It does not yet prove that microphone capture, preprocessing, the quality gate, and classification all fit in internal SRAM at the same time.

## Measured on the ESP32-S3

| Measurement | Result |
|---|---:|
| Student INT8 model file | 188,128 bytes (183.72 KiB) |
| Model runtime placement | Internal SRAM |
| Tensor arena reserved | 131,072 bytes (128.00 KiB) |
| Tensor arena actually used | 117,132 bytes (114.39 KiB) |
| Free internal heap after model/interpreter initialization | 76,124 bytes (74.34 KiB) |
| PSRAM allocations | 0 |
| Firmware `.bin` size in flash | 398,144 bytes (388.81 KiB) |
| Inference latency, 20 clips, mean | 2,109.804 ms |
| Inference latency, 20 clips, median | 2,109.797 ms |
| Inference latency, min–max | 2,109.796–2,109.939 ms |
| Desktop LiteRT versus ESP32 output parity | 20/20 exact INT8 outputs |

The same 20-file set was replayed again on 2026-09-15. It again produced 20/20 exact outputs, 2,109.804 ms mean and 2,109.796 ms median inference latency. The repeated run is saved in `comparison_evaluation/esp32_classifier_retest_2026-09-15.json`.

The 512 KiB figure is the chip's physical on-chip SRAM, not a single completely free application heap. At boot, this firmware reported 416,380 bytes of internal-capable heap after the ESP-IDF runtime and static sections had taken their portions. The live allocation test is therefore more useful than adding file sizes on paper.

## Flash-mapped weights versus SRAM-copied weights

The same 20 inputs were tested twice. In both cases the tensor arena was internal SRAM.

| Weight placement | Median inference | Free internal heap after initialization |
|---|---:|---:|
| Flash/DROM mapping | 2,118.841 ms | 264,256 bytes (258.06 KiB) |
| Internal SRAM copy | 2,109.797 ms | 76,124 bytes (74.34 KiB) |

Copying the 183.72 KiB model into SRAM saved only 9.044 ms per inference, or about 0.43%. For the complete product pipeline, keeping weights in flash and reserving internal SRAM for the tensor arena, DSP working buffers, I2S DMA, and task stacks is the better starting design.

## Accuracy of the deployed INT8 artifact

The full 33,508-file dataset was evaluated with desktop LiteRT using the exact INT8 file embedded in the firmware. Exact output parity on the 20-file hardware golden set links those desktop results to the portable TFLM implementation on the board.

| Dataset | 2-class accuracy | Balanced accuracy | Sensitivity | Specificity | 3-class accuracy | 3-class macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Held-out stethoscope | 71.55% | 74.12% | 77.70% | 70.55% | 67.92% | 43.76% |
| External smartphone combined | 78.16% | 80.36% | 83.52% | 77.20% | 75.42% | 51.26% |
| All sources | 82.97% | 84.71% | 87.90% | 81.51% | 80.26% | 65.44% |

The 20-file ESP32 run gave 90% two-class accuracy and 60% three-class accuracy excluding `B`, but that small subset is only a smoke/parity check and must not replace the full-dataset accuracy above. These are acoustic classification metrics, not clinical diagnostic accuracy.

## Teacher versus SRAM student

| Item | Previous teacher | SRAM student |
|---|---:|---:|
| INT8 model file | 240,112 bytes (234.48 KiB) | 188,128 bytes (183.72 KiB) |
| Measured/required TFLM arena | about 854 KiB | 114.39 KiB used |
| Runtime model data plus TFLM arena fit internal SRAM | No | Yes |
| Portable TFLM inference on ESP32-S3 | 17,831.93 ms | 2,109.80 ms |
| Speed change | baseline | about 8.45x faster |
| Held-out 2-class accuracy | 79.86% | 71.55% |
| External combined 2-class accuracy | 65.59% | 78.16% |

The student gives up held-out stethoscope accuracy but generalizes better on the external smartphone set and is deployable within the internal-SRAM constraint. The earlier approximately 753 ms ESP-NN teacher result is not an accuracy-valid comparison because its output did not match LiteRT. The current student result uses the slower portable TFLM reference kernels and passes parity.

## What is complete and what remains

Complete:

- Classification-only firmware built and flashed on the physical ESP32-S3.
- Student model and tensor arena confirmed in internal SRAM.
- Zero PSRAM allocations confirmed for this test application.
- Exact host/device output parity confirmed on 20 real dataset clips.
- Actual inference latency and post-initialization free internal heap measured.

Still required for the smart-stethoscope PCB pipeline:

- Integrate live INMP441 capture, the exact preprocessing, raw microphone/contact validation, and the quality gate with the student classifier.
- Start with model weights flash-mapped and hot activations/DSP state in internal SRAM; 74.34 KiB is too little headroom for the literal all-internal model-copy layout.
- Measure end-to-end capture, preprocessing, gate, classification latency, peak heap, stack high-water marks, and I2S stability.
- Calibrate gain, placement, contact, motion-rub, speech, and environmental-noise thresholds from labeled real PCB recordings.
- Re-enable ESP-NN only after its outputs pass the same golden-set parity test.

## Quality-gate placement

The gate should be kept in internal SRAM. Its `qg_accum_t` is only about 2 KiB, and its 512-float sorting scratch is another 2 KiB, so the gate adds roughly 4.1 KiB of persistent/static working data. It reuses the frontend frame and power spectrum, so placing it in PSRAM would save little SRAM while adding avoidable external-memory access.

The current complete-pipeline source allocates `qg_accum_t` in PSRAM as part of its older large-workspace layout. When the student model is integrated, move that accumulator to internal SRAM together with the FFT frame, power, spectrum, RFFT state, I2S DMA, and microphone scratch. Keep the 384,000-byte six-second raw recording, 128,000-byte four-second PCM buffer, and 102,656-byte float Log-Mel staging buffer in PSRAM initially.
