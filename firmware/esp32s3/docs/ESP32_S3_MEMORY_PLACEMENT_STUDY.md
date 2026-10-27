# ESP32-S3 SRAM/PSRAM placement study and smart-stethoscope roadmap

Status: classification-only placement is measured; the complete-pipeline layout remains a design recommendation and is not yet implemented.

## Executive decision

The current teacher pipeline cannot fit in the ESP32-S3's 512 KiB on-chip SRAM. Its measured TFLM arena is about 854 KiB, before microphone capture, frontend buffers, FreeRTOS, drivers, and stacks.

The new SRAM student is a credible internal-arena candidate:

| Graph property | Teacher | SRAM student | Change |
|---|---:|---:|---:|
| INT8 TFLite file (flash) | 240,112 B | 188,128 B | -21.6% |
| Estimated MACs | 37,110,784 | 4,011,464 | -89.2% |
| Operators | 29 | 15 | -48.3% |
| Largest activation tensor | 441,728 B | 74,400 B | -83.2% |
| Activation-liveness lower bound | 853,376 B | 100,064 B | -88.3% |
| Actual ESP32 TFLM arena | about 854 KiB, measured | 117,132 B used (114.39 KiB) | measured on hardware |

The earlier 100,064-byte student result was a graph-liveness lower bound. The physical ESP32-S3 test now reports `MicroInterpreter::arena_used_bytes() = 117,132` bytes from a 131,072-byte internal arena.

With both model data and arena copied to internal SRAM, the classifier left 76,124 bytes (74.34 KiB) free and its largest free internal block was 31,744 bytes. With weights flash-mapped and the arena internal, it left 264,256 bytes (258.06 KiB) free. Flash-mapped weights increased median inference latency by only 9.044 ms (0.43%), so the hybrid layout is strongly preferred for the full pipeline.

The recommended first PCB layout is hybrid: keep the model in flash, put the student tensor arena and hot DSP state in internal SRAM, and leave large capture/staging/history buffers in PSRAM. This is the safest route to low latency without first redesigning the six-second capture pipeline.

## What “512 KiB SRAM” actually means

The ESP32-S3 has 512 KiB of on-chip SRAM shared by instructions and data; it is not a 512 KiB application heap. Cache configuration, IRAM code, static DRAM, FreeRTOS, stacks, drivers, DMA, and other components consume part of it. Espressif explicitly notes that available RAM depends on cache size, and provides capability-aware heap APIs to inspect the usable memory at runtime. See the [ESP32-S3 datasheet](https://documentation.espressif.com/esp32_s3_datasheet_en.pdf), [RAM-usage guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-guides/performance/ram-usage.html), and [heap allocator guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-reference/system/mem_alloc.html).

The existing `build/model_test.elf` statically occupies approximately:

| Internal section | Bytes |
|---|---:|
| `.iram0.vectors` | 1,027 |
| `.iram0.text` | 57,291 |
| `.dram0.data` | 12,708 |
| `.dram0.bss` | 3,896 |
| Total visible static internal use | 74,922 (73.2 KiB) |

This is a baseline from the existing build, not a fresh linked student build. Dynamic startup allocations, stacks, DMA, heap fragmentation, and the tensor arena are not included.

## Current firmware buffers

The current source allocates these large objects in PSRAM:

| Buffer | Calculation | Bytes |
|---|---:|---:|
| Six-second raw INMP441 recording | 96,000 × `int32_t` | 384,000 |
| Four-second PCM | 64,000 × `int16_t` | 128,000 |
| Log-Mel workspace | 64 × 401 × `float` | 102,656 |
| Quality accumulator | 512 float energies plus state | about 2 KiB |
| TFLM arena reservation | configured | 2,097,152 |

The model flatbuffer is embedded in flash. Its 188,128-byte file size does **not** need to be added to the tensor arena or internal heap.

With the current allocation lifetimes, even the student's lower-bound arena cannot coexist with all capture/frontend objects inside 512 KiB: 384,000 + 128,000 + 102,656 + 100,064 + 74,922 already exceeds 770 KiB, before stacks, DMA, scratch, and safety margin.

## Recommended memory placement

| Module/data | First-choice memory | Reason |
|---|---|---|
| INT8 TFLite flatbuffer and weights | Flash/DROM | Read-only, persistent, and too costly to duplicate in SRAM |
| Student tensor arena and activations | Internal SRAM | Every operator repeatedly reads/writes activations; this is the highest-value SRAM use |
| ESP-NN/TFLM operator scratch | Internal SRAM | Hot and sometimes subject to alignment/SIMD constraints |
| I2S DMA descriptors and DMA buffers | Internal DMA-capable SRAM | Correct and predictable for the current ESP-IDF/I2S configuration |
| FFT frame, power, spectrum, RFFT state | Internal SRAM | Small, hot, reused 401 times per clip |
| Gate accumulator and live capture statistics | Internal SRAM | Small and latency-sensitive |
| Task stacks and ISR-accessed state | Internal SRAM | Default/safest ESP-IDF placement |
| Six-second raw capture | PSRAM initially | Largest current allocation; not touched during model inference |
| Four-second PCM and float Log-Mel staging | PSRAM initially | Large sequential frontend buffers; release or reuse before inference |
| Saved WAV/history, BLE/Wi-Fi upload queues | PSRAM | Capacity-oriented and not in the inference hot loop |

The current gate uses about 2 KiB for `qg_accum_t` and another 2 KiB for its static sorting scratch, approximately 4.1 KiB total. It should be moved from the older source's PSRAM allocation to internal SRAM when integrating the student. The large six-second recording, PCM, and float Log-Mel staging buffers remain the appropriate PSRAM users.

External PSRAM is mapped through cache, and large accesses eventually run at external-memory speed; flash and PSRAM also share the memory/cache path. Keeping activations internal avoids adding repeated external traffic on top of flash weight reads. Espressif documents these behaviors and DMA limitations in its [external-RAM guide](https://docs.espressif.com/projects/esp-idf/en/v5.4.1/esp32s3/api-guides/external-ram.html) and [flash/PSRAM guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/flash_psram_config.html).

ESP-NN uses ESP32-S3 assembly/vector optimizations and its published examples show large speedups, but those numbers are model- and memory-dependent and cannot be applied to this lung model. Use them as motivation to measure, not as a latency forecast. See [Espressif ESP-NN](https://github.com/espressif/esp-nn).

## If a zero-PSRAM product is mandatory

The existing buffer lifetime must change; merely moving the current allocations to internal SRAM will fail.

A feasible design to prototype later is phase-overlaid memory:

1. Keep the six-second user interaction but discard the first one-second margin as it arrives, retain only the middle four seconds, and discard the last margin. This reduces raw `int32_t` retention from 384,000 to 256,000 bytes without changing the four-second model input.
2. Compute raw DC, RMS, peak, and clipping statistics online while capturing the middle segment.
3. Compact the retained 24-bit samples to 16-bit PCM in the same 256 KiB block after gain is selected.
4. Reuse the freed half of that block for the 102,656-byte float Log-Mel workspace; PCM plus Log-Mel then fit in the same 256 KiB allocation.
5. After normalization and quantization, release/overlay the frontend workspace before inference. Copy the final 25,664-byte INT8 feature into the interpreter input only after `AllocateTensors()`.

This phase-overlay design is not implemented. It needs careful ownership and lifetime tests, repeated-capture soak tests, and a measured largest-free-block margin. A total-free-heap number is insufficient because a fragmented heap may not provide one contiguous arena; record both `heap_caps_get_free_size(MALLOC_CAP_INTERNAL)` and `heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)`.

## Required on-PCB memory and latency experiment

Build two student variants without changing preprocessing or thresholds:

1. Portable TFLM with arena in internal SRAM.
2. ESP-NN with arena in internal SRAM, with bit-exact/fast approximation options left conservative until parity passes.

For each variant, log at boot, after frontend allocation, before inference, and after inference:

- internal free bytes, largest block, and minimum-ever free bytes;
- PSRAM free bytes;
- requested arena size and `arena_used_bytes()`;
- capture time, preprocessing/gate time, `Invoke()` time, and total result time;
- quantized input hash, raw output bytes, and dequantized probabilities;
- CPU frequency, PSRAM mode/frequency, ESP-IDF version, TFLM/ESP-NN commit, and build SHA.

Use a fixed golden set containing N, C, W, B, threshold-near cases, silence, clipping, noise, and poor-contact PCB recordings. Portable TFLM must match host LiteRT first. ESP-NN must then match portable outputs or remain disabled. Do not trade correctness for the published benchmark speed.

## End-to-end product pipeline

The intended product flow is:

```text
INMP441 I2S capture (6 s)
  -> wiring/no-signal and raw-RMS/contact check
  -> DC removal and bounded gain
  -> 4 s center segment
  -> FFT + Log-Mel frontend
  -> clipping/noise/signal-quality gate
       RETRY -> reason-specific user alert
       PASS  -> INT8 classifier
                  -> 2-class: normal / abnormal
                  -> 3-class: normal / crackle / wheeze
  -> save/share raw audio, quality metrics, probabilities, firmware/model versions
```

The product output should never emit `both`; when both heads fire, the three-class result uses the dominant probability while the two-class result remains abnormal.

Two gate details need correction/calibration before product release. First, low-amplitude audio mostly passes `qg_scan_pcm` because that function peak-normalizes before its RMS calculation; the independent pre-gain raw-RMS contact check is therefore mandatory. Second, the clipping fraction is also calculated after peak normalization. A clean steady sine wave can consequently be reported as `clipping_saturation` even when it never reached ADC/full-scale clipping. The product design should use raw ADC/post-gain saturation counts for true clipping and reserve a separate reason for tonal/interference or low-dynamic-range signals. These changes are intentionally not implemented during this study.

The reference products point to three important requirements beyond a classifier. StethoMe guides chest placement, returns results, and supports sharing recordings with a doctor; Stemoscope emphasizes capture/streaming and remote auscultation; Skeeper describes on-device AI, preprocessing, original-sound preservation, and device-side analysis. These are product requirements to study, not evidence that our present prototype is clinically equivalent. Sources: [StethoMe](https://www.stethome.com/en-gb), [Stemoscope](https://stemoscope.com/), and [Skeeper AI](https://ismartsound.com/solutions/skeeper-ai/).

## Real-world validation gates before PCB/product readiness

1. **Golden parity:** identical features and INT8 output bytes across host LiteRT, portable TFLM, and—if enabled—ESP-NN.
2. **Engineering capture set:** labeled good contact, no contact, loose contact, clothing, cable rub, hand motion, speech, ambient music/TV, fan noise, quiet room, and clinical-like breath sounds recorded through the actual PCB enclosure.
3. **Gate validation:** report good-recording acceptance and bad-recording rejection separately by failure type. Tune on a calibration subset and lock thresholds before testing.
4. **Classifier validation:** report patient-disjoint sensitivity, specificity, balanced accuracy, F1, AUROC, and confusion matrices for two heads plus the product-facing two- and three-class decisions.
5. **Domain validation:** report every microphone/enclosure/site/age group separately. Do not hide a weak site inside an overall average.
6. **Timing and power:** measure capture, frontend, gate, inference, wireless transfer, current draw, battery life, and thermal behavior on the final PCB.
7. **Usability:** verify that each RETRY reason produces a specific action—improve contact, stop rubbing/moving, reduce environmental noise, or check microphone wiring.
8. **Clinical/regulatory:** before marketing medical diagnostic claims, define intended use, risk controls, clinician labeling protocol, prospective validation, privacy/security, and the applicable regulatory path.

## Current decision boundary

- Teacher: accuracy/reference model; PSRAM-required in the current TFLM implementation.
- Student: classification deployment candidate; model data and arena have been proven in internal SRAM with 20/20 exact host/device output parity.
- Recommended architecture for the first PCB test: hybrid internal-SRAM arena plus PSRAM capture/frontend staging.
- Strict all-internal design: possible only after the phase-overlay frontend redesign and a measured heap/fragmentation margin.
- Full microphone/frontend/gate placement should not be merged until the integrated build reports heap, largest-block, stack, and end-to-end timing measurements.
