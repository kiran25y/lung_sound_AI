# Hanil AI prebuilt package for ESP32-S3

This package contains the complete lung-sound AI pipeline as a prebuilt static
library plus a public C API. It includes microphone capture, placement checks,
the signal/noise quality gate, exact log-Mel preprocessing, INT8 TFLite Micro
inference, and the final two-class/three-class decision layer.

## Compatibility

- Target: ESP32-S3
- ESP-IDF: 6.0.x (built with 6.0.2)
- Toolchain: `xtensa-esp-elf` GCC 15.2.0
- Inference backend: portable TFLite Micro INT8 kernels
- Model: 188,128-byte SRAM student
- Audio contract: 16 kHz mono, 4-second analysis window
- Microphone default: INMP441, BCLK GPIO 2, WS GPIO 1, SD GPIO 4

The `.a` file is ABI-specific. If the ESP-IDF or compiler version changes,
rebuild the source component in `components/hanil_ai` rather than reusing this
archive.

## Files

| Path | Purpose |
|---|---|
| `lib/libhanil_ai.a` | complete prebuilt AI pipeline |
| `include/hanil_ai_types.h` | common constants, status, timing, memory |
| `include/hanil_microphone.h` | I2S capture and placement API |
| `include/hanil_quality_gate.h` | signal/noise quality-gate API |
| `include/hanil_frontend.h` | log-Mel frontend API |
| `include/hanil_classifier.h` | INT8 classifier API |
| `include/hanil_pipeline.h` | end-to-end pipeline API |
| `example/main.cc` | minimal live-inference example |

## Integration

1. Create `components/hanil_ai` in the target project and extract all package
   contents there. If using the unpacked repository directory instead, copy
   its contents (not the parent directory) to `components/hanil_ai`.
2. Add `hanil_ai` to the application component requirements:

```cmake
idf_component_register(SRCS "main.cc" REQUIRES hanil_ai)
```

3. Include `hanil_pipeline.h` and use `hanil_pipeline_create()` followed by
   either `hanil_pipeline_capture_and_run()` for the live microphone or
   `hanil_pipeline_run_pcm()` for an existing 64,000-sample PCM16 buffer.

The top-level API returns a prediction only after both microphone placement
and quality-gate checks pass. Inspect `prediction_valid`, `microphone_result`,
and `frontend.gate_result` before using the classification.

The application must enable PSRAM because the 6-second microphone recording,
4-second PCM buffer, and log-Mel staging buffer are deliberately kept outside
internal SRAM. The model, tensor arena, gate state, FFT workspaces, and I2S
scratch buffer use internal SRAM.

## Build verification

This archive was built and link-tested with ESP-IDF 6.0.2. The independent
smoke firmware in `tests/prebuilt_package_smoke` includes no Hanil source files
and links only this `.a` file and the six public headers. `SHA256SUMS.txt`
contains the integrity hashes for the package contents.

## Memory placement

- Internal SRAM: model copy, tensor arena, quality-gate state, FFT workspaces,
  and I2S scratch buffer.
- PSRAM: raw microphone recording, PCM staging, and log-Mel staging.

This is the measured mixed-memory layout used by the final firmware. The model
and tensor arena remain in internal SRAM for predictable inference latency.

