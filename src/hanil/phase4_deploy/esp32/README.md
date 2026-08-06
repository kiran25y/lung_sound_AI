# ESP32-S3 integration (5.2 target / 5.6 PCB)

## Firmware stack
- **ESP-IDF** ≥ 5.1
- **esp-tflite-micro** (TFLite Micro port) + **ESP-NN** (INT8 SIMD kernels — enable it,
  it is the difference between ~40 ms and ~300 ms inference)
- **ESP-DSP** for the fixed-point log-Mel front end

## Resource budget (0.6–1.2M INT8 student)
| Resource | Budget | Notes |
|---|---|---|
| Model (flash) | ~150–500 KB | `student_int8.tflite` via `xxd -i` |
| Tensor arena (PSRAM) | ~60–300 KB | static buffer in PSRAM |
| Inference | ~40–120 ms | with ESP-NN |
| Front end (log-Mel) | ~2–5 ms | ESP-DSP |

## Front-end parity — the #1 on-device pitfall
The device log-Mel MUST match `common/features.py` exactly:
`sr=16000, n_fft=400 (25 ms), hop=160 (10 ms), n_mels=64, f_min=50, f_max=2000`,
followed by `log(mel + 1e-6)` and per-clip mean/var standardisation. Re-verify a few
frames against the Python reference before trusting board results.

## Embed the model
```bash
xxd -i runs/export/student_int8.tflite > main/model_data.cc
# then reference g_model in your ESP-IDF main
```

## Inference skeleton (C++)
```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model_data.cc"      // g_model, g_model_len

constexpr int kArenaSize = 300 * 1024;                 // in PSRAM
static uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM);

void run_inference(const int8_t* logmel /* n_mels*frames, quantized */) {
    const tflite::Model* model = tflite::GetModel(g_model);
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected(); resolver.AddAveragePool2D();
    resolver.AddReshape(); resolver.AddSoftmax();
    resolver.AddAdd(); resolver.AddMul();

    static tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
    interp.AllocateTensors();
    TfLiteTensor* in = interp.input(0);
    memcpy(in->data.int8, logmel, in->bytes);
    interp.Invoke();
    TfLiteTensor* out = interp.output(0);
    // out->data.int8[0..2] -> {normal, crackle, wheeze}; dequantize with out->params
}
```

## Quality gate on-device (Phase 2)
Run `noise_check` + `snr_check` (ported from `phase2_quality/*.py`) BEFORE inference.
Only classify clips that pass the gate; drive the LED/vibration from `alert.py`'s state
machine. This mirrors the Python reference so behaviour is identical to training-time QC.

## 5.6 PCB notes
- MEMS mic (e.g. Knowles SPH0645, I2S), 24-bit, capture 48 kHz then downsample to 16 kHz.
- ESP32-S3 module with ≥8 MB PSRAM, ≥8 MB flash.
- Built-in WiFi/BLE for app streaming + OTA model updates.
- Consumer-grade lifecycle — see the report's product-positioning note before making
  any medical (CE Class IIa) claim.
