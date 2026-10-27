# Hanil AI ESP32-S3 component

`hanil_ai` packages the complete lung-sound pipeline as one ESP-IDF static
library. ESP-IDF produces `libhanil_ai.a`; applications use only the public
headers in `include/`.

## Modules

| Header | Module |
|---|---|
| `hanil_ai_types.h` | shared constants, status, timing, and memory reporting |
| `hanil_microphone.h` | INMP441 I2S capture, center crop, gain, placement checks |
| `hanil_quality_gate.h` | clipping, silence, noise/rustle, and signal-quality gate |
| `hanil_frontend.h` | exact 16 kHz, 4 s, 64 x 401 log-Mel preprocessing |
| `hanil_classifier.h` | SRAM INT8 TFLite Micro classifier and decision layer |
| `hanil_pipeline.h` | complete microphone-to-classification orchestration |

## Add to another ESP-IDF application

Copy this directory to `<project>/components/hanil_ai`, then add `hanil_ai` to
the application's component requirements:

```cmake
idf_component_register(SRCS "main.cc" REQUIRES hanil_ai)
```

Use the top-level interface:

```cpp
#include "hanil_pipeline.h"

hanil_pipeline_config_t config = hanil_pipeline_default_config();
hanil_pipeline_t *pipeline = nullptr;
ESP_ERROR_CHECK(hanil_pipeline_create(&config, &pipeline) == HANIL_AI_OK
                    ? ESP_OK : ESP_FAIL);

hanil_pipeline_result_t result{};
hanil_pipeline_capture_and_run(pipeline, &result);
```

The archive is target-specific. Build it with the same ESP-IDF, target,
compiler, TFLite Micro component, and configuration as the final application.
The component source package is the recommended integration method because
ESP-IDF then links all required framework and TFLite dependencies correctly.

