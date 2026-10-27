// Classification-only SRAM acceptance test for the compact INT8 student.
//
// The model can either be memory-mapped from flash or copied into internal
// SRAM at boot. All mutable TFLM inference state, including input/output
// tensors and kernel scratch buffers, is forced into the ESP32-S3's internal
// byte-addressable SRAM. No PSRAM allocation is made by this program.
//
// USB protocol (little endian):
//   "FEA1" + uint32(25664) + quantized INT8 [1,64,401,1]
#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstring>

#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_memory_utils.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const uint8_t model_start[] asm("_binary_sram_student_int8_tflite_start");
extern const uint8_t model_end[] asm("_binary_sram_student_int8_tflite_end");

namespace {

constexpr char kTag[] = "CLASSIFIER_SRAM";
// Hardware-measured optimal use is 117,132 bytes. Keep a 13,940-byte margin
// while returning the rest of internal SRAM to the future frontend and gate.
constexpr size_t kArenaCapacity = 128 * 1024;
constexpr size_t kFeatureBytes = 64 * 401;
constexpr float kCrackleThreshold = 0.49f;
constexpr float kWheezeThreshold = 0.28f;

struct CommandHeader {
    char magic[4];
    uint32_t payload_bytes;
};

uint32_t fnv1a(const int8_t *data, size_t bytes)
{
    uint32_t value = 2166136261U;
    for (size_t i = 0; i < bytes; ++i) {
        value ^= static_cast<uint8_t>(data[i]);
        value *= 16777619U;
    }
    return value;
}

bool read_exact(void *destination, size_t bytes)
{
    auto *dst = static_cast<uint8_t *>(destination);
    size_t received = 0;
    while (received < bytes) {
        const uint32_t request = static_cast<uint32_t>(
            std::min<size_t>(bytes - received, 4096));
        const int count = usb_serial_jtag_read_bytes(
            dst + received, request, pdMS_TO_TICKS(5000));
        if (count <= 0) {
            ESP_LOGE(kTag, "receive timeout (%u/%u bytes)",
                     static_cast<unsigned>(received), static_cast<unsigned>(bytes));
            return false;
        }
        received += static_cast<size_t>(count);
    }
    return true;
}

bool discard_bytes(size_t bytes)
{
    uint8_t scratch[128];
    while (bytes > 0) {
        const size_t chunk = std::min(bytes, sizeof(scratch));
        if (!read_exact(scratch, chunk)) {
            return false;
        }
        bytes -= chunk;
    }
    return true;
}

void log_internal_heap(const char *stage)
{
    constexpr uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    ESP_LOGI(kTag,
             "SRAM stage=%s free=%u largest=%u minimum_free=%u total=%u",
             stage,
             static_cast<unsigned>(heap_caps_get_free_size(caps)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(caps)),
             static_cast<unsigned>(heap_caps_get_minimum_free_size(caps)),
             static_cast<unsigned>(heap_caps_get_total_size(caps)));
}

void run_prediction(tflite::MicroInterpreter &interpreter,
                    TfLiteTensor *input, TfLiteTensor *output)
{
    const uint32_t input_hash = fnv1a(input->data.int8, input->bytes);
    const int64_t start_us = esp_timer_get_time();
    if (interpreter.Invoke() != kTfLiteOk) {
        ESP_LOGE(kTag, "Invoke failed");
        return;
    }
    const int64_t latency_us = esp_timer_get_time() - start_us;

    const float output_scale = output->params.scale;
    const int output_zp = output->params.zero_point;
    const int crackle_q = static_cast<int>(output->data.int8[0]);
    const int wheeze_q = static_cast<int>(output->data.int8[1]);
    const float crackle_logit = (crackle_q - output_zp) * output_scale;
    const float wheeze_logit = (wheeze_q - output_zp) * output_scale;
    const float p_crackle = 1.0f / (1.0f + expf(-crackle_logit));
    const float p_wheeze = 1.0f / (1.0f + expf(-wheeze_logit));
    const float p_abnormal = 1.0f - (1.0f - p_crackle) * (1.0f - p_wheeze);
    const bool crackle = p_crackle >= kCrackleThreshold;
    const bool wheeze = p_wheeze >= kWheezeThreshold;
    const char *class2 = crackle || wheeze ? "abnormal" : "normal";
    const char *class3 = "normal";
    if (crackle && wheeze) {
        class3 = p_crackle >= p_wheeze ? "crackle" : "wheeze";
    } else if (crackle) {
        class3 = "crackle";
    } else if (wheeze) {
        class3 = "wheeze";
    }

    ESP_LOGI(kTag,
             "RESULT source=host_features p_crackle=%.6f p_wheeze=%.6f "
             "p_abnormal=%.6f class2=%s class3=%s latency_ms=%.3f "
             "latency_us=%" PRId64 " qhash=%08" PRIx32 " out_q=[%d,%d]",
             p_crackle, p_wheeze, p_abnormal, class2, class3,
             latency_us / 1000.0f, latency_us, input_hash, crackle_q, wheeze_q);
}

}  // namespace

extern "C" void app_main(void)
{
    log_internal_heap("boot");

    const size_t model_bytes = static_cast<size_t>(model_end - model_start);
    ESP_LOGI(kTag, "MODEL_STORAGE bytes=%u kib=%.2f in_flash_drom=%s",
             static_cast<unsigned>(model_bytes), model_bytes / 1024.0f,
             esp_ptr_in_drom(model_start) ? "yes" : "no");

    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, kArenaCapacity, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!arena) {
        ESP_LOGE(kTag, "internal tensor arena allocation failed: %u bytes",
                 static_cast<unsigned>(kArenaCapacity));
        log_internal_heap("arena_allocation_failed");
        return;
    }
    ESP_LOGI(kTag, "ARENA capacity=%u address=%p internal=%s",
             static_cast<unsigned>(kArenaCapacity), arena,
             esp_ptr_internal(arena) ? "yes" : "no");
    log_internal_heap("arena_allocated");

    const uint8_t *model_data = model_start;
#if defined(HANIL_CLASSIFIER_COPY_MODEL_TO_SRAM)
    auto *model_copy = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, model_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!model_copy) {
        ESP_LOGE(kTag, "internal model-copy allocation failed: %u bytes",
                 static_cast<unsigned>(model_bytes));
        log_internal_heap("model_copy_failed");
        return;
    }
    std::memcpy(model_copy, model_start, model_bytes);
    model_data = model_copy;
    ESP_LOGI(kTag,
             "MODEL_RUNTIME bytes=%u address=%p location=internal_sram internal=%s",
             static_cast<unsigned>(model_bytes), model_data,
             esp_ptr_internal(model_data) ? "yes" : "no");
    log_internal_heap("model_copied");
#else
    ESP_LOGI(kTag,
             "MODEL_RUNTIME bytes=%u address=%p location=flash_drom internal=no",
             static_cast<unsigned>(model_bytes), model_data);
#endif

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(kTag, "TFLite schema mismatch model=%d runtime=%d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();
    resolver.AddFullyConnected();

    static tflite::MicroInterpreter interpreter(
        model, resolver, arena, kArenaCapacity);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(kTag, "AllocateTensors failed with %u-byte internal arena",
                 static_cast<unsigned>(kArenaCapacity));
        return;
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);
    if (input->type != kTfLiteInt8 || input->bytes != kFeatureBytes ||
        output->type != kTfLiteInt8 || output->bytes < 2) {
        ESP_LOGE(kTag,
                 "unexpected tensor contract input_type=%d input_bytes=%u "
                 "output_type=%d output_bytes=%u",
                 input->type, static_cast<unsigned>(input->bytes), output->type,
                 static_cast<unsigned>(output->bytes));
        return;
    }

    const size_t arena_used = interpreter.arena_used_bytes();
    ESP_LOGI(kTag,
             "MODEL input=[1,64,401,1] input_scale=%.9g input_zp=%d "
             "output=[1,2] output_scale=%.9g output_zp=%d",
             input->params.scale, static_cast<int>(input->params.zero_point),
             output->params.scale, static_cast<int>(output->params.zero_point));
    ESP_LOGI(kTag,
             "CLASSIFIER_MEMORY arena_used=%u arena_used_kib=%.2f "
             "arena_capacity=%u internal=%s model_runtime=%s psram_allocations=0",
             static_cast<unsigned>(arena_used), arena_used / 1024.0f,
             static_cast<unsigned>(kArenaCapacity),
             esp_ptr_internal(arena) ? "yes" : "no",
             esp_ptr_internal(model_data) ? "internal_sram" : "flash_drom");
    log_internal_heap("tensors_ready");

    usb_serial_jtag_driver_config_t usb_config =
        USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_config.rx_buffer_size = 8192;
    usb_config.tx_buffer_size = 1024;
    if (!usb_serial_jtag_is_driver_installed()) {
        ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_config));
    }

    while (true) {
        ESP_LOGI(kTag,
                 "READY classification_only=yes internal_sram=yes "
                 "PCM1=0_bytes FEA1=%u_bytes thresholds=[%.2f,%.2f]",
                 static_cast<unsigned>(kFeatureBytes),
                 kCrackleThreshold, kWheezeThreshold);

        CommandHeader header{};
        if (!read_exact(&header, sizeof(header))) {
            continue;
        }
        if (std::memcmp(header.magic, "FEA1", 4) != 0) {
            ESP_LOGE(kTag, "unsupported command %.4s length=%u (use FEA1)",
                     header.magic, static_cast<unsigned>(header.payload_bytes));
            discard_bytes(header.payload_bytes);
            continue;
        }
        if (header.payload_bytes != kFeatureBytes) {
            ESP_LOGE(kTag, "FEA1 length %u, expected %u",
                     static_cast<unsigned>(header.payload_bytes),
                     static_cast<unsigned>(kFeatureBytes));
            discard_bytes(header.payload_bytes);
            continue;
        }
        if (!read_exact(input->data.int8, kFeatureBytes)) {
            continue;
        }
        ESP_LOGI(kTag, "RECEIVED source=host_features bytes=%u qhash=%08" PRIx32,
                 static_cast<unsigned>(header.payload_bytes),
                 fnv1a(input->data.int8, input->bytes));
        run_prediction(interpreter, input, output);
    }
}
