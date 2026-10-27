#include <cinttypes>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hanil_pipeline.h"

namespace {

constexpr char kTag[] = "HANIL_EXAMPLE";

void log_memory(void)
{
    hanil_ai_memory_t memory{};
    hanil_ai_get_memory(&memory);
    ESP_LOGI(kTag,
             "MEMORY internal_free=%u internal_largest=%u internal_total=%u "
             "psram_free=%u psram_total=%u",
             static_cast<unsigned>(memory.internal_free_bytes),
             static_cast<unsigned>(memory.internal_largest_block_bytes),
             static_cast<unsigned>(memory.internal_total_bytes),
             static_cast<unsigned>(memory.psram_free_bytes),
             static_cast<unsigned>(memory.psram_total_bytes));
}

}  // namespace

extern "C" void app_main(void)
{
    hanil_pipeline_config_t config = hanil_pipeline_default_config();
    hanil_pipeline_t *pipeline = nullptr;
    const hanil_ai_status_t create_status =
        hanil_pipeline_create(&config, &pipeline);
    if (create_status != HANIL_AI_OK) {
        ESP_LOGE(kTag, "pipeline create failed: %s",
                 hanil_ai_status_string(create_status));
        return;
    }
    ESP_LOGI(kTag,
             "READY package=hanil_ai model_bytes=%u arena_used=%u "
             "sample_rate=%d audio_seconds=%d",
             static_cast<unsigned>(hanil_pipeline_model_size(pipeline)),
             static_cast<unsigned>(hanil_pipeline_tensor_arena_used(pipeline)),
             HANIL_AI_SAMPLE_RATE, HANIL_AI_AUDIO_SECONDS);
    log_memory();

    while (true) {
        ESP_LOGI(kTag, "Place the chestpiece firmly; capture starts now");
        hanil_pipeline_result_t result{};
        const hanil_ai_status_t status =
            hanil_pipeline_capture_and_run(pipeline, &result);
        if (status != HANIL_AI_OK) {
            ESP_LOGE(kTag, "pipeline failed: %s", hanil_ai_status_string(status));
        } else if (result.microphone_result != HANIL_MIC_CAPTURE_OK) {
            ESP_LOGW(kTag, "RETRY reason=%s raw_rms_dbfs=%.1f",
                     hanil_microphone_capture_reason(result.microphone_result),
                     result.microphone_metrics.raw_rms_dbfs);
        } else if (result.frontend.gate_result != HANIL_QG_PASS) {
            ESP_LOGW(kTag, "RETRY reason=%s quality=%.2f",
                     hanil_qg_reason(result.frontend.gate_result),
                     result.frontend.gate_metrics.quality_score);
        } else if (result.prediction_valid) {
            ESP_LOGI(kTag,
                     "RESULT class2=%s class3=%s crackle=%.6f wheeze=%.6f "
                     "capture_ms=%.2f compute_ms=%.2f inference_ms=%.2f",
                     result.classification.class2,
                     result.classification.class3,
                     result.classification.crackle_probability,
                     result.classification.wheeze_probability,
                     result.timing.capture_ms, result.timing.compute_ms,
                     result.timing.inference_ms);
        }
        log_memory();
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}

