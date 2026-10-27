#include "esp_log.h"
#include "hanil_pipeline.h"

extern "C" void app_main(void)
{
    hanil_pipeline_config_t config = hanil_pipeline_default_config();
    hanil_pipeline_t *pipeline = nullptr;
    hanil_ai_status_t status = hanil_pipeline_create(&config, &pipeline);
    if (status != HANIL_AI_OK) {
        ESP_LOGE("HANIL_APP", "init failed: %s", hanil_ai_status_string(status));
        return;
    }

    hanil_pipeline_result_t result{};
    status = hanil_pipeline_capture_and_run(pipeline, &result);
    if (status != HANIL_AI_OK) {
        ESP_LOGE("HANIL_APP", "pipeline failed: %s",
                 hanil_ai_status_string(status));
    } else if (!result.prediction_valid) {
        ESP_LOGW("HANIL_APP", "RETRY mic=%s gate=%s",
                 hanil_microphone_capture_reason(result.microphone_result),
                 hanil_qg_reason(result.frontend.gate_result));
    } else {
        ESP_LOGI("HANIL_APP", "class2=%s class3=%s crackle=%.4f wheeze=%.4f",
                 result.classification.class2, result.classification.class3,
                 result.classification.crackle_probability,
                 result.classification.wheeze_probability);
    }
    hanil_pipeline_destroy(pipeline);
}

