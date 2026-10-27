#include "esp_log.h"
#include "hanil_pipeline.h"

extern "C" void app_main(void)
{
    hanil_pipeline_config_t config = hanil_pipeline_default_config();
    config.initialize_microphone = false;
    hanil_pipeline_t *pipeline = nullptr;
    const hanil_ai_status_t status = hanil_pipeline_create(&config, &pipeline);
    ESP_LOGI("HANIL_SMOKE", "status=%s model=%u arena=%u",
             hanil_ai_status_string(status),
             static_cast<unsigned>(hanil_pipeline_model_size(pipeline)),
             static_cast<unsigned>(hanil_pipeline_tensor_arena_used(pipeline)));
    hanil_pipeline_destroy(pipeline);
}

