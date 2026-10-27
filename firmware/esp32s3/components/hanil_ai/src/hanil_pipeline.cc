#include "hanil_pipeline.h"

#include <cstring>
#include <new>

#include "esp_heap_caps.h"
#include "esp_timer.h"

struct hanil_pipeline {
    hanil_pipeline_config_t config{};
    hanil_classifier_t *classifier = nullptr;
    hanil_frontend_t *frontend = nullptr;
    hanil_microphone_t *microphone = nullptr;
    int16_t *pcm = nullptr;
};

extern "C" hanil_pipeline_config_t hanil_pipeline_default_config(void)
{
    hanil_pipeline_config_t config{};
    config.classifier = hanil_classifier_default_config();
    config.microphone = hanil_microphone_default_config();
    config.quality_gate = hanil_qg_default_config();
    config.initialize_microphone = true;
    return config;
}

extern "C" hanil_ai_status_t hanil_pipeline_create(
    const hanil_pipeline_config_t *config, hanil_pipeline_t **pipeline_out)
{
    if (!pipeline_out) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    *pipeline_out = nullptr;
    const hanil_pipeline_config_t selected =
        config ? *config : hanil_pipeline_default_config();
    auto *pipeline = new (std::nothrow) hanil_pipeline_t();
    if (!pipeline) {
        return HANIL_AI_OUT_OF_MEMORY;
    }
    pipeline->config = selected;
    hanil_ai_status_t status = hanil_classifier_create(
        &selected.classifier, &pipeline->classifier);
    if (status != HANIL_AI_OK) {
        hanil_pipeline_destroy(pipeline);
        return status;
    }
    status = hanil_frontend_create(&pipeline->frontend);
    if (status != HANIL_AI_OK) {
        hanil_pipeline_destroy(pipeline);
        return status;
    }
    pipeline->pcm = static_cast<int16_t *>(heap_caps_malloc(
        HANIL_AI_AUDIO_SAMPLES * sizeof(int16_t),
        MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM));
    if (!pipeline->pcm) {
        hanil_pipeline_destroy(pipeline);
        return HANIL_AI_OUT_OF_MEMORY;
    }
    if (selected.initialize_microphone) {
        status = hanil_microphone_create(&selected.microphone,
                                          &pipeline->microphone);
        if (status != HANIL_AI_OK) {
            hanil_pipeline_destroy(pipeline);
            return status;
        }
    }
    *pipeline_out = pipeline;
    return HANIL_AI_OK;
}

extern "C" void hanil_pipeline_destroy(hanil_pipeline_t *pipeline)
{
    if (!pipeline) {
        return;
    }
    hanil_microphone_destroy(pipeline->microphone);
    hanil_frontend_destroy(pipeline->frontend);
    hanil_classifier_destroy(pipeline->classifier);
    heap_caps_free(pipeline->pcm);
    delete pipeline;
}

extern "C" hanil_ai_status_t hanil_pipeline_run_pcm(
    hanil_pipeline_t *pipeline, const int16_t *pcm, size_t pcm_samples,
    hanil_pipeline_result_t *result)
{
    if (!pipeline || !pcm || pcm_samples != HANIL_AI_AUDIO_SAMPLES || !result) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    std::memset(result, 0, sizeof(*result));
    result->microphone_result = HANIL_MIC_CAPTURE_OK;
    const int64_t start = esp_timer_get_time();
    size_t feature_count = 0;
    int8_t *features = hanil_classifier_input_buffer(
        pipeline->classifier, &feature_count);
    if (!features || feature_count != HANIL_AI_FEATURE_COUNT) {
        result->status = HANIL_AI_MODEL_ERROR;
        return result->status;
    }
    result->status = hanil_frontend_process(
        pipeline->frontend, pcm, pcm_samples,
        hanil_classifier_input_scale(pipeline->classifier),
        hanil_classifier_input_zero_point(pipeline->classifier), features,
        feature_count, &pipeline->config.quality_gate, &result->frontend);
    if (result->status != HANIL_AI_OK) {
        return result->status;
    }
    result->timing = result->frontend.timing;
    if (result->frontend.gate_result != HANIL_QG_PASS) {
        result->timing.end_to_end_ms =
            static_cast<float>(esp_timer_get_time() - start) / 1000.0f;
        return HANIL_AI_OK;
    }
    result->status = hanil_classifier_predict(
        pipeline->classifier, features, feature_count, &result->classification);
    if (result->status != HANIL_AI_OK) {
        return result->status;
    }
    result->prediction_valid = true;
    result->timing.inference_ms = result->classification.inference_ms;
    result->timing.compute_ms = result->frontend.timing.compute_ms +
                                result->classification.inference_ms;
    result->timing.end_to_end_ms =
        static_cast<float>(esp_timer_get_time() - start) / 1000.0f;
    return HANIL_AI_OK;
}

extern "C" hanil_ai_status_t hanil_pipeline_capture_and_run(
    hanil_pipeline_t *pipeline, hanil_pipeline_result_t *result)
{
    if (!pipeline || !result) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    std::memset(result, 0, sizeof(*result));
    if (!pipeline->microphone) {
        result->status = HANIL_AI_NOT_READY;
        return result->status;
    }
    const int64_t start = esp_timer_get_time();
    result->microphone_result = hanil_microphone_capture(
        pipeline->microphone, pipeline->pcm, HANIL_AI_AUDIO_SAMPLES,
        &result->microphone_metrics);
    result->timing.capture_ms = result->microphone_metrics.capture_ms;
    if (result->microphone_result != HANIL_MIC_CAPTURE_OK) {
        result->status = HANIL_AI_OK;
        result->timing.end_to_end_ms =
            static_cast<float>(esp_timer_get_time() - start) / 1000.0f;
        return HANIL_AI_OK;
    }

    hanil_pipeline_result_t inference{};
    const hanil_ai_status_t status = hanil_pipeline_run_pcm(
        pipeline, pipeline->pcm, HANIL_AI_AUDIO_SAMPLES, &inference);
    const hanil_microphone_metrics_t microphone_metrics =
        result->microphone_metrics;
    *result = inference;
    result->microphone_result = HANIL_MIC_CAPTURE_OK;
    result->microphone_metrics = microphone_metrics;
    result->timing.capture_ms = microphone_metrics.capture_ms;
    result->timing.end_to_end_ms =
        static_cast<float>(esp_timer_get_time() - start) / 1000.0f;
    return status;
}

extern "C" size_t hanil_pipeline_model_size(
    const hanil_pipeline_t *pipeline)
{
    return pipeline ? hanil_classifier_model_size(pipeline->classifier) : 0;
}

extern "C" size_t hanil_pipeline_tensor_arena_used(
    const hanil_pipeline_t *pipeline)
{
    return pipeline ? hanil_classifier_arena_used(pipeline->classifier) : 0;
}

