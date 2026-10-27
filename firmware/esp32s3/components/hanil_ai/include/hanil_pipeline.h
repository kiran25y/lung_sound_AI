#pragma once

#include "hanil_ai_types.h"
#include "hanil_classifier.h"
#include "hanil_frontend.h"
#include "hanil_microphone.h"
#include "hanil_quality_gate.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hanil_pipeline hanil_pipeline_t;

typedef struct {
    hanil_classifier_config_t classifier;
    hanil_microphone_config_t microphone;
    hanil_qg_config_t quality_gate;
    bool initialize_microphone;
} hanil_pipeline_config_t;

typedef struct {
    hanil_ai_status_t status;
    hanil_microphone_capture_result_t microphone_result;
    hanil_microphone_metrics_t microphone_metrics;
    hanil_frontend_result_t frontend;
    hanil_classification_t classification;
    bool prediction_valid;
    hanil_ai_timing_t timing;
} hanil_pipeline_result_t;

hanil_pipeline_config_t hanil_pipeline_default_config(void);
hanil_ai_status_t hanil_pipeline_create(const hanil_pipeline_config_t *config,
                                        hanil_pipeline_t **pipeline);
void hanil_pipeline_destroy(hanil_pipeline_t *pipeline);
hanil_ai_status_t hanil_pipeline_run_pcm(hanil_pipeline_t *pipeline,
                                         const int16_t *pcm,
                                         size_t pcm_samples,
                                         hanil_pipeline_result_t *result);
hanil_ai_status_t hanil_pipeline_capture_and_run(
    hanil_pipeline_t *pipeline, hanil_pipeline_result_t *result);
size_t hanil_pipeline_model_size(const hanil_pipeline_t *pipeline);
size_t hanil_pipeline_tensor_arena_used(const hanil_pipeline_t *pipeline);

#ifdef __cplusplus
}
#endif

