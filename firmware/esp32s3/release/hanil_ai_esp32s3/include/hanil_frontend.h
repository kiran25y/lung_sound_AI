#pragma once

#include "hanil_ai_types.h"
#include "hanil_quality_gate.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hanil_frontend hanil_frontend_t;

typedef struct {
    hanil_qg_result_t gate_result;
    hanil_qg_metrics_t gate_metrics;
    hanil_ai_timing_t timing;
    float feature_mean;
    float feature_stddev;
    uint32_t feature_hash;
} hanil_frontend_result_t;

hanil_ai_status_t hanil_frontend_create(hanil_frontend_t **frontend);
void hanil_frontend_destroy(hanil_frontend_t *frontend);
hanil_ai_status_t hanil_frontend_process(
    hanil_frontend_t *frontend, const int16_t *pcm, size_t pcm_samples,
    float quant_scale, int quant_zero_point, int8_t *quantized_features,
    size_t feature_capacity, const hanil_qg_config_t *quality_config,
    hanil_frontend_result_t *result);

#ifdef __cplusplus
}
#endif

