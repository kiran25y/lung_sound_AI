#pragma once

#include "hanil_ai_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hanil_microphone hanil_microphone_t;

typedef enum {
    HANIL_MIC_CAPTURE_OK = 0,
    HANIL_MIC_CAPTURE_INVALID,
    HANIL_MIC_CAPTURE_POOR_CONTACT,
    HANIL_MIC_CAPTURE_TOO_LOUD,
} hanil_microphone_capture_result_t;

typedef struct {
    int sample_rate;
    int bclk_gpio;
    int ws_gpio;
    int data_gpio;
    int record_seconds;
    int analysis_seconds;
    float target_pcm_peak;
    float max_digital_gain;
    float minimum_raw_rms_dbfs;
    float placement_minimum_dbfs;
    float placement_maximum_dbfs;
} hanil_microphone_config_t;

typedef struct {
    float raw_peak_dbfs;
    float raw_rms_dbfs;
    float raw_clip_fraction;
    float dc_offset;
    float digital_gain;
    float pcm_rms;
    int32_t pcm_peak;
    float capture_ms;
} hanil_microphone_metrics_t;

hanil_microphone_config_t hanil_microphone_default_config(void);
hanil_ai_status_t hanil_microphone_create(
    const hanil_microphone_config_t *config, hanil_microphone_t **microphone);
void hanil_microphone_destroy(hanil_microphone_t *microphone);
hanil_microphone_capture_result_t hanil_microphone_capture(
    hanil_microphone_t *microphone, int16_t *pcm, size_t pcm_samples,
    hanil_microphone_metrics_t *metrics);
const char *hanil_microphone_capture_reason(
    hanil_microphone_capture_result_t result);

#ifdef __cplusplus
}
#endif

