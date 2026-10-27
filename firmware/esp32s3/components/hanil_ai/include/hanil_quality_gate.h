#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HANIL_QG_PASS = 0,
    HANIL_QG_RETRY_CLIPPING,
    HANIL_QG_RETRY_SILENCE,
    HANIL_QG_RETRY_NOISE,
    HANIL_QG_RETRY_LOW_SIGNAL,
} hanil_qg_result_t;

typedef struct {
    float clip_level;
    float clip_frac_max;
    float rms_min;
    float flatness_max;
    float zcr_max;
    float quality_min;
} hanil_qg_config_t;

typedef struct {
    int n_frames;
    double sum_flat;
    double sum_zcr;
    float band_energy[512];
    int cap;
    float sort_scratch[512];
} hanil_qg_accum_t;

typedef struct {
    float clip_frac;
    float rms;
    float flatness;
    float zcr;
    float quality_score;
} hanil_qg_metrics_t;

hanil_qg_config_t hanil_qg_default_config(void);
void hanil_qg_reset(hanil_qg_accum_t *accum);
void hanil_qg_accumulate(hanil_qg_accum_t *accum, const float *frame,
                         int frame_len, const float *power, int n_bins,
                         int sample_rate, int n_fft);
void hanil_qg_scan_pcm(const int16_t *pcm, int n, float clip_level,
                       float *clip_frac_out, float *rms_out);
hanil_qg_result_t hanil_qg_decide(hanil_qg_accum_t *accum,
                                  const hanil_qg_config_t *config,
                                  hanil_qg_metrics_t *metrics);
const char *hanil_qg_reason(hanil_qg_result_t result);

#ifdef __cplusplus
}
#endif

