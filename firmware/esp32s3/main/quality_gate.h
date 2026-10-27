#pragma once

// On-device quality gate, evaluated before classification.
// The thresholds are calibrated for the exact 16 kHz / 400-point frontend in
// this firmware. Feed the full 201-bin power spectrum for every frame.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QG_PASS = 0,
    QG_RETRY_CLIPPING,
    QG_RETRY_SILENCE,
    QG_RETRY_NOISE,
    QG_RETRY_LOW_SIGNAL,
} qg_result_t;

typedef struct {
    float clip_level;
    float clip_frac_max;
    float rms_min;
    float flatness_max;
    float zcr_max;
    float quality_min;
} qg_config_t;

typedef struct {
    int n_frames;
    double sum_flat;
    double sum_zcr;
    float band_energy[512];
    int cap;
    // Sorting workspace is owned by the gate so callers can place and verify
    // the complete mutable quality-gate state in internal SRAM.
    float sort_scratch[512];
} qg_accum_t;

typedef struct {
    float clip_frac;
    float rms;
    float flatness;
    float zcr;
    float quality_score;
} qg_metrics_t;

qg_config_t qg_default_config(void);
void qg_reset(qg_accum_t *accum);
void qg_accumulate(qg_accum_t *accum, const float *frame, int frame_len,
                   const float *power, int n_bins, int sample_rate, int n_fft);
void qg_scan_pcm(const int16_t *pcm, int n, float clip_level,
                 float *clip_frac_out, float *rms_out);
qg_result_t qg_decide(qg_accum_t *accum, const qg_config_t *config,
                      qg_metrics_t *metrics);
const char *qg_reason(qg_result_t result);

#ifdef __cplusplus
}
#endif
