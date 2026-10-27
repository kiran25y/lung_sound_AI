#pragma once
// On-device quality gate — noise check + signal-quality check BEFORE classification.
// Ports the validated checks from phase2_quality_gate.py. Reuses the STFT power spectra your
// mel front end already computes (feed one frame at a time), so it adds almost no cost.
//
// Flow:  capture PCM -> (front end computes per-frame power) -> gate -> PASS ? classify : RETRY
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    QG_PASS = 0,
    QG_RETRY_CLIPPING,     // saturated input
    QG_RETRY_SILENCE,      // no signal (mic not on skin / too quiet)
    QG_RETRY_NOISE,        // broadband noise / rustle
    QG_RETRY_LOW_SIGNAL,   // lung signal too weak vs background
} qg_result_t;

typedef struct {
    float clip_level;      // sample considered clipped at |x|>=this (norm.)   default 0.99
    float clip_frac_max;   // reject if clipped fraction >                     default 0.01
    float rms_min;         // reject if normalized RMS <  (silence)            default 0.004
    float flatness_max;    // noise if spectral flatness >                     default 0.50
    float zcr_max;         // noise if zero-crossing rate >                    default 0.15
    float quality_min;     // reject if signal-quality score <  (relative)     default 10.0
} qg_config_t;

qg_config_t qg_default_config(void);

typedef struct {
    int    n_frames;
    double sum_flat, sum_zcr;
    float  band_energy[512];   // per-frame breath-band energy (401 frames used)
    int    cap;
} qg_accum_t;

void qg_reset(qg_accum_t *a);

// Call once per STFT frame, from inside your front-end loop:
//   frame[]  = the windowed time-domain frame (length frame_len)
//   power[]  = its power spectrum (length n_bins = n_fft/2+1)
void qg_accumulate(qg_accum_t *a, const float *frame, int frame_len,
                   const float *power, int n_bins, int sample_rate, int n_fft);

// Call once on the raw PCM (time-domain clipping + silence).
void qg_scan_pcm(const int16_t *pcm, int n, float *clip_frac_out, float *rms_out);

// Final decision. Also returns the relative signal-quality score (NOT calibrated dB).
qg_result_t qg_decide(const qg_accum_t *a, float clip_frac, float rms,
                      const qg_config_t *cfg, float *quality_score_out);

const char *qg_reason(qg_result_t r);
