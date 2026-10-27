#include "quality_gate.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

qg_config_t qg_default_config(void)
{
    qg_config_t config;
    config.clip_level = 0.99f;
    // Calibrated Python pipeline value, retained for comparable saturation checks.
    config.clip_frac_max = 0.0263f;
    config.rms_min = 0.004f;
    // Measured with this C gate's power-flatness definition. In 48 real ICBHI
    // clips max flatness was 0.0302; seeded white noise measured 0.5597.
    config.flatness_max = 0.05f;
    config.zcr_max = 0.10f;
    // The original placeholder (10.0) rejected valid clips. 3.0 retains 47/48
    // calibration clips while rejecting the deliberately poor low-signal case.
    config.quality_min = 3.0f;
    return config;
}

void qg_reset(qg_accum_t *accum)
{
    memset(accum, 0, sizeof(*accum));
    accum->cap = 512;
}

void qg_accumulate(qg_accum_t *accum, const float *frame, int frame_len,
                   const float *power, int n_bins, int sample_rate, int n_fft)
{
    const float bin_hz = (float)sample_rate / (float)n_fft;
    int k0 = (int)(100.0f / bin_hz);
    if (k0 < 1) {
        k0 = 1;
    }

    double log_sum = 0.0;
    double linear_sum = 0.0;
    int count = 0;
    for (int k = k0; k < n_bins; ++k) {
        const float value = power[k] + 1.0e-10f;
        log_sum += log((double)value);
        linear_sum += value;
        ++count;
    }
    if (count > 0) {
        const double geometric = exp(log_sum / count);
        const double arithmetic = linear_sum / count;
        accum->sum_flat += geometric / (arithmetic + 1.0e-12);
    }

    int crossings = 0;
    for (int i = 1; i < frame_len; ++i) {
        if ((frame[i] >= 0.0f) != (frame[i - 1] >= 0.0f)) {
            ++crossings;
        }
    }
    accum->sum_zcr += (double)crossings / (frame_len - 1);

    int k1 = (int)(1000.0f / bin_hz);
    if (k1 >= n_bins) {
        k1 = n_bins - 1;
    }
    double band_energy = 0.0;
    for (int k = k0; k <= k1; ++k) {
        band_energy += power[k];
    }
    if (accum->n_frames < accum->cap) {
        accum->band_energy[accum->n_frames] = (float)band_energy;
    }
    ++accum->n_frames;
}

void qg_scan_pcm(const int16_t *pcm, int n, float clip_level,
                 float *clip_frac_out, float *rms_out)
{
    // Peak normalization intentionally matches phase2_quality_gate.py.
    int peak = 1;
    for (int i = 0; i < n; ++i) {
        int value = pcm[i];
        int magnitude = value < 0 ? -value : value;
        if (magnitude > peak) {
            peak = magnitude;
        }
    }

    double clipped = 0.0;
    double sum_squared = 0.0;
    const float inverse_peak = 1.0f / peak;
    for (int i = 0; i < n; ++i) {
        const float value = pcm[i] * inverse_peak;
        if (fabsf(value) >= clip_level) {
            clipped += 1.0;
        }
        sum_squared += (double)value * value;
    }
    *clip_frac_out = (float)(clipped / n);
    *rms_out = (float)sqrt(sum_squared / n);
}

static int compare_float(const void *left, const void *right)
{
    const float a = *(const float *)left;
    const float b = *(const float *)right;
    return (a > b) - (a < b);
}

qg_result_t qg_decide(qg_accum_t *accum, const qg_config_t *config,
                      qg_metrics_t *metrics)
{
    metrics->flatness = accum->n_frames
                            ? (float)(accum->sum_flat / accum->n_frames)
                            : 0.0f;
    metrics->zcr = accum->n_frames
                       ? (float)(accum->sum_zcr / accum->n_frames)
                       : 0.0f;

    const int frame_count = accum->n_frames < accum->cap
                                ? accum->n_frames
                                : accum->cap;
    metrics->quality_score = 0.0f;
    if (frame_count >= 5) {
        memcpy(accum->sort_scratch, accum->band_energy,
               frame_count * sizeof(float));
        qsort(accum->sort_scratch, frame_count, sizeof(float), compare_float);

        const int q20 = frame_count / 5;
        const int q60 = (frame_count * 6) / 10;
        double quiet = 0.0;
        for (int i = 0; i <= q20; ++i) {
            quiet += accum->sort_scratch[i];
        }
        quiet /= q20 + 1;
        double active = 0.0;
        int active_count = 0;
        for (int i = q60; i < frame_count; ++i) {
            active += accum->sort_scratch[i];
            ++active_count;
        }
        active /= active_count > 0 ? active_count : 1;
        metrics->quality_score =
            10.0f * log10f((float)(active / (quiet + 1.0e-12)) + 1.0e-9f);
    }

    if (metrics->clip_frac > config->clip_frac_max) {
        return QG_RETRY_CLIPPING;
    }
    if (metrics->rms < config->rms_min) {
        return QG_RETRY_SILENCE;
    }
    if (metrics->flatness > config->flatness_max &&
        metrics->zcr > config->zcr_max) {
        return QG_RETRY_NOISE;
    }
    if (metrics->quality_score < config->quality_min) {
        return QG_RETRY_LOW_SIGNAL;
    }
    return QG_PASS;
}

const char *qg_reason(qg_result_t result)
{
    switch (result) {
        case QG_PASS:
            return "pass";
        case QG_RETRY_CLIPPING:
            return "clipping_saturation";
        case QG_RETRY_SILENCE:
            return "silence_no_contact";
        case QG_RETRY_NOISE:
            return "noise_rustle";
        case QG_RETRY_LOW_SIGNAL:
            return "low_signal_quality";
        default:
            return "unknown";
    }
}
