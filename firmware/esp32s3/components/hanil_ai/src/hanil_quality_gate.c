#include "hanil_quality_gate.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

hanil_qg_config_t hanil_qg_default_config(void)
{
    hanil_qg_config_t config;
    config.clip_level = 0.99f;
    config.clip_frac_max = 0.0263f;
    config.rms_min = 0.004f;
    config.flatness_max = 0.05f;
    config.zcr_max = 0.10f;
    config.quality_min = 3.0f;
    return config;
}

void hanil_qg_reset(hanil_qg_accum_t *accum)
{
    memset(accum, 0, sizeof(*accum));
    accum->cap = 512;
}

void hanil_qg_accumulate(hanil_qg_accum_t *accum, const float *frame,
                         int frame_len, const float *power, int n_bins,
                         int sample_rate, int n_fft)
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

void hanil_qg_scan_pcm(const int16_t *pcm, int n, float clip_level,
                       float *clip_frac_out, float *rms_out)
{
    int peak = 1;
    for (int i = 0; i < n; ++i) {
        const int value = pcm[i];
        const int magnitude = value < 0 ? -value : value;
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

hanil_qg_result_t hanil_qg_decide(hanil_qg_accum_t *accum,
                                  const hanil_qg_config_t *config,
                                  hanil_qg_metrics_t *metrics)
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
        return HANIL_QG_RETRY_CLIPPING;
    }
    if (metrics->rms < config->rms_min) {
        return HANIL_QG_RETRY_SILENCE;
    }
    if (metrics->flatness > config->flatness_max &&
        metrics->zcr > config->zcr_max) {
        return HANIL_QG_RETRY_NOISE;
    }
    if (metrics->quality_score < config->quality_min) {
        return HANIL_QG_RETRY_LOW_SIGNAL;
    }
    return HANIL_QG_PASS;
}

const char *hanil_qg_reason(hanil_qg_result_t result)
{
    switch (result) {
        case HANIL_QG_PASS:
            return "pass";
        case HANIL_QG_RETRY_CLIPPING:
            return "clipping_saturation";
        case HANIL_QG_RETRY_SILENCE:
            return "silence_no_contact";
        case HANIL_QG_RETRY_NOISE:
            return "noise_rustle";
        case HANIL_QG_RETRY_LOW_SIGNAL:
            return "low_signal_quality";
        default:
            return "unknown";
    }
}

