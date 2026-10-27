#include "quality_gate.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

qg_config_t qg_default_config(void)
{
    qg_config_t c;
    c.clip_level    = 0.99f;
    c.clip_frac_max = 0.01f;
    c.rms_min       = 0.004f;
    c.flatness_max  = 0.50f;
    c.zcr_max       = 0.15f;
    c.quality_min   = 10.0f;   // relative energy-ratio score (tune vs Python gate)
    return c;
}

void qg_reset(qg_accum_t *a) { memset(a, 0, sizeof(*a)); a->cap = 512; }

void qg_accumulate(qg_accum_t *a, const float *frame, int frame_len,
                   const float *power, int n_bins, int sample_rate, int n_fft)
{
    // --- spectral flatness over the >100 Hz band (geometric mean / arithmetic mean) ---
    float bin_hz = (float)sample_rate / (float)n_fft;
    int k0 = (int)(100.0f / bin_hz); if (k0 < 1) k0 = 1;
    double log_sum = 0.0, lin_sum = 0.0; int cnt = 0;
    for (int k = k0; k < n_bins; ++k) {
        float p = power[k] + 1e-10f;
        log_sum += log((double)p); lin_sum += (double)p; cnt++;
    }
    if (cnt > 0) {
        double geo = exp(log_sum / cnt), ari = lin_sum / cnt;
        a->sum_flat += geo / (ari + 1e-12);
    }

    // --- zero-crossing rate of the (windowed) frame ---
    int zc = 0;
    for (int i = 1; i < frame_len; ++i)
        if ((frame[i] >= 0.0f) != (frame[i - 1] >= 0.0f)) zc++;
    a->sum_zcr += (double)zc / (double)(frame_len - 1);

    // --- breath-band (100–1000 Hz) energy for the signal-quality ratio ---
    int k1 = (int)(1000.0f / bin_hz); if (k1 >= n_bins) k1 = n_bins - 1;
    double be = 0.0;
    for (int k = k0; k <= k1; ++k) be += power[k];
    if (a->n_frames < a->cap) a->band_energy[a->n_frames] = (float)be;
    a->n_frames++;
}

void qg_scan_pcm(const int16_t *pcm, int n, float *clip_frac_out, float *rms_out)
{
    // peak-normalize (matches feature_config), then measure clipping + RMS
    int peak = 1;
    for (int i = 0; i < n; ++i) { int v = pcm[i] < 0 ? -pcm[i] : pcm[i]; if (v > peak) peak = v; }
    double clipped = 0.0, sumsq = 0.0;
    float inv = 1.0f / (float)peak;
    for (int i = 0; i < n; ++i) {
        float x = pcm[i] * inv;
        if (fabsf(x) >= 0.99f) clipped += 1.0;
        sumsq += (double)x * x;
    }
    *clip_frac_out = (float)(clipped / n);
    *rms_out = (float)sqrt(sumsq / n);
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

qg_result_t qg_decide(const qg_accum_t *a, float clip_frac, float rms,
                      const qg_config_t *cfg, float *quality_score_out)
{
    float flat = a->n_frames ? (float)(a->sum_flat / a->n_frames) : 0.0f;
    float zcr  = a->n_frames ? (float)(a->sum_zcr  / a->n_frames) : 0.0f;

    // signal-quality: energy in the loud (breath-active) frames vs the quiet frames
    int nf = a->n_frames < a->cap ? a->n_frames : a->cap;
    float score = 0.0f;
    if (nf >= 5) {
        static float tmp[512];
        memcpy(tmp, a->band_energy, nf * sizeof(float));
        qsort(tmp, nf, sizeof(float), cmp_float);
        int q20 = nf / 5, q60 = (nf * 6) / 10;       // 20th & 60th percentiles
        double quiet = 0; for (int i = 0; i < q20 + 1; ++i) quiet += tmp[i];
        quiet /= (q20 + 1);
        double active = 0; int na = 0; for (int i = q60; i < nf; ++i) { active += tmp[i]; na++; }
        active /= (na > 0 ? na : 1);
        score = 10.0f * log10f((float)(active / (quiet + 1e-12)) + 1e-9f);
    }
    if (quality_score_out) *quality_score_out = score;

    if (clip_frac > cfg->clip_frac_max)          return QG_RETRY_CLIPPING;
    if (rms < cfg->rms_min)                       return QG_RETRY_SILENCE;
    if (flat > cfg->flatness_max && zcr > cfg->zcr_max) return QG_RETRY_NOISE;
    if (score < cfg->quality_min)                 return QG_RETRY_LOW_SIGNAL;
    return QG_PASS;
}

const char *qg_reason(qg_result_t r)
{
    switch (r) {
        case QG_PASS:             return "pass";
        case QG_RETRY_CLIPPING:   return "clipping/saturation";
        case QG_RETRY_SILENCE:    return "silence/no-contact";
        case QG_RETRY_NOISE:      return "noise/rustle";
        case QG_RETRY_LOW_SIGNAL: return "low signal quality";
        default:                  return "unknown";
    }
}
