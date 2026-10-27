#include "hanil_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "signal/src/rfft.h"

extern const uint8_t hanil_mel_fb_start[] asm("_binary_mel_fb_npy_start");
extern const uint8_t hanil_mel_fb_end[] asm("_binary_mel_fb_npy_end");
extern const uint8_t hanil_window_start[] asm("_binary_window_npy_start");
extern const uint8_t hanil_window_end[] asm("_binary_window_npy_end");

namespace {

constexpr int kFft = 400;
constexpr int kHop = 160;
constexpr int kPad = kFft / 2;
constexpr int kFreqBins = kFft / 2 + 1;
constexpr float kLogOffset = 1.0e-6f;
constexpr float kNormEps = 1.0e-5f;

const float *npy_f32(const uint8_t *start, const uint8_t *end,
                     size_t expected_values)
{
    const size_t bytes = static_cast<size_t>(end - start);
    if (bytes < 16 || std::memcmp(start, "\x93NUMPY", 6) != 0 || start[6] != 1) {
        return nullptr;
    }
    const size_t header_bytes = static_cast<size_t>(start[8]) |
                                (static_cast<size_t>(start[9]) << 8);
    const size_t data_offset = 10 + header_bytes;
    if (data_offset + expected_values * sizeof(float) > bytes ||
        (data_offset & 3U) != 0) {
        return nullptr;
    }
    return reinterpret_cast<const float *>(start + data_offset);
}

void *internal_alloc(size_t bytes)
{
    return heap_caps_malloc(bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
}

void *psram_alloc(size_t bytes)
{
    return heap_caps_malloc(bytes, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
}

int reflected_index(int index)
{
    if (index < 0) {
        return -index;
    }
    if (index >= HANIL_AI_AUDIO_SAMPLES) {
        return 2 * HANIL_AI_AUDIO_SAMPLES - 2 - index;
    }
    return index;
}

uint32_t fnv1a(const int8_t *data, size_t bytes)
{
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<uint8_t>(data[i]);
        hash *= 16777619u;
    }
    return hash;
}

}  // namespace

struct hanil_frontend {
    const float *mel_fb = nullptr;
    const float *window = nullptr;
    float *logmel = nullptr;
    float *frame = nullptr;
    float *power = nullptr;
    Complex<float> *spectrum = nullptr;
    void *rfft_storage = nullptr;
    void *rfft_state = nullptr;
    hanil_qg_accum_t *quality_accum = nullptr;
    int max_mel_bin = 0;
};

extern "C" hanil_ai_status_t hanil_frontend_create(
    hanil_frontend_t **frontend_out)
{
    if (!frontend_out) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    *frontend_out = nullptr;
    auto *frontend = new (std::nothrow) hanil_frontend_t();
    if (!frontend) {
        return HANIL_AI_OUT_OF_MEMORY;
    }
    frontend->mel_fb = npy_f32(hanil_mel_fb_start, hanil_mel_fb_end,
                               HANIL_AI_MEL_BANDS * kFreqBins);
    frontend->window = npy_f32(hanil_window_start, hanil_window_end, kFft);
    frontend->logmel = static_cast<float *>(
        psram_alloc(HANIL_AI_FEATURE_COUNT * sizeof(float)));
    frontend->quality_accum = static_cast<hanil_qg_accum_t *>(
        heap_caps_aligned_alloc(16, sizeof(hanil_qg_accum_t),
                                MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
    frontend->frame = static_cast<float *>(internal_alloc(kFft * sizeof(float)));
    frontend->power =
        static_cast<float *>(internal_alloc(kFreqBins * sizeof(float)));
    frontend->spectrum = static_cast<Complex<float> *>(
        internal_alloc(kFreqBins * sizeof(Complex<float>)));
    const size_t rfft_bytes = tflm_signal::RfftFloatGetNeededMemory(kFft);
    frontend->rfft_storage = internal_alloc(rfft_bytes);
    if (!frontend->mel_fb || !frontend->window || !frontend->logmel ||
        !frontend->quality_accum || !frontend->frame || !frontend->power ||
        !frontend->spectrum || !frontend->rfft_storage) {
        hanil_frontend_destroy(frontend);
        return HANIL_AI_OUT_OF_MEMORY;
    }
    frontend->rfft_state = tflm_signal::RfftFloatInit(
        kFft, frontend->rfft_storage, rfft_bytes);
    if (!frontend->rfft_state) {
        hanil_frontend_destroy(frontend);
        return HANIL_AI_FRONTEND_ERROR;
    }
    for (int k = 0; k < kFreqBins; ++k) {
        for (int m = 0; m < HANIL_AI_MEL_BANDS; ++m) {
            if (frontend->mel_fb[m * kFreqBins + k] != 0.0f) {
                frontend->max_mel_bin = std::max(frontend->max_mel_bin, k);
            }
        }
    }
    *frontend_out = frontend;
    return HANIL_AI_OK;
}

extern "C" void hanil_frontend_destroy(hanil_frontend_t *frontend)
{
    if (!frontend) {
        return;
    }
    heap_caps_free(frontend->logmel);
    heap_caps_free(frontend->quality_accum);
    heap_caps_free(frontend->frame);
    heap_caps_free(frontend->power);
    heap_caps_free(frontend->spectrum);
    heap_caps_free(frontend->rfft_storage);
    delete frontend;
}

extern "C" hanil_ai_status_t hanil_frontend_process(
    hanil_frontend_t *frontend, const int16_t *pcm, size_t pcm_samples,
    float quant_scale, int quant_zero_point, int8_t *quantized_features,
    size_t feature_capacity, const hanil_qg_config_t *quality_config,
    hanil_frontend_result_t *result)
{
    if (!frontend || !pcm || pcm_samples != HANIL_AI_AUDIO_SAMPLES ||
        quant_scale <= 0.0f || !quantized_features ||
        feature_capacity < HANIL_AI_FEATURE_COUNT || !quality_config || !result) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    std::memset(result, 0, sizeof(*result));
    double sum = 0.0;
    double sum_sq = 0.0;
    const int64_t start = esp_timer_get_time();
    int64_t gate_us = 0;
    int64_t gate_start = esp_timer_get_time();
    hanil_qg_reset(frontend->quality_accum);
    gate_us += esp_timer_get_time() - gate_start;

    for (int t = 0; t < HANIL_AI_MEL_FRAMES; ++t) {
        for (int n = 0; n < kFft; ++n) {
            const int source_index = reflected_index(t * kHop + n - kPad);
            frontend->frame[n] =
                (static_cast<float>(pcm[source_index]) / 32768.0f) *
                frontend->window[n];
        }
        tflm_signal::RfftFloatApply(frontend->rfft_state, frontend->frame,
                                    frontend->spectrum);
        for (int k = 0; k < kFreqBins; ++k) {
            frontend->power[k] =
                frontend->spectrum[k].real * frontend->spectrum[k].real +
                frontend->spectrum[k].imag * frontend->spectrum[k].imag;
        }
        gate_start = esp_timer_get_time();
        hanil_qg_accumulate(frontend->quality_accum, frontend->frame, kFft,
                            frontend->power, kFreqBins, HANIL_AI_SAMPLE_RATE,
                            kFft);
        gate_us += esp_timer_get_time() - gate_start;

        for (int m = 0; m < HANIL_AI_MEL_BANDS; ++m) {
            const float *filter = frontend->mel_fb + m * kFreqBins;
            float mel_energy = 0.0f;
            for (int k = 0; k <= frontend->max_mel_bin; ++k) {
                mel_energy += filter[k] * frontend->power[k];
            }
            const float value = logf(mel_energy + kLogOffset);
            frontend->logmel[m * HANIL_AI_MEL_FRAMES + t] = value;
            sum += value;
            sum_sq += static_cast<double>(value) * value;
        }
    }

    gate_start = esp_timer_get_time();
    hanil_qg_scan_pcm(pcm, HANIL_AI_AUDIO_SAMPLES, quality_config->clip_level,
                      &result->gate_metrics.clip_frac,
                      &result->gate_metrics.rms);
    result->gate_result = hanil_qg_decide(frontend->quality_accum,
                                          quality_config,
                                          &result->gate_metrics);
    gate_us += esp_timer_get_time() - gate_start;
    const int64_t gate_done = esp_timer_get_time();
    result->timing.gate_ms = static_cast<float>(gate_us) / 1000.0f;
    result->timing.frontend_ms =
        static_cast<float>(gate_done - start - gate_us) / 1000.0f;
    if (result->gate_result != HANIL_QG_PASS) {
        result->timing.compute_ms =
            static_cast<float>(gate_done - start) / 1000.0f;
        return HANIL_AI_OK;
    }

    const double count = HANIL_AI_FEATURE_COUNT;
    result->feature_mean = static_cast<float>(sum / count);
    const double variance = std::max(
        0.0, sum_sq / count - (sum / count) * (sum / count));
    result->feature_stddev = static_cast<float>(sqrt(variance));
    const float divisor = result->feature_stddev + kNormEps;
    const int64_t quantize_start = esp_timer_get_time();
    for (int i = 0; i < HANIL_AI_FEATURE_COUNT; ++i) {
        const float normalized =
            (frontend->logmel[i] - result->feature_mean) / divisor;
        int quantized = static_cast<int>(
            lrintf(normalized / quant_scale + quant_zero_point));
        quantized = std::max(-128, std::min(127, quantized));
        quantized_features[i] = static_cast<int8_t>(quantized);
    }
    result->timing.quantize_ms =
        static_cast<float>(esp_timer_get_time() - quantize_start) / 1000.0f;
    result->timing.compute_ms =
        static_cast<float>(esp_timer_get_time() - start) / 1000.0f;
    result->feature_hash = fnv1a(quantized_features, HANIL_AI_FEATURE_COUNT);
    return HANIL_AI_OK;
}

