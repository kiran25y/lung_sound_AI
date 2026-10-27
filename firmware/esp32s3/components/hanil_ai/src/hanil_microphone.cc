#include "hanil_microphone.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

namespace {

constexpr int kScratchSamples = 256;
constexpr int kDmaFrames = 512;

}  // namespace

struct hanil_microphone {
    hanil_microphone_config_t config{};
    i2s_chan_handle_t rx = nullptr;
    int32_t *scratch = nullptr;
    int32_t *recording = nullptr;
    int record_samples = 0;
};

extern "C" hanil_microphone_config_t hanil_microphone_default_config(void)
{
    hanil_microphone_config_t config{};
    config.sample_rate = HANIL_AI_SAMPLE_RATE;
    config.bclk_gpio = 2;
    config.ws_gpio = 1;
    config.data_gpio = 4;
    config.record_seconds = 6;
    config.analysis_seconds = HANIL_AI_AUDIO_SECONDS;
    config.target_pcm_peak = 8192.0f;
    config.max_digital_gain = 32.0f;
    config.minimum_raw_rms_dbfs = -70.0f;
    config.placement_minimum_dbfs = -55.0f;
    config.placement_maximum_dbfs = -6.0f;
    return config;
}

extern "C" hanil_ai_status_t hanil_microphone_create(
    const hanil_microphone_config_t *config,
    hanil_microphone_t **microphone_out)
{
    if (!microphone_out) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    *microphone_out = nullptr;
    const hanil_microphone_config_t selected =
        config ? *config : hanil_microphone_default_config();
    if (selected.sample_rate != HANIL_AI_SAMPLE_RATE ||
        selected.analysis_seconds != HANIL_AI_AUDIO_SECONDS ||
        selected.record_seconds < selected.analysis_seconds ||
        selected.max_digital_gain < 1.0f || selected.target_pcm_peak <= 0.0f) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    auto *microphone = new (std::nothrow) hanil_microphone_t();
    if (!microphone) {
        return HANIL_AI_OUT_OF_MEMORY;
    }
    microphone->config = selected;
    microphone->record_samples = selected.sample_rate * selected.record_seconds;
    microphone->scratch = static_cast<int32_t *>(heap_caps_malloc(
        kScratchSamples * sizeof(int32_t),
        MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
    microphone->recording = static_cast<int32_t *>(heap_caps_malloc(
        microphone->record_samples * sizeof(int32_t),
        MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM));
    if (!microphone->scratch || !microphone->recording) {
        hanil_microphone_destroy(microphone);
        return HANIL_AI_OUT_OF_MEMORY;
    }

    i2s_chan_config_t channel_config =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    channel_config.dma_desc_num = 6;
    channel_config.dma_frame_num = kDmaFrames;
    channel_config.auto_clear = true;
    if (i2s_new_channel(&channel_config, nullptr, &microphone->rx) != ESP_OK) {
        hanil_microphone_destroy(microphone);
        return HANIL_AI_MICROPHONE_ERROR;
    }
    i2s_std_config_t standard_config = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(
            static_cast<uint32_t>(selected.sample_rate)),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = static_cast<gpio_num_t>(selected.bclk_gpio),
            .ws = static_cast<gpio_num_t>(selected.ws_gpio),
            .dout = I2S_GPIO_UNUSED,
            .din = static_cast<gpio_num_t>(selected.data_gpio),
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    standard_config.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    if (i2s_channel_init_std_mode(microphone->rx, &standard_config) != ESP_OK ||
        i2s_channel_enable(microphone->rx) != ESP_OK) {
        hanil_microphone_destroy(microphone);
        return HANIL_AI_MICROPHONE_ERROR;
    }
    *microphone_out = microphone;
    return HANIL_AI_OK;
}

extern "C" void hanil_microphone_destroy(hanil_microphone_t *microphone)
{
    if (!microphone) {
        return;
    }
    if (microphone->rx) {
        i2s_channel_disable(microphone->rx);
        i2s_del_channel(microphone->rx);
    }
    heap_caps_free(microphone->scratch);
    heap_caps_free(microphone->recording);
    delete microphone;
}

namespace {

int read_raw24(hanil_microphone_t *microphone, int32_t *output, int samples)
{
    int produced = 0;
    while (produced < samples) {
        const int wanted = std::min(kScratchSamples, samples - produced);
        size_t bytes_read = 0;
        if (i2s_channel_read(microphone->rx, microphone->scratch,
                             wanted * sizeof(int32_t), &bytes_read, 1000) != ESP_OK) {
            return -1;
        }
        const int received = static_cast<int>(bytes_read / sizeof(int32_t));
        for (int i = 0; i < received; ++i) {
            if (output) {
                output[produced + i] = microphone->scratch[i] >> 8;
            }
        }
        produced += received;
        if (received == 0) {
            break;
        }
    }
    return produced;
}

}  // namespace

extern "C" hanil_microphone_capture_result_t hanil_microphone_capture(
    hanil_microphone_t *microphone, int16_t *pcm, size_t pcm_samples,
    hanil_microphone_metrics_t *metrics)
{
    if (!microphone || !pcm || pcm_samples != HANIL_AI_AUDIO_SAMPLES || !metrics) {
        return HANIL_MIC_CAPTURE_INVALID;
    }
    std::memset(metrics, 0, sizeof(*metrics));
    const int drain_samples = microphone->config.sample_rate / 5;
    if (read_raw24(microphone, nullptr, drain_samples) != drain_samples) {
        return HANIL_MIC_CAPTURE_INVALID;
    }
    const int64_t start = esp_timer_get_time();
    if (read_raw24(microphone, microphone->recording,
                   microphone->record_samples) != microphone->record_samples) {
        return HANIL_MIC_CAPTURE_INVALID;
    }
    metrics->capture_ms = static_cast<float>(esp_timer_get_time() - start) / 1000.0f;

    const int crop_offset =
        (microphone->record_samples - HANIL_AI_AUDIO_SAMPLES) / 2;
    const int32_t *crop = microphone->recording + crop_offset;
    int64_t raw_sum = 0;
    double raw_sum_squared = 0.0;
    int raw_clipped = 0;
    for (int i = 0; i < HANIL_AI_AUDIO_SAMPLES; ++i) {
        const int32_t value = crop[i];
        raw_sum += value;
        raw_sum_squared += static_cast<double>(value) * value;
        if (std::abs(static_cast<int64_t>(value)) >= 8304722) {
            ++raw_clipped;
        }
    }
    const double raw_mean =
        static_cast<double>(raw_sum) / HANIL_AI_AUDIO_SAMPLES;
    const double raw_variance = std::max(
        0.0, raw_sum_squared / HANIL_AI_AUDIO_SAMPLES - raw_mean * raw_mean);
    const float raw_rms = static_cast<float>(sqrt(raw_variance));
    int32_t raw_peak = 0;
    for (int i = 0; i < HANIL_AI_AUDIO_SAMPLES; ++i) {
        const int64_t centered =
            static_cast<int64_t>(llround(crop[i] - raw_mean));
        raw_peak = std::max<int32_t>(
            raw_peak, static_cast<int32_t>(std::abs(centered)));
    }
    metrics->raw_peak_dbfs = raw_peak > 0
        ? 20.0f * log10f(raw_peak / 8388608.0f) : -120.0f;
    metrics->raw_rms_dbfs = raw_rms > 0
        ? 20.0f * log10f(raw_rms / 8388608.0f) : -120.0f;
    metrics->raw_clip_fraction =
        static_cast<float>(raw_clipped) / HANIL_AI_AUDIO_SAMPLES;
    metrics->dc_offset = static_cast<float>(raw_mean);
    if (metrics->raw_rms_dbfs < microphone->config.minimum_raw_rms_dbfs) {
        return HANIL_MIC_CAPTURE_INVALID;
    }

    const float wanted_gain = raw_peak > 0
        ? (microphone->config.target_pcm_peak * 256.0f) / raw_peak : 1.0f;
    metrics->digital_gain = std::max(
        1.0f, std::min(microphone->config.max_digital_gain, wanted_gain));
    double sum_squared = 0.0;
    for (int i = 0; i < HANIL_AI_AUDIO_SAMPLES; ++i) {
        const double centered = crop[i] - raw_mean;
        int32_t value = static_cast<int32_t>(
            lrint(centered * metrics->digital_gain / 256.0));
        value = std::max<int32_t>(INT16_MIN, std::min<int32_t>(INT16_MAX, value));
        pcm[i] = static_cast<int16_t>(value);
        const int32_t magnitude = value < 0 ? -value : value;
        metrics->pcm_peak = std::max(metrics->pcm_peak, magnitude);
        sum_squared += static_cast<double>(value) * value;
    }
    metrics->pcm_rms = static_cast<float>(
        sqrt(sum_squared / HANIL_AI_AUDIO_SAMPLES));
    if (metrics->raw_clip_fraction > 0.01f ||
        metrics->raw_rms_dbfs > microphone->config.placement_maximum_dbfs) {
        return HANIL_MIC_CAPTURE_TOO_LOUD;
    }
    if (metrics->raw_rms_dbfs <
        microphone->config.placement_minimum_dbfs) {
        return HANIL_MIC_CAPTURE_POOR_CONTACT;
    }
    return HANIL_MIC_CAPTURE_OK;
}

extern "C" const char *hanil_microphone_capture_reason(
    hanil_microphone_capture_result_t result)
{
    switch (result) {
        case HANIL_MIC_CAPTURE_OK: return "ok";
        case HANIL_MIC_CAPTURE_INVALID: return "mic_capture_invalid";
        case HANIL_MIC_CAPTURE_POOR_CONTACT: return "poor_chest_contact";
        case HANIL_MIC_CAPTURE_TOO_LOUD: return "too_loud_or_clipping";
        default: return "unknown";
    }
}

