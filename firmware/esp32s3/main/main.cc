// Real respiratory-audio inference for ESP32-S3.
//
// USB protocol (little endian):
//   "PCM1" + uint32(128000) + 64000 signed PCM16 samples
//       -> exact on-device 16 kHz/4 s log-Mel frontend, then TFLM inference
//   "FEA1" + uint32(25664) + quantized INT8 [1,64,401,1]
//       -> reference/parity path that bypasses the frontend
//   "MIC1" + uint32(0)
//       -> capture 6 seconds from INMP441 and center-crop the middle 4 seconds
//
// The window and Slaney mel filterbank are the exact arrays exported by the
// training repository. They are embedded as .npy assets and read in-place.
#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstring>

#include "driver/i2s_std.h"
#include "driver/usb_serial_jtag.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_memory_utils.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "quality_gate.h"
#include "signal/src/rfft.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if defined(HANIL_STUDENT_PIPELINE_SRAM)
extern const uint8_t model_start[] asm("_binary_sram_student_int8_tflite_start");
extern const uint8_t model_end[] asm("_binary_sram_student_int8_tflite_end");
#else
extern const uint8_t model_start[] asm("_binary_model_int8_tflm_tflite_start");
extern const uint8_t model_end[] asm("_binary_model_int8_tflm_tflite_end");
#endif
extern const uint8_t mel_fb_start[] asm("_binary_mel_fb_npy_start");
extern const uint8_t mel_fb_end[] asm("_binary_mel_fb_npy_end");
extern const uint8_t window_start[] asm("_binary_window_npy_start");
extern const uint8_t window_end[] asm("_binary_window_npy_end");

namespace {

constexpr char kTag[] = "LUNGAI";
constexpr int kSampleRate = 16000;
constexpr int kSamples = 64000;
constexpr int kFft = 400;
constexpr int kHop = 160;
constexpr int kPad = kFft / 2;
constexpr int kFreqBins = kFft / 2 + 1;
constexpr int kMels = 64;
constexpr int kFrames = 401;
constexpr int kFeatureCount = kMels * kFrames;
#if defined(HANIL_STUDENT_PIPELINE_SRAM)
// Hardware measurement for this model is 117,132 bytes. The 128 KiB arena
// Measured arena use is 117,132 bytes. A 118 KiB arena leaves 3,700 bytes of
// safety margin while recovering enough contiguous internal SRAM for the
// complete model + arena + quality-gate build.
constexpr size_t kArenaBytes = 118 * 1024;
#else
constexpr size_t kArenaBytes = 2 * 1024 * 1024;
#endif
constexpr float kLogOffset = 1.0e-6f;
constexpr float kNormEps = 1.0e-5f;
constexpr float kCrackleThreshold = 0.49f;
#if defined(HANIL_STUDENT_PIPELINE_SRAM)
constexpr float kWheezeThreshold = 0.28f;
#else
constexpr float kWheezeThreshold = 0.52f;
#endif
#if defined(HANIL_USE_ESP_NN)
constexpr char kKernelBackend[] = "esp_nn_optimized_int8_static_shapes";
#else
constexpr char kKernelBackend[] = "tflm_portable_int8_static_shapes";
#endif
// Proven hardware map from the current board's working INMP441 test firmware.
constexpr gpio_num_t kMicBclk = GPIO_NUM_2;
constexpr gpio_num_t kMicWs = GPIO_NUM_1;
constexpr gpio_num_t kMicData = GPIO_NUM_4;
constexpr int kMicScratchSamples = 256;
constexpr int kMicDmaFrames = 512;
constexpr int kMicDrainSamples = kSampleRate / 5;
constexpr int kMicRecordSamples = kSampleRate * 6;
constexpr int kMicCropOffset = (kMicRecordSamples - kSamples) / 2;
constexpr int kMicTargetPcmPeak = 8192;  // -12.0 dBFS after bounded auto-gain.
constexpr float kMicMaxDigitalGain = 32.0f;  // +30.1 dB maximum.
constexpr float kMicMinRawRmsDbfs = -70.0f;
constexpr float kMicPlacementMinDbfs = -55.0f;
constexpr float kMicPlacementMaxDbfs = -6.0f;
constexpr uint32_t kPcmPayloadBytes = kSamples * sizeof(int16_t);
constexpr uint32_t kFeaturePayloadBytes = kFeatureCount * sizeof(int8_t);

struct CommandHeader {
    char magic[4];
    uint32_t payload_bytes;
};

struct FrontendWorkspace {
    int16_t *pcm = nullptr;
    float *logmel = nullptr;
    float *frame = nullptr;
    float *power = nullptr;
    Complex<float> *spectrum = nullptr;
    void *rfft_storage = nullptr;
    void *rfft_state = nullptr;
    qg_accum_t *quality_accum = nullptr;
    int32_t *mic_recording = nullptr;
    int max_mel_bin = 0;
};

struct MicrophoneInput {
    i2s_chan_handle_t rx = nullptr;
    int32_t *scratch = nullptr;
    bool ready = false;
};

enum class MicrophoneCaptureResult {
    kOk,
    kCaptureInvalid,
    kPoorContact,
    kTooLoudOrClipping,
};

struct PipelineTiming {
    float frontend_ms = 0.0f;
    float gate_ms = 0.0f;
    float quantize_ms = 0.0f;
    float preprocess_total_ms = 0.0f;
};

const float *npy_f32(const uint8_t *start, const uint8_t *end, size_t expected_values)
{
    const size_t bytes = static_cast<size_t>(end - start);
    if (bytes < 16 || std::memcmp(start, "\x93NUMPY", 6) != 0 || start[6] != 1) {
        ESP_LOGE(kTag, "invalid embedded NPY header");
        return nullptr;
    }
    const size_t header_bytes = static_cast<size_t>(start[8]) |
                                (static_cast<size_t>(start[9]) << 8);
    const size_t data_offset = 10 + header_bytes;
    if (data_offset + expected_values * sizeof(float) > bytes || (data_offset & 3U) != 0) {
        ESP_LOGE(kTag, "invalid embedded NPY size/alignment");
        return nullptr;
    }
    return reinterpret_cast<const float *>(start + data_offset);
}

void *psram_alloc(size_t bytes)
{
    return heap_caps_malloc(bytes, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
}

void *internal_alloc(size_t bytes)
{
    return heap_caps_malloc(bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
}

void log_memory(const char *stage)
{
    constexpr uint32_t internal_caps = MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL;
    ESP_LOGI(kTag,
             "MEMORY stage=%s internal_free=%u internal_largest=%u "
             "internal_minimum_free=%u internal_total=%u psram_free=%u "
             "psram_largest=%u psram_total=%u",
             stage,
             static_cast<unsigned>(heap_caps_get_free_size(internal_caps)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(internal_caps)),
             static_cast<unsigned>(heap_caps_get_minimum_free_size(internal_caps)),
             static_cast<unsigned>(heap_caps_get_total_size(internal_caps)),
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)),
             static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM)));
}

bool read_exact(void *destination, size_t bytes)
{
    auto *dst = static_cast<uint8_t *>(destination);
    size_t received = 0;
    while (received < bytes) {
        const size_t remaining = bytes - received;
        const uint32_t request = static_cast<uint32_t>(std::min<size_t>(remaining, 4096));
        const int n = usb_serial_jtag_read_bytes(dst + received, request, pdMS_TO_TICKS(5000));
        if (n > 0) {
            received += static_cast<size_t>(n);
        } else {
            ESP_LOGW(kTag, "receive timeout (%u/%u bytes)",
                     static_cast<unsigned>(received), static_cast<unsigned>(bytes));
        }
    }
    return true;
}

void discard_bytes(size_t bytes)
{
    uint8_t scratch[256];
    while (bytes > 0) {
        const size_t chunk = std::min(bytes, sizeof(scratch));
        read_exact(scratch, chunk);
        bytes -= chunk;
    }
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

int reflected_index(int index)
{
    // Matches np.pad(x, 200, mode="reflect") / torch reflect padding.
    if (index < 0) {
        return -index;
    }
    if (index >= kSamples) {
        return 2 * kSamples - 2 - index;
    }
    return index;
}

bool prepare_frontend(FrontendWorkspace &ws, const float *mel_fb)
{
    ws.pcm = static_cast<int16_t *>(psram_alloc(kPcmPayloadBytes));
    ws.logmel = static_cast<float *>(psram_alloc(kFeatureCount * sizeof(float)));
    // The complete mutable gate state, including its sorting workspace, is
    // deliberately internal. Large raw-audio and log-Mel staging buffers stay
    // in PSRAM because they are not accessed by inference kernels.
    ws.quality_accum = static_cast<qg_accum_t *>(
        heap_caps_aligned_alloc(16, sizeof(qg_accum_t),
                                MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
    // Retain all signed 24 microphone bits until gain calibration is complete.
    ws.mic_recording = static_cast<int32_t *>(
        psram_alloc(kMicRecordSamples * sizeof(int32_t)));
    ws.frame = static_cast<float *>(internal_alloc(kFft * sizeof(float)));
    ws.power = static_cast<float *>(internal_alloc(kFreqBins * sizeof(float)));
    ws.spectrum = static_cast<Complex<float> *>(
        internal_alloc(kFreqBins * sizeof(Complex<float>)));
    const size_t rfft_bytes = tflm_signal::RfftFloatGetNeededMemory(kFft);
    ws.rfft_storage = internal_alloc(rfft_bytes);
    if (!ws.pcm || !ws.logmel || !ws.quality_accum || !ws.mic_recording ||
        !ws.frame || !ws.power || !ws.spectrum || !ws.rfft_storage) {
        ESP_LOGE(kTag, "frontend workspace allocation failed");
        return false;
    }
    ws.rfft_state = tflm_signal::RfftFloatInit(kFft, ws.rfft_storage, rfft_bytes);
    if (!ws.rfft_state) {
        ESP_LOGE(kTag, "400-point RFFT initialization failed");
        return false;
    }

    // The filterbank is zero above 2 kHz, so calculate only spectrum bins used
    // by at least one mel band. This is exact, not an approximation.
    for (int k = 0; k < kFreqBins; ++k) {
        for (int m = 0; m < kMels; ++m) {
            if (mel_fb[m * kFreqBins + k] != 0.0f) {
                ws.max_mel_bin = std::max(ws.max_mel_bin, k);
            }
        }
    }
    ESP_LOGI(kTag,
             "frontend ready: 400-point full-spectrum RFFT bins=0..%d mel_bins=0..%d",
             kFreqBins - 1, ws.max_mel_bin);
    ESP_LOGI(kTag,
             "GATE_MEMORY mutable_bytes=%u address=%p location=internal_sram "
             "internal=%s band_energy_bytes=%u sort_workspace_bytes=%u",
             static_cast<unsigned>(sizeof(qg_accum_t)), ws.quality_accum,
             esp_ptr_internal(ws.quality_accum) ? "yes" : "no",
             512U * static_cast<unsigned>(sizeof(float)),
             512U * static_cast<unsigned>(sizeof(float)));
    ESP_LOGI(kTag,
             "FRONTEND_MEMORY pcm_psram=%u logmel_psram=%u mic_recording_psram=%u "
             "frame_internal=%u power_internal=%u spectrum_internal=%u",
             static_cast<unsigned>(kPcmPayloadBytes),
             static_cast<unsigned>(kFeatureCount * sizeof(float)),
             static_cast<unsigned>(kMicRecordSamples * sizeof(int32_t)),
             static_cast<unsigned>(kFft * sizeof(float)),
             static_cast<unsigned>(kFreqBins * sizeof(float)),
             static_cast<unsigned>(kFreqBins * sizeof(Complex<float>)));
    log_memory("frontend_ready");
    return true;
}

bool prepare_microphone(MicrophoneInput &mic)
{
    mic.scratch = static_cast<int32_t *>(
        internal_alloc(kMicScratchSamples * sizeof(int32_t)));
    if (!mic.scratch) {
        ESP_LOGE(kTag, "I2S scratch-buffer allocation failed");
        return false;
    }

    i2s_chan_config_t channel_config =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    channel_config.dma_desc_num = 6;
    channel_config.dma_frame_num = kMicDmaFrames;
    channel_config.auto_clear = true;
    esp_err_t error = i2s_new_channel(&channel_config, nullptr, &mic.rx);
    if (error != ESP_OK) {
        ESP_LOGE(kTag, "I2S channel creation failed: %s", esp_err_to_name(error));
        return false;
    }

    i2s_std_config_t standard_config = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = kMicBclk,
            .ws = kMicWs,
            .dout = I2S_GPIO_UNUSED,
            .din = kMicData,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    // The microphone L/R pad is tied to ground on this board.
    standard_config.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    error = i2s_channel_init_std_mode(mic.rx, &standard_config);
    if (error != ESP_OK) {
        ESP_LOGE(kTag, "I2S standard-mode initialization failed: %s",
                 esp_err_to_name(error));
        return false;
    }
    error = i2s_channel_enable(mic.rx);
    if (error != ESP_OK) {
        ESP_LOGE(kTag, "I2S channel enable failed: %s", esp_err_to_name(error));
        return false;
    }
    mic.ready = true;
    ESP_LOGI(kTag,
             "MIC_CONFIG source=current_board_test type=INMP441 "
             "format=philips rate=%d bits=32 mode=mono_left raw_shift=8 "
             "auto_gain_target_dbfs=-12.0 auto_gain_max_db=30.1 "
             "bclk_gpio=%d ws_gpio=%d data_gpio=%d",
             kSampleRate, static_cast<int>(kMicBclk), static_cast<int>(kMicWs),
             static_cast<int>(kMicData));
    return true;
}

int read_microphone_raw24(MicrophoneInput &mic, int32_t *output, int samples)
{
    int produced = 0;
    while (produced < samples) {
        const int wanted = std::min(kMicScratchSamples, samples - produced);
        size_t bytes_read = 0;
        const esp_err_t error = i2s_channel_read(
            mic.rx, mic.scratch, wanted * sizeof(int32_t), &bytes_read, 1000);
        if (error != ESP_OK) {
            ESP_LOGE(kTag, "I2S read failed: %s", esp_err_to_name(error));
            return -1;
        }
        const int received = static_cast<int>(bytes_read / sizeof(int32_t));
        for (int i = 0; i < received; ++i) {
            // INMP441: signed 24-bit sample, MSB-aligned in a 32-bit slot.
            // Do not discard the lower eight valid bits before auto-gain.
            if (output) {
                output[produced + i] = mic.scratch[i] >> 8;
            }
        }
        produced += received;
        if (received == 0) {
            break;
        }
    }
    return produced;
}

MicrophoneCaptureResult capture_microphone(MicrophoneInput &mic, FrontendWorkspace &ws)
{
    if (!mic.ready) {
        ESP_LOGE(kTag, "microphone is not initialized");
        return MicrophoneCaptureResult::kCaptureInvalid;
    }

    ESP_LOGI(kTag, "MIC_CAPTURE state=starting record_s=6 crop_s=4");
    if (read_microphone_raw24(mic, nullptr, kMicDrainSamples) != kMicDrainSamples) {
        return MicrophoneCaptureResult::kCaptureInvalid;
    }
    const int64_t capture_start = esp_timer_get_time();
    if (read_microphone_raw24(mic, ws.mic_recording, kMicRecordSamples) !=
        kMicRecordSamples) {
        return MicrophoneCaptureResult::kCaptureInvalid;
    }

    const int32_t *crop = ws.mic_recording + kMicCropOffset;
    int64_t raw_sum = 0;
    double raw_sum_squared = 0.0;
    int raw_clipped = 0;
    for (int i = 0; i < kSamples; ++i) {
        const int32_t value = crop[i];
        raw_sum += value;
        raw_sum_squared += static_cast<double>(value) * value;
        if (std::abs(static_cast<int64_t>(value)) >= 8304722) {  // 0.99 * 2^23
            ++raw_clipped;
        }
    }
    const double raw_mean = static_cast<double>(raw_sum) / kSamples;
    const double raw_variance = std::max(
        0.0, raw_sum_squared / kSamples - raw_mean * raw_mean);
    const float raw_rms = static_cast<float>(sqrt(raw_variance));
    int32_t raw_peak = 0;
    for (int i = 0; i < kSamples; ++i) {
        const int64_t centered = static_cast<int64_t>(llround(crop[i] - raw_mean));
        raw_peak = std::max<int32_t>(
            raw_peak, static_cast<int32_t>(std::abs(centered)));
    }

    const float raw_peak_dbfs = raw_peak > 0
        ? 20.0f * log10f(raw_peak / 8388608.0f) : -120.0f;
    const float raw_rms_dbfs = raw_rms > 0
        ? 20.0f * log10f(raw_rms / 8388608.0f) : -120.0f;
    const float raw_clip_frac = static_cast<float>(raw_clipped) / kSamples;
    if (raw_rms_dbfs < kMicMinRawRmsDbfs) {
        ESP_LOGE(kTag,
                 "MIC_INVALID silence/no-signal raw_rms_dbfs=%.1f minimum=%.1f",
                 raw_rms_dbfs, kMicMinRawRmsDbfs);
        return MicrophoneCaptureResult::kCaptureInvalid;
    }

    const float wanted_gain = raw_peak > 0
        ? (static_cast<float>(kMicTargetPcmPeak) * 256.0f) / raw_peak
        : 1.0f;
    const float digital_gain = std::max(
        1.0f, std::min(kMicMaxDigitalGain, wanted_gain));
    int32_t peak = 0;
    int clipped = 0;
    double sum_squared = 0.0;
    for (int i = 0; i < kSamples; ++i) {
        const double centered = crop[i] - raw_mean;
        int32_t value = static_cast<int32_t>(lrint(centered * digital_gain / 256.0));
        value = std::max<int32_t>(INT16_MIN, std::min<int32_t>(INT16_MAX, value));
        ws.pcm[i] = static_cast<int16_t>(value);
        const int32_t magnitude = value < 0 ? -value : value;
        peak = std::max(peak, magnitude);
        sum_squared += static_cast<double>(value) * value;
        if (magnitude >= 32760) {
            ++clipped;
        }
    }
    const float rms = static_cast<float>(sqrt(sum_squared / kSamples));
    const char *placement = raw_clip_frac > 0.01f || raw_rms_dbfs > kMicPlacementMaxDbfs
        ? "too_loud_or_clipping"
        : (raw_rms_dbfs < kMicPlacementMinDbfs ? "improve_chest_contact" : "good");
    ESP_LOGI(kTag,
             "MIC_CAPTURE state=complete recorded=%d crop=%d time_ms=%.1f "
             "raw_peak_dbfs=%.1f raw_rms_dbfs=%.1f raw_clip_frac=%.6f dc=%.1f "
             "gain=%.2fx gain_db=%.1f peak=%" PRId32 " rms=%.1f "
             "peak_dbfs=%.1f rms_dbfs=%.1f post_clip=%d placement=%s",
             kMicRecordSamples, kSamples,
             (esp_timer_get_time() - capture_start) / 1000.0f,
             raw_peak_dbfs, raw_rms_dbfs, raw_clip_frac, raw_mean,
             digital_gain, 20.0f * log10f(digital_gain), peak, rms,
             peak > 0 ? 20.0f * log10f(peak / 32768.0f) : -120.0f,
             rms > 0 ? 20.0f * log10f(rms / 32768.0f) : -120.0f,
             clipped, placement);
    ESP_LOGI(kTag,
             "MIC_PLACEMENT status=%s raw_rms_dbfs=%.1f recommended=[%.1f,%.1f] "
             "instruction=hold_sensor_firmly_on_chest_avoid_rubbing",
             placement, raw_rms_dbfs, kMicPlacementMinDbfs,
             kMicPlacementMaxDbfs);
    if (raw_clip_frac > 0.01f || raw_rms_dbfs > kMicPlacementMaxDbfs) {
        return MicrophoneCaptureResult::kTooLoudOrClipping;
    }
    if (raw_rms_dbfs < kMicPlacementMinDbfs) {
        return MicrophoneCaptureResult::kPoorContact;
    }
    return MicrophoneCaptureResult::kOk;
}

qg_result_t preprocess_pcm(FrontendWorkspace &ws, const float *window,
                           const float *mel_fb, TfLiteTensor *input,
                           const qg_config_t &quality_config,
                           qg_metrics_t &quality_metrics,
                           PipelineTiming &timing)
{
    double sum = 0.0;
    double sum_sq = 0.0;
    const int64_t t0 = esp_timer_get_time();
    int64_t gate_us = 0;
    int64_t gate_start = esp_timer_get_time();
    qg_reset(ws.quality_accum);
    gate_us += esp_timer_get_time() - gate_start;

    for (int t = 0; t < kFrames; ++t) {
        for (int n = 0; n < kFft; ++n) {
            const int source_index = reflected_index(t * kHop + n - kPad);
            ws.frame[n] =
                (static_cast<float>(ws.pcm[source_index]) / 32768.0f) * window[n];
        }

        tflm_signal::RfftFloatApply(ws.rfft_state, ws.frame, ws.spectrum);
        for (int k = 0; k < kFreqBins; ++k) {
            ws.power[k] = ws.spectrum[k].real * ws.spectrum[k].real +
                          ws.spectrum[k].imag * ws.spectrum[k].imag;
        }
        gate_start = esp_timer_get_time();
        qg_accumulate(ws.quality_accum, ws.frame, kFft, ws.power, kFreqBins,
                      kSampleRate, kFft);
        gate_us += esp_timer_get_time() - gate_start;

        for (int m = 0; m < kMels; ++m) {
            const float *filter = mel_fb + m * kFreqBins;
            float mel_energy = 0.0f;
            for (int k = 0; k <= ws.max_mel_bin; ++k) {
                mel_energy += filter[k] * ws.power[k];
            }
            const float value = logf(mel_energy + kLogOffset);
            ws.logmel[m * kFrames + t] = value;
            sum += value;
            sum_sq += static_cast<double>(value) * value;
        }
    }

    gate_start = esp_timer_get_time();
    qg_scan_pcm(ws.pcm, kSamples, quality_config.clip_level,
                &quality_metrics.clip_frac, &quality_metrics.rms);
    const qg_result_t quality_result =
        qg_decide(ws.quality_accum, &quality_config, &quality_metrics);
    gate_us += esp_timer_get_time() - gate_start;
    const int64_t decision_done = esp_timer_get_time();
    timing.gate_ms = gate_us / 1000.0f;
    timing.frontend_ms = (decision_done - t0 - gate_us) / 1000.0f;
    timing.preprocess_total_ms = (decision_done - t0) / 1000.0f;
    ESP_LOGI(kTag,
             "GATE decision=%s reason=%s clip_frac=%.6f rms=%.6f flatness=%.6f "
             "zcr=%.6f quality=%.2f gate_ms=%.3f frontend_ms=%.3f "
             "preprocess_ms=%.3f",
             quality_result == QG_PASS ? "PASS" : "RETRY", qg_reason(quality_result),
             quality_metrics.clip_frac, quality_metrics.rms, quality_metrics.flatness,
             quality_metrics.zcr, quality_metrics.quality_score, timing.gate_ms,
             timing.frontend_ms, timing.preprocess_total_ms);
    if (quality_result != QG_PASS) {
        return quality_result;
    }

    const double count = kFeatureCount;
    const float mean = static_cast<float>(sum / count);
    const double variance = std::max(0.0, sum_sq / count - (sum / count) * (sum / count));
    const float stddev = static_cast<float>(sqrt(variance));
    const float divisor = stddev + kNormEps;
    const float scale = input->params.scale;
    const int zero_point = input->params.zero_point;

    const int64_t quantize_start = esp_timer_get_time();
    for (int i = 0; i < kFeatureCount; ++i) {
        const float normalized = (ws.logmel[i] - mean) / divisor;
        int quantized = static_cast<int>(lrintf(normalized / scale + zero_point));
        quantized = std::max(-128, std::min(127, quantized));
        input->data.int8[i] = static_cast<int8_t>(quantized);
    }
    timing.quantize_ms = (esp_timer_get_time() - quantize_start) / 1000.0f;
    timing.preprocess_total_ms = (esp_timer_get_time() - t0) / 1000.0f;

    ESP_LOGI(kTag,
             "FRONTEND mean=%.6f std=%.6f frontend_ms=%.3f gate_ms=%.3f "
             "quantize_ms=%.3f preprocess_ms=%.3f qhash=%08" PRIx32,
             mean, stddev, timing.frontend_ms, timing.gate_ms,
             timing.quantize_ms, timing.preprocess_total_ms,
             fnv1a(input->data.int8, input->bytes));
    return QG_PASS;
}

float run_prediction(tflite::MicroInterpreter &interpreter, TfLiteTensor *input,
                     TfLiteTensor *output, const char *source)
{
    const uint32_t input_hash = fnv1a(input->data.int8, input->bytes);
    const int64_t t0 = esp_timer_get_time();
    if (interpreter.Invoke() != kTfLiteOk) {
        ESP_LOGE(kTag, "Invoke failed");
        return -1.0f;
    }
    const float latency_ms = (esp_timer_get_time() - t0) / 1000.0f;

    const float output_scale = output->params.scale;
    const int output_zp = output->params.zero_point;
    const float crackle_logit = (static_cast<int>(output->data.int8[0]) - output_zp) * output_scale;
    const float wheeze_logit = (static_cast<int>(output->data.int8[1]) - output_zp) * output_scale;
    const float p_crackle = 1.0f / (1.0f + expf(-crackle_logit));
    const float p_wheeze = 1.0f / (1.0f + expf(-wheeze_logit));
    const float p_abnormal = 1.0f - (1.0f - p_crackle) * (1.0f - p_wheeze);
    const bool crackle = p_crackle >= kCrackleThreshold;
    const bool wheeze = p_wheeze >= kWheezeThreshold;
    const char *class2 = crackle || wheeze ? "abnormal" : "normal";
    const char *class3 = "normal";
    if (crackle && wheeze) {
        class3 = p_crackle >= p_wheeze ? "crackle" : "wheeze";
    } else if (crackle) {
        class3 = "crackle";
    } else if (wheeze) {
        class3 = "wheeze";
    }

    ESP_LOGI(kTag,
             "RESULT source=%s p_crackle=%.6f p_wheeze=%.6f p_abnormal=%.6f "
             "class2=%s class3=%s latency_ms=%.2f qhash=%08" PRIx32 " out_q=[%d,%d]",
             source, p_crackle, p_wheeze, p_abnormal, class2, class3, latency_ms, input_hash,
             static_cast<int>(output->data.int8[0]), static_cast<int>(output->data.int8[1]));
    return latency_ms;
}

}  // namespace

extern "C" void app_main(void)
{
    log_memory("boot");
    const float *mel_fb = npy_f32(mel_fb_start, mel_fb_end, kMels * kFreqBins);
    const float *window = npy_f32(window_start, window_end, kFft);
    if (!mel_fb || !window) {
        return;
    }

    const size_t model_bytes = static_cast<size_t>(model_end - model_start);
    ESP_LOGI(kTag, "MODEL_STORAGE bytes=%u kib=%.2f in_flash_drom=%s",
             static_cast<unsigned>(model_bytes), model_bytes / 1024.0f,
             esp_ptr_in_drom(model_start) ? "yes" : "no");

    const uint8_t *model_data = model_start;
#if defined(HANIL_STUDENT_PIPELINE_SRAM) && defined(HANIL_PIPELINE_COPY_MODEL_TO_SRAM)
    // Allocate the larger model block before the arena. Reversing this order
    // fragments the internal heap into blocks smaller than the 188 KiB model
    // even though the total free memory is sufficient.
    auto *model_copy = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, model_bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
    if (!model_copy) {
        ESP_LOGE(kTag, "internal model-copy allocation failed: %u bytes",
                 static_cast<unsigned>(model_bytes));
        log_memory("model_copy_failed");
        return;
    }
    std::memcpy(model_copy, model_start, model_bytes);
    model_data = model_copy;
    ESP_LOGI(kTag,
             "MODEL_RUNTIME bytes=%u address=%p location=internal_sram internal=%s",
             static_cast<unsigned>(model_bytes), model_data,
             esp_ptr_internal(model_data) ? "yes" : "no");
    log_memory("model_copied");
#else
    ESP_LOGI(kTag,
             "MODEL_RUNTIME bytes=%u address=%p location=flash_drom internal=no",
             static_cast<unsigned>(model_bytes), model_data);
#endif

#if defined(HANIL_STUDENT_PIPELINE_SRAM)
    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, kArenaBytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
#else
    auto *arena = static_cast<uint8_t *>(psram_alloc(kArenaBytes));
#endif
    if (!arena) {
        ESP_LOGE(kTag, "tensor arena allocation failed");
        log_memory("arena_allocation_failed");
        return;
    }
    ESP_LOGI(kTag, "ARENA capacity=%u address=%p location=%s internal=%s",
             static_cast<unsigned>(kArenaBytes), arena,
             esp_ptr_internal(arena) ? "internal_sram" : "psram",
             esp_ptr_internal(arena) ? "yes" : "no");
    log_memory("arena_allocated");

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(kTag, "TFLite schema mismatch");
        return;
    }

#if defined(HANIL_STUDENT_PIPELINE_SRAM)
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
#else
    static tflite::MicroMutableOpResolver<30> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddFullyConnected();
    resolver.AddAdd(); resolver.AddMul(); resolver.AddSub();
    resolver.AddMean(); resolver.AddReshape(); resolver.AddExpandDims();
    resolver.AddStridedSlice(); resolver.AddSlice(); resolver.AddTranspose(); resolver.AddPad();
    resolver.AddSpaceToBatchNd(); resolver.AddBatchToSpaceNd();
    resolver.AddQuantize(); resolver.AddDequantize();
    resolver.AddRelu(); resolver.AddRelu6(); resolver.AddMaximum(); resolver.AddMinimum();
    resolver.AddSoftmax(); resolver.AddLogistic(); resolver.AddConcatenation(); resolver.AddPack();
    resolver.AddShape();
#endif

    static tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaBytes);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(kTag, "AllocateTensors failed");
        return;
    }
    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);
    if (input->type != kTfLiteInt8 || input->bytes != kFeaturePayloadBytes ||
        output->type != kTfLiteInt8 || output->bytes < 2) {
        ESP_LOGE(kTag, "unexpected model tensor contract");
        return;
    }

    ESP_LOGI(kTag,
             "MODEL input=[1,64,401,1] input_scale=%.9g input_zp=%d "
             "output_scale=%.9g output_zp=%d arena_used_bytes=%u "
             "arena_used_kib=%.2f arena_internal=%s model_runtime=%s "
             "kernel_backend=%s",
             input->params.scale, static_cast<int>(input->params.zero_point),
             output->params.scale, static_cast<int>(output->params.zero_point),
             static_cast<unsigned>(interpreter.arena_used_bytes()),
             interpreter.arena_used_bytes() / 1024.0f,
             esp_ptr_internal(arena) ? "yes" : "no",
             esp_ptr_internal(model_data) ? "internal_sram" : "flash_drom",
             kKernelBackend);
    log_memory("tensors_ready");

    FrontendWorkspace frontend;
    if (!prepare_frontend(frontend, mel_fb)) {
        return;
    }
    MicrophoneInput microphone;
    const bool microphone_ready = prepare_microphone(microphone);
    log_memory("microphone_ready");
    const qg_config_t quality_config = qg_default_config();
    ESP_LOGI(kTag,
             "GATE_CONFIG clip_max=%.4f rms_min=%.4f flatness_max=%.3f "
             "zcr_max=%.3f quality_min=%.1f",
             quality_config.clip_frac_max, quality_config.rms_min,
             quality_config.flatness_max, quality_config.zcr_max,
             quality_config.quality_min);
    bool have_pcm_features = false;

    usb_serial_jtag_driver_config_t usb_config = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_config.rx_buffer_size = 8192;
    usb_config.tx_buffer_size = 1024;
    if (!usb_serial_jtag_is_driver_installed()) {
        ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_config));
    }
    log_memory("usb_ready");

    while (true) {
        ESP_LOGI(kTag,
                 "READY build=%s MIC1=live_6s_crop4s mic_ready=%s PCM1=%u_bytes "
                 "FEA1=%u_bytes CMP1=%u_bytes thresholds=[%.2f,%.2f]",
#if defined(HANIL_STUDENT_PIPELINE_SRAM)
                 "student_pipeline_sram",
#else
                 "reference_pipeline",
#endif
                 microphone_ready ? "yes" : "no",
                 static_cast<unsigned>(kPcmPayloadBytes),
                 static_cast<unsigned>(kFeaturePayloadBytes),
                 static_cast<unsigned>(kFeaturePayloadBytes),
                 kCrackleThreshold, kWheezeThreshold);

        CommandHeader header{};
        read_exact(&header, sizeof(header));
        if (std::memcmp(header.magic, "MIC1", 4) == 0) {
            if (header.payload_bytes != 0) {
                ESP_LOGE(kTag, "MIC1 length %u, expected 0",
                         static_cast<unsigned>(header.payload_bytes));
                discard_bytes(header.payload_bytes);
                continue;
            }
            ESP_LOGI(kTag, "RECEIVED source=live_i2s command=MIC1");
            const int64_t command_start = esp_timer_get_time();
            const int64_t capture_start = command_start;
            const MicrophoneCaptureResult capture_result = microphone_ready
                ? capture_microphone(microphone, frontend)
                : MicrophoneCaptureResult::kCaptureInvalid;
            if (capture_result == MicrophoneCaptureResult::kCaptureInvalid) {
                have_pcm_features = false;
                ESP_LOGI(kTag,
                         "GATE decision=RETRY reason=mic_capture_invalid clip_frac=0.000000 "
                         "rms=0.000000 flatness=0.000000 zcr=0.000000 quality=-90.00 "
                         "preprocess_ms=0.0");
                ESP_LOGI(kTag, "ACTION CHECK_MIC_WIRING");
                ESP_LOGI(kTag,
                         "PIPELINE_TIMING source=live_i2s status=RETRY capture_ms=%.3f "
                         "frontend_ms=0.000 gate_ms=0.000 quantize_ms=0.000 "
                         "inference_ms=-1.000 compute_ms=0.000 end_to_end_ms=%.3f",
                         (esp_timer_get_time() - capture_start) / 1000.0f,
                         (esp_timer_get_time() - command_start) / 1000.0f);
                log_memory("after_live_i2s_retry");
                continue;
            }
            const float capture_ms = (esp_timer_get_time() - capture_start) / 1000.0f;
            if (capture_result != MicrophoneCaptureResult::kOk) {
                have_pcm_features = false;
                const char *reason = capture_result == MicrophoneCaptureResult::kPoorContact
                    ? "poor_chest_contact"
                    : "microphone_too_loud_or_clipping";
                ESP_LOGI(kTag,
                         "GATE decision=RETRY reason=%s clip_frac=0.000000 "
                         "rms=0.000000 flatness=0.000000 zcr=0.000000 quality=-90.00 "
                         "gate_ms=0.000 frontend_ms=0.000 preprocess_ms=0.000",
                         reason);
                ESP_LOGI(kTag, "ACTION RETRY_RECORDING reason=%s", reason);
                ESP_LOGI(kTag,
                         "PIPELINE_TIMING source=live_i2s status=RETRY capture_ms=%.3f "
                         "frontend_ms=0.000 gate_ms=0.000 quantize_ms=0.000 "
                         "inference_ms=-1.000 compute_ms=0.000 end_to_end_ms=%.3f",
                         capture_ms,
                         (esp_timer_get_time() - command_start) / 1000.0f);
                log_memory("after_live_i2s_retry");
                continue;
            }
            qg_metrics_t quality_metrics{};
            PipelineTiming timing{};
            const qg_result_t quality_result =
                preprocess_pcm(frontend, window, mel_fb, input, quality_config,
                               quality_metrics, timing);
            if (quality_result != QG_PASS) {
                have_pcm_features = false;
                ESP_LOGI(kTag, "ACTION RETRY_RECORDING reason=%s", qg_reason(quality_result));
                ESP_LOGI(kTag,
                         "PIPELINE_TIMING source=live_i2s status=RETRY capture_ms=%.3f "
                         "frontend_ms=%.3f gate_ms=%.3f quantize_ms=%.3f "
                         "inference_ms=-1.000 compute_ms=%.3f end_to_end_ms=%.3f",
                         capture_ms, timing.frontend_ms, timing.gate_ms,
                         timing.quantize_ms, timing.preprocess_total_ms,
                         (esp_timer_get_time() - command_start) / 1000.0f);
                log_memory("after_live_i2s_retry");
                continue;
            }
            have_pcm_features = false;
            const float inference_ms =
                run_prediction(interpreter, input, output, "live_i2s");
            ESP_LOGI(kTag,
                     "PIPELINE_TIMING source=live_i2s status=PASS capture_ms=%.3f "
                     "frontend_ms=%.3f gate_ms=%.3f quantize_ms=%.3f "
                     "inference_ms=%.3f compute_ms=%.3f end_to_end_ms=%.3f",
                     capture_ms, timing.frontend_ms, timing.gate_ms,
                     timing.quantize_ms, inference_ms,
                     timing.preprocess_total_ms + inference_ms,
                     (esp_timer_get_time() - command_start) / 1000.0f);
        } else if (std::memcmp(header.magic, "PCM1", 4) == 0) {
            if (header.payload_bytes != kPcmPayloadBytes) {
                ESP_LOGE(kTag, "PCM1 length %u, expected %u",
                         static_cast<unsigned>(header.payload_bytes),
                         static_cast<unsigned>(kPcmPayloadBytes));
                discard_bytes(header.payload_bytes);
                continue;
            }
            read_exact(frontend.pcm, kPcmPayloadBytes);
            ESP_LOGI(kTag, "RECEIVED source=on_device_pcm bytes=%u",
                     static_cast<unsigned>(header.payload_bytes));
            const int64_t compute_start = esp_timer_get_time();
            qg_metrics_t quality_metrics{};
            PipelineTiming timing{};
            const qg_result_t quality_result =
                preprocess_pcm(frontend, window, mel_fb, input, quality_config,
                               quality_metrics, timing);
            if (quality_result != QG_PASS) {
                have_pcm_features = false;
                ESP_LOGI(kTag, "ACTION RETRY_RECORDING reason=%s", qg_reason(quality_result));
                ESP_LOGI(kTag,
                         "PIPELINE_TIMING source=on_device_pcm status=RETRY capture_ms=0.000 "
                         "frontend_ms=%.3f gate_ms=%.3f quantize_ms=%.3f "
                         "inference_ms=-1.000 compute_ms=%.3f end_to_end_ms=%.3f",
                         timing.frontend_ms, timing.gate_ms, timing.quantize_ms,
                         timing.preprocess_total_ms,
                         (esp_timer_get_time() - compute_start) / 1000.0f);
                log_memory("after_on_device_pcm_retry");
                continue;
            }
            // TFLM may reuse/mutate the input arena during Invoke(). Preserve
            // the pre-invoke bytes in the now-disposable PCM buffer for CMP1.
            std::memcpy(frontend.pcm, input->data.int8, kFeaturePayloadBytes);
            have_pcm_features = true;
            const float inference_ms =
                run_prediction(interpreter, input, output, "on_device_pcm");
            ESP_LOGI(kTag,
                     "PIPELINE_TIMING source=on_device_pcm status=PASS capture_ms=0.000 "
                     "frontend_ms=%.3f gate_ms=%.3f quantize_ms=%.3f "
                     "inference_ms=%.3f compute_ms=%.3f end_to_end_ms=%.3f",
                     timing.frontend_ms, timing.gate_ms, timing.quantize_ms,
                     inference_ms, timing.preprocess_total_ms + inference_ms,
                     (esp_timer_get_time() - compute_start) / 1000.0f);
        } else if (std::memcmp(header.magic, "FEA1", 4) == 0) {
            if (header.payload_bytes != kFeaturePayloadBytes) {
                ESP_LOGE(kTag, "FEA1 length %u, expected %u",
                         static_cast<unsigned>(header.payload_bytes),
                         static_cast<unsigned>(kFeaturePayloadBytes));
                discard_bytes(header.payload_bytes);
                continue;
            }
            read_exact(input->data.int8, kFeaturePayloadBytes);
            ESP_LOGI(kTag, "RECEIVED source=host_features bytes=%u qhash=%08" PRIx32,
                     static_cast<unsigned>(header.payload_bytes),
                     fnv1a(input->data.int8, input->bytes));
            const float inference_ms =
                run_prediction(interpreter, input, output, "host_features");
            ESP_LOGI(kTag,
                     "PIPELINE_TIMING source=host_features status=PASS capture_ms=0.000 "
                     "frontend_ms=0.000 gate_ms=0.000 quantize_ms=0.000 "
                     "inference_ms=%.3f compute_ms=%.3f end_to_end_ms=%.3f",
                     inference_ms, inference_ms, inference_ms);
            have_pcm_features = false;
        } else if (std::memcmp(header.magic, "CMP1", 4) == 0) {
            if (header.payload_bytes != kFeaturePayloadBytes) {
                ESP_LOGE(kTag, "CMP1 length %u, expected %u",
                         static_cast<unsigned>(header.payload_bytes),
                         static_cast<unsigned>(kFeaturePayloadBytes));
                discard_bytes(header.payload_bytes);
                continue;
            }
            auto *reference = reinterpret_cast<int8_t *>(frontend.logmel);
            read_exact(reference, kFeaturePayloadBytes);
            if (!have_pcm_features) {
                ESP_LOGE(kTag, "CMP1 requires a preceding PCM1 command");
                continue;
            }
            unsigned mismatched = 0;
            unsigned total_abs = 0;
            int max_abs = 0;
            const auto *device_features = reinterpret_cast<const int8_t *>(frontend.pcm);
            for (size_t i = 0; i < kFeaturePayloadBytes; ++i) {
                const int difference = std::abs(static_cast<int>(reference[i]) -
                                                static_cast<int>(device_features[i]));
                if (difference != 0) {
                    ++mismatched;
                }
                total_abs += static_cast<unsigned>(difference);
                max_abs = std::max(max_abs, difference);
            }
            ESP_LOGI(kTag,
                     "PARITY mismatched=%u/%u max_abs_q=%d mean_abs_q=%.8f "
                     "reference_qhash=%08" PRIx32 " device_qhash=%08" PRIx32,
                     mismatched, static_cast<unsigned>(kFeaturePayloadBytes), max_abs,
                     static_cast<double>(total_abs) / kFeaturePayloadBytes,
                     fnv1a(reference, kFeaturePayloadBytes),
                     fnv1a(device_features, kFeaturePayloadBytes));
        } else {
            ESP_LOGW(kTag, "unknown command magic %02x %02x %02x %02x",
                     static_cast<unsigned char>(header.magic[0]),
                     static_cast<unsigned char>(header.magic[1]),
                     static_cast<unsigned char>(header.magic[2]),
                     static_cast<unsigned char>(header.magic[3]));
            if (header.payload_bytes <= kPcmPayloadBytes) {
                discard_bytes(header.payload_bytes);
            }
        }
        log_memory("after_command");
    }
}
