#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HANIL_AI_SAMPLE_RATE 16000
#define HANIL_AI_AUDIO_SECONDS 4
#define HANIL_AI_AUDIO_SAMPLES 64000
#define HANIL_AI_MEL_BANDS 64
#define HANIL_AI_MEL_FRAMES 401
#define HANIL_AI_FEATURE_COUNT (HANIL_AI_MEL_BANDS * HANIL_AI_MEL_FRAMES)

typedef enum {
    HANIL_AI_OK = 0,
    HANIL_AI_INVALID_ARGUMENT,
    HANIL_AI_OUT_OF_MEMORY,
    HANIL_AI_MODEL_ERROR,
    HANIL_AI_FRONTEND_ERROR,
    HANIL_AI_MICROPHONE_ERROR,
    HANIL_AI_NOT_READY,
} hanil_ai_status_t;

typedef struct {
    float frontend_ms;
    float gate_ms;
    float quantize_ms;
    float inference_ms;
    float compute_ms;
    float capture_ms;
    float end_to_end_ms;
} hanil_ai_timing_t;

typedef struct {
    size_t internal_free_bytes;
    size_t internal_largest_block_bytes;
    size_t internal_minimum_free_bytes;
    size_t internal_total_bytes;
    size_t psram_free_bytes;
    size_t psram_largest_block_bytes;
    size_t psram_total_bytes;
} hanil_ai_memory_t;

const char *hanil_ai_status_string(hanil_ai_status_t status);
void hanil_ai_get_memory(hanil_ai_memory_t *memory);

#ifdef __cplusplus
}
#endif

