#include "hanil_ai_types.h"

#include <cstring>

#include "esp_heap_caps.h"

extern "C" const char *hanil_ai_status_string(hanil_ai_status_t status)
{
    switch (status) {
        case HANIL_AI_OK: return "ok";
        case HANIL_AI_INVALID_ARGUMENT: return "invalid_argument";
        case HANIL_AI_OUT_OF_MEMORY: return "out_of_memory";
        case HANIL_AI_MODEL_ERROR: return "model_error";
        case HANIL_AI_FRONTEND_ERROR: return "frontend_error";
        case HANIL_AI_MICROPHONE_ERROR: return "microphone_error";
        case HANIL_AI_NOT_READY: return "not_ready";
        default: return "unknown";
    }
}

extern "C" void hanil_ai_get_memory(hanil_ai_memory_t *memory)
{
    if (!memory) {
        return;
    }
    std::memset(memory, 0, sizeof(*memory));
    constexpr uint32_t internal_caps = MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL;
    memory->internal_free_bytes = heap_caps_get_free_size(internal_caps);
    memory->internal_largest_block_bytes =
        heap_caps_get_largest_free_block(internal_caps);
    memory->internal_minimum_free_bytes =
        heap_caps_get_minimum_free_size(internal_caps);
    memory->internal_total_bytes = heap_caps_get_total_size(internal_caps);
    memory->psram_free_bytes = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    memory->psram_largest_block_bytes =
        heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    memory->psram_total_bytes = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
}

