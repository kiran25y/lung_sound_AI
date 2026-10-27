#pragma once

#include "hanil_ai_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hanil_classifier hanil_classifier_t;

typedef struct {
    bool copy_model_to_internal_sram;
    size_t tensor_arena_bytes;
    float crackle_threshold;
    float wheeze_threshold;
} hanil_classifier_config_t;

typedef struct {
    float crackle_probability;
    float wheeze_probability;
    float abnormal_probability;
    bool crackle_detected;
    bool wheeze_detected;
    const char *class2;
    const char *class3;
    int8_t output_quantized[2];
    uint32_t input_hash;
    float inference_ms;
} hanil_classification_t;

hanil_classifier_config_t hanil_classifier_default_config(void);
hanil_ai_status_t hanil_classifier_create(
    const hanil_classifier_config_t *config, hanil_classifier_t **classifier);
void hanil_classifier_destroy(hanil_classifier_t *classifier);
float hanil_classifier_input_scale(const hanil_classifier_t *classifier);
int hanil_classifier_input_zero_point(const hanil_classifier_t *classifier);
size_t hanil_classifier_arena_used(const hanil_classifier_t *classifier);
size_t hanil_classifier_model_size(const hanil_classifier_t *classifier);
int8_t *hanil_classifier_input_buffer(hanil_classifier_t *classifier,
                                      size_t *feature_count);
hanil_ai_status_t hanil_classifier_predict(
    hanil_classifier_t *classifier, const int8_t *features,
    size_t feature_count, hanil_classification_t *classification);

#ifdef __cplusplus
}
#endif

