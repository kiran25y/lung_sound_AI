#include "hanil_classifier.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const uint8_t hanil_model_start[]
    asm("_binary_sram_student_int8_tflite_start");
extern const uint8_t hanil_model_end[]
    asm("_binary_sram_student_int8_tflite_end");

namespace {

constexpr size_t kDefaultArenaBytes = 118 * 1024;

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

struct hanil_classifier {
    hanil_classifier_config_t config{};
    uint8_t *model_copy = nullptr;
    uint8_t *arena = nullptr;
    size_t model_bytes = 0;
    tflite::MicroMutableOpResolver<4> *resolver = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
};

extern "C" hanil_classifier_config_t hanil_classifier_default_config(void)
{
    hanil_classifier_config_t config{};
    config.copy_model_to_internal_sram = true;
    config.tensor_arena_bytes = kDefaultArenaBytes;
    config.crackle_threshold = 0.49f;
    config.wheeze_threshold = 0.28f;
    return config;
}

extern "C" hanil_ai_status_t hanil_classifier_create(
    const hanil_classifier_config_t *config, hanil_classifier_t **classifier_out)
{
    if (!classifier_out) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    *classifier_out = nullptr;
    const hanil_classifier_config_t selected =
        config ? *config : hanil_classifier_default_config();
    if (selected.tensor_arena_bytes == 0 ||
        selected.crackle_threshold < 0.0f || selected.crackle_threshold > 1.0f ||
        selected.wheeze_threshold < 0.0f || selected.wheeze_threshold > 1.0f) {
        return HANIL_AI_INVALID_ARGUMENT;
    }

    auto *classifier = new (std::nothrow) hanil_classifier_t();
    if (!classifier) {
        return HANIL_AI_OUT_OF_MEMORY;
    }
    classifier->config = selected;
    classifier->model_bytes =
        static_cast<size_t>(hanil_model_end - hanil_model_start);
    const uint8_t *model_data = hanil_model_start;
    if (selected.copy_model_to_internal_sram) {
        classifier->model_copy = static_cast<uint8_t *>(heap_caps_aligned_alloc(
            16, classifier->model_bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
        if (!classifier->model_copy) {
            hanil_classifier_destroy(classifier);
            return HANIL_AI_OUT_OF_MEMORY;
        }
        std::memcpy(classifier->model_copy, hanil_model_start,
                    classifier->model_bytes);
        model_data = classifier->model_copy;
    }

    classifier->arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, selected.tensor_arena_bytes,
        MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
    if (!classifier->arena) {
        hanil_classifier_destroy(classifier);
        return HANIL_AI_OUT_OF_MEMORY;
    }

    const tflite::Model *model = tflite::GetModel(model_data);
    if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
        hanil_classifier_destroy(classifier);
        return HANIL_AI_MODEL_ERROR;
    }
    classifier->resolver =
        new (std::nothrow) tflite::MicroMutableOpResolver<4>();
    if (!classifier->resolver) {
        hanil_classifier_destroy(classifier);
        return HANIL_AI_OUT_OF_MEMORY;
    }
    classifier->resolver->AddConv2D();
    classifier->resolver->AddDepthwiseConv2D();
    classifier->resolver->AddMean();
    classifier->resolver->AddFullyConnected();

    classifier->interpreter = new (std::nothrow) tflite::MicroInterpreter(
        model, *classifier->resolver, classifier->arena,
        selected.tensor_arena_bytes);
    if (!classifier->interpreter ||
        classifier->interpreter->AllocateTensors() != kTfLiteOk) {
        hanil_classifier_destroy(classifier);
        return HANIL_AI_MODEL_ERROR;
    }
    classifier->input = classifier->interpreter->input(0);
    classifier->output = classifier->interpreter->output(0);
    if (!classifier->input || !classifier->output ||
        classifier->input->type != kTfLiteInt8 ||
        classifier->input->bytes != HANIL_AI_FEATURE_COUNT ||
        classifier->output->type != kTfLiteInt8 || classifier->output->bytes < 2) {
        hanil_classifier_destroy(classifier);
        return HANIL_AI_MODEL_ERROR;
    }
    *classifier_out = classifier;
    return HANIL_AI_OK;
}

extern "C" void hanil_classifier_destroy(hanil_classifier_t *classifier)
{
    if (!classifier) {
        return;
    }
    delete classifier->interpreter;
    delete classifier->resolver;
    heap_caps_free(classifier->arena);
    heap_caps_free(classifier->model_copy);
    delete classifier;
}

extern "C" float hanil_classifier_input_scale(
    const hanil_classifier_t *classifier)
{
    return classifier && classifier->input ? classifier->input->params.scale : 0.0f;
}

extern "C" int hanil_classifier_input_zero_point(
    const hanil_classifier_t *classifier)
{
    return classifier && classifier->input ? classifier->input->params.zero_point : 0;
}

extern "C" size_t hanil_classifier_arena_used(
    const hanil_classifier_t *classifier)
{
    return classifier && classifier->interpreter
               ? classifier->interpreter->arena_used_bytes()
               : 0;
}

extern "C" size_t hanil_classifier_model_size(
    const hanil_classifier_t *classifier)
{
    return classifier ? classifier->model_bytes : 0;
}

extern "C" int8_t *hanil_classifier_input_buffer(
    hanil_classifier_t *classifier, size_t *feature_count)
{
    if (feature_count) {
        *feature_count = classifier && classifier->input
                             ? classifier->input->bytes
                             : 0;
    }
    return classifier && classifier->input ? classifier->input->data.int8 : nullptr;
}

extern "C" hanil_ai_status_t hanil_classifier_predict(
    hanil_classifier_t *classifier, const int8_t *features,
    size_t feature_count, hanil_classification_t *classification)
{
    if (!classifier || !features || !classification ||
        feature_count != HANIL_AI_FEATURE_COUNT || !classifier->interpreter) {
        return HANIL_AI_INVALID_ARGUMENT;
    }
    std::memset(classification, 0, sizeof(*classification));
    if (features != classifier->input->data.int8) {
        std::memcpy(classifier->input->data.int8, features, feature_count);
    }
    classification->input_hash = fnv1a(features, feature_count);
    const int64_t start = esp_timer_get_time();
    if (classifier->interpreter->Invoke() != kTfLiteOk) {
        return HANIL_AI_MODEL_ERROR;
    }
    classification->inference_ms =
        static_cast<float>(esp_timer_get_time() - start) / 1000.0f;
    const float scale = classifier->output->params.scale;
    const int zero_point = classifier->output->params.zero_point;
    classification->output_quantized[0] = classifier->output->data.int8[0];
    classification->output_quantized[1] = classifier->output->data.int8[1];
    const float crackle_logit =
        (static_cast<int>(classification->output_quantized[0]) - zero_point) * scale;
    const float wheeze_logit =
        (static_cast<int>(classification->output_quantized[1]) - zero_point) * scale;
    classification->crackle_probability = 1.0f / (1.0f + expf(-crackle_logit));
    classification->wheeze_probability = 1.0f / (1.0f + expf(-wheeze_logit));
    classification->abnormal_probability =
        1.0f - (1.0f - classification->crackle_probability) *
                   (1.0f - classification->wheeze_probability);
    classification->crackle_detected =
        classification->crackle_probability >= classifier->config.crackle_threshold;
    classification->wheeze_detected =
        classification->wheeze_probability >= classifier->config.wheeze_threshold;
    classification->class2 =
        classification->crackle_detected || classification->wheeze_detected
            ? "abnormal"
            : "normal";
    classification->class3 = "normal";
    if (classification->crackle_detected && classification->wheeze_detected) {
        classification->class3 =
            classification->crackle_probability >= classification->wheeze_probability
                ? "crackle"
                : "wheeze";
    } else if (classification->crackle_detected) {
        classification->class3 = "crackle";
    } else if (classification->wheeze_detected) {
        classification->class3 = "wheeze";
    }
    return HANIL_AI_OK;
}

