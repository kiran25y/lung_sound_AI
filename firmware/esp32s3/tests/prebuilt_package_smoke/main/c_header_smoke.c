#include "hanil_ai_types.h"
#include "hanil_classifier.h"
#include "hanil_frontend.h"
#include "hanil_microphone.h"
#include "hanil_pipeline.h"
#include "hanil_quality_gate.h"

/* Compile-only proof that every public header is valid from C as well as C++. */
int hanil_ai_c_header_smoke(void)
{
    hanil_pipeline_config_t config = hanil_pipeline_default_config();
    return (config.microphone.sample_rate == HANIL_AI_SAMPLE_RATE) ? 0 : 1;
}
