idf_component_register(
    SRCS
        "src/hanil_ai_types.cc"
        "src/hanil_classifier.cc"
        "src/hanil_frontend.cc"
        "src/hanil_microphone.cc"
        "src/hanil_pipeline.cc"
        "src/hanil_quality_gate.c"
    INCLUDE_DIRS "include"
    REQUIRES
        esp_driver_i2s
        esp_timer
        esp_psram
        espressif__esp-tflite-micro
    EMBED_FILES
        "assets/sram_student_int8.tflite"
        "assets/mel_fb.npy"
        "assets/window.npy"
)

target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)

if(HANIL_PIPELINE_COPY_MODEL_TO_SRAM)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
                               HANIL_PIPELINE_COPY_MODEL_TO_SRAM=1)
endif()

