if(HANIL_AI_LIBRARY_EXAMPLE)
    idf_component_register(SRCS "library_package_example.cc"
                           INCLUDE_DIRS "."
                           REQUIRES hanil_ai)
elseif(HANIL_CLASSIFICATION_ONLY_SRAM)
    idf_component_register(SRCS "classification_only_sram.cc"
                           INCLUDE_DIRS "."
                           REQUIRES esp_driver_usb_serial_jtag esp_timer
                                    espressif__esp-tflite-micro
                           EMBED_FILES "model/sram_student_int8.tflite")
    if(HANIL_CLASSIFIER_COPY_MODEL_TO_SRAM)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE
                                   HANIL_CLASSIFIER_COPY_MODEL_TO_SRAM=1)
    endif()
elseif(HANIL_STUDENT_PIPELINE_SRAM)
    idf_component_register(SRCS "main.cc" "quality_gate.c"
                           INCLUDE_DIRS "."
                           REQUIRES esp_driver_i2s esp_driver_usb_serial_jtag esp_timer esp_psram
                                    espressif__esp-tflite-micro
                           EMBED_FILES "model/sram_student_int8.tflite"
                                       "frontend/mel_fb.npy"
                                       "frontend/window.npy")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
                               HANIL_STUDENT_PIPELINE_SRAM=1)
    if(HANIL_PIPELINE_COPY_MODEL_TO_SRAM)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE
                                   HANIL_PIPELINE_COPY_MODEL_TO_SRAM=1)
    endif()
else()
    idf_component_register(SRCS "main.cc" "quality_gate.c"
                           INCLUDE_DIRS "."
                           REQUIRES esp_driver_i2s esp_driver_usb_serial_jtag esp_timer esp_psram
                                    espressif__esp-tflite-micro
                           EMBED_FILES "model/model_int8_tflm.tflite"
                                       "frontend/mel_fb.npy"
                                       "frontend/window.npy")
endif()
