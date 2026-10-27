cmake_minimum_required(VERSION 3.16)

option(HANIL_CLASSIFICATION_ONLY_SRAM
       "Build the student classifier-only internal-SRAM test" OFF)
option(HANIL_CLASSIFIER_COPY_MODEL_TO_SRAM
       "Copy classifier weights from flash into internal SRAM at boot" ON)
option(HANIL_STUDENT_PIPELINE_SRAM
       "Build the complete frontend + SRAM quality gate + student classifier" OFF)
option(HANIL_AI_LIBRARY_EXAMPLE
       "Build the reusable hanil_ai static-library integration example" OFF)
option(HANIL_PIPELINE_COPY_MODEL_TO_SRAM
       "Copy compact pipeline model weights from flash into internal SRAM" ON)

# Register the reusable project components before loading ESP-IDF's project
# machinery. This also keeps minimal builds deterministic when the component
# manager is disabled.
list(APPEND EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/components")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

if(HANIL_CLASSIFICATION_ONLY_SRAM OR HANIL_STUDENT_PIPELINE_SRAM OR
   HANIL_AI_LIBRARY_EXAMPLE)
    idf_build_set_property(MINIMAL_BUILD ON)
endif()

# Offline/recovery builds can disable the component manager after the locked
# managed components have already been downloaded.
if("$ENV{IDF_COMPONENT_MANAGER}" STREQUAL "0")
    list(APPEND EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/managed_components")
endif()

project(model_test)

# Keep portable integer kernels as the parity-safe default.  The switch allows
# an optimized ESP-NN candidate to be built and measured against the same
# static model without editing the source between experiments.
option(HANIL_USE_ESP_NN "Enable optimized ESP-NN TensorFlow Lite kernels" OFF)
idf_component_get_property(tflm_component espressif__esp-tflite-micro COMPONENT_LIB)
if(HANIL_USE_ESP_NN)
    message(STATUS "Hanil inference backend: optimized ESP-NN")
    idf_component_get_property(main_component main COMPONENT_LIB)
    target_compile_definitions(${main_component} PRIVATE HANIL_USE_ESP_NN=1)
else()
    message(STATUS "Hanil inference backend: portable TFLM integer kernels")
    target_compile_options(${tflm_component} PRIVATE -UESP_NN)
endif()

# Component-manager metadata normally injects this dependency. Recreate it for
# offline builds so the TFLM ESP-NN wrappers inherit esp_nn.h and link ESP-NN.
if("$ENV{IDF_COMPONENT_MANAGER}" STREQUAL "0" AND
   NOT HANIL_CLASSIFICATION_ONLY_SRAM AND
    NOT HANIL_STUDENT_PIPELINE_SRAM AND
    NOT HANIL_AI_LIBRARY_EXAMPLE)
    idf_component_get_property(tflm_component espressif__esp-tflite-micro COMPONENT_LIB)
    idf_component_get_property(esp_nn_component espressif__esp-nn COMPONENT_LIB)
    target_link_libraries(${tflm_component} PRIVATE ${esp_nn_component})
endif()
