cmake_minimum_required(VERSION 3.16)

list(APPEND EXTRA_COMPONENT_DIRS
     "${CMAKE_CURRENT_LIST_DIR}/../../release/hanil_ai_esp32s3"
     "${CMAKE_CURRENT_LIST_DIR}/../../managed_components/espressif__esp-tflite-micro")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
idf_build_set_property(MINIMAL_BUILD ON)
set(SDKCONFIG_DEFAULTS "${CMAKE_CURRENT_LIST_DIR}/../../sdkconfig.defaults")
project(hanil_ai_prebuilt_smoke)

idf_component_get_property(tflm_component espressif__esp-tflite-micro COMPONENT_LIB)
target_compile_options(${tflm_component} PRIVATE -UESP_NN)

