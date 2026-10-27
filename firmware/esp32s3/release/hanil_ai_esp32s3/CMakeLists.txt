idf_component_register(
    INCLUDE_DIRS "include"
    REQUIRES esp_driver_i2s esp_timer esp_psram espressif__esp-tflite-micro
)

add_prebuilt_library(hanil_ai_prebuilt "lib/libhanil_ai.a"
                     REQUIRES esp_driver_i2s esp_timer esp_psram
                              espressif__esp-tflite-micro)
target_link_libraries(${COMPONENT_LIB} INTERFACE hanil_ai_prebuilt)

