# Final ESP32 S3 AI Release

This directory contains only the final checkpoint, deployable INT8 model,
flashable ESP32-S3 binaries, final result files, and final engineering report.

## Contents

- `checkpoint/sram_student_best.pt`: final PyTorch SRAM-student checkpoint.
- `model/sram_student_int8.tflite`: final full-integer INT8 TFLite model.
- `firmware/`: validated ESP32-S3 full-pipeline binaries and flash arguments.
- `results/training_results.json`: final training and clean-set metrics.
- `results/int8_parity.json`: FP32-to-INT8 quantization parity result.
- `results/esp32_full_pipeline.json`: final on-board memory, latency, gate,
  parity, accuracy, and microphone test result.
- `report/Pediatric_LungSound_AI_Final_Report.docx`: final simplified report.
- `SHA256SUMS.txt`: integrity hashes for all release artifacts.

## Flash

From the repository root, with `esptool.py` installed and the board connected
to `COM4`:

```powershell
esptool.py --chip esp32s3 -p COM4 -b 460800 write_flash `
  --flash-mode dio --flash-freq 80m --flash-size 16MB `
  0x0 release/final_ai/firmware/bootloader/bootloader.bin `
  0x8000 release/final_ai/firmware/partition_table/partition-table.bin `
  0x10000 release/final_ai/firmware/model_test.bin
```

The build targets an ESP32-S3 with 16 MB flash and 8 MB octal PSRAM. It uses
portable TFLite Micro INT8 kernels because the optimized ESP-NN path has not
yet passed output-parity validation.

## Important limitation

This is an engineering-validation release. The live microphone path works, but
the final chestpiece still needs acoustic-coupling and quality-gate calibration
before real clinical recordings can be trusted.
