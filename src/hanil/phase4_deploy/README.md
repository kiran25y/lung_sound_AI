# Phase 4 — Deployment (ESP32-S3)

Full export chain (PyTorch training -> INT8 TFLite for the ESP32-S3):

```
best.pt (PyTorch student)
   │  export_onnx.py        →  student.onnx
   │  onnx_to_tf.py         →  saved_model/
   │  quantize_tflite.py    →  student_int8.tflite   (representative dataset = both devices + noise)
   │  verify_tflite.py      →  FP32 vs INT8 parity report
   ▼
 esp32/  (xxd -i model  →  model_data.cc  →  ESP-IDF + ESP-TFLite-Micro + ESP-NN)
```

## 5.4 Deployment strategy (ESP32-only)
- **Ship:** the 0.6–1.2M INT8 student. Teacher (AST) stays in the cloud.
- **Phone:** the same student runs via TFLite/LiteRT (Android) or Core ML (iOS) — a
  few ms, no quantization needed. Reuse the Phase-2 quality gate before inference.
- **OTA:** push new `.tflite` over the ESP32-S3's built-in WiFi.

## 5.5 Quantization notes
- INT8 with int8 IO is required for ESP-NN acceleration.
- Budget 1–3% accuracy loss; check per-class crackle/wheeze sensitivity. If it drops
  more than ~2–3 pts, retrain with quantization-aware training (QAT).
- Representative dataset must cover both devices + the noise range (handled in
  `quantize_tflite.py`).

## One-liners
```bash
python -m hanil.phase4_deploy.export_onnx    --config configs/exp3_final.yaml --ckpt runs/final/best.pt --onnx runs/export/student.onnx
python -m hanil.phase4_deploy.onnx_to_tf     --onnx runs/export/student.onnx --saved_model runs/export/saved_model
python -m hanil.phase4_deploy.quantize_tflite --config configs/exp3_final.yaml --saved_model runs/export/saved_model --tflite runs/export/student_int8.tflite
python -m hanil.phase4_deploy.verify_tflite  --config configs/exp3_final.yaml --ckpt runs/final/best.pt --tflite runs/export/student_int8.tflite
```
