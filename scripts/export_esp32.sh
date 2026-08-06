#!/usr/bin/env bash
set -e
CFG=configs/exp3_final.yaml
CKPT=runs/final/best.pt
mkdir -p runs/export
python -m hanil.phase4_deploy.export_onnx     --config $CFG --ckpt $CKPT --onnx runs/export/student.onnx
python -m hanil.phase4_deploy.onnx_to_tf      --onnx runs/export/student.onnx --saved_model runs/export/saved_model
python -m hanil.phase4_deploy.quantize_tflite --config $CFG --saved_model runs/export/saved_model --tflite runs/export/student_int8.tflite
python -m hanil.phase4_deploy.verify_tflite   --config $CFG --ckpt $CKPT --tflite runs/export/student_int8.tflite
