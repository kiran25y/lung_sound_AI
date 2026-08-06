"""Model factory — picks teacher/student/mobilenet from config."""
from __future__ import annotations
import torch.nn as nn

from ..mixstyle import MixStyle


def build_model(cfg) -> nn.Module:
    arch = cfg.model["arch"]
    num_classes = len(cfg.classes)
    if arch == "teacher_ast":
        from .teacher_ast import ASTTeacher
        return ASTTeacher(num_classes=num_classes)
    if arch == "student_cnn":
        from .student_cnn import StudentCNN
        ms = None
        dr = cfg.get_path("device_robust.mixstyle", {})
        if dr and dr.get("enable"):
            ms = MixStyle(p=float(dr.get("p", 0.5)), alpha=float(dr.get("alpha", 0.3)))
        return StudentCNN(num_classes=num_classes,
                          width=float(cfg.model.get("student_width", 1.0)),
                          dropout=float(cfg.model.get("dropout", 0.1)),
                          mixstyle=ms)
    if arch == "mobilenetv3s":
        import torchvision
        m = torchvision.models.mobilenet_v3_small(num_classes=num_classes)
        m.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1, bias=False)  # 1-channel input
        return m
    raise ValueError(f"unknown arch: {arch}")
