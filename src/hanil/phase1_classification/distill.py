"""Phase 2.5 — Knowledge distillation (teacher AST -> student CNN)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combines hard-label loss with KL(teacher||student) on softened logits."""

    def __init__(self, base_loss: nn.Module, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.base = base_loss
        self.t = temperature
        self.alpha = alpha

    def forward(self, student_logits, targets, teacher_logits=None):
        hard = self.base(student_logits, targets)
        if teacher_logits is None:
            return hard
        s = F.log_softmax(student_logits / self.t, dim=1)
        tch = F.softmax(teacher_logits / self.t, dim=1)
        soft = F.kl_div(s, tch, reduction="batchmean") * (self.t ** 2)
        return self.alpha * soft + (1.0 - self.alpha) * hard
