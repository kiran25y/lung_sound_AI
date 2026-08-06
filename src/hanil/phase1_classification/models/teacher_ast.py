"""Phase 2.5 — Cloud teacher: Audio Spectrogram Transformer (AudioSet-pretrained).

Wraps HuggingFace AST. Trained only to distil the student; never deployed.
Falls back to a small ViT-like transformer if `transformers` is unavailable so
the repo still imports in minimal environments.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ASTTeacher(nn.Module):
    def __init__(self, num_classes=3, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        try:
            from transformers import ASTModel
            self.backbone = ASTModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self._hf = True
        except Exception:                       # offline / no transformers
            self.backbone = _TinyTransformer()
            hidden = self.backbone.hidden
            self._hf = False
        self.embed_dim = hidden
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, spec, return_embedding: bool = False):
        # spec: [B, 1, n_mels, frames]. AST expects [B, frames, n_mels].
        x = spec.squeeze(1).transpose(1, 2)
        if self._hf:
            out = self.backbone(input_values=x).pooler_output
        else:
            out = self.backbone(x)
        logits = self.head(out)
        if return_embedding:
            return logits, out
        return logits


class _TinyTransformer(nn.Module):
    """Lightweight stand-in used only when HF AST cannot be downloaded."""

    def __init__(self, n_mels=64, hidden=192, depth=4, heads=4):
        super().__init__()
        self.hidden = hidden
        self.proj = nn.Linear(n_mels, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)

    def forward(self, x):                        # x [B, frames, n_mels]
        h = self.proj(x)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.enc(h)
        return h[:, 0]
