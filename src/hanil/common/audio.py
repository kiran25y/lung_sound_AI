"""Audio IO + fixed-length cropping. Everything is resampled to a single rate."""
from __future__ import annotations
import numpy as np
import torch
import torchaudio


def _read_audio(path: str):
    """Load (wav[C, T] float32, sr). soundfile first (no torchcodec dependency),
    torchaudio as fallback for formats soundfile can't read."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)   # [T, C]
        return torch.from_numpy(data.T.copy()), sr                  # [C, T]
    except Exception:
        wav, sr = torchaudio.load(path)
        return wav.float(), sr


def load_wav(path: str, target_sr: int) -> torch.Tensor:
    """Return mono waveform [1, T] at target_sr, peak-normalised."""
    wav, sr = _read_audio(path)
    if wav.shape[0] > 1:                      # downmix to mono
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav


def fix_length(wav: torch.Tensor, num_samples: int, training: bool = True) -> torch.Tensor:
    """Crop (random in training, centre otherwise) or zero-pad to num_samples."""
    t = wav.shape[-1]
    if t == num_samples:
        return wav
    if t > num_samples:
        start = int(torch.randint(0, t - num_samples + 1, (1,))) if training else (t - num_samples) // 2
        return wav[..., start:start + num_samples]
    pad = num_samples - t
    left = 0 if training else pad // 2
    return torch.nn.functional.pad(wav, (left, pad - left))


def rms(x: np.ndarray | torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        return float(torch.sqrt(torch.mean(x.float() ** 2) + 1e-12))
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
