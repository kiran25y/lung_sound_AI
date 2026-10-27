#!/usr/bin/env python3
"""feature_config.py — the SINGLE source of truth for audio -> log-Mel preprocessing.

Why this file exists: on-device accuracy depends on the firmware mel EXACTLY matching the
training mel. The Slaney-vs-HTK mel formula, the window, `center` padding, the power, the log
offset and the per-clip normalization are all silent mismatch traps. This file:

  1. Pins every parameter in one config (FeatureConfig).
  2. Provides a pure-numpy reference `logmel_numpy` that mirrors what firmware must compute
     (the executable spec the C/ESP-DSP code is validated against).
  3. Exports the exact Mel filterbank + window as tables (mel_fb.npy / window.npy / .h),
     so firmware uses the SAME coefficients instead of re-deriving Slaney — removing the
     single biggest source of parity error.

Deployment note: on ESP32-S3 the mel is computed in firmware (ESP-DSP FFT), and the quantized
model takes the log-Mel feature. TFLite-Micro does not reliably support STFT/mel ops, so do
NOT bake the mel into the model. Match `logmel_numpy` in firmware, then feed its output to the
INT8 model.

  python feature_config.py --list                       # print every parameter + library
  python feature_config.py --export-fb --out fw/         # write mel_fb / window tables for firmware
  python feature_config.py --selftest --wav clip.wav     # numpy-vs-torchaudio parity (needs torch)
"""
from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass, asdict, field
import numpy as np


@dataclass
class FeatureConfig:
    # --- library / provenance ---
    reference_impl: str = "torchaudio.transforms.MelSpectrogram + natural-log + per-clip norm"
    firmware_target: str = "ESP32-S3 / ESP-DSP FFT (mel in firmware, model takes log-Mel)"
    # --- framing ---
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    n_fft: int = 400            # 25 ms
    win_length: int = 400
    hop_length: int = 160       # 10 ms
    window: str = "hann_periodic"   # periodic Hann (torch default, periodic=True)
    center: bool = True             # reflect-pad n_fft//2 each side (replicable on a full 4 s buffer)
    pad_mode: str = "reflect"
    # --- spectrum ---
    power: float = 2.0          # power spectrogram (|.|^2)
    # --- mel filterbank ---
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 2000.0
    mel_scale: str = "slaney"   # slaney (NOT htk) — must match firmware table
    mel_norm: str = "slaney"    # triangular-area normalization
    # --- log + normalization ---
    log_type: str = "ln"        # natural log
    log_offset: float = 1e-6    # log(mel + offset)
    normalize: str = "per_clip_meanstd"   # (x - mean)/(std + norm_eps) over (mel, time)
    norm_eps: float = 1e-5

    @property
    def n_frames(self):
        n = int(self.clip_seconds * self.sample_rate)
        if self.center:
            n = n + 2 * (self.n_fft // 2)
        return 1 + (n - self.n_fft) // self.hop_length

    def to_json(self, path):
        json.dump(asdict(self), open(path, "w"), indent=2)
        return path

    @classmethod
    def from_json(cls, path):
        return cls(**{k: v for k, v in json.load(open(path)).items()
                      if k in cls.__dataclass_fields__})


DEFAULT = FeatureConfig()


# ── Mel filterbank (Slaney) — matches torchaudio mel_scale='slaney', norm='slaney' ──
def _hz_to_mel_slaney(f):
    f = np.asarray(f, float); f_sp = 200.0 / 3.0
    mel = f / f_sp
    log_t = f >= 1000.0
    mel[log_t] = 15.0 + np.log(f[log_t] / 1000.0) / (np.log(6.4) / 27.0)
    return mel


def _mel_to_hz_slaney(mel):
    mel = np.asarray(mel, float); f_sp = 200.0 / 3.0
    hz = f_sp * mel
    log_t = mel >= 15.0
    hz[log_t] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mel[log_t] - 15.0))
    return hz


def mel_filterbank(cfg: FeatureConfig = DEFAULT):
    """Return the [n_mels, n_fft//2+1] Slaney mel matrix (with slaney area norm)."""
    n_freq = cfg.n_fft // 2 + 1
    fft_hz = np.linspace(0, cfg.sample_rate / 2.0, n_freq)
    m_min, m_max = _hz_to_mel_slaney([cfg.f_min])[0], _hz_to_mel_slaney([cfg.f_max])[0]
    m_pts = np.linspace(m_min, m_max, cfg.n_mels + 2)
    hz_pts = _mel_to_hz_slaney(m_pts)
    fb = np.zeros((cfg.n_mels, n_freq), dtype=np.float64)
    for i in range(cfg.n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_hz - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_hz) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
        if cfg.mel_norm == "slaney":
            fb[i] *= 2.0 / max(hi - lo, 1e-9)      # area normalization
    return fb.astype(np.float32)


def _hann_periodic(n):
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float64)


def logmel_numpy(wav, cfg: FeatureConfig = DEFAULT, fb=None):
    """FIRMWARE REFERENCE: raw waveform -> log-Mel [n_mels, n_frames]. Pure numpy.
    Firmware (ESP-DSP) must reproduce this exactly (using the exported fb/window tables)."""
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if cfg.center:
        pad = cfg.n_fft // 2
        x = np.pad(x, pad, mode=cfg.pad_mode)
    win = _hann_periodic(cfg.win_length)
    fb = mel_filterbank(cfg) if fb is None else fb
    n_freq = cfg.n_fft // 2 + 1
    n_frames = 1 + (len(x) - cfg.n_fft) // cfg.hop_length
    mel = np.empty((cfg.n_mels, n_frames), dtype=np.float64)
    for t in range(n_frames):
        frame = x[t * cfg.hop_length: t * cfg.hop_length + cfg.n_fft] * win
        spec = np.abs(np.fft.rfft(frame, n=cfg.n_fft)) ** cfg.power   # [n_freq]
        mel[:, t] = fb @ spec
    logmel = np.log(mel + cfg.log_offset)
    if cfg.normalize == "per_clip_meanstd":
        logmel = (logmel - logmel.mean()) / (logmel.std() + cfg.norm_eps)
    return logmel.astype(np.float32)


def logmel_torch(wav, cfg: FeatureConfig = DEFAULT):
    """The training front end (torchaudio) built from the SAME config — for parity checks."""
    import torch, torchaudio
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length,
        hop_length=cfg.hop_length, f_min=cfg.f_min, f_max=cfg.f_max, n_mels=cfg.n_mels,
        power=cfg.power, norm=cfg.mel_norm, mel_scale=cfg.mel_scale, center=cfg.center,
        pad_mode=cfg.pad_mode)
    w = torch.as_tensor(np.asarray(wav, dtype=np.float32)).reshape(1, -1)
    m = torch.log(mel(w) + cfg.log_offset)
    m = (m - m.mean()) / (m.std() + cfg.norm_eps)
    return m.squeeze(0).numpy()


# ── firmware table export ────────────────────────────────────────────────────
def export_tables(cfg: FeatureConfig, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fb = mel_filterbank(cfg); win = _hann_periodic(cfg.win_length).astype(np.float32)
    np.save(os.path.join(out_dir, "mel_fb.npy"), fb)
    np.save(os.path.join(out_dir, "window.npy"), win)
    cfg.to_json(os.path.join(out_dir, "feature_config.json"))
    # C headers so firmware links the EXACT coefficients (no Slaney re-derivation)
    with open(os.path.join(out_dir, "mel_fb.h"), "w") as f:
        f.write(f"// Slaney mel filterbank [{fb.shape[0]}][{fb.shape[1]}], sr={cfg.sample_rate}, "
                f"n_fft={cfg.n_fft}, f=[{cfg.f_min},{cfg.f_max}] Hz\n")
        f.write(f"#define MEL_N {fb.shape[0]}\n#define MEL_FREQ {fb.shape[1]}\n")
        f.write("static const float MEL_FB[MEL_N][MEL_FREQ] = {\n")
        for row in fb:
            f.write("  {" + ",".join(f"{v:.8g}f" for v in row) + "},\n")
        f.write("};\n")
    with open(os.path.join(out_dir, "window.h"), "w") as f:
        f.write(f"#define WIN_N {len(win)}\nstatic const float HANN_WIN[WIN_N] = {{\n  ")
        f.write(",".join(f"{v:.8g}f" for v in win)); f.write("\n};\n")
    print(f"[export] mel_fb.npy/.h [{fb.shape}], window.npy/.h [{len(win)}], "
          f"feature_config.json -> {out_dir}")


def print_table(cfg: FeatureConfig):
    print("PREPROCESSING PARAMETERS (single source of truth)")
    print("-" * 68)
    for k, v in asdict(cfg).items():
        print(f"  {k:<18} = {v}")
    print(f"  {'n_frames (derived)':<18} = {cfg.n_frames}")
    print("-" * 68)
    print("Firmware must reproduce logmel_numpy() using the exported mel_fb/window tables.")


def main():
    ap = argparse.ArgumentParser(description="Feature-extraction config, reference & firmware export")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--export-fb", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="numpy vs torchaudio parity")
    ap.add_argument("--config", default=None, help="load a FeatureConfig JSON (else defaults)")
    ap.add_argument("--wav", default=None, help="a wav for the parity self-test")
    ap.add_argument("--out", default="fw")
    args = ap.parse_args()
    cfg = FeatureConfig.from_json(args.config) if args.config else DEFAULT
    if args.list or not (args.export_fb or args.selftest):
        print_table(cfg)
    if args.export_fb:
        export_tables(cfg, args.out)
    if args.selftest:
        if args.wav:
            import soundfile as sf
            w, sr = sf.read(args.wav, dtype="float32", always_2d=True); w = w.mean(1)
            if sr != cfg.sample_rate:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(int(sr), cfg.sample_rate); w = resample_poly(w, cfg.sample_rate // g, int(sr) // g)
            n = int(cfg.clip_seconds * cfg.sample_rate)
            w = np.pad(w, (0, max(0, n - len(w))))[:n]
        else:
            rng = np.random.default_rng(0); t = np.arange(int(cfg.clip_seconds * cfg.sample_rate))
            w = (0.2 * np.sin(2 * np.pi * 300 * t / cfg.sample_rate) + rng.normal(0, 0.02, len(t))).astype("float32")
        a = logmel_numpy(w, cfg)
        try:
            b = logmel_torch(w, cfg)
            d = np.abs(a - b)
            print(f"[selftest] numpy vs torchaudio: shape {a.shape} vs {b.shape}  "
                  f"max|Δ|={d.max():.4f}  mean|Δ|={d.mean():.4f}  "
                  f"({'MATCH' if d.max() < 0.05 else 'CHECK filterbank/window/center'})")
        except Exception as e:
            print(f"[selftest] numpy OK (shape {a.shape}); torchaudio unavailable ({e})")


if __name__ == "__main__":
    main()