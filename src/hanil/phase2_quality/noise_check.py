"""Phase 3.1 — Noise Check Algorithm.

Decides whether a recording contains too much non-breathing sound before it is
sent to the classifier. Pure-DSP, no learning — mirrors the fixed-point version
that runs on the ESP32. Two representations of each 10 ms frame are used:
  * band-passed 100–2000 Hz  -> RMS energy & its variance (breathing presence)
  * wideband, DC-removed      -> spectral flatness & zero-crossing rate (noisiness)

Why two: measuring flatness/ZCR on the *band-passed* signal flattens broadband
hiss inside the band and hides it. Broadband noise (fan, hiss, rustle) is only
distinguishable as "flat + high-ZCR" when looked at wideband. The band-passed
energy is what tells us whether in-band breathing is actually present.

Decision (matches the project spec):
  * RMS variance < rms_var_min                  -> no_breathing (flat line)
  * flatness > flatness_max AND ZCR > zcr_max   -> too_noisy (random/hiss)
  * otherwise                                   -> pass to the SNR check

Thresholds are defaults calibrated on synthetic breath/white/flatline signals;
re-calibrate from labelled good/bad clips via an ROC sweep and lock for deployment.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfilt


@dataclass
class NoiseCheckConfig:
    sample_rate: int = 16000
    band: tuple = (100.0, 2000.0)   # lung-sound band for the energy/RMS analysis
    highpass_hz: float = 100.0      # DC/rumble removal for the flatness/ZCR analysis
    frame_ms: float = 10.0
    flatness_max: float = 0.50      # mean wideband flatness above -> too noisy
    zcr_max: float = 0.15           # mean wideband ZCR above -> hiss/rustle
    rms_var_min: float = 1e-5       # in-band RMS variance below -> flat line


@dataclass
class NoiseCheckResult:
    passed: bool
    reason: str
    mean_flatness: float
    mean_zcr: float
    rms_variance: float


class NoiseChecker:
    def __init__(self, cfg: NoiseCheckConfig = NoiseCheckConfig()):
        self.cfg = cfg
        nyq = cfg.sample_rate / 2.0
        self.sos_band = butter(4, [cfg.band[0] / nyq, cfg.band[1] / nyq],
                               btype="band", output="sos")
        self.sos_hp = butter(2, cfg.highpass_hz / nyq, btype="high", output="sos")
        self.frame = int(cfg.sample_rate * cfg.frame_ms / 1000.0)

    def _frames(self, x: np.ndarray) -> np.ndarray:
        n = len(x) // self.frame
        return x[: n * self.frame].reshape(n, self.frame) if n else x[None, :]

    @staticmethod
    def _spectral_flatness(frame: np.ndarray) -> float:
        mag = np.abs(np.fft.rfft(frame)) + 1e-10
        gmean = np.exp(np.mean(np.log(mag)))
        amean = np.mean(mag)
        return float(gmean / amean)

    @staticmethod
    def _zcr(frame: np.ndarray) -> float:
        return float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))

    def check(self, wav: np.ndarray) -> NoiseCheckResult:
        raw = np.asarray(wav, dtype=np.float64).flatten()
        x_band = sosfilt(self.sos_band, raw)     # for energy / breathing presence
        x_wide = sosfilt(self.sos_hp, raw)       # for flatness / ZCR (noisiness)
        fb, fw = self._frames(x_band), self._frames(x_wide)
        rms = np.sqrt(np.mean(fb ** 2, axis=1) + 1e-12)
        flat = np.array([self._spectral_flatness(f) for f in fw])
        zcr = np.array([self._zcr(f) for f in fw])

        mean_flat, mean_zcr, rms_var = float(flat.mean()), float(zcr.mean()), float(rms.var())
        c = self.cfg
        if rms_var < c.rms_var_min:
            return NoiseCheckResult(False, "no_breathing", mean_flat, mean_zcr, rms_var)
        if mean_flat > c.flatness_max and mean_zcr > c.zcr_max:
            return NoiseCheckResult(False, "too_noisy", mean_flat, mean_zcr, rms_var)
        return NoiseCheckResult(True, "ok", mean_flat, mean_zcr, rms_var)
