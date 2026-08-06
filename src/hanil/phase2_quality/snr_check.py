"""Phase 3.2 — Real-time SNR Check.

Streaming SNR estimate over a 1 s sliding window, updated every 100 ms. Signal =
low-band (100–600 Hz) energy where breathing lives; noise floor = EMA of the
minimum band energy in quiet gaps. Returns SNR in dB. Hysteresis is applied by the
alert layer, not here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfilt


@dataclass
class SNRConfig:
    sample_rate: int = 16000
    signal_band: tuple = (100.0, 600.0)
    window_s: float = 1.0
    hop_s: float = 0.1
    ema_alpha: float = 0.1


class RealtimeSNR:
    def __init__(self, cfg: SNRConfig = SNRConfig()):
        self.cfg = cfg
        nyq = cfg.sample_rate / 2.0
        self.sos = butter(4, [cfg.signal_band[0] / nyq, cfg.signal_band[1] / nyq],
                          btype="band", output="sos")
        self.win = int(cfg.sample_rate * cfg.window_s)
        self.hop = int(cfg.sample_rate * cfg.hop_s)
        self.noise_floor = None
        self._buf = np.zeros(0, dtype=np.float64)

    def reset(self):
        self.noise_floor = None
        self._buf = np.zeros(0, dtype=np.float64)

    def _band_energy(self, frame: np.ndarray) -> float:
        y = sosfilt(self.sos, frame)
        return float(np.mean(y ** 2) + 1e-12)

    def update(self, samples: np.ndarray) -> list:
        """Feed a chunk of samples; return list of SNR_dB readings (one per hop)."""
        self._buf = np.concatenate([self._buf, np.asarray(samples, np.float64).flatten()])
        out = []
        while len(self._buf) >= self.win:
            frame = self._buf[: self.win]
            energy = self._band_energy(frame)
            if self.noise_floor is None:
                self.noise_floor = energy
            else:
                # track the floor downward quickly, upward slowly
                a = self.cfg.ema_alpha
                self.noise_floor = min(self.noise_floor,
                                       (1 - a) * self.noise_floor + a * energy)
            snr_db = 10.0 * np.log10(energy / max(self.noise_floor, 1e-12))
            out.append(float(snr_db))
            self._buf = self._buf[self.hop:]
        return out
