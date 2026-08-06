"""Unit tests for Phase 2 DSP quality modules (no torch needed)."""
import numpy as np
from hanil.phase2_quality.noise_check import NoiseChecker, NoiseCheckConfig
from hanil.phase2_quality.snr_check import RealtimeSNR, SNRConfig
from hanil.phase2_quality.alert import AlertStateMachine, AlertState


def _tone(freq, sr=16000, dur=1.0, amp=0.3):
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_noise_check_passes_breathing_like_signal():
    # a 300 Hz tone with amplitude modulation resembles in-band breathing energy
    sr = 16000
    t = np.arange(sr) / sr
    sig = (0.3 * (1 + 0.5 * np.sin(2 * np.pi * 0.4 * t))) * np.sin(2 * np.pi * 300 * t)
    res = NoiseChecker(NoiseCheckConfig(sample_rate=sr)).check(sig)
    assert res.passed is True
    assert res.reason == "ok"


def test_noise_check_rejects_flatline():
    sr = 16000
    res = NoiseChecker(NoiseCheckConfig(sample_rate=sr)).check(np.zeros(sr))
    assert res.passed is False
    assert res.reason == "no_breathing"


def test_noise_check_rejects_white_noise():
    sr = 16000
    rng = np.random.default_rng(0)
    res = NoiseChecker(NoiseCheckConfig(sample_rate=sr)).check(rng.normal(0, 0.3, sr))
    assert res.passed is False
    assert res.reason == "too_noisy"


def test_realtime_snr_returns_readings():
    sr = 16000
    snr = RealtimeSNR(SNRConfig(sample_rate=sr))
    readings = snr.update(_tone(300, sr, 2.0))
    assert len(readings) > 0
    assert all(np.isfinite(r) for r in readings)


def test_alert_hysteresis_needs_three_bad_windows():
    sm = AlertStateMachine(consecutive_red=3)
    assert sm.step(5.0).state == AlertState.ORANGE   # 1st bad
    assert sm.step(5.0).state == AlertState.ORANGE   # 2nd bad
    assert sm.step(5.0).state == AlertState.RED      # 3rd bad -> RED
    assert sm.step(25.0).state == AlertState.GREEN   # recovers, resets run


def test_alert_no_breathing():
    sm = AlertStateMachine()
    fb = sm.step(30.0, breathing_present=False)
    assert fb.state == AlertState.NO_BREATHING
    assert fb.vibrate == "double"
