"""Phase 3.3 — Alert Logic & User Feedback.

Maps a stream of SNR readings + noise-check verdict to a user-facing state with
hysteresis (only escalate to RED after 3 consecutive bad windows = ~300 ms) so
brief noises don't trigger false alarms. Mirrors the on-device state machine.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class AlertState(Enum):
    GREEN = "green"        # SNR > 20 dB — good
    YELLOW = "yellow"      # 10–20 dB — acceptable, hold steady
    ORANGE = "orange"      # <10 dB once — warning
    RED = "red"            # <10 dB x3 — stop, move to quiet area
    NO_BREATHING = "no_breathing"


@dataclass
class AlertFeedback:
    state: AlertState
    vibrate: str           # none | short | double
    message: str


MESSAGES = {
    AlertState.GREEN:  ("none",   "Recording quality good"),
    AlertState.YELLOW: ("none",   "Some noise detected, hold steady"),
    AlertState.ORANGE: ("none",   "Noise increasing"),
    AlertState.RED:    ("short",  "Noise too high - move to a quiet area"),
    AlertState.NO_BREATHING: ("double", "No breathing detected"),
}


class AlertStateMachine:
    def __init__(self, red_threshold_db: float = 10.0, green_threshold_db: float = 20.0,
                 consecutive_red: int = 3):
        self.red_db = red_threshold_db
        self.green_db = green_threshold_db
        self.consecutive_red = consecutive_red
        self._bad_run = 0

    def reset(self):
        self._bad_run = 0

    def step(self, snr_db: float, breathing_present: bool = True) -> AlertFeedback:
        if not breathing_present:
            self._bad_run = 0
            return self._fb(AlertState.NO_BREATHING)
        if snr_db >= self.green_db:
            self._bad_run = 0
            return self._fb(AlertState.GREEN)
        if snr_db >= self.red_db:
            self._bad_run = 0
            return self._fb(AlertState.YELLOW)
        # below RED threshold
        self._bad_run += 1
        if self._bad_run >= self.consecutive_red:
            return self._fb(AlertState.RED)
        return self._fb(AlertState.ORANGE)

    @staticmethod
    def _fb(state: AlertState) -> AlertFeedback:
        vib, msg = MESSAGES[state]
        return AlertFeedback(state=state, vibrate=vib, message=msg)


def gate_recording(noise_result, snr_readings) -> bool:
    """Convenience gate: True = accept clip for classification."""
    if not noise_result.passed:
        return False
    if not snr_readings:
        return False
    # accept if median SNR is at least the yellow threshold
    med = sorted(snr_readings)[len(snr_readings) // 2]
    return med >= 10.0
