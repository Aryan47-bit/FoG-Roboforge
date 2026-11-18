from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    dataset: str
    subject: str
    session: str
    recording: str
    source: Path
    fs: float
    timestamps: np.ndarray
    signals: np.ndarray
    labels: np.ndarray
    channels: tuple[str, ...]
    units: tuple[str, ...]

    @property
    def subject_key(self):
        return f"{self.dataset}:{self.subject}"

    def validate(self):
        n = len(self.timestamps)
        if not self.subject or not self.session or not self.recording:
            raise ValueError("Verified subject, session, and recording IDs are required")
        if self.fs <= 0 or n < 2:
            raise ValueError("Recording needs a positive rate and at least two samples")
        if self.signals.shape != (n, len(self.channels)) or self.labels.shape != (n,):
            raise ValueError("Signal, channel, timestamp, or label shape mismatch")
        if len(set(self.channels)) != len(self.channels) or len(self.units) != len(self.channels):
            raise ValueError("Channel names must be unique and have declared units")
        if not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if not np.isin(self.labels, [-1, 0, 1]).all():
            raise ValueError("Labels must be -1 (invalid), 0, or 1")
        dt = np.median(np.diff(self.timestamps))
        if not np.isclose(dt, 1 / self.fs, rtol=0.08):
            raise ValueError(f"Timestamp interval {dt} does not match {self.fs} Hz")
        return self
