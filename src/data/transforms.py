from dataclasses import replace

import numpy as np
from scipy.signal import butter, sosfilt


def valid_segments(record):
    """Never bridge invalid annotations, nonfinite sensors, or recording gaps."""
    record.validate()
    valid = (record.labels >= 0) & np.isfinite(record.signals).all(axis=1)
    gaps = np.r_[False, np.diff(record.timestamps) > 1.6 / record.fs]
    start = None
    for i in range(len(valid) + 1):
        boundary = i == len(valid) or not valid[i] or gaps[i]
        if boundary and start is not None:
            if i - start >= 2:
                yield replace(record, recording=f"{record.recording}:{start}",
                              timestamps=record.timestamps[start:i], signals=record.signals[start:i],
                              labels=record.labels[start:i])
            start = None
        if i < len(valid) and valid[i] and start is None:
            start = i


def resample_causal(record, target_hz=100):
    """Causal low-pass plus previous-sample hold; no future input interpolation.

    Filter delay remains in the signals. Drop one second of filter warm-up.
    Call on a contiguous valid segment, never on a recording spanning gaps.
    """
    record.validate()
    if target_hz <= 0 or not np.isfinite(record.signals).all() or np.any(record.labels < 0):
        raise ValueError("Resampling requires positive rate and a valid finite segment")
    if np.any(np.diff(record.timestamps) > 1.6 / record.fs):
        raise ValueError("Split timing gaps before resampling")
    cutoff = 0.4 * min(record.fs, target_hz)
    filtered = sosfilt(butter(4, cutoff, fs=record.fs, output="sos"), record.signals, axis=0)
    t = np.arange(record.timestamps[0] + 1, record.timestamps[-1], 1 / target_hz)
    if len(t) < 2:
        raise ValueError("Segment too short after one-second filter warm-up")
    indices = np.searchsorted(record.timestamps, t, side="right") - 1
    return replace(record, fs=target_hz, timestamps=t, signals=filtered[indices].astype("float32"),
                   labels=record.labels[indices]).validate()


class Standardizer:
    """Streaming channel statistics fitted exclusively on training subjects."""

    def fit(self, records, splits):
        count, mean, m2 = 0, None, None
        self.channels = None
        for record in records:
            if record.subject_key not in splits:
                raise ValueError("Recording absent from split manifest")
            if splits[record.subject_key] != "train":
                raise ValueError("Standardizer may only fit training subjects")
            x = record.signals.astype("float64")
            if not np.isfinite(x).all() or not len(x):
                raise ValueError("Normalization needs finite, nonempty signals")
            if self.channels is None:
                self.channels, self.units = record.channels, record.units
                mean, m2 = np.zeros(x.shape[1]), np.zeros(x.shape[1])
            if (record.channels, record.units) != (self.channels, self.units):
                raise ValueError("Cannot combine different channel schemas or units")
            delta = x.mean(axis=0) - mean
            total = count + len(x)
            m2 += ((x - x.mean(axis=0)) ** 2).sum(axis=0) + delta ** 2 * count * len(x) / total
            mean += delta * len(x) / total
            count = total
        if not count:
            raise ValueError("No training samples")
        self.mean, self.scale = mean, np.maximum(np.sqrt(m2 / count), 1e-6)
        return self

    def transform(self, record):
        if (record.channels, record.units) != (self.channels, self.units):
            raise ValueError("Normalization schema mismatch")
        return replace(record, signals=((record.signals - self.mean) / self.scale).astype("float32"))
