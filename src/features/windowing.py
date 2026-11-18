from dataclasses import dataclass

import numpy as np


@dataclass
class Windows:
    signals: np.ndarray
    targets: np.ndarray
    decision_times: np.ndarray
    starts: np.ndarray


def window_indices(record, window_seconds=2, stride_seconds=0.1, pre_onset_seconds=5, post_onset_seconds=1):
    """Label by decision time relative to observed 0->1 transitions.

    Exclude censored tail negatives and ongoing FoG outside the post-onset band.
    A recording starting in FoG has no observed onset. Never manufacture one.
    """
    record.validate()
    size, stride = round(window_seconds * record.fs), round(stride_seconds * record.fs)
    if size < 1 or stride < 1 or min(pre_onset_seconds, post_onset_seconds) < 0:
        raise ValueError("Invalid window or horizon configuration")
    if np.any(record.labels < 0) or not np.isfinite(record.signals).all():
        raise ValueError("Window only valid segments")
    if np.any(np.diff(record.timestamps) > 1.6 / record.fs):
        raise ValueError("Windowing must not cross timing gaps")
    onsets = record.timestamps[1:][(record.labels[:-1] == 0) & (record.labels[1:] == 1)]
    starts = np.arange(0, max(0, len(record.timestamps) - size + 1), stride)
    end = starts + size - 1
    decisions = record.timestamps[end]
    positive = np.zeros(len(starts), dtype=bool)
    for onset in onsets:
        positive |= (decisions >= onset - pre_onset_seconds) & (decisions <= onset + post_onset_seconds)
    known_negative = (record.labels[end] == 0) & (decisions + pre_onset_seconds <= record.timestamps[-1])
    keep = positive | known_negative
    starts, decisions, positive = starts[keep], decisions[keep], positive[keep]
    return starts, positive.astype("int8"), decisions


def make_windows(record, window_seconds=2, stride_seconds=0.1, pre_onset_seconds=5, post_onset_seconds=1):
    starts, targets, decisions = window_indices(record, window_seconds, stride_seconds,
                                               pre_onset_seconds, post_onset_seconds)
    size = round(window_seconds * record.fs)
    x = np.stack([record.signals[s:s+size] for s in starts]) if len(starts) else np.empty((0, size, len(record.channels)), dtype="float32")
    return Windows(x, targets, decisions, starts)
