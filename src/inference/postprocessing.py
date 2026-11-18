from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlarmConfig:
    threshold: float
    hysteresis: float = .1
    min_duration: float = .5
    refractory: float = 10.

    def __post_init__(self):
        if not 0 <= self.threshold <= 1 or not 0 <= self.hysteresis <= 1:
            raise ValueError("Threshold and hysteresis must be in [0, 1]")
        if not np.isfinite([self.min_duration, self.refractory]).all() or min(self.min_duration, self.refractory) < 0:
            raise ValueError("Durations must be finite and nonnegative")


def alarm_times(times, probabilities, config):
    """Causal hysteresis with sustained start confirmation and no backdating.

    Call separately for each contiguous recording. An active alarm must end
    before a new one can begin; refractory time alone never retriggers it.
    """
    times, probabilities = np.asarray(times), np.asarray(probabilities)
    if times.shape != probabilities.shape or times.ndim != 1:
        raise ValueError("Aligned one-dimensional probabilities and times required")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Times must be finite and increasing")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    active, pending, last = False, None, -np.inf
    alarms = []
    for time, probability in zip(times, probabilities):
        if active:
            if probability < max(0, config.threshold - config.hysteresis):
                active = False
            continue
        if time - last < config.refractory:
            pending = None
            continue
        if probability > min(1, config.threshold + config.hysteresis):
            if pending is None:
                pending = time
            if time - pending >= config.min_duration - 1e-9:
                alarms.append(float(time))
                last, active, pending = time, True, None
        else:
            pending = None
    return np.asarray(alarms)
