import numpy as np
from scipy.signal import welch


def imu_features(windows, fs):
    """Time and Welch band-power features, returned with stable feature names."""
    x = np.asarray(windows)
    if x.ndim != 3 or x.shape[1] < 2 or not np.isfinite(x).all() or fs <= 0:
        raise ValueError("Expected finite (windows, time, channels) signals and positive rate")
    stats = {"mean": x.mean(axis=1), "std": x.std(axis=1),
             "rms": np.sqrt((x*x).mean(axis=1)), "min": x.min(axis=1), "max": x.max(axis=1)}
    f, psd = welch(x, fs=fs, axis=1, nperseg=min(x.shape[1], 256))
    for lo, hi in ((0.5, 3), (3, 8), (8, 12)):
        mask = (f >= lo) & (f < hi)
        stats[f"power_{lo}_{hi}"] = psd[:, mask].sum(axis=1) * (f[1] - f[0])
    names = tuple(f"channel_{c}:{name}" for name in stats for c in range(x.shape[2]))
    return np.concatenate(list(stats.values()), axis=1).astype("float32"), names
