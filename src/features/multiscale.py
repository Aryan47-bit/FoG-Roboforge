"""Rotation-invariant features and recent-versus-past gait changes."""
import numpy as np

from src.features.imu_features import imu_features


def invariant_features(windows, fs):
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 9 or x.shape[1] < round(5*fs):
        raise ValueError("Expected five seconds of nine accelerometer channels")
    if not np.isfinite(x).all():
        raise ValueError("Signals must be finite")
    sensors = x.reshape(len(x), x.shape[1], 3, 3)
    parts = []
    for seconds in (1, 2, 5):
        recent = sensors[:, -round(seconds*fs):]
        centered = recent-recent.mean(axis=1, keepdims=True)
        magnitude = np.linalg.norm(recent, axis=-1)
        dynamic = np.linalg.norm(centered, axis=-1)
        features, _ = imu_features(np.concatenate([magnitude, dynamic], axis=-1), fs)
        covariance = np.einsum('btsi,btsj->bsij', centered, centered)/recent.shape[1]
        eigenvalues = np.linalg.eigvalsh(covariance).reshape(len(x), -1)
        jerk = np.linalg.norm(np.diff(recent, axis=1), axis=-1)
        parts.extend([features, eigenvalues, jerk.mean(axis=1), jerk.std(axis=1)])
    # Within-window changes reduce reliance on subject-specific absolute amplitude.
    first, last = sensors[:, :round(fs)], sensors[:, -round(fs):]
    def energy(a):
        return np.mean(np.sum((a-a.mean(axis=1, keepdims=True))**2, axis=-1), axis=1)
    parts.append(np.log((energy(last)+1e-8)/(energy(first)+1e-8)))
    return np.concatenate(parts, axis=1).astype("float32")


def subject_class_weights(subjects, labels):
    """Equal total weight per subject/class group present in training only."""
    subjects, labels = np.asarray(subjects), np.asarray(labels)
    if subjects.shape != labels.shape or not len(labels):
        raise ValueError("Nonempty aligned subjects and labels required")
    result = np.zeros(len(labels))
    for subject in np.unique(subjects):
        for label in np.unique(labels[subjects == subject]):
            mask = (subjects == subject) & (labels == label)
            result[mask] = 1/mask.sum()
    return result/result.mean()
