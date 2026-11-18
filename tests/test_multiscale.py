import numpy as np

from src.features.multiscale import invariant_features, subject_class_weights


def test_independent_sensor_rotations_preserve_features():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 500, 3, 3))
    rotated = x.copy()
    for sensor in range(3):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotated[:, :, sensor] = x[:, :, sensor] @ rotation
    np.testing.assert_allclose(invariant_features(x.reshape(2, 500, 9), 100),
                               invariant_features(rotated.reshape(2, 500, 9), 100), atol=1e-5)


def test_subject_class_groups_have_equal_weight():
    subjects = np.array(['a']*10+['b']*4)
    labels = np.array([0]*8+[1]*2+[0]*3+[1])
    weights = subject_class_weights(subjects, labels)
    totals = [weights[(subjects == s) & (labels == y)].sum() for s in ('a', 'b') for y in (0, 1)]
    np.testing.assert_allclose(totals, totals[0])
    assert np.isclose(weights.mean(), 1)
