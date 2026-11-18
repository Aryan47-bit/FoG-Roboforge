from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
from src.data.base_dataset import Recording
from src.training.compare_features import extract, classification


def test_features_are_finite_and_do_not_use_future_samples():
    t = np.arange(1000)/100
    x = np.tile(np.sin(t)[:, None], (1, 9)).astype("float32")
    r = Recording("fixture", "1", "s", "r", Path("fixture"), 100, t, x,
                  np.zeros(len(t), dtype="int8"), tuple(str(i) for i in range(9)), ("g",)*9)
    first = extract(r, np.array([0]), 500)
    assert first.shape == (1, 144)
    assert np.isfinite(first).all()
    r.signals[500:] = 100
    np.testing.assert_array_equal(extract(r, np.array([0]), 500), first)


def test_silent_model_does_not_pass_balanced_accuracy():
    result = classification(np.array([0, 0, 0, 1]), np.zeros(4), .5)
    assert result["accuracy"] == .75
    assert result["balanced_accuracy"] == .5
    assert result["sensitivity"] == 0
