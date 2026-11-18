from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.base_dataset import Recording
from src.data.daphnet_dataset import load_daphnet
from src.data.fogstar_dataset import CHANNELS, load_fogstar
from src.data.multimodal_dataset import load_multimodal
from src.data.splits import loso, subject_split
from src.data.transforms import Standardizer, resample_causal, valid_segments
from src.features.imu_features import imu_features
from src.features.windowing import make_windows


def recording(fs=100, seconds=30):
    t = np.arange(fs * seconds) / fs
    return Recording("synthetic", "01", "session", "record", Path("fixture"), fs,
                     t, np.sin(2*np.pi*t)[:, None].astype("float32"),
                     ((t >= 15) & (t < 18)).astype("int8"), ("acc_x",), ("g",))


def test_daphnet_labels_and_units(tmp_path):
    a = np.zeros((3, 11))
    a[:, 0] = [0, 16, 31]
    a[:, 1:10] = 1000
    a[:, -1] = [0, 1, 2]
    np.savetxt(tmp_path / "S01R01.txt", a)
    r = next(load_daphnet(tmp_path))
    assert r.subject_key == "daphnet:S01"
    np.testing.assert_array_equal(r.labels, [-1, 0, 1])
    np.testing.assert_allclose(r.signals, 1)


def test_fogstar_interleaved_chunks_preserve_recordings(tmp_path):
    rows = []
    for i in range(12):
        for subject in [1, 2]:
            rows.append({**dict.fromkeys(CHANNELS, 1.), "subjectID": subject,
                         "sessionID": 1, "taskID": 1, "timestamp": i/60, "fog": int(i > 8)})
    path = tmp_path / "sensor.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    records = list(load_fogstar(path, chunksize=5))
    assert len(records) == 2
    assert all(len(r.timestamps) == 12 for r in records)
    assert records[0].signals.shape == (12, 24)


def test_multimodal_requires_identity_and_explicit_schema(tmp_path):
    pd.DataFrame({"time": [0, .002, .004], "eeg": [1, 2, 3], "label": [0, 0, 1]}).to_csv(tmp_path / "aligned.csv", index=False)
    entry = dict(path="aligned.csv", subject="verified-01", session="visit-1", recording="task-1",
                 sample_rate_hz=500, time_column="time", time_scale_seconds=1,
                 label_column="label", channels=["eeg"], units=["uV"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([entry]))
    assert next(load_multimodal(manifest)).subject_key == "multimodal:verified-01"
    entry["subject"] = ""
    manifest.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="IDs"):
        next(load_multimodal(manifest))


def test_subject_splits_and_loso_are_disjoint():
    keys = [f"data:{i}" for i in range(10)]
    splits = subject_split(keys)
    assert splits == subject_split(keys[::-1] + keys)
    assert [list(splits.values()).count(s) for s in ["train", "validation", "test"]] == [6, 2, 2]
    folds = list(loso(keys))
    assert len(folds) == 10
    assert all(list(f.values()).count("test") == 1 for f in folds)
    assert {next(k for k, v in f.items() if v == "test") for f in folds} == set(keys)


def test_invalid_samples_and_gaps_split_windows():
    r = recording()
    r.labels[1000] = -1
    r.signals[2000] = np.nan
    r.timestamps[2500:] += 1
    segments = list(valid_segments(r))
    assert [len(s.timestamps) for s in segments] == [1000, 999, 499, 500]
    assert all(np.isfinite(s.signals).all() for s in segments)
    with pytest.raises(ValueError):
        make_windows(r)


def test_causal_resampling_does_not_see_future():
    r = recording(fs=500)
    changed = replace(r, signals=r.signals.copy())
    changed.signals[r.timestamps >= 10] += 100
    first, second = resample_causal(r), resample_causal(changed)
    np.testing.assert_array_equal(first.signals[first.timestamps < 10], second.signals[second.timestamps < 10])
    assert first.fs == 100
    assert np.std(first.signals) == pytest.approx(1 / np.sqrt(2), rel=.03)


def test_normalization_rejects_validation_subjects():
    r = recording()
    with pytest.raises(ValueError, match="training"):
        Standardizer().fit([r], {r.subject_key: "validation"})
    normalizer = Standardizer().fit([r], {r.subject_key: "train"})
    normalized = normalizer.transform(r)
    assert abs(normalized.signals.mean()) < 1e-6
    assert normalized.signals.std() == pytest.approx(1)


def test_prediction_labels_censoring_and_history():
    r = recording()
    w = make_windows(r)
    positive_times = w.decision_times[w.targets == 1]
    assert positive_times.min() >= 10
    assert positive_times.max() <= 16
    assert not np.any((w.decision_times > 16) & (w.decision_times < 18))
    assert w.decision_times.max() <= 24.99
    np.testing.assert_array_equal(w.signals[:, -1, 0], r.signals[w.starts+199, 0])


def test_recording_start_in_fog_is_not_a_new_onset():
    r = recording()
    r.labels[:] = 1
    assert len(make_windows(r).targets) == 0


def test_feature_bandpower_known_sine():
    t = np.arange(200) / 100
    x = np.sin(2*np.pi*5*t)[None, :, None]
    features, names = imu_features(x, 100)
    assert features[0, names.index("channel_0:power_3_8")] == pytest.approx(.5, abs=.01)
    assert features[0, names.index("channel_0:power_8_12")] < .001
