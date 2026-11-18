from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data.base_dataset import Recording
from src.evaluation.metrics import event_counts, summarize
from src.evaluation.threshold_optimizer import calibrate
from src.inference.postprocessing import AlarmConfig, alarm_times


def trace():
    t = np.arange(6000)/100
    r = Recording("fixture", "01", "s", "r", Path("fixture"), 100, t, np.zeros((len(t), 1)),
                  ((t >= 30) & (t < 35)).astype("int8"), ("x",), ("g",))
    times = np.arange(5, 59.9, .1)
    p = np.zeros(len(times))
    p[(times >= 27) & (times < 29)] = .95
    return r, times, p


def test_alarm_confirmation_is_not_backdated_or_retriggered():
    t = np.arange(300)/10
    p = np.ones(300)*.9
    np.testing.assert_allclose(alarm_times(t, p, AlarmConfig(.5)), [.5])
    p[:] = 0
    p[10:14] = .9  # less than 0.5 seconds
    assert len(alarm_times(t, p, AlarmConfig(.5))) == 0


def test_event_matching_exposure_and_false_alarms():
    r, t, p = trace()
    p[(t >= 8) & (t < 10)] = .95
    c = event_counts(r, t, p, AlarmConfig(.5))
    assert c["detected"] == c["episodes"] == 1
    assert c["false_alarms"] == 1
    assert c["non_fog_seconds"] == pytest.approx(44.99)
    assert -3 < c["latencies_seconds"][0] < -2
    assert summarize([c])["fa_per_hour"] == pytest.approx(3600/44.99)


def test_trace_gaps_rejected_and_boundary_events_censored():
    r, t, p = trace()
    with pytest.raises(ValueError, match="uninterrupted"):
        event_counts(r, np.delete(t, np.s_[20:30]), np.delete(p, np.s_[20:30]), AlarmConfig(.5))
    r = replace(r, labels=((r.timestamps >= 7) & (r.timestamps < 10)).astype("int8"))
    assert event_counts(r, t, p*0, AlarmConfig(.5))["episodes"] == 0


def test_calibration_rejects_test_data_and_reports_infeasibility():
    r, t, p = trace()
    with pytest.raises(ValueError, match="validation-only"):
        calibrate([(r, t, p)], {r.subject_key: "test"})
    result = calibrate([(r, t, p)], {r.subject_key: "validation"}, thresholds=[.5])
    assert result["best"]["metrics"]["pooled"]["episode_sensitivity"] == 1
    p[(t >= 8) & (t < 10)] = .95
    result = calibrate([(r, t, p)], {r.subject_key: "validation"}, thresholds=[.5])
    assert result["status"] == "no_feasible_threshold"
    assert result["best"] is None


def test_no_event_validation_cannot_calibrate():
    r, t, p = trace()
    r = replace(r, labels=np.zeros_like(r.labels))
    assert calibrate([(r, t, p*0)], {r.subject_key: "validation"}, thresholds=[.99])["best"] is None
