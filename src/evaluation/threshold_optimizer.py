import numpy as np

from src.evaluation.metrics import evaluate_traces
from src.inference.postprocessing import AlarmConfig


def calibrate(traces, splits, thresholds=None, target_fa=2.):
    """Maximize pooled event sensitivity with < target FA/hr for EVERY val subject.

    No feasible threshold returns an explicit failure, never a silent fallback.
    Event-free or exposure-free validation cannot establish a usable predictor.
    """
    if not traces or any(splits.get(r.subject_key) != "validation" for r, _, _ in traces):
        raise ValueError("Calibration requires validation-only recordings")
    candidates = []
    for threshold in np.linspace(.01, .99, 99) if thresholds is None else thresholds:
        config = AlarmConfig(float(threshold))
        metrics = evaluate_traces(traces, config)
        subjects = metrics["subjects"].values()
        feasible = (metrics["pooled"]["episodes"] > 0 and
                    all(s["fa_per_hour"] is not None and s["fa_per_hour"] < target_fa for s in subjects))
        candidates.append({"threshold": float(threshold), "feasible": feasible, "metrics": metrics})
    feasible = [c for c in candidates if c["feasible"]]
    best = max(feasible, key=lambda c: (c["metrics"]["pooled"]["episode_sensitivity"],
                                      -c["metrics"]["pooled"]["fa_per_hour"], c["threshold"])) if feasible else None
    return {"status": "feasible" if best else "no_feasible_threshold", "best": best,
            "sensitivity_target_met": bool(best and best["metrics"]["pooled"]["episode_sensitivity"] > .9),
            "target_fa_per_hour_strict": target_fa, "constraint": "each_validation_subject", "sweep": candidates}
