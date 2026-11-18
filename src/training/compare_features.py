"""Full-recording subject-independent comparison of feature-based FoG models."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier

from src.data.daphnet_dataset import load_daphnet
from src.data.splits import subject_split
from src.data.transforms import valid_segments, resample_causal
from src.evaluation.threshold_optimizer import calibrate
from src.features.imu_features import imu_features
from src.features.windowing import window_indices
from src.features.multiscale import invariant_features, subject_class_weights


def extract(record, starts, size, batch=256, representation="original"):
    rows = []
    for offset in range(0, len(starts), batch):
        x = np.stack([record.signals[s:s+size] for s in starts[offset:offset+batch]])
        if representation == "invariant":
            rows.append(invariant_features(x, record.fs))
            continue
        # Orientation-robust magnitudes supplement all nine original axes.
        magnitudes = np.sqrt((x.reshape(len(x), size, 3, 3)**2).sum(axis=-1))
        x = np.concatenate([x, magnitudes], axis=-1)
        base, _ = imu_features(x, record.fs)
        differences = np.diff(x, axis=1)
        dynamic = np.concatenate([differences.std(axis=1), np.mean(np.abs(differences), axis=1),
                                  np.percentile(x, 75, axis=1)-np.percentile(x, 25, axis=1)], axis=1)
        # imu_features order: mean/std/rms/min/max and three power bands.
        freeze_ratio = np.log1p(base[:, 6*12:7*12] / (base[:, 5*12:6*12]+1e-8))
        rows.append(np.concatenate([base, dynamic, freeze_ratio], axis=1))
    return np.concatenate(rows).astype("float32")


def classification(y, p, threshold):
    predicted = p > threshold
    positive, negative = y == 1, y == 0
    sensitivity = float(predicted[positive].mean()) if positive.any() else None
    specificity = float((~predicted[negative]).mean()) if negative.any() else None
    return {"sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            "accuracy": float((predicted == y).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--representation", choices=["original", "invariant"], default="original")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted((args.root/"daphnet").rglob("S*R*.txt"))
    splits = subject_split(["daphnet:"+p.stem[:3] for p in paths], seed=42)
    train_x, train_y, train_subjects, validation, manifest = [], [], [], [], []
    for record in load_daphnet(args.root/"daphnet"):
        split = splits[record.subject_key]
        if split == "test":
            continue
        digest = hashlib.sha256(record.source.read_bytes()).hexdigest()
        for segment in valid_segments(record):
            if len(segment.timestamps)/segment.fs < 20:
                continue
            r = resample_causal(segment)
            size = round(5*r.fs)
            starts, labels, _ = window_indices(r, window_seconds=5)
            if not len(starts):
                continue
            if split == "train":
                train_x.append(extract(r, starts, size, representation=args.representation))
                train_y.append(labels)
                train_subjects.extend([r.subject_key]*len(labels))
            else:
                continuous = np.arange(0, len(r.timestamps)-size+1, 10)
                validation.append((r, continuous, extract(r, continuous, size, representation=args.representation), starts//10, labels))
            manifest.append({"subject": r.subject_key, "recording": r.recording, "split": split,
                             "source": str(r.source), "sha256": digest, "samples": len(r.timestamps)})
        print("prepared", record.recording, split, flush=True)
    x, y = np.concatenate(train_x), np.concatenate(train_y)
    print("training", x.shape, "positive_windows", int(y.sum()), flush=True)
    models = {
        "extra_trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=10, max_features=.7,
                                             class_weight="balanced", random_state=42, n_jobs=4),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features="sqrt",
                                                 class_weight="balanced_subsample", random_state=42, n_jobs=4),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                                                  l2_regularization=10, early_stopping=False, random_state=42),
    }
    results = []
    if args.representation == "invariant":
        models.pop("random_forest")
        models["extra_trees"].set_params(class_weight=None)
    all_y = np.concatenate([entry[4] for entry in validation])
    for name, model in models.items():
        print("fitting", name, flush=True)
        if args.representation == "invariant":
            model.fit(x, y, sample_weight=subject_class_weights(train_subjects, y))
        elif name == "hist_gradient_boosting":
            weights = np.where(y == 1, len(y)/(2*(y == 1).sum()), len(y)/(2*(y == 0).sum()))
            model.fit(x, y, sample_weight=weights)
        else:
            model.fit(x, y)
        traces, prediction_arrays, window_probs = [], {}, []
        for i, (r, starts, features, keep, _) in enumerate(validation):
            p = model.predict_proba(features)[:, 1]
            times = r.timestamps[starts+499]
            traces.append((r, times, p))
            prediction_arrays[f"times_{i}"] = times
            prediction_arrays[f"probabilities_{i}"] = p
            window_probs.append(p[keep])
        p = np.concatenate(window_probs)
        calibration = calibrate(traces, splits)
        # Compare balanced accuracy at the SAME probability threshold used by
        # the alarm FSM; do not quote favorable accuracy at a different cutoff.
        for candidate in calibration["sweep"]:
            candidate["window_metrics"] = classification(all_y, p, candidate["threshold"])
        feasible = [c for c in calibration["sweep"] if c["feasible"]]
        best = max(feasible, key=lambda c: (c["window_metrics"]["balanced_accuracy"],
                                           c["metrics"]["pooled"]["episode_sensitivity"],
                                           -c["metrics"]["pooled"]["fa_per_hour"])) if feasible else None
        result = {"model": name, "unconstrained_at_0_5": classification(all_y, p, .5),
                  "selected_by": "maximum_window_balanced_accuracy_subject_to_each_subject_FA_per_hour_below_2",
                  "selected": best, "targets_met": bool(best and best["window_metrics"]["balanced_accuracy"] > .9
                                                         and best["metrics"]["pooled"]["episode_sensitivity"] > .9),
                  "calibration": calibration}
        results.append(result)
        with (args.output/f"{name}.pkl").open("wb") as handle:
            pickle.dump({"model": model, "window_seconds": 5, "sample_rate_hz": 100,
                         "features": args.representation,
                         "subject_class_weighted": args.representation == "invariant",
                         "threshold": best["threshold"] if best else None}, handle)
        np.savez_compressed(args.output/f"{name}_validation_probabilities.npz", **prediction_arrays)
        report = {"dataset": "daphnet", "test_evaluated": False, "seed": 42, "splits": splits,
                  "representation": args.representation, "subject_class_weighted": args.representation == "invariant",
                  "train_windows": len(y), "validation_windows": len(all_y), "features": x.shape[1],
                  "manifest": manifest, "results": results,
                  "caveat": "Exploratory selection on two validation subjects; no independent test claim. Fixed default alarm policy."}
        (args.output/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("model", "unconstrained_at_0_5", "selected", "targets_met")}), flush=True)


if __name__ == "__main__":
    main()
