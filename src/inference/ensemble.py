"""Replay the equal-weight CNN, Extra Trees and Random Forest Daphnet ensemble."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.data.daphnet_dataset import load_daphnet
from src.data.transforms import Standardizer, resample_causal, valid_segments
from src.inference.postprocessing import AlarmConfig, alarm_times
from src.models.cnn import ConvPredictor
from src.training.compare_features import extract
from src.training.trainer import predict_continuous


class DaphnetEnsemble:
    def __init__(self, reports):
        reports = Path(reports)
        self.config = AlarmConfig(json.loads((reports / "accuracy_first.json").read_text())["threshold"])
        checkpoint = torch.load(reports / "cnn_full_001/model.pt", map_location="cpu", weights_only=True)
        self.scaler = Standardizer()
        self.scaler.channels = tuple(checkpoint["input_channels"])
        self.scaler.units = tuple(checkpoint["units"])
        self.scaler.mean = np.array(checkpoint["normalization_mean"])
        self.scaler.scale = np.array(checkpoint["normalization_scale"])
        self.cnn = ConvPredictor(len(self.scaler.channels))
        self.cnn.load_state_dict(checkpoint["state_dict"])
        self.trees = []
        for name in ("extra_trees", "random_forest"):
            with (reports / "feature_comparison_001" / f"{name}.pkl").open("rb") as handle:
                self.trees.append(pickle.load(handle)["model"])

    def predict(self, record):
        """Yield decision timestamps, probabilities, classifications and alarm starts per segment."""
        for segment in valid_segments(record):
            if segment.timestamps[-1] - segment.timestamps[0] < 6:
                continue
            prepared = resample_causal(segment, 100)
            starts = np.arange(0, len(prepared.timestamps) - 499, 10)
            if not len(starts):
                continue
            normalized = self.scaler.transform(prepared)
            _, times, cnn = predict_continuous(self.cnn, [normalized])[0]
            features = extract(prepared, starts, 500)
            probabilities = np.mean([cnn] + [tree.predict_proba(features)[:, 1] for tree in self.trees], axis=0)
            alarms = alarm_times(times, probabilities, self.config)
            yield times, probabilities, probabilities > self.config.threshold, np.isin(times, alarms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Daphnet SxxRxx.txt recording")
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, required=True, help="New CSV file")
    args = parser.parse_args()
    torch.set_num_threads(2)
    record = next(load_daphnet(args.recording))
    ensemble = DaphnetEnsemble(args.reports)
    with args.output.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["decision_seconds", "probability", "positive_window", "alarm_start"])
        for times, probabilities, positive, alarms in ensemble.predict(record):
            writer.writerows(zip(times, probabilities, positive.astype(int), alarms.astype(int)))


if __name__ == "__main__":
    main()
