"""Small real-data optimization check, not an evaluation or performance claim."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.daphnet_dataset import load_daphnet
from src.data.splits import subject_split
from src.data.transforms import Standardizer, resample_causal, valid_segments
from src.features.windowing import make_windows
from src.models.tcn import TCN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/foundation_smoke.json"))
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    paths = sorted((args.root / "daphnet").rglob("S*R*.txt"))
    splits = subject_split(["daphnet:" + p.stem[:3] for p in paths])
    selected = None
    for record in load_daphnet(args.root / "daphnet"):
        if splits[record.subject_key] != "train":
            continue
        for segment in valid_segments(record):
            if segment.timestamps[-1] - segment.timestamps[0] >= 15:
                selected = resample_causal(segment)
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError("No sufficiently long training segment")
    scaler = Standardizer().fit([selected], splits)
    # Five-second context matches the planned TCN history; two-second windows
    # remain the generic feature-pipeline default.
    windows = make_windows(scaler.transform(selected), window_seconds=5)
    if len(windows.targets) < 2:
        raise ValueError("Not enough uncensored windows for smoke training")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(windows.targets), min(32, len(windows.targets)), replace=False)
    x = torch.from_numpy(windows.signals[indices])
    y = torch.from_numpy(windows.targets[indices].astype("float32"))
    model = TCN(x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(float(loss.detach()))
    report = {"purpose": "Training plumbing smoke test only; no validation/test performance",
              "subject": selected.subject_key, "recording": selected.recording,
              "split": splits[selected.subject_key], "split_manifest": splits,
              "seed": 42, "sample_rate_hz": 100, "window_seconds": 5,
              "batch_shape": list(x.shape), "positive_batch_windows": int(y.sum()),
              "receptive_field_samples": model.receptive_field, "losses": losses}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
