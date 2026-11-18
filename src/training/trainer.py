"""Subject-isolated Daphnet TCN baseline with continuous validation calibration."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.datamodule import WindowDataset
from src.data.daphnet_dataset import load_daphnet
from src.data.splits import subject_split
from src.data.transforms import Standardizer, resample_causal, valid_segments
from src.evaluation.threshold_optimizer import calibrate
from src.models.tcn import TCN
from src.models.cnn import ConvPredictor
from src.inference.postprocessing import AlarmConfig


@torch.no_grad()
def predict_continuous(model, records, batch_size=64, window_seconds=5, stride_seconds=.1):
    model.eval()
    device = next(model.parameters()).device
    traces = []
    for record in records:
        size, stride = round(window_seconds*record.fs), round(stride_seconds*record.fs)
        starts = np.arange(0, len(record.timestamps)-size+1, stride)
        probabilities = []
        for offset in range(0, len(starts), batch_size):
            x = np.stack([record.signals[s:s+size] for s in starts[offset:offset+batch_size]])
            probabilities.extend(torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().tolist())
        traces.append((record, record.timestamps[starts+size-1], np.asarray(probabilities)))
    return traces


@torch.no_grad()
def balanced_accuracy(model, loader):
    model.eval()
    device = next(model.parameters()).device
    tp = tn = positives = negatives = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prediction = model(x) >= 0
        positives += int((y == 1).sum())
        negatives += int((y == 0).sum())
        tp += int((prediction & (y == 1)).sum())
        tn += int((~prediction & (y == 0)).sum())
    if not positives or not negatives:
        raise ValueError("Validation needs both classes for balanced accuracy")
    return .5 * (tp/positives + tn/negatives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True, help="New run directory (never overwritten)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["tcn", "cnn"], default="tcn")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-seconds", type=float, default=None, help="Debug only: truncate each segment")
    parser.add_argument("--max-train-windows", type=int, default=None, help="Debug only: seeded training subset")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("Epochs, patience and batch size must be positive")
    if args.max_seconds is not None and args.max_seconds < 20:
        parser.error("--max-seconds must be at least 20")
    if args.max_train_windows is not None and args.max_train_windows < 2:
        parser.error("--max-train-windows must be at least 2")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    paths = sorted((args.root / "daphnet").rglob("S*R*.txt"))
    splits = subject_split(["daphnet:"+p.stem[:3] for p in paths], args.seed)
    records = {"train": [], "validation": []}
    for record in load_daphnet(args.root / "daphnet"):
        split = splits[record.subject_key]
        if split == "test":
            continue  # Never normalize, train, select, or calibrate on test subjects.
        for segment in valid_segments(record):
            if args.max_seconds is not None:
                n = round(args.max_seconds*segment.fs)
                segment = replace(segment, timestamps=segment.timestamps[:n], signals=segment.signals[:n], labels=segment.labels[:n])
            if len(segment.timestamps) / segment.fs < 20:
                continue
            records[split].append(resample_causal(segment))
    if not all(records.values()):
        raise ValueError("No usable training or validation recordings")
    scaler = Standardizer().fit(records["train"], splits)
    records = {split: [scaler.transform(r) for r in group] for split, group in records.items()}
    train, val = WindowDataset(records["train"]), WindowDataset(records["validation"])
    indices = np.arange(len(train))
    if args.max_train_windows:
        indices = rng.choice(indices, min(len(indices), args.max_train_windows), replace=False)
    labels = train.targets[indices]
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("Training subset needs both classes; increase debug limits")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(train, indices.tolist()), batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    model = (TCN(len(scaler.channels)) if args.model == "tcn" else ConvPredictor(len(scaler.channels))).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    weight = torch.tensor(float((labels == 0).sum() / (labels == 1).sum()), device=args.device)
    history, best_score, stale, best_state = [], -1., 0, None
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0., 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach())*len(y)
            count += len(y)
        score = balanced_accuracy(model, val_loader)
        history.append({"epoch": epoch+1, "train_loss": loss_sum/count, "validation_balanced_accuracy_at_0_5": score})
        print(json.dumps(history[-1]), flush=True)
        scheduler.step(epoch+1)
        if score > best_score:
            best_score, stale = score, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    traces = predict_continuous(model, records["validation"], args.batch_size)
    calibration = calibrate(traces, splits)
    selected = calibration["best"]
    config = asdict(AlarmConfig(selected["threshold"])) if selected else None
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()}, "architecture": args.model,
                "input_channels": list(scaler.channels), "units": list(scaler.units),
                "normalization_mean": scaler.mean.tolist(), "normalization_scale": scaler.scale.tolist(),
                "sample_rate_hz": 100, "window_seconds": 5, "stride_seconds": .1,
                "alarm_config": config, "splits": splits}, args.output / "model.pt")
    report = {"status": "debug_run" if args.max_seconds or args.max_train_windows else "baseline_run",
              "test_evaluated": False, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "splits": splits, "train_windows_used": len(indices), "validation_windows": len(val),
              "recordings": {s: [{"subject": r.subject_key, "recording": r.recording, "source": str(r.source),
                                   "samples": len(r.timestamps)} for r in group] for s, group in records.items()},
              "history": history, "calibration": calibration}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"calibration_status": calibration["status"], "best": selected}), flush=True)


if __name__ == "__main__":
    main()
