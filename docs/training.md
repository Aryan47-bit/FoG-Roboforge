# Training and validation baseline

The runnable baseline trains a causal TCN on Daphnet's nine accelerometer axes.
It is a research model, not the single-MPU6050 deployment model. Hardware is
unchanged.

```powershell
python -m src.training.trainer --output reports/tcn_baseline_001
```

Defaults: 100 epochs, batch size 64, five-second context, 0.1-second stride,
100 Hz sampling, AdamW at 3e-4, weight decay 1e-4, cosine warm restarts,
gradient clipping at 1, and patience 15. This baseline uses class-weighted BCE
and deterministic CPU execution by default. `--model cnn --device cuda` selects
the compact CNN and GPU execution; `--model tcn` retains the original baseline.
Lightning, mixed precision, MLflow, and additional training datasets remain future work.

Split subjects before preparation. Only training subjects fit normalization and
class weights. Lazy indexing avoids storing overlapping copies of signals.
Validation balanced accuracy at probability 0.5 selects the epoch. Continuous
validation predictions then calibrate the alarm threshold. This is an initial
baseline, not the final FA-first architecture/ensemble selection procedure.
Test subjects do not enter normalization, training, selection, or calibration.

## Small integration run

```powershell
python -m src.training.trainer --output reports/tcn_debug_001 --epochs 2 --max-seconds 60 --max-train-windows 128 --batch-size 32
```

Use a fresh output directory; existing directories are rejected. Debug flags
truncate segments or sample training windows; metrics are integration diagnostics
and must not be compared with full-cohort targets.

Each run saves `model.pt`: weights, normalization, channel/unit metadata, window
geometry, subject splits, and alarm configuration (null if calibration fails).
`report.json` records arguments, recording inventory, training history, and the
complete threshold sweep with per-subject metrics. This is not an embedded export.

The first debug run used 128 training windows for two epochs. Best validation
balanced accuracy was approximately 50.15%. Its selected alarm threshold detected
0 of 4 scored episodes, with zero false alarms. It failed the sensitivity target;
no held-out test performance was measured.

## Event metric contract

- Evaluate every 0.1-second decision, including ongoing FoG and tail windows;
  the causal alarm state sees the entire trace. Reset between valid segments.
- Require probability above threshold plus 0.1 for 0.5 seconds to start, and
  record the actual confirmation time. End below threshold minus 0.1; enforce
  a 10-second refractory interval. Sustained activity cannot retrigger.
- Match each alarm and onset at most once, from five seconds before through
  one second after onset. Negative latency means early warning. Separately
  count warnings with 5-10 second lead time; the current five-second horizon
  does not establish the broader ten-second prediction objective.
- Score only onsets with their full matching interval observed. Exclude the
  last five seconds from scored exposure because future onset labels are
  unavailable. Exclude startup/filter/context time from monitoring exposure.
- Integrate non-FoG time once from the prepared sample timeline. False alarms
  are unmatched normal-gait alarm events. Alerts near boundary-censored or
  already matched events, and late ongoing-FoG alerts, are `other_alarms`.
- Every validation subject must have normal-gait exposure and FA/hr strictly
  below 2. Maximize pooled event sensitivity among feasible thresholds.
  Event-free validation or an infeasible sweep returns failure. A silent model
  may satisfy FA/hr while failing sensitivity; inspect `sensitivity_target_met`.

Remaining work includes confidence intervals, frozen-model test evaluation,
dataset fingerprints, multimodal integration, and embedded compression/export.

## Alternative-model experiments

The full-recording comparison trains Extra Trees, Random Forest and histogram
gradient boosting using 144 motion/spectral features, plus a separate compact
CNN on the same fixed subject partition. See
[measured comparison results](../reports/model_comparison.md). None met both
the accuracy and false-alarm targets. The feature-model reports include source
SHA256 fingerprints and saved validation probability traces.

```powershell
python -m src.training.compare_features --output reports/feature_comparison_new
python -m src.training.trainer --output reports/cnn_full_new --model cnn --device cuda --epochs 10 --patience 4 --batch-size 128
```

Feature models require scikit-learn (`pip install -e ".[research]"`). Experiment
directories must be new. These runs use all eligible recordings, not the earlier
debug truncation. Test subjects remain untouched by selection and calibration.

## Rotation-invariant features at multiple time scales

```powershell
python -m src.training.compare_features --representation invariant --output reports/invariant_comparison_new
```

This approach changes the representation and training weights, rather than only
the classifier. It combines acceleration magnitudes, centered motion magnitudes,
covariance eigenvalues, and magnitude of sample-to-sample changes over trailing
1-, 2-, and 5-second intervals. A within-window energy ratio compares recent
motion with earlier motion. All features use only the observed five-second
history; no future samples or validation-fitted statistics are used.

Training gives equal total weight to each observed subject/class group. Extra
Trees and gradient boosting use these weights; their previous class weighting
is disabled to avoid applying imbalance correction twice. The subject split,
prediction labels, alarm policy, and FA constraint remain the same as the earlier
comparison. The run changes both representation and weighting, so it does not
isolate their individual contributions. Rotation invariance is numerically
tested under a different orthogonal transform for each sensor.
