# Accelerometer-only FoG rebuild

Completed run: 20 RTX 4050 experiments (270 epochs total), 12 automated tests,
1,000 host C++ normalization cases and 1,050 alarm transitions. The selected
1,669-parameter model exports to a 9,008-byte full-INT8 TFLite artifact.
INT8 held-out sensor-stream event sensitivity is 53.9%, with 32.8 false events/hour
and a 1.96-second median delay. **Not deployment-ready.** See
[results](reports/RESULTS.md) and [export validation](artifacts/deployment/validation.json).
The full multimodal release and physical ESP32 validation remain outstanding.

Independent implementation created from the local raw datasets. Source datasets,
the historical implementation, and its trained weights are not overwritten.
This directory sits under `raw` because that was the writable workspace supplied
for this session; `_rebuild/artifacts` contains derived data, never source data.

## Reproduce

Run these commands from the `raw` directory in PowerShell:

```powershell
python -m venv --system-site-packages _rebuild/.venv
& _rebuild/.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
& _rebuild/.venv/Scripts/python.exe _rebuild/data_pipeline.py
& _rebuild/.venv/Scripts/python.exe -m unittest discover -s _rebuild/tests -v
& _rebuild/.venv/Scripts/python.exe _rebuild/train.py --epochs 20 --suite extended
& _rebuild/.venv/Scripts/python.exe _rebuild/export_model.py
& _rebuild/.venv/Scripts/python.exe _rebuild/make_report.py
```

The environment inherits the installed NumPy, SciPy, pandas, sklearn and TensorFlow
packages. CUDA PyTorch is installed locally. TensorFlow is used only for conversion
and CPU evaluation; **neural training must use CUDA** and aborts without it. The
user replaced the historical Intel `tf_arc` requirement with the RTX 4050.
`reports/gpu_verification.json` records real forward/backward/optimizer checks.

## Pipeline

- `data_pipeline.py`: source SHA256, schemas, rate/quality audit, disk-backed
  FoG-STAR grouping, unit conversion to m/s², source anti-alias filtering,
  causal 50 Hz resampling, recording-boundary preservation, compressed caches,
  fixed participant split manifests, and overlap labels.
- `models.py`: small CNN, residual CNN and causal TCN. All accept `(100,3)`.
- `train.py`: train-only normalization, dataset/participant balanced sampling,
  weighted binary loss, early stopping, last and best checkpoints, validation-only
  alarm tuning, test predictions and per-dataset/participant/side metrics.
- `evaluation.py`: window metrics and causal event replay.
- `export_model.py`: explicit weight mapping to Keras, numerical parity,
  training-only INT8 calibration, held-out quantization evaluation, C headers.
- `firmware/`: shared preprocessing and alarm state, host parity test, ESP32
  integration candidate retaining existing pins. Board validation remains required.
- `collect.py`: serial recording with pseudonymous metadata. Unlabelled recordings
  are never automatically made negative training examples.

## Dataset evidence

Daphnet: 17 recordings, 10 participants, documented 64 Hz, milli-g. Timestamp
rounding produces 15/16 ms steps; `(N-1)/(t_last-t_first)` confirms 64 Hz.
Do not use `1/median(diff(t))`, which incorrectly yields 62.5 Hz. Only the first
three acceleration columns (shank) are used. Label 0 is excluded; 1→0 and 2→1.

FoG-STAR: 22 participants, 106 subject/session/task groups, both ankles. Official
[Zenodo v3](https://zenodo.org/records/17838806) documents g, 60 Hz and binary `fog`.
The official timestamp description says milliseconds, but the local CSV increments
by 1/60; the loader explicitly validates the seconds encoding. Labels and sensor
rows stay within each task. Missing acceleration breaks a segment instead of
being interpolated. Severity is not an ordinal clinical severity assumption:
the source codes distinguish shuffling, trembling and akinesia.

Multimodal: `multimodal_parkinson` contains processing scripts and unaligned
sample segments. The full labelled participant release is absent. Samples have
no verified participant identity/alignment and are excluded. The dataset's
three labelled examples contain 1501, 1001 and 501 rows at the repository's
interpolated 500 Hz (about six seconds total). Their left/right accelerometer
columns are identical; the raw LShank/RShank files are byte-for-byte identical.
See `reports/multimodal_sample_audit.json`. They are useful schema fixtures, not
independent bilateral participant data. The dataset's
[filtered release](https://data.mendeley.com/datasets/r8gmbtv7w2/3) is needed to
run the three-source matrix. Its file-list endpoint returned HTTP 403 during this
session. No duplicate source downloads or fabricated labels were introduced.

## Signal and evaluation conventions

Native axes are retained: no anatomically unsupported axis negation or rotation.
Either-ankle support is tested on recorded FoG-STAR sides, not asserted for arbitrary
mountings. Source resampling uses a causal fourth-order 20 Hz Butterworth filter,
then previous-sample hold onto 50 Hz timestamps. Filter delay is retained.
One second of filter warm-up is excluded; windows require two more seconds of data.
Any invalid sample, missing sample or gap over 1.6 source sample intervals breaks
continuity. Values are clipped to the existing device's ±2 g range.

Normalization is fitted on training recordings only. Dynamic-input experiments
subtract a causal .02 EMA gravity estimate; the default keeps gravity. Positive
window thresholds of 20%, 50%, and 80% are compared using validation data. Stride
is 10 samples (200 ms); window length is 100 samples. Window-size/stride ablations
and orientation augmentation are not yet implemented.

Splits are deterministic, namespaced and participant-disjoint, including both
ankles. Two-dataset holdout experiments exclude the held-out dataset from training
and validation. Cross-side experiments train on one side and evaluate both sides
of held-out participants. Participant bootstrap intervals are reported, keeping
both sides together. The extended suite includes three grouped development folds,
excluding every frozen final test participant, and a uniform-sampling comparison.
Small participant counts still limit population-level conclusions.

Predictions are timestamped at window end, never backdated. Event matching uses
one-to-one positive temporal overlap, no tolerance; duplicate detections count as
false events. False alarms/hour uses valid observable sensor hours including FoG.
Missing intervals and warm-up are excluded and reported. Events truncated by
unobservable periods are evaluated only over the observed part. Full-recording
episode sensitivity including these periods is not established. Bilateral pooled
metrics count two correlated sensor observations; they are not unique episode counts.

Validation selection maximizes event F1 minus `0.002 * false alarms/hour`; it is an
engineering tradeoff, not a clinical acceptance criterion. Test data never fits
normalization, thresholds, representation selection or quantization calibration.
Completed experiments are reused. To change configuration, use a new experiment
name/output version; `last.pt` contains optimizer, scheduler and RNG state and
resumes training from the most recently completed epoch.

## Firmware and collection

Pins are preserved: SDA 21, SCL 22, motor 13, BOOT 0, LED 2, MPU6050 address 0x68,
400 kHz I²C, ±2 g, 50 Hz. Copy generated `model_data.h` and `deployment_config.h`
from `artifacts/deployment` beside the sketch before compiling against a compatible
TensorFlow Lite Micro/ESP32 Arduino 2.x toolchain. No board toolchain was available
in this workspace. The arena is a provisional allocation, not measured usage.

Sampling and inference use separate tasks with a one-slot latest-window queue.
Timeouts, stale windows, saturation and sensor failures suppress cues. Each cue is
a bounded 150 ms pulse; continuous positives cannot leave the motor on. Motor is
disabled by default; serial `V` enables it and `v` or BOOT disables it. A software
fault that prevents all scheduling still depends on hardware watchdog/reset and
the documented gate pull-down; watchdog behavior has not been validated.

The offline source anti-alias filter and the MPU6050 hardware frequency response
are **not** established as equivalent. Host parity covers clipping/normalization
and alarm transitions. It does not prove source-filter, sensor, or TFLM board parity.
Deployment remains blocked on that characterization, operator/toolchain checks,
measured memory/latency, and long-duration testing.

Collection requires `pyserial` installed in the local environment:

```powershell
& _rebuild/.venv/Scripts/python.exe _rebuild/collect.py --port COM3 --participant pilot01 --side L --mounting lateral-shank --seconds 60 --out _rebuild/artifacts/collection/session01
```

Use pseudonymous identifiers and appropriate consent/supervision for human data.
The utility records m/s², timestamps, status and mounting metadata; labels must be
independently annotated before supervised use. Automatic import and annotation UI
are not yet implemented.

## Status

See `reports/RESULTS.md`, `reports/experiment_summary.json`, and
`artifacts/deployment/validation.json` after execution. Missing artifacts mean that
stage has not completed. No clinical readiness, board performance, or improvement
over the old model is implied by a successful training/export run.
