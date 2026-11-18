# Data contract and implementation status

Implemented: recording schema; Daphnet and Fogstar loaders; a manifest-based
multimodal loader; deterministic subject splits and LOSO folds; causal resampling;
training-only normalization; onset-relative windows; initial IMU features; a
causal TCN and real-data training smoke entry point.

Raw files and `_rebuild` are unchanged. Hardware remains ESP32-WROOM-32 plus
MPU6050 at 50 Hz with the original wiring. Research processing uses 100 Hz.

## Dataset semantics

- Daphnet has nine accelerometer axes from ankle, thigh, and trunk, in milligravity.
  The loader converts to g and maps labels 0/1/2 to invalid/normal/FoG. It does not
  invent gyroscopes or bilateral shank channels.
- Fogstar retains all 24 sensor channels and groups by subject/session/task even
  when CSV chunks or rows interleave. Acceleration is in g; gyroscope scaling is
  explicitly `source_gyro` until verified. Do not mix it with device deg/s data.
  Severity extraction and auxiliary training are pending.
- The bundled multimodal examples lack verified patient identities and alignment
  provenance. They are excluded from subject-wise experiments. The loader accepts
  aligned numeric-timestamp CSVs only through an explicit manifest. Do not assign
  task IDs as subject IDs or claim the sample files represent ten patients.

Multimodal manifest example (replace all values with verified release metadata):

```json
[
  {
    "path": "aligned/subject01_session01.csv",
    "subject": "subject01",
    "session": "session01",
    "recording": "task01",
    "sample_rate_hz": 500,
    "time_column": "seconds",
    "time_scale_seconds": 1,
    "label_column": "Label",
    "channels": ["LShankACCX", "EMG1"],
    "units": ["g", "uV"]
  }
]
```

Channel names and units are ordered, explicit metadata. Headerless CSVs also
require a `columns` array. A unified multimodal tensor and modality masks remain
to be implemented; different channel schemas cannot share a normalizer.

## Time and labels

Split invalid samples and time gaps before resampling. The causal low-pass filter
and previous-sample hold never interpolate using future sensor samples; retain
filter delay and discard a one-second startup interval. Upsampling adds no new
information. Evaluate latency with these delays included.

Window targets are based on the final observed timestamp relative to genuine
0-to-1 FoG transitions: five seconds before through one second after onset.
Ongoing FoG outside that interval is excluded from this prediction objective.
Tail negatives lacking five seconds of future annotations are censored. A
segment already in FoG has no known onset. These prediction windows must not be
used to estimate uninterrupted monitoring exposure or event-level false alarms;
that evaluation needs full continuous recordings and original event annotations.

The generic window is two seconds with a 0.1-second stride. The TCN smoke uses
five-second windows and a 509-sample theoretical receptive field at 100 Hz.

## Run checks

```powershell
python -m pip install -e ".[dev,train]"
python -m pytest
python -m src.training.smoke --root . --steps 5
```

The local `_rebuild/.venv/Scripts/python.exe` also has PyTorch and can run the
checks without changing that environment. Smoke output goes to
`reports/foundation_smoke.json`. This is a small training-only optimization
check; its losses are not validation results or evidence of target attainment.

The next increment adds a lazy window dataset, Daphnet training/validation CLI,
continuous event metrics, and validation-only calibration. See
[Training and validation baseline](training.md) for commands and metric semantics.
Verified dataset/unit manifests, a modality-aware Lightning DataModule, MLflow,
additional architectures, full subject-wise evaluation, and embedded export
remain outstanding. The remaining plan is not complete.
