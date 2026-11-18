# Alternative-model comparison

Full eligible Daphnet recordings; fixed seed-42 subject split. Training: 103,621 windows, 144 features for tree models; CNN uses the same historical five-second windows directly. No test subjects used for model selection or calibration.

Validation: subjects S01 and S08, 37 fully observable episodes, 2,290.04 seconds (0.636 hours) of non-FoG exposure. Short segments below 20 seconds, startup, and censored intervals are excluded by the documented metric contract.

## Operating points satisfying observed FA/hr < 2 for each validation subject

| Model | Threshold | Balanced accuracy | Window sensitivity | Window specificity | Episodes caught | Observed FA/hr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| extra_trees | 0.35 | 57.27% | 17.97% | 96.58% | 2/37 | 0.00 |
| random_forest | 0.65 | 50.00% | 0.00% | 100.00% | 0/37 | 0.00 |
| hist_gradient_boosting | 0.83 | 51.89% | 4.46% | 99.32% | 0/37 | 0.00 |
| cnn | 0.88 | 56.36% | 39.22% | 73.50% | 0/37 | 0.00 |

These probability thresholds maximize window balanced accuracy among the false-alarm-feasible candidates. Window classification uses the stated threshold; event alarms additionally apply the fixed 0.1 hysteresis margin, 0.5-second confirmation and 10-second refractory interval. They are distinct metrics. The original CNN trainer checkpoint uses its separate event-sensitivity-first calibration policy; its comparison threshold is recorded in `cnn_full_001/comparison_operating_point.json`.

## Default probability threshold 0.5 (before FA constraint)

| Model | Balanced accuracy | Episode sensitivity | FA/hr (pooled) |
| --- | ---: | ---: | ---: |
| extra_trees | 50.85% | 0.00% | 0.00 |
| random_forest | 50.17% | 0.00% | 1.57 |
| hist_gradient_boosting | 56.72% | 18.92% | 29.87 |
| cnn | 65.77% | 18.92% | 62.88 |

## Conclusion

None meets the joint >90% balanced accuracy, >90% sensitivity and <2 false alarms/hour targets. Extra Trees catches two episodes under the observed FA constraint; their median alarm latency is -2.635 seconds. No compared model produces a scored warning 5-10 seconds before onset at its constrained operating point.

Zero observed false alarms over this short validation exposure does not establish a population FA rate below 2/hour. These are exploratory validation results, not independent test results. The held-out subjects remain reserved. Models use nine Daphnet accelerometer axes; they are not drop-in firmware for a single six-axis MPU6050.

## Reproduce

```powershell
python -m src.training.compare_features --output reports/feature_comparison_new
python -m src.training.trainer --output reports/cnn_full_new --model cnn --device cuda --epochs 10 --patience 4 --batch-size 128
```

CNN stopped after seven epochs; epoch three had the best validation balanced accuracy. Saved tree estimators, validation probability traces, source SHA256 hashes, subject splits, and complete threshold sweeps are in `feature_comparison_001`. CNN weights and training history are in `cnn_full_001`.

Next experiments should preserve the evaluation contract: verify source units and sensor compatibility for Fogstar training, evaluate additional subject folds, and compare onset detection with strict pre-onset prediction as separately labeled objectives. Do not relax the prediction horizon or substitute plain accuracy to claim the original targets.
