# Training and evaluation report

Selected using validation only: **combined_cnn_gravity_0.2**.

| Experiment | Test PR-AUC | Event sensitivity | False events/hour | Median delay (s) |
|---|---:|---:|---:|---:|
| daphnet_cnn | 0.36817241043254983 | 0.348 | 13.06 | 1.3999999999987267 |
| daphnet_rescnn | 0.5005128714483656 | 0.591 | 26.11 | 1.1999999999989086 |
| daphnet_tcn | 0.4803705811417704 | 0.530 | 23.50 | 1.479999999998654 |
| fogstar_cnn | 0.5925819516304499 | 0.806 | 92.00 | 0.0 |
| fogstar_rescnn | 0.42893526551058114 | 0.861 | 103.30 | 0.5799999999994725 |
| fogstar_tcn | 0.4559160094802041 | 0.833 | 92.00 | 0.0 |
| combined_cnn | 0.36034787790526285 | 0.431 | 27.14 | 1.8400000000196997 |
| combined_rescnn | 0.40805129722568206 | 0.490 | 43.54 | 1.159999999998945 |
| combined_tcn | 0.48691615283594913 | 0.549 | 46.37 | 1.7199999999984357 |
| heldout_daphnet | 0.14208516232566307 | 0.241 | 20.52 | 0.8999999999991815 |
| heldout_fogstar | 0.1655516060381657 | 0.000 | 0.00 | None |
| train_side_L | 0.4460848555088417 | 0.917 | 117.83 | 1.1199999999989814 |
| train_side_R | 0.3918773221764102 | 0.417 | 75.86 | 0.0 |
| combined_cnn_dynamic_0.5 | 0.6135087809371474 | 0.892 | 84.25 | 0.5599999999994907 |
| combined_cnn_gravity_0.2 | 0.4381440953317814 | 0.559 | 32.80 | 1.9599999999982174 |
| combined_cnn_gravity_0.8 | 0.2720576812349533 | 0.725 | 59.37 | 1.170000000012351 |
| combined_cnn_uniform | 0.47311401057714797 | 0.618 | 36.76 | 1.7599999999983993 |
| development_cv_0 | 0.26708939460170616 | 0.163 | 23.82 | 0.0 |
| development_cv_1 | 0.4596636841918612 | 0.549 | 46.24 | 0.8200000000089176 |
| development_cv_2 | 0.15926632313697234 | 0.272 | 24.55 | 1.7599999999983993 |

## Combining datasets

The following comparison uses validation-selected individual models and the validation-selected combined model, evaluated on the same frozen participants within each dataset. Event metrics share ground truth even when window-overlap labels differ.

- daphnet: individual `daphnet_rescnn` sensitivity 0.591, false events/hour 26.11; combined sensitivity 0.439, false events/hour 16.54.
- fogstar: individual `fogstar_rescnn` sensitivity 0.861, false events/hour 103.30; combined sensitivity 0.778, false events/hour 62.95.

## Either-side observations

- L ankle, held-out participants: sensitivity 0.611, false events/hour 35.34.
- R ankle, held-out participants: sensitivity 0.944, false events/hour 90.82.

## Export and quantization

Float TFLite: 12,268 bytes. Full INT8 TFLite: 9,008 bytes. Input/output are INT8; input shape is 1 × 100 × 3.
PyTorch/Keras maximum absolute probability difference: 2.09e-07.
INT8 held-out event sensitivity: 0.539; false events/hour: 32.80; median delay: 1.96 s.
Quantization event-sensitivity change: -1.96 percentage points. Maximum calibration-window probability difference: 0.060.
All exported operator names are covered by the candidate firmware resolver. This is a static operator check, not a TFLM compile or hardware compatibility result.
Export artifacts: `artifacts/deployment/`. Model header and normalization/decision config are also placed beside the firmware sketch.

## Limits


These are sensor-stream metrics. Bilateral FoG-STAR observations are correlated; pooled event counts do not represent unique clinical episodes.
Matching is chronological, one-to-one, strictly positive overlap after the first available window-end decision. Duplicate predicted events count as false events. No tolerance or backdating is applied.
False-event rate divides by valid observed sensor time, including FoG; warm-up and missing intervals are excluded. Full-recording sensitivity, including unobservable episodes, has not been established.
Small held-out groups have high uncertainty. Native axes remain dataset-specific. Either-ankle conclusions apply only to these recorded placements.
Training used the RTX 4050. Public-dataset evaluation does not establish clinical performance or arbitrary mounting tolerance.
The full multimodal annotated release is missing. Three-source comparisons and that dataset baseline are blocked.
Board memory, latency, firmware compilation, hardware-in-the-loop parity, and long-duration stability have not been measured. This is not deployment-ready.