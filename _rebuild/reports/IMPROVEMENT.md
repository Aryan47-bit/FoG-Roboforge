# Second development round

Previously reported test set is a reused benchmark, not fresh independent certification. No thresholds are fit on it.

| Model | Validation score | Benchmark sensitivity | False events/hour | Median delay |
|---|---:|---:|---:|---:|
| combined_cnn_gravity_0.2 | 0.571 | 0.559 | 32.80 | 1.9599999999982174 |
| v2_longtcn_gravity_0.2 | 0.528 | 0.431 | 22.62 | 1.920000000019627 |
| v2_longtcn_dynamic_0.2 | 0.335 | 0.794 | 73.51 | 1.8399999999983265 |
| v2_longtcn_dynamic_0.5 | 0.355 | 0.755 | 61.64 | 1.999999999998181 |

Validation-selected: combined_cnn_gravity_0.2. Engineering target met: False.
The target of 90% sensitivity, <=2 false events/hour and <=2 s median delay is provisional, not a clinical acceptance criterion.
Existing deployment artifacts are preserved. Fresh participant data, source/device filter parity, board compilation, resource measurements and hardware validation remain necessary.
Validation-only target sweep: 84 configurations across the three new models. Best sensitivity subject to <=2 false events/hour was 20.4%, at 1.35 false events/hour and 3.74 s median delay. No configuration met the provisional joint target. All 12 automated tests passed after the architecture change. This round did not establish a superior deployment candidate.

