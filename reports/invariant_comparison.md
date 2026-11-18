# Rotation-invariant multiscale experiment

Changed the representation and training weights: 192 rotation-invariant features over trailing 1/2/5-second intervals, recent-versus-past motion energy ratios, and equal total weight per observed training subject/class group. Extra Trees and histogram gradient boosting were retrained on all 103,621 eligible training windows.

Same seed-42 Daphnet split, five-second context, prediction labels, validation recordings, and alarm policy as the previous experiment. Validation contains 37 scored episodes and 0.636 hours of non-FoG exposure. Held-out test subjects were not used for selection or calibration. No hardware changes.

| Model | Balanced accuracy at 0.5 | FA/hr at 0.5 | Constrained threshold | Constrained balanced accuracy | Constrained FA/hr | Episodes caught |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| extra_trees | 64.16% | 40.87 | 0.78 | 52.73% | 0.00 | 0/37 |
| hist_gradient_boosting | 57.80% | 160.35 | 0.88 | 57.00% | 0.00 | 0/37 |

The constrained threshold maximizes window balanced accuracy among candidates with observed FA/hr <2 for each validation subject. Window accuracy uses the probability threshold; alarm events additionally use the unchanged 0.1 hysteresis, 0.5-second confirmation, and 10-second refractory period.

Neither model meets the joint targets. Extra Trees improved its default-threshold balanced accuracy from 50.85% to 64.16%, but under the false-alarm constraint it fell to 52.73% and detected no episodes. Gradient boosting reached 57.00% constrained balanced accuracy but also detected no episodes. The previous feature-based Extra Trees result (57.27%, 2/37 episodes) remains better under the constraint.

Zero false alarms on this short validation sample does not establish a population false-alarm rate below 2/hour. Repeated model selection on the same two validation subjects remains exploratory; independent test evaluation is still reserved. No 5-10-second warning target was demonstrated.

```powershell
python -m src.training.compare_features --representation invariant --output reports/invariant_comparison_new
```

The run directory contains saved estimators, complete threshold sweeps, validation probabilities, subject splits, and source-file SHA256 hashes. The test suite passes 22 tests, including independent per-sensor rotations and subject/class weighting checks.
