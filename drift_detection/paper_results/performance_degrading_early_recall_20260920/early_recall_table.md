# Early recall and detector cost

Seeds: 42, 43, 44. Trials per seed: 1000. Detection metrics are percentages, mean +/- sample standard deviation across seeds.

Recall is cumulative detection by the stated number of samples after onset, counted from one. FPR is the fraction of positive no-drift windows of 200 samples, with endpoints 200, 210, ..., 500 within each clean trace. SADA evaluates each window at its endpoint. A label-aware window is positive if the detector raises at least one native alarm within that interval; detector state is retained across overlapping windows and reset after alarms.

SADA uses a 200-sample window, PCA 128, the original calibration quantile and frozen thresholds, and a stride of 10 arrivals. Its first post-onset checkpoint contains 190 clean and 10 shifted samples. Label-aware methods retain their native per-sample updates, calibration warmup and reset rules. Missing detections are explicit.

| Method | Window FPR | Recall <= 10 | Recall <= 20 | Recall <= 30 | Recall <= 40 | Recall <= 50 | Recall <= 100 | Recall <= 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SADA | 4.05 +/- 0.56 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| DDM | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| EDDM | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 34.53 +/- 2.58 |
| ADWIN | 0.93 +/- 0.16 | 0.03 +/- 0.06 | 0.87 +/- 0.12 | 16.27 +/- 1.07 | 83.10 +/- 0.62 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| HDDM-W | 2.72 +/- 0.82 | 71.30 +/- 1.39 | 99.23 +/- 0.32 | 99.70 +/- 0.20 | 99.73 +/- 0.15 | 99.80 +/- 0.10 | 99.80 +/- 0.10 | 99.80 +/- 0.10 |
| Page-Hinkley | 11.19 +/- 1.03 | 0.03 +/- 0.06 | 0.40 +/- 0.20 | 2.53 +/- 0.51 | 10.93 +/- 0.75 | 39.23 +/- 2.06 | 93.87 +/- 0.59 | 96.23 +/- 0.57 |
| KSWIN | 1.26 +/- 0.10 | 1.50 +/- 0.50 | 46.60 +/- 1.25 | 99.50 +/- 0.26 | 99.60 +/- 0.10 | 99.70 +/- 0.20 | 99.73 +/- 0.15 | 99.73 +/- 0.15 |
| OPTWIN-style | 2.85 +/- 0.22 | 4.70 +/- 0.36 | 99.87 +/- 0.06 | 99.87 +/- 0.06 | 99.87 +/- 0.06 | 99.87 +/- 0.06 | 99.87 +/- 0.06 | 99.87 +/- 0.06 |

The three seeds repeat resampling from fixed cached pools; their standard deviation describes this Monte Carlo variation. The drift pool contains high-supervised-loss examples and does not represent all types of natural drift. Detector thresholds are fixed independently of this evaluation.

## Detector computation

One CPU thread on an AMD EPYC 9654; 300 shared no-change traces from seed 42, with ten timed repeats each. Each trace has a 300-sample clean prefix followed by 200 timed arrivals. SADA performs 20 scoring calls; label-aware methods perform 200 updates including resets after alarms. Input construction, PCA projection, calibration, warmup and state cloning are excluded.

| Method | Detector ms per 200 new arrivals |
|---|---:|
| SADA | 0.650 |
| DDM | 0.161 |
| EDDM | 0.093 |
| ADWIN | 13.478 |
| HDDM-W | 0.497 |
| Page-Hinkley | 0.160 |
| KSWIN | 9.200 |
| OPTWIN-style | 7.068 |

SADA attains 100% recall at its first 10-sample checkpoint with a window FPR of 4.05 +/- 0.56%. Its FPR exceeds ADWIN's 0.93 +/- 0.16% under the same window definition.

## Released results and reproduction

- `early_recall_config.json`: frozen configurations, environment versions, commands and source/input hashes.
- `early_recall_runs.json`, `early_recall_summary.json`, `early_recall_summary.csv`, `seed_*/metrics.json`: per-seed and aggregated window FPR and recall.
- `window_fpr_reviewed.json`: per-seed positive-window counts, denominators and the window-positive rules.
- `reproduction_audit.json`: fixed-cadence recall and delay comparisons.
- `independent_audit.json`: input and recall audit metadata.

Full trial-index arrays, input features/losses, per-trial detector traces and timing arrays are external experiment artifacts. The generation and full trial-audit entry points require those inputs.
