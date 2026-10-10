# Task-Held-Out Detector Hyperparameter Sensitivity

This is a one-at-a-time diagnostic around the fixed main setting `V+VT / PCA=128 / K=200 / q=0.99`. Each cell averages the seven held-out-task test folds; no setting in this table is used to retune the reported main detector.

## PCA dim.

| Value | Macro | FPR | R@5% | R@10% | Worst-task Macro | Worst-task R@5% |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 95.84 | 4.42 | 88.20 | 97.43 | 73.23 (T4) | 18.50 (T4) |
| **128** | 96.84 | 2.60 | 88.42 | 99.45 | 79.15 (T4) | 18.97 (T4) |
| 192 | 97.52 | 1.83 | 88.76 | 99.92 | 83.58 (T4) | 21.30 (T4) |
| 256 | 98.10 | 1.78 | 89.74 | 99.99 | 87.60 (T4) | 28.17 (T4) |

## Window

| Value | Macro | FPR | R@5% | R@10% | Worst-task Macro | Worst-task R@5% |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 94.06 | 2.06 | 85.46 | 92.93 | 60.26 (T4) | 7.80 (T4) |
| 150 | 95.98 | 2.59 | 87.92 | 97.71 | 73.13 (T4) | 15.50 (T4) |
| **200** | 96.84 | 2.60 | 88.42 | 99.45 | 79.15 (T4) | 18.97 (T4) |
| 300 | 98.06 | 6.18 | 91.22 | 100.00 | 89.47 (T4) | 38.57 (T4) |

## Quantile

| Value | Macro | FPR | R@5% | R@10% | Worst-task Macro | Worst-task R@5% |
|---:|---:|---:|---:|---:|---:|---:|
| 0.95 | 97.45 | 14.36 | 92.68 | 99.97 | 89.30 (T4) | 48.77 (T4) |
| 0.97 | 97.45 | 8.81 | 91.24 | 99.90 | 86.53 (T4) | 38.67 (T4) |
| **0.99** | 96.84 | 2.60 | 88.42 | 99.45 | 79.15 (T4) | 18.97 (T4) |
| 0.995 | 96.34 | 1.27 | 87.57 | 98.98 | 74.98 (T4) | 12.97 (T4) |
