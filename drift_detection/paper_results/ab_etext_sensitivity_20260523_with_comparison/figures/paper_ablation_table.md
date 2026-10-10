# Paper ablation table

Seed 42; PCA 128; window 200; calibration quantile 0.99. Macro accuracy is the arithmetic mean over drift ratios 0, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, and 0.20. Values match `../../论文消融结果表.csv`. The original extended-ratio diagnostic is retained in `../historical_extended_ratios/`.

| Paper stream | Code group | FPR (%) | R@5% (%) | R@10% (%) | R@12% (%) | R@20% (%) | Macro accuracy (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| V | A | 1.30 | 6.00 | 26.00 | 34.50 | 92.50 | 43.11 |
| VT | E_text | 3.10 | 64.80 | 99.50 | 100.00 | 100.00 | 93.66 |
| V+VT | A+E_text | 4.00 | 66.30 | 99.50 | 100.00 | 100.00 | 94.00 |
