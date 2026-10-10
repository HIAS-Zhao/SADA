# Task-Held-Out Detector Evaluation

- Seven folds hold out one shifted task at a time.
- Hyperparameter selection uses only the other six shifted tasks and the no-drift development half.
- Final FPR uses the disjoint no-drift test half; held-out task windows are never used for selection.

| Protocol | FPR | R@5 | R@10 | R@20 | Macro | Worst-task macro |
|---|---:|---:|---:|---:|---:|---:|
| Nested task-held-out selection | 0.92% | 86.29% | 99.15% | 100.00% | 95.92% | 71.99% |
| Fixed main configuration | 2.60% | 88.42% | 99.45% | 100.00% | 96.84% | 79.15% |

## Nested Selection by Held-Out Task

| Task | Selected config | Dev macro | Test macro | FPR | R@5 |
|---:|---|---:|---:|---:|---:|
| 1 | E_text, PCA 256, W 300, q=0.99 | 99.64% | 99.95% | 0.60% | 100.00% |
| 2 | E_text, PCA 256, W 300, q=0.97 | 99.32% | 99.76% | 2.93% | 100.00% |
| 3 | E_text, PCA 256, W 300, q=0.99 | 99.68% | 99.94% | 0.73% | 100.00% |
| 4 | E_text, PCA 256, W 150, q=0.995 | 99.97% | 71.99% | 0.07% | 4.00% |
| 5 | E_text, PCA 256, W 300, q=0.99 | 99.66% | 99.94% | 0.73% | 100.00% |
| 6 | E_text, PCA 256, W 300, q=0.99 | 99.68% | 99.94% | 0.67% | 100.00% |
| 7 | E_text, PCA 256, W 300, q=0.99 | 99.72% | 99.94% | 0.70% | 100.00% |

## Fixed Main Configuration by Held-Out Task

| Task | Test macro | FPR | R@5 | R@10 | R@20 |
|---:|---:|---:|---:|---:|---:|
| 1 | 99.79% | 2.53% | 100.00% | 100.00% | 100.00% |
| 2 | 99.76% | 2.93% | 100.00% | 100.00% | 100.00% |
| 3 | 99.80% | 2.37% | 100.00% | 100.00% | 100.00% |
| 4 | 79.15% | 2.57% | 18.97% | 96.17% | 100.00% |
| 5 | 99.76% | 2.83% | 100.00% | 100.00% | 100.00% |
| 6 | 99.79% | 2.47% | 100.00% | 100.00% | 100.00% |
| 7 | 99.79% | 2.50% | 100.00% | 100.00% | 100.00% |
