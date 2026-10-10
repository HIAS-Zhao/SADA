# Drift detection experiments

This directory is the paper's detector reproducibility directory. `src/` implements feature streams and detectors; `scripts/` contains the final comparison, ablation, held-out sensitivity, early-recall, label-aware, and overhead entry points; `paper_results/` contains the selected final result snapshots.

The fixed main configuration is `V+VT` (code feature group `A+E_text`), PCA dimension 128, window size 200, and quantile 0.99. The final paper reports 94.07% macro score, 4.10% FPR, and 65.37% recall at the 5% shift ratio for the main label-free comparison. The result folders are dated experiment identifiers, not a request to combine historical runs.

Paper feature labels `V`, `VT`, and `V+VT` correspond to code groups `A`, `E_text`, and `A+E_text`. The comparison exports use `SADA` and `Global Fréchet`; their internal scorer lookup keys are `Mahalanobis` and `DriftLens (Frechet)`, respectively. Module names, function names, and scorer keys identify the implementation used by the reproduction and overhead scripts.

Run scripts from this directory after preparing the original dataset and feature caches. Paths are resolved relative to the experiment root or supplied by the script; no data is bundled.

## Final comparison command

After materializing the workspace, run from its `drift_detection_exp/` directory. Repeat with seeds 42, 43, and 44:

```bash
python scripts/run_best_mahalanobis_baseline_comparison.py \
  --features-dir features_pooled_e_fusion --feature-group A+E_text \
  --fusion-rule max_norm --pca-dim 128 --window-size 200 \
  --calibration-quantile 0.99 --calibration-windows 10000 \
  --windows-per-ratio 1000 --seed 42 \
  --drift-ratios 0 .05 .06 .07 .08 .09 .10 .12 .14 .16 .18 .20 \
  --output-dir outputs/paper_seed42
```

For the paper's seed-42 ablation, restrict `run_ab_etext_sensitivity.py` to `--feature-groups A E_text A+E_text --fusion-rules max_norm --pca-dims 128 --quantiles .99 --window-sizes 200 --seed 42` and the same drift ratios. Single-stream cases use their single-stream rule internally. Historical defaults in archived scripts are not necessarily the paper configuration; pass the explicit options above.

The held-out sensitivity table can be rebuilt without features or a GPU:

```bash
python drift_detection/scripts/summarize_taskheldout_hyperparameter_sensitivity.py \
  --input drift_detection/paper_results/task_heldout_detector_20260726/candidate_fold_scores.csv \
  --output-dir outputs/heldout_sensitivity
```

Run that command from the repository root. It verifies coverage of all seven held-out tasks for each of the twelve parameter settings. `paper_results/论文消融结果表.csv` is the canonical paper ablation table; `paper_results/ab_etext_sensitivity_20260523_with_comparison/figures/paper_ablation_table.csv` and `.md` provide the matching English version. Its macro score averages 12 drift ratios from 0% through 20%. The original seven-configuration tables averaging 14 ratios through 50% are retained under `historical_extended_ratios/`. Original machine-readable evaluation arrays remain in `paper_configurations.json`.
