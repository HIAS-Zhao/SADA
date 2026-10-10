# Resource split and ship classification accuracy

## Resource panel

Panel (a) uses normalized load factors 0.5x, 1x, 2x, and 4x. Its source is `framework/paper_results/qwen25_selections.json`, with all five groups and seeds 42/43/44 at each factor. Mean inference budgets are 8.7301%, 17.4603%, 38.9041%, and 77.8083%; the complementary share is the allocated retraining budget. These values are recorded selections.

## Case and source

Panel (b) shows ship type classification (G3, `v2_g3_A9_B5`), Qwen2.5, at absolute arrival rate 0.2 samples/s: 110 ordered held-out samples, 550 s horizon, 3 s upstream wait, and seeds 42/43/44. This absolute-rate case uses the original selected policies; its settings differ from the normalized-load resource panel and the subsequent target80 admission policies.

The original experiment sources are `upstream_delay_sensitivity_formal_new_manifest_0to5s_20260907/group_delay_rows.csv` and `dtop80_current_pool_recompute_20260726/all_model_rtest_only_formal_profiles_new_manifest/seed*/pressure/qwen25/ar0p2/group_detail_native.csv`. These external result and prediction bundles are not included in this repository. The builder reads their prediction/config manifests and checks all three SADA and NoRetrain scores.

Mean over seeds: readiness 421.8248 s; before readiness SADA 47.0577%, NoRetrain 49.4115%; after readiness SADA 67.9785%, NoRetrain 52.0086%; full window SADA 51.8182%, NoRetrain 50.0000%. The individual ready times are 411.27147, 424.160296, and 430.042627 s.

## Smoothing and display

Reflected Gaussian smoothing uses sigma 15 samples (75 s), truncate 4, and the original chronological outcomes. SADA is filtered separately within each seed's pre/post-ready segments; NoRetrain is filtered continuously with the same bandwidth. The reflected filter preserves SADA segment means and both methods' full-window means to 1e-12. The plotted curves include all three seeds. Original and smoothed records are provided separately. Both methods use the same 35–80% y-axis range.

This filtering summarizes variation retrospectively; it is not an online estimator. Separate SADA segments preserve its known model-replacement boundary; the baseline has no replacement.

## Case selection

This case was selected post hoc from five Qwen2.5 normalized tasks, five stateful tasks across three seeds, and ship cases at six absolute arrival rates (0.025/0.05/0.075/0.1/0.15/0.2 samples/s). The selected case illustrates a pre-ready disadvantage, a post-ready advantage, and a positive full-window difference. All three seeds are included. It is an illustrative example and does not establish average behavior across tasks or loads. The original case-search archive is external and is not included in this release.

## Files and reproduction

- `validation.json`: protocol, metrics, smoothing settings, selection scope, and source hashes.
- `raw_sample_values.csv`, `smoothed_sample_values.csv`, `plotted_trace.csv`, `seed_metrics.csv`, `resource_split.csv`: released numerical inputs and summaries.
- `build_ship_figure.py`: original source-reconstruction and figure-generation script, requiring Python, NumPy, SciPy, Matplotlib, and the external experiment bundles described above.

The builder generates combined/separate figures and a raw-versus-smoothed comparison in this directory. Generated PNG/PDF/SVG files are not bundled with this release.
