# Resource and accuracy analysis at all nine normalized loads

This analysis covers all nine Qwen2.5 normalized loads and all five deployable methods from the formal experiment outputs, using their recorded policies, coverage factors, and 3 s upstream delay.

## Provenance and verification

- Canonical aggregate: `@@WORKSPACE@@/drift_lora_project/results/model_specific_delay_normalized_20260914/normalized_load_mean_std.csv`.
- Qwen2.5 baseline source: `upstream_delay_sensitivity_formal_new_manifest_0to5s_20260907/group_delay_rows.csv`, normalized loads, delay 3 s, seeds 42/43/44.
- Qwen2.5 SADA source: `qwen25_target80_admission_gate_3s_20260914/target80_group_actual_results.csv` and per-seed `admission_policy_selections.csv`.
- Preserved five groups, 110 ordered held-out R_test samples per group, current inference/adaptation profiles and accepted-and-correct scoring. Saved coverage factors and sample-boundary rounding are preserved.
- Independently reproduced 36 baseline mean/std pairs and 9 SADA mean/std pairs. Reconstructed and checked all 540 baseline and 135 SADA group/seed/load utilities from per-sample predictions.
- Periodic, Static and EWC have identical window utilities at all nine loads because their inference policies match and none of their adapters becomes ready inside these windows. Their training completion times differ; they are not identical algorithms.
- Qwen2.5 higher/tied/lower count is 7/2/0. Overall four-family count is 31/2/3.
- `validation.json` stores source hashes and check counts. `all_families_all_methods.csv` copies all 180 current formal mean/std records; `table_v_all_loads.csv` contains the five-by-nine table.

## Figure semantics

- Panel (a): mean selected inference allocation, selected retraining allocation only when a training job exists, and idle residual budget. The unused inference share includes both retraining allocation and idle capacity.
- Panels (b–d): resource budgets reconstructed from saved decisions and completion times, averaged over all 15 group/seed decisions. This is not measured GPU utilization. Upon completion, training budget is shown as released; inference allocation is not increased beyond the recorded selection.
- The 3 s upstream phase uses the original NoRetrain configuration only for training decisions. The original Qwen2.5 no-update branch uses its selected configuration immediately, faithfully retaining the current evaluation convention.
- Panels (e–g): fixed bins of 10 consecutive samples per group, averaging all five groups and three seeds. No best-case group or seed was selected. Areas mark differences against NoRetrain evaluated on the same samples; the full-window means exactly reproduce Table V.
- 0.25x: 14/15 adapters ready, mean accuracy 0.4884848485 versus 0.42. 0.5x: 3/15 ready, 0.4254545455 versus 0.42. Higher loads select no adapter.
- The data show intermittent pre-readiness disadvantages and later gains, not a universal smooth decline-then-rise curve. Resource allocation and inference configuration both change, and sample composition also affects the temporal trajectory; this is not an isolated causal ablation of allocation.
- The builder generates `cumulative_accuracy_all_loads.pdf` with all nine SADA/NoRetrain cumulative accuracy trajectories; this generated PDF is not bundled with the release.

## Reproduction

`build_results.py` requires NumPy, Matplotlib, and the original external profiles, manifests, and prediction bundles listed above. Those external inputs are not included in this repository. With those inputs available, the script checks the sources and regenerates CSV, PDF, SVG, PNG, and `table_v.tex` outputs in this directory.
