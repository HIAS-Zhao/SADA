# Framework experiments

This directory is the paper's full-framework reproducibility directory. `scripts/` contains the seed-42/43/44 replay, training, validation, and aggregation entry points; `paper_results/` contains the 945 group-level rows, 420 seed-level cells, 140 three-seed aggregates, and the Chinese paper-format table; `src/` contains the archived modules that those entry points import.

Run `python scripts/verify_paper_results.py` to recalculate the published aggregate table from the released group-level CSV files. It checks row counts, weighted seed scores, means, and sample standard deviations (ddof=1).

On Linux, to restore the original experiment layout, run `python scripts/prepare_workspace.py --workspace <empty-workspace> --qwen25-python <python> --vlm-python <python>`. Use absolute interpreter paths. The restored run still needs the datasets, checkpoints, third-party code, and accelerator environment that are excluded from this repository. The `source_manifest.json` records the original relative source locations, source hashes, and published hashes; machine-specific prefixes are replaced by tokens and text formatting is normalized. The archived scripts in `src/` may retain older names because the final experiment imports their functions.

`ThiefLoRA` is the dynamic scheduler key in archived allocation inputs, and `Ours` identifies its schedule in the materialization code. Published framework tables use `SADA`. These keys are retained for loading the corresponding experiment artifacts.

From the restored `drift_lora_project/results/framework_seeds43_44_20260921/` directory, the original sequence is `run_lane.py qwen25`, `run_lane.py qwen35`, `train_specialists.py`, missing-prediction evaluation where required, `qwen25_replay.py score`, `replay_bundles.py`, and `aggregate.py`. Read each entry point and supply its referenced training manifests and profiles first; the frozen CSV verification command above needs none of those external inputs. Original scripts select specific GPU indices; adapt those indices to the available hardware.
