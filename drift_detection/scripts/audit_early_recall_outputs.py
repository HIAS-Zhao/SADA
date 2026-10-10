#!/usr/bin/env python3
"""Independently check trial counts, delays, statistics, and frozen input hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir
    config = json.loads((root / "early_recall_config.json").read_text())
    assert config["fpr_type"] == "window"
    summary = json.loads((root / "early_recall_summary.json").read_text())
    runs = json.loads((root / "early_recall_runs.json").read_text())
    by_seed = {(row["method"], row["seed"]): row for row in runs}
    pool = set(np.load(root / "aligned_signals.npz")["degrading_indices"].tolist())
    count = config["max_trials"] or config["trials"]
    horizon = config["clean_prefix"]
    window_size = config["sada"]["window_size"]
    endpoints = np.arange(window_size, horizon + 1, config["sada_stride"])
    windows_per_seed = count * len(endpoints)
    checks = 0
    for source, expected in config["input_sha256"].items():
        digest = hashlib.sha256()
        with Path(source).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        observed = digest.hexdigest()
        assert observed == expected, source
        checks += 1
    for seed in config["seeds"]:
        seed_dir = root / f"seed_{seed}"
        with np.load(seed_dir / "trial_indices.npz") as trials:
            for kind in ["clean", "drift", "fpr"]:
                length = config["drift_suffix"] if kind == "drift" else horizon
                assert trials[kind].shape == (count, length), (seed, kind)
                assert all(len(set(row.tolist())) == length for row in trials[kind]), (seed, kind)
            assert set(trials["drift"].ravel().tolist()).issubset(pool), seed
        with np.load(seed_dir / "sada_alarm_checks.npz") as alarms:
            assert alarms["clean_checkpoints"].tolist() == endpoints.tolist()
            assert alarms["drift_checkpoints"].tolist() == list(range(config["sada_stride"], config["drift_suffix"] + 1, config["sada_stride"]))
            sada_window_alarms = alarms["fpr_alarms"].copy()
            assert sada_window_alarms.shape == (count, len(endpoints))
            sada_records = json.loads((seed_dir / "SADA_traces.json").read_text())
            for index, record in enumerate(sada_records):
                hits = [int(n) for n, alarm in zip(alarms["drift_checkpoints"], alarms["drift_alarms"][index]) if alarm]
                assert record["first_alarm_after_onset"] == (min(hits) if hits else None)
                assert record["alarm_windows"] == int(sada_window_alarms[index].sum())
                assert record["windows"] == len(endpoints)
                checks += 3
        for group in summary:
            method = group["method"]
            records = json.loads((seed_dir / f"{method}_traces.json").read_text())
            assert len(records) == count
            assert [row["trial"] for row in records] == list(range(count))
            assert all(row["missed"] == (row["first_alarm_after_onset"] is None) for row in records)
            assert all(row["detected"] != row["missed"] for row in records)
            assert all(not {"false_positive", "first_alarm_in_fpr_sequence", "prechange_alarm_count"}.intersection(row) for row in records)
            if method == "SADA":
                window_alarms = sada_window_alarms
            else:
                with np.load(seed_dir / f"{method}_alarm_checks.npz") as alarms:
                    assert alarms["sample_alarms"].shape == (count, horizon)
                    assert alarms["window_checkpoints"].tolist() == endpoints.tolist()
                    window_alarms = np.column_stack([
                        alarms["sample_alarms"][:, end - window_size:end].any(axis=1)
                        for end in endpoints
                    ])
                    assert np.array_equal(window_alarms, alarms["window_alarms"]), (method, seed)
                checks += 3
            assert [row["alarm_windows"] for row in records] == window_alarms.sum(axis=1).tolist()
            assert all(row["windows"] == len(endpoints) for row in records)
            run = by_seed[(method, seed)]
            assert run["fpr_type"] == "window"
            assert run["alarm_windows"] == int(window_alarms.sum())
            assert run["windows"] == windows_per_seed
            assert "prechange_false_alarm_rate" not in run
            checks += 6
            values = {"fpr": int(window_alarms.sum()) / windows_per_seed,
                      "miss_rate": sum(row["missed"] for row in records) / count,
                      "mean_delay_censored": statistics.mean(row["first_alarm_after_onset"] or config["drift_suffix"] for row in records)}
            for budget in [*config["budgets"], config["drift_suffix"]]:
                values[f"r{budget}"] = sum(row["detected"] and row["first_alarm_after_onset"] <= budget for row in records) / count
            terminal_recall = f"r{config['drift_suffix']}"
            assert abs(values[terminal_recall] + values["miss_rate"] - 1) < 1e-12
            for metric, value in values.items():
                assert abs(value - by_seed[(method, seed)][metric]) < 1e-12, (method, seed, metric)
                checks += 1
    for group in summary:
        method = group["method"]
        assert group["fpr_type"] == "window"
        assert group["alarm_windows"] == sum(by_seed[(method, seed)]["alarm_windows"] for seed in config["seeds"])
        assert group["windows"] == windows_per_seed * len(config["seeds"])
        checks += 3
        for key in [key for key in group if key.endswith("_mean")]:
            metric = key[:-5]
            values = [by_seed[(method, seed)][metric] for seed in config["seeds"]]
            assert abs(statistics.mean(values) - group[key]) < 1e-12, (method, metric, "mean")
            if len(values) > 1:
                assert abs(statistics.stdev(values) - group[f"{metric}_std"]) < 1e-12, (method, metric, "sd")
            else:
                assert group[f"{metric}_std"] is None, (method, metric, "sd")
            checks += 2
    historical = json.loads((root / "reproduction_audit.json").read_text())
    assert historical["full_historical_trial_count"] == (count == config["trials"])
    if historical["full_historical_trial_count"]:
        assert historical["all_metrics_match"]
    for comparison in historical["comparisons"]:
        assert set(comparison["differences"]) == {"mean_delay_censored", "r50", "r100", "r200"}
    result = {"passed": True, "numeric_checks": checks, "methods": len(summary), "seeds": config["seeds"], "fpr_type": "window",
              "fpr_windows_per_method": windows_per_seed * len(config["seeds"]),
              "trials_per_method": count * len(config["seeds"]),
              "historical_metric_comparisons": sum(len(row["differences"]) for row in historical["comparisons"])}
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
