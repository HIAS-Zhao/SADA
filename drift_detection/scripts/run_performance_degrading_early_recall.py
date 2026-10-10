#!/usr/bin/env python3
"""Early recall on the frozen 500+500 high-loss change-point experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_name, "1")

import numpy as np
import scipy
import sklearn
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_label_aware_oracle_comparison import (  # noqa: E402
    BINARY_METHODS,
    METHOD_ORDER,
    detector_candidates,
    empirical_cdf_transform,
    load_ordered_losses,
)
from run_mahalanobis_fusion_search import prepare_stream_data  # noqa: E402
from run_performance_degrading_change import build_trial_indices, sada_detected  # noqa: E402
from src.label_aware_detectors import detects_after_onset, warm_detector  # noqa: E402

BUDGETS = (10, 20, 30, 40, 50, 100, 200)
HISTORICAL_DIR = PROJECT_ROOT / "results_detection/performance_degrading_change_20260714"
METRICS = ("fpr", *(f"r{n}" for n in BUDGETS), "r500", "miss_rate", "mean_delay_censored")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_checkpoints(horizon: int, window_size: int, stride: int) -> np.ndarray:
    if not 0 < stride <= window_size <= horizon:
        raise ValueError("Require 0 < stride <= window_size <= horizon")
    return np.arange(window_size, horizon + 1, stride, dtype=np.int64)


def first_alarm(alarms: np.ndarray, checkpoints: np.ndarray) -> int | None:
    if alarms.shape != checkpoints.shape:
        raise ValueError("Alarm and checkpoint arrays must have the same shape")
    hits = np.flatnonzero(alarms)
    return int(checkpoints[hits[0]]) if hits.size else None


def label_alarm_trace(warmed_detector: Any, values: np.ndarray) -> np.ndarray:
    """Record every native alarm, retaining state and resetting after each alarm."""
    detector = warmed_detector.clone()
    alarms = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if detector.update(float(value)):
            alarms[index] = True
            detector.reset()
    return alarms


def windows_with_alarm(sample_alarms: np.ndarray, window_size: int, endpoints: np.ndarray) -> np.ndarray:
    """Mark intervals [endpoint - window_size, endpoint) containing any alarm."""
    if sample_alarms.ndim != 1 or endpoints.ndim != 1:
        raise ValueError("Require one-dimensional alarms and endpoints")
    if window_size < 1 or np.any(endpoints < window_size) or np.any(endpoints > len(sample_alarms)):
        raise ValueError("Window endpoints must lie inside the recorded trace")
    cumulative = np.concatenate(([0], np.cumsum(sample_alarms, dtype=np.int64)))
    return cumulative[endpoints] - cumulative[endpoints - window_size] > 0


def make_trace(trial: int, delay: int | None, alarm_windows: int | None = None, windows: int | None = None) -> dict[str, Any]:
    trace = {
        "trial": trial,
        "detected": delay is not None,
        "first_alarm_after_onset": delay,
        "missed": delay is None,
    }
    if (alarm_windows is None) != (windows is None):
        raise ValueError("Window numerator and denominator must be provided together")
    if alarm_windows is not None:
        if windows <= 0 or not 0 <= alarm_windows <= windows:
            raise ValueError("Require 0 <= alarm_windows <= windows and a positive denominator")
        trace.update(alarm_windows=int(alarm_windows), windows=int(windows))
    return trace


def summarize_traces(method: str, seed: int, traces: list[dict[str, Any]], horizon: int = 500, include_fpr: bool = True) -> dict[str, Any]:
    if not traces:
        raise ValueError("Cannot summarize empty trials")
    detected = np.asarray([row["detected"] for row in traces], dtype=bool)
    delays = np.asarray([
        row["first_alarm_after_onset"] if row["detected"] else horizon for row in traces
    ], dtype=np.int64)
    for row, delay in zip(traces, delays):
        if row["missed"] == row["detected"] or (row["first_alarm_after_onset"] is None) == row["detected"]:
            raise ValueError("Inconsistent detection and missing-delay flags")
        if not 1 <= delay <= horizon:
            raise ValueError("Detection delays are one-based and bounded by the horizon")
    result = {
        "method": method,
        "seed": seed,
        "trials": len(traces),
        "mean_delay_censored": float(delays.mean()),
        "miss_rate": float(np.mean(~detected)),
    }
    if include_fpr:
        alarm_windows = sum(row["alarm_windows"] for row in traces)
        windows = sum(row["windows"] for row in traces)
        if windows <= 0 or not 0 <= alarm_windows <= windows:
            raise ValueError("Invalid window-level FPR counts")
        result.update(fpr=alarm_windows / windows, fpr_type="window",
                      alarm_windows=alarm_windows, windows=windows)
    for budget in (*BUDGETS, horizon):
        result[f"r{budget}"] = float(np.mean(detected & (delays <= budget)))
    return result


def frozen_factories(config: dict[str, Any]) -> dict[str, Any]:
    factories = {}
    for method, candidates in detector_candidates().items():
        selected = config["selected_detector_configs"][method]["selected"]["config"]
        matches = [factory for candidate, factory in candidates if candidate == selected]
        if len(matches) != 1:
            raise ValueError(f"No unique historical configuration for {method}: {selected}")
        factories[method] = matches[0]
    return factories


def label_traces(method: str, seed: int, factory: Any, signals: dict[str, np.ndarray], trials: dict[str, np.ndarray], window_size: int = 200, stride: int = 10, output_dir: Path | None = None) -> list[dict[str, Any]]:
    warmed = warm_detector(factory(), signals["calibration"])
    traces = []
    endpoints = clean_checkpoints(trials["fpr"].shape[1], window_size, stride)
    sample_alarms = np.zeros(trials["fpr"].shape, dtype=bool)
    window_alarms = np.zeros((len(trials["fpr"]), len(endpoints)), dtype=bool)
    start = time.monotonic()
    for trial, (fpr_idx, clean_idx, drift_idx) in enumerate(zip(trials["fpr"], trials["clean"], trials["drift"])):
        sample_alarms[trial] = label_alarm_trace(warmed, signals["clean"][fpr_idx])
        window_alarms[trial] = windows_with_alarm(sample_alarms[trial], window_size, endpoints)
        detected, delay_index, _ = detects_after_onset(
            warmed, signals["clean"][clean_idx], signals["drift"][drift_idx]
        )
        traces.append(make_trace(
            trial,
            int(delay_index) + 1 if detected else None,
            int(window_alarms[trial].sum()), len(endpoints),
        ))
        if (trial + 1) % 250 == 0:
            print(f"seed={seed} {method}: {trial + 1}/{len(trials['clean'])} trials, {time.monotonic() - start:.1f}s", flush=True)
    if output_dir is not None:
        np.savez_compressed(output_dir / f"{method}_alarm_checks.npz", sample_alarms=sample_alarms,
                            window_checkpoints=endpoints, window_alarms=window_alarms)
    return traces


def sada_traces(features_dir: Path, config: dict[str, Any], trials: dict[str, np.ndarray], seed: int, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = config["sada"]
    window_size = settings["window_size"]
    streams = prepare_stream_data(
        features_dir, ["A", "E_text"], settings["pca_dim"], window_size,
        config["sada_calibration_windows"], config["sada_cov_eps"], seed,
    )
    thresholds = {
        stream: float(np.quantile(data["calibration_scores"], settings["quantile"]))
        for stream, data in streams.items()
    }
    for stream, data in streams.items():
        data["detector"].save_artifacts(output_dir / "sada_calibration" / stream)
    write_json(output_dir / "sada_thresholds.json", thresholds)
    clean_points = clean_checkpoints(config["clean_prefix"], window_size, config["sada_stride"])
    drift_points = np.arange(config["sada_stride"], config["drift_suffix"] + 1, config["sada_stride"])
    old_drift_mask = np.isin(drift_points, settings["delay_checkpoints"])
    count = len(trials["clean"])
    fpr_alarms = np.zeros((count, len(clean_points)), dtype=bool)
    drift_alarms = np.zeros((count, len(drift_points)), dtype=bool)
    traces, historical = [], []
    start = time.monotonic()
    for trial, (fpr_idx, clean_idx, drift_idx) in enumerate(zip(trials["fpr"], trials["clean"], trials["drift"])):
        for column, point in enumerate(clean_points):
            fpr_alarms[trial, column] = sada_detected(
                streams, thresholds, fpr_idx[point - window_size:point], drift_idx, 0, window_size
            )
        for column, point in enumerate(drift_points):
            drift_alarms[trial, column] = sada_detected(streams, thresholds, clean_idx, drift_idx, int(point), window_size)
        traces.append(make_trace(
            trial, first_alarm(drift_alarms[trial], drift_points),
            int(fpr_alarms[trial].sum()), len(clean_points),
        ))
        historical.append(make_trace(
            trial, first_alarm(drift_alarms[trial, old_drift_mask], drift_points[old_drift_mask]),
        ))
        if (trial + 1) % 250 == 0:
            print(f"seed={seed} SADA: {trial + 1}/{count} trials, {time.monotonic() - start:.1f}s", flush=True)
    np.savez_compressed(output_dir / "sada_alarm_checks.npz", clean_checkpoints=clean_points,
                        drift_checkpoints=drift_points, fpr_alarms=fpr_alarms,
                        drift_alarms=drift_alarms)
    return traces, historical


def run_seed(seed: int, config: dict[str, Any], signals: dict[str, np.ndarray], degrading_indices: np.ndarray, features_dir: Path, output_root: Path, max_trials: int | None) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    output_dir = output_root / f"seed_{seed}"
    output_dir.mkdir()
    # Generate the complete historical draw before slicing a smoke test.
    trials = build_trial_indices(len(signals["clean"]), degrading_indices, config["clean_prefix"],
                                 config["drift_suffix"], config["trials"], seed)
    if max_trials is not None:
        trials = {key: value[:max_trials] for key, value in trials.items()}
    np.savez_compressed(output_dir / "trial_indices.npz", **trials)
    runs = []
    traces, historical = sada_traces(features_dir, config, trials, seed, output_dir)
    write_json(output_dir / "SADA_traces.json", traces)
    write_json(output_dir / "SADA_historical_traces.json", historical)
    write_json(output_dir / "SADA_historical_metrics.json", summarize_traces("SADA", seed, historical, include_fpr=False))
    runs.append(summarize_traces("SADA", seed, traces))
    write_json(output_dir / "metrics.json", runs)
    for method, factory in frozen_factories(config).items():
        method_signals = {
            key: (values >= config["degrading_cdf_threshold"]).astype(np.uint8) if method in BINARY_METHODS else values
            for key, values in signals.items()
        }
        traces = label_traces(method, seed, factory, method_signals, trials,
                              window_size=config["sada"]["window_size"], stride=config["sada_stride"], output_dir=output_dir)
        write_json(output_dir / f"{method}_traces.json", traces)
        runs.append(summarize_traces(method, seed, traces))
        write_json(output_dir / "metrics.json", runs)
        print(f"seed={seed} {method} complete: FPR={runs[-1]['fpr']:.4f}, R10={runs[-1]['r10']:.4f}, R40={runs[-1]['r40']:.4f}", flush=True)
    return runs


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for method in METHOD_ORDER:
        subset = [row for row in runs if row["method"] == method]
        row = {"method": method, "n_seeds": len(subset), "trials_per_seed": subset[0]["trials"],
               "fpr_type": "window", "alarm_windows": sum(item["alarm_windows"] for item in subset),
               "windows": sum(item["windows"] for item in subset)}
        for metric in METRICS:
            values = np.asarray([item[metric] for item in subset])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(row)
    return summary


def write_window_fpr(output_dir: Path, runs: list[dict[str, Any]], summary: list[dict[str, Any]], config: dict[str, Any]) -> None:
    window_size = config["sada"]["window_size"]
    value = {
        "description": "Window-level FPR values for the main label-aware table.",
        "source_experiment": output_dir.name,
        "fpr_type": "window",
        "seeds": config["seeds"],
        "trials_per_seed": summary[0]["trials_per_seed"],
        "normal_sequence_length": config["clean_prefix"],
        "window_size": window_size,
        "stride": config["sada_stride"],
        "window_endpoints": clean_checkpoints(config["clean_prefix"], window_size, config["sada_stride"]).tolist(),
        "sada_window_positive_rule": f"Score the {window_size}-sample window at its endpoint and compare with the original threshold.",
        "label_aware_window_positive_rule": f"At least one native detector alarm during the same {window_size}-sample interval; retain state across window boundaries and reset after alarms.",
        "aggregation": "Mean and sample standard deviation across seeds, percentages.",
        "runs": [{"seed": row["seed"], "method": row["method"], "alarm_windows": row["alarm_windows"],
                  "windows": row["windows"], "window_fpr": row["fpr"]} for row in runs],
        "summary": [],
    }
    for row in summary:
        per_seed = [100 * item["fpr"] for item in runs if item["method"] == row["method"]]
        value["summary"].append({"method": row["method"], "window_fpr_pct": 100 * row["fpr_mean"],
                                 "window_sd_pct": 100 * row["fpr_std"] if row["fpr_std"] is not None else None,
                                 "per_seed_pct": per_seed, "alarm_windows": row["alarm_windows"], "windows": row["windows"]})
    write_json(output_dir / "window_fpr_reviewed.json", value)


def audit_reproduction(output_dir: Path, historical_dir: Path, runs: list[dict[str, Any]], full_trials: bool) -> dict[str, Any]:
    previous = json.loads((historical_dir / "performance_degrading_runs.json").read_text())
    expected = {(row["method"], row["seed"]): row for row in previous}
    comparisons = []
    for row in runs:
        if row["method"] == "SADA":
            row = json.loads((output_dir / f"seed_{row['seed']}" / "SADA_historical_metrics.json").read_text())
        old = expected[(row["method"], row["seed"])]
        differences = {
            key: row[key] - old["mean_delay" if key == "mean_delay_censored" else key]
            for key in ("mean_delay_censored", "r50", "r100", "r200")
        }
        comparisons.append({"method": row["method"], "seed": row["seed"], "differences": differences})
    matches = all(abs(value) < 1e-12 for row in comparisons for value in row["differences"].values())
    return {"full_historical_trial_count": full_trials, "all_metrics_match": matches, "comparisons": comparisons}


def write_report(output_dir: Path, summary: list[dict[str, Any]], config: dict[str, Any], audit: dict[str, Any]) -> None:
    metrics = ("fpr", *(f"r{n}" for n in BUDGETS))
    window_size = config["sada"]["window_size"]
    stride = config["sada_stride"]
    endpoints = clean_checkpoints(config["clean_prefix"], window_size, stride)
    lines = [
        f"# Early Recall: Frozen {config['clean_prefix']}+{config['drift_suffix']} High-Loss Experiment", "",
        f"Seeds: {config['seeds']}. Trials per seed: {summary[0]['trials_per_seed']}. Values are percentages, mean +/- sample SD across seeds.", "",
        f"Recall is cumulative detection by the indicated number of samples after onset, counted from one. FPR is the fraction of positive {window_size}-sample clean windows, checked every {stride} samples from endpoint {endpoints[0]} through {endpoints[-1]}. A SADA window is positive when its score exceeds the frozen threshold. A label-aware window is positive when at least one native detector alarm falls inside its interval; detector state is retained across windows and reset after alarms. Misses are explicit; they do not count as detection at sample {config['drift_suffix']}.", "",
        f"SADA retains its original {window_size}-sample window, PCA, calibration, and thresholds. It checks every {stride} arrivals, starting at the first full window (sample {window_size}). After onset it checks multiples of {stride} through sample {config['drift_suffix']}; at {stride}, its window contains {window_size - stride} clean and {stride} shifted samples. Label-aware methods retain their original sample-by-sample updates, native clocks, calibration warmup, and reset rules.", "",
        "| Method | Window-level FPR | " + " | ".join(f"Recall <= {n}" for n in BUDGETS) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for row in summary:
        cells = [f"{100 * row[f'{metric}_mean']:.2f} +/- {100 * row[f'{metric}_std']:.2f}" if row[f"{metric}_std"] is not None else f"{100 * row[f'{metric}_mean']:.2f} (one seed)" for metric in metrics]
        lines.append("| " + " | ".join([row["method"], *cells]) + " |")
    lines += ["", f"Historical-cadence reproduction (censored mean delay and recall at 50/100/200): {'PASS' if audit['full_historical_trial_count'] and audit['all_metrics_match'] else 'see reproduction_audit.json' }.", "",
              "The three seeds repeat resampling from fixed cached pools; the reported SD describes this Monte Carlo variation. The selected drift pool contains high-supervised-loss examples and does not represent all types of natural drift. No detector thresholds were retuned using this experiment.", "",
              f"The historical SADA runtime is for one {window_size}-sample scoring call. A stride of {stride} requires {window_size / stride:g} such calls per {window_size} new arrivals after warmup; that historical single-call time must not be presented as total cost under this denser schedule.", ""]
    (output_dir / "early_recall_table.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, default=HISTORICAL_DIR)
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "features_pooled_e_fusion")
    parser.add_argument("--loss-dir", type=Path, default=PROJECT_ROOT / "results_detection/label_aware_oracle_20260714/supervised_loss")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-trials", type=int, help="Smoke test only: slice after generating all 1,000 historical trials")
    args = parser.parse_args()
    config_path = args.historical_dir / "performance_degrading_config.json"
    config = json.loads(config_path.read_text())
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(config["seeds"]):
        raise ValueError("Require positive workers and unique historical seeds")
    if args.max_trials is not None and not 1 <= args.max_trials <= config["trials"]:
        raise ValueError("max-trials must be within the historical trial count")
    frozen_factories(config)
    config.update(seeds=args.seeds, budgets=list(BUDGETS), sada_stride=10,
                  sada_calibration_windows=10000, sada_cov_eps=1e-6, fpr_type="window")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    losses = {
        key: load_ordered_losses(args.loss_dir, args.features_dir, split)
        for key, split in (("calibration", "1_Threshold_Cal"), ("clean", "2_Stream_Sim_NoDrift"), ("drift", "2_Stream_Sim_Drift"))
    }
    signals = {key: empirical_cdf_transform(losses["calibration"], values) for key, values in losses.items()}
    degrading_indices = np.flatnonzero(signals["drift"] >= config["degrading_cdf_threshold"])
    if len(degrading_indices) != config["degrading_pool_size"]:
        raise ValueError("The high-loss pool no longer matches the historical experiment")
    np.savez_compressed(args.output_dir / "aligned_signals.npz", **signals, degrading_indices=degrading_indices)
    input_paths = [config_path, args.historical_dir / "performance_degrading_runs.json"]
    input_paths += [args.features_dir / split / "features.pt" for split in ("0_Reference_Fit", "1_Threshold_Cal", "2_Stream_Sim_NoDrift", "2_Stream_Sim_Drift")]
    input_paths += [args.loss_dir / f"{split}.jsonl" for split in ("1_Threshold_Cal", "2_Stream_Sim_NoDrift", "2_Stream_Sim_Drift")]
    input_paths += [Path(__file__), PROJECT_ROOT / "scripts/run_performance_degrading_change.py",
                    PROJECT_ROOT / "scripts/run_label_aware_oracle_comparison.py",
                    PROJECT_ROOT / "scripts/run_mahalanobis_fusion_search.py",
                    PROJECT_ROOT / "src/label_aware_detectors.py", PROJECT_ROOT / "src/mahalanobis_detector.py"]
    config["input_sha256"] = {str(path.resolve()): sha256(path) for path in input_paths}
    config["environment"] = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__}
    config["command"] = sys.argv
    config["workers"] = args.workers
    config["max_trials"] = args.max_trials
    config["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(args.output_dir / "early_recall_config.json", config)
    print(f"Frozen experiment: {len(degrading_indices)} high-loss samples, seeds={args.seeds}, workers={args.workers}", flush=True)
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(run_seed, seed, config, signals, degrading_indices, args.features_dir, args.output_dir, args.max_trials) for seed in args.seeds]
        for future in as_completed(futures):
            runs.extend(future.result())
    runs.sort(key=lambda row: (row["seed"], METHOD_ORDER.index(row["method"])))
    summary = aggregate(runs)
    audit = audit_reproduction(args.output_dir, args.historical_dir, runs, args.max_trials in (None, config["trials"]))
    write_json(args.output_dir / "early_recall_runs.json", runs)
    write_json(args.output_dir / "early_recall_summary.json", summary)
    write_json(args.output_dir / "reproduction_audit.json", audit)
    write_window_fpr(args.output_dir, runs, summary, config)
    with (args.output_dir / "early_recall_summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_report(args.output_dir, summary, config, audit)
    if audit["full_historical_trial_count"] and not audit["all_metrics_match"]:
        raise RuntimeError("Historical reproduction differs; inspect saved audit before using results")
    write_json(args.output_dir / "completed.json", {"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": len(runs), "reproduction_passed": audit["all_metrics_match"] if audit["full_historical_trial_count"] else None})
    print(f"Complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
