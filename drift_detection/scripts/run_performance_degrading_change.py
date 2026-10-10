#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_label_aware_oracle_comparison import (  # noqa: E402
    BINARY_METHODS,
    METHOD_ORDER,
    empirical_cdf_transform,
    load_ordered_losses,
    sample_stationary_windows,
    select_detector_configs,
)
from run_mahalanobis_fusion_search import prepare_stream_data  # noqa: E402
from src.label_aware_detectors import detects_after_onset, warm_detector  # noqa: E402

BUDGETS = [50, 100, 200]
SADA_DELAY_CHECKPOINTS = [50, 100, 200, 300, 400, 500]
LABEL_METHODS = [method for method in METHOD_ORDER if method != "SADA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Performance-degrading 500+500 change-point experiment.")
    parser.add_argument(
        "--loss-dir",
        type=Path,
        default=PROJECT_ROOT / "results_detection" / "label_aware_oracle_20260714" / "supervised_loss",
    )
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "features_pooled_e_fusion")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results_detection" / "performance_degrading_change_20260714",
    )
    parser.add_argument("--degrading-cdf-threshold", type=float, default=0.70)
    parser.add_argument("--clean-prefix", type=int, default=500)
    parser.add_argument("--drift-suffix", type=int, default=500)
    parser.add_argument("--sada-window-size", type=int, default=200)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--target-calibration-fpr", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--calibration-seed", type=int, default=20260714)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--calibration-score-windows", type=int, default=10000)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    return parser.parse_args()


def build_trial_indices(
    no_drift_len: int,
    degrading_indices: np.ndarray,
    clean_prefix: int,
    drift_suffix: int,
    trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    clean = np.stack(
        [
            rng.choice(no_drift_len, clean_prefix, replace=no_drift_len < clean_prefix)
            for _ in range(trials)
        ]
    )
    drift = np.stack(
        [
            rng.choice(
                degrading_indices,
                drift_suffix,
                replace=len(degrading_indices) < drift_suffix,
            )
            for _ in range(trials)
        ]
    )
    fpr = np.stack(
        [
            rng.choice(no_drift_len, clean_prefix, replace=no_drift_len < clean_prefix)
            for _ in range(trials)
        ]
    )
    return {"clean": clean, "drift": drift, "fpr": fpr}


def calibrate_factories(
    calibration_percentiles: np.ndarray,
    failure_threshold: float,
    calibration_windows: int,
    horizon: int,
    seed: int,
    target_fpr: float,
) -> tuple[dict[str, Callable[[], Any]], dict[str, Any], dict[str, np.ndarray]]:
    calibration_failures = (calibration_percentiles >= failure_threshold).astype(np.uint8)
    calibration_by_method = {
        method: calibration_failures if method in BINARY_METHODS else calibration_percentiles
        for method in LABEL_METHODS
    }
    windows_by_method = {
        method: sample_stationary_windows(
            signal,
            count=calibration_windows,
            window_size=horizon,
            seed=seed,
        )
        for method, signal in calibration_by_method.items()
    }
    factories, audit = select_detector_configs(
        calibration_by_method=calibration_by_method,
        calibration_windows_by_method=windows_by_method,
        target_fpr=target_fpr,
    )
    return factories, audit, calibration_by_method


def evaluate_label_method(
    method: str,
    factory: Callable[[], Any],
    calibration_signal: np.ndarray,
    no_drift_signal: np.ndarray,
    drift_signal: np.ndarray,
    trials: dict[str, np.ndarray],
    seed: int,
    drift_suffix: int,
) -> dict[str, Any]:
    warmed = warm_detector(factory(), calibration_signal)
    delays: list[int] = []

    for clean_indices, drift_indices in zip(
        trials["clean"],
        trials["drift"],
    ):
        detected, delay_index, _ = detects_after_onset(
            warmed,
            no_drift_signal[clean_indices],
            drift_signal[drift_indices],
        )
        delays.append((int(delay_index) + 1) if detected and delay_index is not None else drift_suffix)

    row: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "mean_delay": float(np.mean(delays)),
    }
    for budget in BUDGETS:
        row[f"r{budget}"] = float(np.mean(np.asarray(delays) <= budget))
    return row


def sada_detected(
    stream_data: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
    clean_indices: np.ndarray,
    drift_indices: np.ndarray,
    checkpoint: int,
    window_size: int,
) -> bool:
    if checkpoint < 0 or checkpoint > len(drift_indices):
        raise ValueError(
            f"checkpoint={checkpoint} must be within [0, {len(drift_indices)}]"
        )
    n_drift = min(checkpoint, window_size)
    n_clean = window_size - n_drift
    drift_start = checkpoint - n_drift
    normalized_scores = []
    for stream in ["A", "E_text"]:
        parts = []
        if n_clean > 0:
            parts.append(stream_data[stream]["no_drift_projected"][clean_indices[-n_clean:]])
        if n_drift > 0:
            parts.append(
                stream_data[stream]["drift_projected"][
                    drift_indices[drift_start:checkpoint]
                ]
            )
        window = np.concatenate(parts, axis=0)
        score = stream_data[stream]["detector"].score_projected_window(window)
        normalized_scores.append(score / max(thresholds[stream], 1e-12))
    return max(normalized_scores) > 1.0


def evaluate_sada(
    features_dir: Path,
    trials: dict[str, np.ndarray],
    seed: int,
    clean_prefix: int,
    drift_suffix: int,
    window_size: int,
    pca_dim: int,
    quantile: float,
    calibration_windows: int,
    cov_eps: float,
) -> dict[str, Any]:
    stream_data = prepare_stream_data(
        features_dir=features_dir,
        stream_names=["A", "E_text"],
        pca_dim=pca_dim,
        window_size=window_size,
        calibration_windows=calibration_windows,
        cov_eps=cov_eps,
        seed=seed,
    )
    thresholds = {
        stream: float(np.quantile(stream_data[stream]["calibration_scores"], quantile))
        for stream in ["A", "E_text"]
    }

    delays: list[int] = []
    for clean_indices, drift_indices in tqdm(
        zip(trials["clean"], trials["drift"]),
        total=len(trials["clean"]),
        desc=f"seed={seed} SADA",
        leave=False,
    ):
        delay = drift_suffix
        for checkpoint in SADA_DELAY_CHECKPOINTS:
            if checkpoint > drift_suffix:
                continue
            if sada_detected(
                stream_data,
                thresholds,
                clean_indices=clean_indices,
                drift_indices=drift_indices,
                checkpoint=checkpoint,
                window_size=window_size,
            ):
                delay = checkpoint
                break
        delays.append(delay)

    row: dict[str, Any] = {
        "method": "SADA",
        "seed": seed,
        "mean_delay": float(np.mean(delays)),
    }
    for budget in BUDGETS:
        row[f"r{budget}"] = float(np.mean(np.asarray(delays) <= budget))
    return row


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    metrics = ["r50", "r100", "r200", "mean_delay"]
    for method in METHOD_ORDER:
        subset = [row for row in runs if row["method"] == method]
        row: dict[str, Any] = {"method": method, "n_runs": len(subset)}
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in subset], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(row)
    return output


def best_methods(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    best: dict[str, set[str]] = {}
    for metric in ["r50", "r100", "r200", "mean_delay"]:
        values = {row["method"]: float(row[f"{metric}_mean"]) for row in rows}
        optimum = min(values.values()) if metric == "mean_delay" else max(values.values())
        best[metric] = {
            method for method, value in values.items() if math.isclose(value, optimum, abs_tol=1e-12)
        }
    return best


def metric_cell(row: dict[str, Any], metric: str, best: dict[str, set[str]], latex: bool = False) -> str:
    scale = 1.0 if metric == "mean_delay" else 100.0
    value = f"{float(row[f'{metric}_mean']) * scale:.2f}±{float(row[f'{metric}_std']) * scale:.2f}"
    if latex:
        value = value.replace("±", "$\\pm$")
        return f"\\textbf{{{value}}}" if row["method"] in best[metric] else value
    return f"**{value}**" if row["method"] in best[metric] else value


def write_outputs(output_dir: Path, runs: list[dict[str, Any]], summary: list[dict[str, Any]], config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "performance_degrading_runs.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "performance_degrading_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "performance_degrading_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = ["method", "n_runs"] + [
        field
        for metric in ["r50", "r100", "r200", "mean_delay"]
        for field in (f"{metric}_mean", f"{metric}_std")
    ]
    with (output_dir / "performance_degrading_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in summary)

    best = best_methods(summary)
    clean_prefix = int(config["clean_prefix"])
    drift_suffix = int(config["drift_suffix"])
    cdf_threshold = float(config["degrading_cdf_threshold"])
    headers = [
        "Method",
        "R@50",
        "R@100",
        "R@200",
        "Mean Delay ↓",
    ]
    markdown = [
        "# Performance-Degrading Change-Point Experiment",
        "",
        f"Protocol: {clean_prefix} clean samples followed by a contiguous post-change "
        f"segment of {drift_suffix} samples drawn without replacement from the high-loss drift pool. "
        f"The drift pool contains samples with calibration-ECDF supervised loss >= {cdf_threshold:.2f}. "
        f"Missed detections are censored at {drift_suffix} samples.",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    metrics = ["r50", "r100", "r200", "mean_delay"]
    for row in summary:
        markdown.append(
            "| "
            + " | ".join([row["method"]] + [metric_cell(row, metric, best) for metric in metrics])
            + " |"
        )
    (output_dir / "performance_degrading_table.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Performance-degrading change-point results. Each stream contains "
        f"{clean_prefix} clean samples followed by a contiguous post-change segment of "
        f"{drift_suffix} samples drawn without replacement from the high-loss drift pool. "
        "Lower delay and higher recall are better.}",
        "\\label{tab:performance_degrading_change}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & ".join(headers).replace("%", "\\%") + " \\\\",
        "\\midrule",
    ]
    for row in summary:
        latex.append(
            " & ".join([row["method"]] + [metric_cell(row, metric, best, latex=True) for metric in metrics])
            + " \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    (output_dir / "performance_degrading_table.tex").write_text(
        "\n".join(latex) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.clean_prefix < args.sada_window_size:
        raise ValueError(
            f"clean_prefix={args.clean_prefix} must be at least "
            f"sada_window_size={args.sada_window_size}"
        )
    if args.drift_suffix < max(BUDGETS):
        raise ValueError(
            f"drift_suffix={args.drift_suffix} must be at least the largest "
            f"recall budget ({max(BUDGETS)})"
        )
    calibration_losses = load_ordered_losses(args.loss_dir, args.features_dir, "1_Threshold_Cal")
    no_drift_losses = load_ordered_losses(args.loss_dir, args.features_dir, "2_Stream_Sim_NoDrift")
    drift_losses = load_ordered_losses(args.loss_dir, args.features_dir, "2_Stream_Sim_Drift")

    calibration_percentiles = empirical_cdf_transform(calibration_losses, calibration_losses)
    no_drift_percentiles = empirical_cdf_transform(calibration_losses, no_drift_losses)
    drift_percentiles = empirical_cdf_transform(calibration_losses, drift_losses)
    degrading_indices = np.flatnonzero(drift_percentiles >= args.degrading_cdf_threshold)
    if len(degrading_indices) < args.drift_suffix:
        raise RuntimeError(
            f"High-loss drift pool has only {len(degrading_indices)} samples, "
            f"fewer than drift_suffix={args.drift_suffix}"
        )

    factories, calibration_audit, calibration_by_method = calibrate_factories(
        calibration_percentiles=calibration_percentiles,
        failure_threshold=args.degrading_cdf_threshold,
        calibration_windows=args.calibration_windows,
        horizon=args.clean_prefix,
        seed=args.calibration_seed,
        target_fpr=args.target_calibration_fpr,
    )
    no_drift_failures = (no_drift_percentiles >= args.degrading_cdf_threshold).astype(np.uint8)
    drift_failures = (drift_percentiles >= args.degrading_cdf_threshold).astype(np.uint8)
    no_drift_by_method = {
        method: no_drift_failures if method in BINARY_METHODS else no_drift_percentiles
        for method in LABEL_METHODS
    }
    drift_by_method = {
        method: drift_failures if method in BINARY_METHODS else drift_percentiles
        for method in LABEL_METHODS
    }

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        trials = build_trial_indices(
            no_drift_len=len(no_drift_losses),
            degrading_indices=degrading_indices,
            clean_prefix=args.clean_prefix,
            drift_suffix=args.drift_suffix,
            trials=args.trials,
            seed=seed,
        )
        runs.append(
            evaluate_sada(
                features_dir=args.features_dir,
                trials=trials,
                seed=seed,
                clean_prefix=args.clean_prefix,
                drift_suffix=args.drift_suffix,
                window_size=args.sada_window_size,
                pca_dim=args.pca_dim,
                quantile=args.quantile,
                calibration_windows=args.calibration_score_windows,
                cov_eps=args.cov_eps,
            )
        )
        for method in tqdm(LABEL_METHODS, desc=f"seed={seed} label-aware"):
            runs.append(
                evaluate_label_method(
                    method=method,
                    factory=factories[method],
                    calibration_signal=calibration_by_method[method],
                    no_drift_signal=no_drift_by_method[method],
                    drift_signal=drift_by_method[method],
                    trials=trials,
                    seed=seed,
                    drift_suffix=args.drift_suffix,
                )
            )

    summary = aggregate_runs(runs)
    config = {
        "protocol": (
            f"{args.clean_prefix}_clean_then_{args.drift_suffix}_sample_"
            "high_loss_postchange_segment"
        ),
        "degrading_cdf_threshold": args.degrading_cdf_threshold,
        "degrading_pool_size": int(len(degrading_indices)),
        "degrading_pool_fraction": float(len(degrading_indices) / len(drift_losses)),
        "degrading_pool_mean_nll": float(drift_losses[degrading_indices].mean()),
        "degrading_pool_mean_cdf": float(drift_percentiles[degrading_indices].mean()),
        "calibration_failure_rate": float(
            np.mean(calibration_percentiles >= args.degrading_cdf_threshold)
        ),
        "no_drift_failure_rate": float(
            np.mean(no_drift_percentiles >= args.degrading_cdf_threshold)
        ),
        "drift_pool_failure_rate": float(
            np.mean(drift_percentiles[degrading_indices] >= args.degrading_cdf_threshold)
        ),
        "clean_prefix": args.clean_prefix,
        "drift_suffix": args.drift_suffix,
        "budgets": BUDGETS,
        "trials": args.trials,
        "seeds": args.seeds,
        "selected_detector_configs": calibration_audit,
        "sada": {
            "feature_group": "A+E_text",
            "fusion_rule": "max_norm",
            "pca_dim": args.pca_dim,
            "quantile": args.quantile,
            "window_size": args.sada_window_size,
            "delay_checkpoints": SADA_DELAY_CHECKPOINTS,
        },
    }
    write_outputs(args.output_dir, runs, summary, config)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
