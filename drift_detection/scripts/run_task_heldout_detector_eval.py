#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import numpy as np
import torch


PROJECT_ROOT = Path("@@WORKSPACE@@/drift_detection_exp")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mahalanobis_detector import MahalanobisDetector


DEFAULT_FEATURES = PROJECT_ROOT / "features_pooled_e_fusion"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results_detection" / "task_heldout_detector_20260726"
)
SPLITS = (
    "0_Reference_Fit",
    "1_Threshold_Cal",
    "2_Stream_Sim_NoDrift",
    "2_Stream_Sim_Drift",
)
STREAMS = ("A", "E_text")
FEATURE_GROUPS = ("A+E_text", "E_text", "A")
PCA_DIMS = (64, 128, 192, 256)
WINDOW_SIZES = (100, 150, 200, 300)
QUANTILES = (0.95, 0.97, 0.99, 0.995)
DRIFT_RATIOS = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)
SELECTION_SEEDS = (142, 143, 144)
TEST_SEEDS = (42, 43, 44)
MAIN_CONFIG = {
    "feature_group": "A+E_text",
    "pca_dim": 128,
    "window_size": 200,
    "quantile": 0.99,
}


def parse_csv_numbers(text: str, caster: Any) -> list[Any]:
    return [caster(part.strip()) for part in text.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_feature_payloads(
    features_dir: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    features: dict[str, dict[str, np.ndarray]] = {}
    tasks: dict[str, np.ndarray] = {}
    for split in SPLITS:
        payload = torch.load(
            features_dir / split / "features.pt",
            map_location="cpu",
            weights_only=False,
        )
        features[split] = {
            stream: payload["features"][stream].float().numpy()
            for stream in STREAMS
        }
        tasks[split] = np.asarray(
            [int(value) for value in payload["metadata"]["task_ids"]],
            dtype=np.int64,
        )
    return features, tasks


def stratified_no_drift_split(
    task_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dev: list[int] = []
    test: list[int] = []
    for task in sorted(np.unique(task_ids)):
        indices = np.flatnonzero(task_ids == task)
        shuffled = indices[rng.permutation(len(indices))]
        cut = len(shuffled) // 2
        dev.extend(int(value) for value in shuffled[:cut])
        test.extend(int(value) for value in shuffled[cut:])
    return np.asarray(sorted(dev), dtype=np.int64), np.asarray(sorted(test), dtype=np.int64)


def window_scores(
    *,
    no_drift: np.ndarray,
    drift: np.ndarray,
    mu: np.ndarray,
    sigma_inv: np.ndarray,
    window_size: int,
    ratios: tuple[float, ...],
    windows_per_ratio: int,
    seed: int,
    chunk_size: int = 100,
) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[float, np.ndarray] = {}
    dim = no_drift.shape[1]
    for ratio in ratios:
        n_drift = int(round(window_size * ratio))
        n_clean = window_size - n_drift
        scores = np.empty(windows_per_ratio, dtype=np.float64)
        for start in range(0, windows_per_ratio, chunk_size):
            count = min(chunk_size, windows_per_ratio - start)
            means = np.zeros((count, dim), dtype=np.float64)
            if n_clean:
                clean_means = np.empty((count, dim), dtype=np.float64)
                replace_clean = len(no_drift) < n_clean
                for index in range(count):
                    chosen = rng.choice(
                        len(no_drift),
                        size=n_clean,
                        replace=replace_clean,
                    )
                    clean_means[index] = no_drift[chosen].mean(axis=0)
                means += (n_clean / window_size) * clean_means
            if n_drift:
                drift_means = np.empty((count, dim), dtype=np.float64)
                replace_drift = len(drift) < n_drift
                for index in range(count):
                    chosen = rng.choice(
                        len(drift),
                        size=n_drift,
                        replace=replace_drift,
                    )
                    drift_means[index] = drift[chosen].mean(axis=0)
                means += (n_drift / window_size) * drift_means
            diff = means - mu
            squared = np.einsum(
                "bi,ij,bj->b",
                diff,
                sigma_inv,
                diff,
                optimize=True,
            )
            scores[start : start + count] = np.sqrt(np.maximum(squared, 0.0))
        out[float(ratio)] = scores
    return out


def calibration_scores(
    *,
    calibration: np.ndarray,
    mu: np.ndarray,
    sigma_inv: np.ndarray,
    window_size: int,
    windows: int,
    seed: int,
) -> np.ndarray:
    scores = window_scores(
        no_drift=calibration,
        drift=calibration,
        mu=mu,
        sigma_inv=sigma_inv,
        window_size=window_size,
        ratios=(0.0,),
        windows_per_ratio=windows,
        seed=seed,
    )
    return scores[0.0]


def group_detection(
    raw_by_stream: dict[str, dict[float, np.ndarray]],
    thresholds: dict[str, float],
    *,
    feature_group: str,
) -> dict[float, float]:
    streams = feature_group.split("+")
    result: dict[float, float] = {}
    for ratio in (0.0, *DRIFT_RATIOS):
        normalized = np.stack(
            [
                raw_by_stream[stream][ratio]
                / max(float(thresholds[stream]), 1e-12)
                for stream in streams
            ],
            axis=0,
        )
        fused = normalized[0] if len(streams) == 1 else np.max(normalized, axis=0)
        result[float(ratio)] = float(np.mean(fused > 1.0))
    return result


def metric_row(
    *,
    rates: dict[float, float],
) -> dict[str, float]:
    fpr = rates[0.0]
    recalls = [rates[ratio] for ratio in DRIFT_RATIOS]
    macro = ((1.0 - fpr) + sum(recalls)) / (1 + len(recalls))
    row: dict[str, float] = {
        "fpr": fpr,
        "macro": macro,
    }
    for ratio in DRIFT_RATIOS:
        row[f"recall_{int(round(ratio * 100)):02d}"] = rates[ratio]
    return row


def evaluate_phase(
    *,
    no_drift_by_stream: dict[str, np.ndarray],
    drift_by_stream: dict[str, np.ndarray],
    detector_by_stream: dict[str, MahalanobisDetector],
    thresholds_by_quantile: dict[float, dict[str, float]],
    window_size: int,
    seeds: tuple[int, ...],
    windows_per_ratio: int,
    heldout_task: int,
    phase: str,
) -> list[dict[str, Any]]:
    per_seed_raw: dict[int, dict[str, dict[float, np.ndarray]]] = {}
    ratios = (0.0, *DRIFT_RATIOS)
    for seed in seeds:
        per_seed_raw[seed] = {}
        for stream in STREAMS:
            detector = detector_by_stream[stream]
            per_seed_raw[seed][stream] = window_scores(
                no_drift=no_drift_by_stream[stream],
                drift=drift_by_stream[stream],
                mu=np.asarray(detector.mu_base, dtype=np.float64),
                sigma_inv=np.asarray(detector.sigma_inv, dtype=np.float64),
                window_size=window_size,
                ratios=ratios,
                windows_per_ratio=windows_per_ratio,
                seed=seed + heldout_task * 10000 + (0 if phase == "dev" else 500000),
            )

    out: list[dict[str, Any]] = []
    for quantile, thresholds in thresholds_by_quantile.items():
        for feature_group in FEATURE_GROUPS:
            seed_metrics = []
            for seed in seeds:
                rates = group_detection(
                    per_seed_raw[seed],
                    thresholds,
                    feature_group=feature_group,
                )
                seed_metrics.append(metric_row(rates=rates))
            row: dict[str, Any] = {
                "heldout_task": heldout_task,
                "phase": phase,
                "feature_group": feature_group,
                "window_size": window_size,
                "quantile": quantile,
                "seeds": len(seeds),
            }
            fields = ["fpr", "macro"] + [
                f"recall_{int(round(ratio * 100)):02d}"
                for ratio in DRIFT_RATIOS
            ]
            for field in fields:
                values = [float(item[field]) for item in seed_metrics]
                row[f"{field}_mean"] = statistics.mean(values)
                row[f"{field}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            out.append(row)
    return out


def run_pca_job(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pca_dim = int(payload["pca_dim"])
    features_dir = Path(payload["features_dir"])
    features, tasks = load_feature_payloads(features_dir)
    no_dev_idx, no_test_idx = stratified_no_drift_split(
        tasks["2_Stream_Sim_NoDrift"],
        seed=int(payload["split_seed"]),
    )
    heldout_tasks = list(payload["heldout_tasks"])
    calibration_windows = int(payload["calibration_windows"])
    selection_windows = int(payload["selection_windows"])
    test_windows = int(payload["test_windows"])
    window_sizes = tuple(int(value) for value in payload["window_sizes"])
    quantiles = tuple(float(value) for value in payload["quantiles"])
    cov_eps = float(payload["cov_eps"])

    detectors: dict[str, MahalanobisDetector] = {}
    projected: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for stream in STREAMS:
        detector = MahalanobisDetector(
            pca_dim=pca_dim,
            window_size=200,
            calibration_windows=calibration_windows,
            calibration_quantile=0.99,
            cov_eps=cov_eps,
            random_seed=int(payload["pca_seed"]),
        )
        detector.fit_reference(
            features["0_Reference_Fit"][stream].astype(np.float64, copy=False)
        )
        detectors[stream] = detector
        for split in (
            "1_Threshold_Cal",
            "2_Stream_Sim_NoDrift",
            "2_Stream_Sim_Drift",
        ):
            projected[stream][split] = detector.transform(
                features[split][stream].astype(np.float64, copy=False)
            )

    all_rows: list[dict[str, Any]] = []
    for window_size in window_sizes:
        thresholds_by_quantile: dict[float, dict[str, float]] = {
            quantile: {} for quantile in quantiles
        }
        for stream in STREAMS:
            detector = detectors[stream]
            scores = calibration_scores(
                calibration=projected[stream]["1_Threshold_Cal"],
                mu=np.asarray(detector.mu_base, dtype=np.float64),
                sigma_inv=np.asarray(detector.sigma_inv, dtype=np.float64),
                window_size=window_size,
                windows=calibration_windows,
                seed=(
                    int(payload["calibration_seed"])
                    + pca_dim * 1000
                    + window_size * 10
                ),
            )
            for quantile in quantiles:
                thresholds_by_quantile[quantile][stream] = float(
                    np.quantile(scores, quantile)
                )

        for heldout_task in heldout_tasks:
            train_drift_idx = np.flatnonzero(
                tasks["2_Stream_Sim_Drift"] != heldout_task
            )
            test_drift_idx = np.flatnonzero(
                tasks["2_Stream_Sim_Drift"] == heldout_task
            )
            dev_rows = evaluate_phase(
                no_drift_by_stream={
                    stream: projected[stream]["2_Stream_Sim_NoDrift"][no_dev_idx]
                    for stream in STREAMS
                },
                drift_by_stream={
                    stream: projected[stream]["2_Stream_Sim_Drift"][
                        train_drift_idx
                    ]
                    for stream in STREAMS
                },
                detector_by_stream=detectors,
                thresholds_by_quantile=thresholds_by_quantile,
                window_size=window_size,
                seeds=tuple(payload["selection_seeds"]),
                windows_per_ratio=selection_windows,
                heldout_task=heldout_task,
                phase="dev",
            )
            test_rows = evaluate_phase(
                no_drift_by_stream={
                    stream: projected[stream]["2_Stream_Sim_NoDrift"][no_test_idx]
                    for stream in STREAMS
                },
                drift_by_stream={
                    stream: projected[stream]["2_Stream_Sim_Drift"][
                        test_drift_idx
                    ]
                    for stream in STREAMS
                },
                detector_by_stream=detectors,
                thresholds_by_quantile=thresholds_by_quantile,
                window_size=window_size,
                seeds=tuple(payload["test_seeds"]),
                windows_per_ratio=test_windows,
                heldout_task=heldout_task,
                phase="test",
            )
            test_by_key = {
                (
                    row["heldout_task"],
                    row["feature_group"],
                    row["window_size"],
                    row["quantile"],
                ): row
                for row in test_rows
            }
            for dev in dev_rows:
                test = test_by_key[
                    (
                        dev["heldout_task"],
                        dev["feature_group"],
                        dev["window_size"],
                        dev["quantile"],
                    )
                ]
                merged = {
                    "heldout_task": heldout_task,
                    "feature_group": dev["feature_group"],
                    "pca_dim": pca_dim,
                    "window_size": window_size,
                    "quantile": dev["quantile"],
                }
                for key, value in dev.items():
                    if key.endswith("_mean") or key.endswith("_std"):
                        merged[f"dev_{key}"] = value
                for key, value in test.items():
                    if key.endswith("_mean") or key.endswith("_std"):
                        merged[f"test_{key}"] = value
                all_rows.append(merged)
        print(
            f"[pca {pca_dim}] completed window={window_size}",
            flush=True,
        )
    return all_rows


def config_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["feature_group"]),
        int(row["pca_dim"]),
        int(row["window_size"]),
        round(float(row["quantile"]), 6),
    )


def select_nested(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["heldout_task"])].append(row)
    selected: list[dict[str, Any]] = []
    fixed: list[dict[str, Any]] = []
    for heldout_task, fold_rows in sorted(by_fold.items()):
        best = max(
            fold_rows,
            key=lambda row: (
                float(row["dev_macro_mean"]),
                -float(row["dev_fpr_mean"]),
                -len(str(row["feature_group"]).split("+")),
                -int(row["pca_dim"]),
                -int(row["window_size"]),
                -float(row["quantile"]),
            ),
        )
        selected.append(dict(best))
        matches = [
            row
            for row in fold_rows
            if config_key(row)
            == (
                MAIN_CONFIG["feature_group"],
                MAIN_CONFIG["pca_dim"],
                MAIN_CONFIG["window_size"],
                MAIN_CONFIG["quantile"],
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one fixed main row for heldout task {heldout_task}"
            )
        fixed.append(dict(matches[0]))
    return selected, fixed


def aggregate_fold_rows(
    rows: list[dict[str, Any]],
    *,
    protocol: str,
) -> dict[str, Any]:
    fields = ["fpr", "macro"] + [
        f"recall_{int(round(ratio * 100)):02d}" for ratio in DRIFT_RATIOS
    ]
    out: dict[str, Any] = {
        "protocol": protocol,
        "folds": len(rows),
    }
    for field in fields:
        values = [float(row[f"test_{field}_mean"]) for row in rows]
        out[f"{field}_mean"] = statistics.mean(values)
        out[f"{field}_std_across_tasks"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
        out[f"{field}_min_task"] = min(values)
        out[f"{field}_max_task"] = max(values)
    return out


def write_report(
    path: Path,
    *,
    aggregate: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
) -> None:
    lines = [
        "# Task-Held-Out Detector Evaluation",
        "",
        "- Seven folds hold out one shifted task at a time.",
        "- Hyperparameter selection uses only the other six shifted tasks and the no-drift development half.",
        "- Final FPR uses the disjoint no-drift test half; held-out task windows are never used for selection.",
        "",
        "| Protocol | FPR | R@5 | R@10 | R@20 | Macro | Worst-task macro |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['protocol']} | {float(row['fpr_mean']):.2%} | "
            f"{float(row['recall_05_mean']):.2%} | "
            f"{float(row['recall_10_mean']):.2%} | "
            f"{float(row['recall_20_mean']):.2%} | "
            f"{float(row['macro_mean']):.2%} | "
            f"{float(row['macro_min_task']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Nested Selection by Held-Out Task",
            "",
            "| Task | Selected config | Dev macro | Test macro | FPR | R@5 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        lines.append(
            f"| {row['heldout_task']} | {row['feature_group']}, "
            f"PCA {row['pca_dim']}, W {row['window_size']}, "
            f"q={float(row['quantile']):g} | "
            f"{float(row['dev_macro_mean']):.2%} | "
            f"{float(row['test_macro_mean']):.2%} | "
            f"{float(row['test_fpr_mean']):.2%} | "
            f"{float(row['test_recall_05_mean']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Fixed Main Configuration by Held-Out Task",
            "",
            "| Task | Test macro | FPR | R@5 | R@10 | R@20 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fixed:
        lines.append(
            f"| {row['heldout_task']} | {float(row['test_macro_mean']):.2%} | "
            f"{float(row['test_fpr_mean']):.2%} | "
            f"{float(row['test_recall_05_mean']):.2%} | "
            f"{float(row['test_recall_10_mean']):.2%} | "
            f"{float(row['test_recall_20_mean']):.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested leave-one-shift-task-out detector evaluation."
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pca-dims", default="64,128,192,256")
    parser.add_argument("--window-sizes", default="100,150,200,300")
    parser.add_argument("--quantiles", default="0.95,0.97,0.99,0.995")
    parser.add_argument("--heldout-tasks", default="1,2,3,4,5,6,7")
    parser.add_argument("--selection-windows", type=int, default=1000)
    parser.add_argument("--test-windows", type=int, default=1000)
    parser.add_argument("--calibration-windows", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    parser.add_argument("--split-seed", type=int, default=20260726)
    parser.add_argument("--pca-seed", type=int, default=20260726)
    parser.add_argument("--calibration-seed", type=int, default=20260726)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty root: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_dims = parse_csv_numbers(args.pca_dims, int)
    window_sizes = parse_csv_numbers(args.window_sizes, int)
    quantiles = parse_csv_numbers(args.quantiles, float)
    heldout_tasks = parse_csv_numbers(args.heldout_tasks, int)
    payloads = [
        {
            "pca_dim": pca_dim,
            "features_dir": str(args.features_dir.resolve()),
            "window_sizes": window_sizes,
            "quantiles": quantiles,
            "heldout_tasks": heldout_tasks,
            "selection_seeds": SELECTION_SEEDS,
            "test_seeds": TEST_SEEDS,
            "selection_windows": args.selection_windows,
            "test_windows": args.test_windows,
            "calibration_windows": args.calibration_windows,
            "cov_eps": args.cov_eps,
            "split_seed": args.split_seed,
            "pca_seed": args.pca_seed,
            "calibration_seed": args.calibration_seed,
        }
        for pca_dim in pca_dims
    ]

    candidate_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_by_pca = {
            executor.submit(run_pca_job, payload): payload["pca_dim"]
            for payload in payloads
        }
        for future in as_completed(future_by_pca):
            pca_dim = future_by_pca[future]
            rows = future.result()
            candidate_rows.extend(rows)
            print(f"[done] PCA {pca_dim}: {len(rows)} candidate folds", flush=True)

    candidate_rows.sort(
        key=lambda row: (
            int(row["heldout_task"]),
            int(row["pca_dim"]),
            int(row["window_size"]),
            float(row["quantile"]),
            str(row["feature_group"]),
        )
    )
    expected = (
        len(heldout_tasks)
        * len(pca_dims)
        * len(window_sizes)
        * len(quantiles)
        * len(FEATURE_GROUPS)
    )
    if len(candidate_rows) != expected:
        raise RuntimeError(
            f"Expected {expected} candidate-fold rows, found {len(candidate_rows)}"
        )
    selected, fixed = select_nested(candidate_rows)
    aggregate = [
        aggregate_fold_rows(selected, protocol="Nested task-held-out selection"),
        aggregate_fold_rows(fixed, protocol="Fixed main configuration"),
    ]
    selected_counts = Counter(config_key(row) for row in selected)
    config_rows = [
        {
            "feature_group": key[0],
            "pca_dim": key[1],
            "window_size": key[2],
            "quantile": key[3],
            "selected_folds": count,
        }
        for key, count in sorted(
            selected_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    write_csv(output_dir / "candidate_fold_scores.csv", candidate_rows)
    write_csv(output_dir / "nested_selected_fold_results.csv", selected)
    write_csv(output_dir / "fixed_main_fold_results.csv", fixed)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    write_csv(output_dir / "selected_config_counts.csv", config_rows)
    write_json(output_dir / "aggregate_summary.json", aggregate)
    write_json(
        output_dir / "task_heldout_protocol.json",
        {
            "features_dir": str(args.features_dir.resolve()),
            "output_dir": str(output_dir),
            "shift_tasks": heldout_tasks,
            "stable_tasks": [8, 9, 10, 11],
            "no_drift_split": (
                "task-stratified deterministic 50/50 development/test split"
            ),
            "selection_drift": (
                "all shifted tasks except the held-out task for that fold"
            ),
            "test_drift": "held-out shifted task only",
            "selection_seeds": SELECTION_SEEDS,
            "test_seeds": TEST_SEEDS,
            "pca_dims": pca_dims,
            "window_sizes": window_sizes,
            "quantiles": quantiles,
            "feature_groups": FEATURE_GROUPS,
            "fusion": (
                "single stream for A or E_text; max normalized score for A+E_text"
            ),
            "calibration_windows": args.calibration_windows,
            "selection_windows_per_ratio": args.selection_windows,
            "test_windows_per_ratio": args.test_windows,
            "drift_ratios": DRIFT_RATIOS,
            "selection_rule": (
                "highest development macro, then lower development FPR, "
                "then deterministic simplicity tie-break"
            ),
            "fixed_main_config": MAIN_CONFIG,
            "test_access_during_selection": False,
        },
    )
    write_report(
        output_dir / "task_heldout_report.md",
        aggregate=aggregate,
        selected=selected,
        fixed=fixed,
    )
    print(f"[ok] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
