#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_mahalanobis_fusion_search import build_sampling_plans  # noqa: E402
from src.label_aware_detectors import (  # noqa: E402
    ADWIN,
    DDM,
    EDDM,
    HDDMW,
    KSWIN,
    OPTWIN,
    PageHinkley,
    detects_after_onset,
    detects_in_window,
    warm_detector,
)

DRIFT_RATIOS = [0.0, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
METHOD_ORDER = ["SADA", "DDM", "EDDM", "ADWIN", "HDDM-W", "Page-Hinkley", "KSWIN", "OPTWIN"]
BINARY_METHODS = {"DDM", "EDDM", "HDDM-W"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-aware oracle drift detectors on supervised-loss events.")
    parser.add_argument(
        "--loss-dir",
        type=Path,
        default=PROJECT_ROOT / "results_detection" / "label_aware_oracle_20260714" / "supervised_loss",
    )
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "features_pooled_e_fusion")
    parser.add_argument(
        "--sada-runs",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "traditional_comparison_pf_repeats_20260524_full_r05_20"
            / "traditional_pf_full_runs.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results_detection" / "label_aware_oracle_20260714",
    )
    parser.add_argument("--loss-quantile", type=float, default=0.95)
    parser.add_argument("--target-calibration-fpr", type=float, default=0.05)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--windows-per-ratio", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--calibration-seed", type=int, default=20260714)
    parser.add_argument(
        "--stream-layout",
        choices=["shuffled", "contiguous_tail"],
        default="shuffled",
    )
    return parser.parse_args()


def load_feature_uids(features_dir: Path, split_name: str) -> list[str]:
    payload = torch.load(features_dir / split_name / "features.pt", map_location="cpu")
    return [str(uid) for uid in payload["metadata"]["uids"]]


def load_ordered_losses(loss_dir: Path, features_dir: Path, split_name: str) -> np.ndarray:
    path = loss_dir / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    successful: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = str(row["uid"])
            if "mean_nll" in row:
                successful[uid] = row
                errors.pop(uid, None)
            else:
                errors[uid] = str(row.get("error", "unknown error"))

    ordered_uids = load_feature_uids(features_dir, split_name)
    missing = [uid for uid in ordered_uids if uid not in successful]
    if missing:
        details = {uid: errors.get(uid, "missing") for uid in missing[:10]}
        raise RuntimeError(f"{split_name}: missing {len(missing)} supervised losses: {details}")
    extras = sorted(set(successful) - set(ordered_uids))
    if extras:
        raise RuntimeError(f"{split_name}: found {len(extras)} loss rows not present in feature metadata: {extras[:10]}")

    values = np.asarray([float(successful[uid]["mean_nll"]) for uid in ordered_uids], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        bad = np.flatnonzero(~np.isfinite(values))
        raise RuntimeError(f"{split_name}: non-finite losses at indices {bad[:10].tolist()}")
    return values


def empirical_cdf_transform(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=np.float64))
    ranks = np.searchsorted(sorted_reference, values, side="right")
    return ranks.astype(np.float64) / len(sorted_reference)


def detector_candidates() -> dict[str, list[tuple[dict[str, Any], Callable[[], Any]]]]:
    candidates: dict[str, list[tuple[dict[str, Any], Callable[[], Any]]]] = {
        "DDM": [],
        "EDDM": [],
        "ADWIN": [],
        "HDDM-W": [],
        "Page-Hinkley": [],
        "KSWIN": [],
        "OPTWIN": [],
    }
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        config = {"warm_start": 30, "warning_threshold": 2.0, "drift_threshold": threshold}
        candidates["DDM"].append((config, lambda cfg=config: DDM(**cfg)))
    for beta in [0.97, 0.95, 0.92, 0.9, 0.85, 0.8, 0.75]:
        config = {"warm_start": 30, "alpha": max(0.98, beta), "beta": beta}
        candidates["EDDM"].append((config, lambda cfg=config: EDDM(**cfg)))
    for delta in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        config = {
            "delta": delta,
            "clock": 8,
            "min_window_length": 10,
            "grace_period": 30,
            "max_window": 400,
            "cut_step": 5,
            "two_sided": True,
        }
        candidates["ADWIN"].append((config, lambda cfg=config: ADWIN(**cfg)))
    for confidence in [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005]:
        config = {
            "drift_confidence": confidence,
            "warning_confidence": min(max(confidence * 5.0, confidence), 0.5),
            "lambda_val": 0.05,
        }
        candidates["HDDM-W"].append((config, lambda cfg=config: HDDMW(**cfg)))
    for threshold in [2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]:
        config = {
            "min_instances": 30,
            "delta": 0.005,
            "threshold": threshold,
            "alpha": 0.9999,
            "mode": "both",
        }
        candidates["Page-Hinkley"].append((config, lambda cfg=config: PageHinkley(**cfg)))
    for alpha in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]:
        config = {"alpha": alpha, "window_size": 100, "stat_size": 30, "seed": 42, "clock": 5}
        candidates["KSWIN"].append((config, lambda cfg=config: KSWIN(**cfg)))
    for delta in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]:
        config = {
            "delta": delta,
            "rigor": 0.5,
            "max_window": 200,
            "min_subwindow": 20,
            "clock": 5,
            "cut_step": 5,
            "use_variance_test": False,
            "two_sided": True,
        }
        candidates["OPTWIN"].append((config, lambda cfg=config: OPTWIN(**cfg)))
    return candidates


def sample_stationary_windows(
    values: np.ndarray,
    count: int,
    window_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.stack(
        [
            rng.choice(
                len(values),
                size=window_size,
                replace=len(values) < window_size,
            )
            for _ in range(count)
        ],
        axis=0,
    )
    return values[indices]


def calibration_fpr(factory: Callable[[], Any], calibration: np.ndarray, windows: np.ndarray) -> float:
    warmed = warm_detector(factory(), calibration)
    detections = 0
    for window in windows:
        detected, _ = detects_in_window(warmed, window)
        detections += int(detected)
    return detections / len(windows)


def select_detector_configs(
    calibration_by_method: dict[str, np.ndarray],
    calibration_windows_by_method: dict[str, np.ndarray],
    target_fpr: float,
) -> tuple[dict[str, Callable[[], Any]], dict[str, dict[str, Any]]]:
    selected_factories: dict[str, Callable[[], Any]] = {}
    audit: dict[str, dict[str, Any]] = {}

    for method, candidates in detector_candidates().items():
        calibration = calibration_by_method[method]
        calibration_windows = calibration_windows_by_method[method]
        candidate_rows = []
        selected_index: int | None = None
        for index, (config, factory) in enumerate(candidates):
            fpr = calibration_fpr(factory, calibration, calibration_windows)
            candidate_rows.append({"config": config, "calibration_fpr": fpr})
            if selected_index is None and fpr <= target_fpr:
                selected_index = index

        if selected_index is None:
            selected_index = min(
                range(len(candidate_rows)),
                key=lambda index: (candidate_rows[index]["calibration_fpr"], index),
            )
        selected_factories[method] = candidates[selected_index][1]
        audit[method] = {
            "selected_index": selected_index,
            "selected": candidate_rows[selected_index],
            "candidates": candidate_rows,
        }
    return selected_factories, audit


def build_signal_windows(
    no_drift_signal: np.ndarray,
    drift_signal: np.ndarray,
    plans: dict[float, dict[str, Any]],
    stream_layout: str,
) -> dict[float, np.ndarray]:
    signal_windows: dict[float, np.ndarray] = {}
    for ratio, plan in plans.items():
        windows = np.empty(
            (len(plan["permutations"]), int(plan["n_clean"]) + int(plan["n_drift"])),
            dtype=np.result_type(no_drift_signal.dtype, drift_signal.dtype),
        )
        for index, permutation in enumerate(plan["permutations"]):
            parts = []
            if int(plan["n_clean"]) > 0:
                parts.append(no_drift_signal[plan["clean_batches"][index]])
            if int(plan["n_drift"]) > 0:
                parts.append(drift_signal[plan["drift_batches"][index]])
            combined = np.concatenate(parts)
            windows[index] = combined[permutation] if stream_layout == "shuffled" else combined
        signal_windows[float(ratio)] = windows
    return signal_windows


def evaluate_detector(
    method: str,
    factory: Callable[[], Any],
    calibration_events: np.ndarray,
    event_windows: dict[float, np.ndarray],
    plans: dict[float, dict[str, Any]],
    stream_layout: str,
    seed: int,
) -> dict[str, Any]:
    warmed = warm_detector(factory(), calibration_events)
    row: dict[str, Any] = {"seed": seed, "method": method}
    delays: dict[str, float | None] = {}

    for ratio in DRIFT_RATIOS:
        detections = 0
        censored_delays = []
        prechange_alarm_windows = 0
        for window in event_windows[ratio]:
            if ratio > 0.0 and stream_layout == "contiguous_tail":
                n_clean = int(plans[ratio]["n_clean"])
                detected, delay, prechange_alarms = detects_after_onset(
                    warmed,
                    window[:n_clean],
                    window[n_clean:],
                )
                prechange_alarm_windows += int(prechange_alarms > 0)
                delay_limit = int(plans[ratio]["n_drift"])
            else:
                detected, delay = detects_in_window(warmed, window)
                delay_limit = len(window)
            detections += int(detected)
            if ratio > 0.0:
                censored_delays.append((int(delay) + 1) if delay is not None else delay_limit)
        rate = detections / len(event_windows[ratio])
        if ratio == 0.0:
            row["fpr"] = rate
        else:
            key = f"r{int(round(ratio * 100)):02d}"
            row[key] = rate
            delay_key = f"d{int(round(ratio * 100)):02d}"
            row[delay_key] = float(np.mean(censored_delays))
            if stream_layout == "contiguous_tail":
                row[f"pre_fa{int(round(ratio * 100)):02d}"] = (
                    prechange_alarm_windows / len(event_windows[ratio])
                )
            delays[key] = row[delay_key]

    accuracies = [1.0 - float(row["fpr"])] + [
        float(row[f"r{int(round(ratio * 100)):02d}"])
        for ratio in DRIFT_RATIOS
        if ratio > 0.0
    ]
    row["mean_accuracy"] = float(np.mean(accuracies))
    row["mean_delay"] = float(
        np.mean(
            [
                float(row[f"d{int(round(ratio * 100)):02d}"])
                for ratio in DRIFT_RATIOS
                if ratio > 0.0
            ]
        )
    )
    row["mean_detection_delay"] = delays
    return row


def load_sada_runs(path: Path, stream_layout: str, window_size: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_rows = payload.get("SADA", payload.get("Mahalanobis"))
    if source_rows is None:
        raise ValueError(f"Missing SADA results in {path}")
    rows = copy.deepcopy(source_rows)
    for row in rows:
        row["method"] = "SADA"
        for ratio in DRIFT_RATIOS:
            if ratio > 0.0:
                delay = (
                    float(int(round(window_size * ratio)))
                    if stream_layout == "contiguous_tail"
                    else float(window_size)
                )
                row[f"d{int(round(ratio * 100)):02d}"] = delay
                if stream_layout == "contiguous_tail":
                    row[f"pre_fa{int(round(ratio * 100)):02d}"] = 0.0
        row["mean_delay"] = float(
            np.mean(
                [
                    row[f"d{int(round(ratio * 100)):02d}"]
                    for ratio in DRIFT_RATIOS
                    if ratio > 0.0
                ]
            )
        )
    return rows


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate = []
    prechange_keys = [
        f"pre_fa{int(round(ratio * 100)):02d}"
        for ratio in DRIFT_RATIOS
        if ratio > 0.0
        and any(f"pre_fa{int(round(ratio * 100)):02d}" in row for row in runs)
    ]
    metric_keys = ["fpr"] + [
        f"r{int(round(ratio * 100)):02d}" for ratio in DRIFT_RATIOS if ratio > 0.0
    ] + [
        f"d{int(round(ratio * 100)):02d}" for ratio in DRIFT_RATIOS if ratio > 0.0
    ] + prechange_keys + ["mean_accuracy", "mean_delay"]
    for method in METHOD_ORDER:
        method_runs = [row for row in runs if row["method"] == method]
        if not method_runs:
            continue
        output: dict[str, Any] = {"method": method, "n_runs": len(method_runs)}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in method_runs], dtype=np.float64)
            output[f"{key}_mean"] = float(values.mean())
            output[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(output)
    return aggregate


def format_metric(mean: float, std: float, bold: bool = False, percent: bool = True) -> str:
    scale = 100.0 if percent else 1.0
    value = f"{mean * scale:.2f}±{std * scale:.2f}"
    return f"**{value}**" if bold else value


def metric_best_methods(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    keys = ["fpr"] + [
        f"r{int(round(ratio * 100)):02d}" for ratio in DRIFT_RATIOS if ratio > 0.0
    ] + ["mean_accuracy", "mean_delay"]
    best: dict[str, set[str]] = {}
    for key in keys:
        values = {row["method"]: float(row[f"{key}_mean"]) for row in rows}
        optimum = min(values.values()) if key in {"fpr", "mean_delay"} else max(values.values())
        best[key] = {method for method, value in values.items() if math.isclose(value, optimum, abs_tol=1e-12)}
    return best


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    prechange_keys = [
        f"pre_fa{int(round(ratio * 100)):02d}"
        for ratio in DRIFT_RATIOS
        if ratio > 0.0
        and any(f"pre_fa{int(round(ratio * 100)):02d}_mean" in row for row in rows)
    ]
    keys = ["fpr"] + [
        f"r{int(round(ratio * 100)):02d}" for ratio in DRIFT_RATIOS if ratio > 0.0
    ] + [
        f"d{int(round(ratio * 100)):02d}" for ratio in DRIFT_RATIOS if ratio > 0.0
    ] + prechange_keys + ["mean_accuracy", "mean_delay"]
    fields = ["setting", "method"]
    for key in keys:
        fields.extend([f"{key}_mean", f"{key}_std"])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"setting": "5%-20%", **{field: row.get(field, "") for field in fields if field != "setting"}})


def delay_description(stream_layout: str) -> str:
    if stream_layout == "contiguous_tail":
        return (
            "Mean Delay is measured from the true drift onset to the first post-change alarm; "
            "misses are censored at the available drift-suffix length. Pre-change alarms are reset "
            "and are not counted as recall. SADA alarms at the end of its fixed window."
        )
    return (
        "Mean Delay is the number of samples from the start of a detection window to the first alarm; "
        "missed windows are censored at 200 samples. SADA is a fixed 200-sample window detector."
    )


def write_markdown(path: Path, rows: list[dict[str, Any]], stream_layout: str = "shuffled") -> None:
    ratios = [ratio for ratio in DRIFT_RATIOS if ratio > 0.0]
    headers = ["Setting", "Method", "FPR"] + [f"R@{int(round(ratio * 100))}%" for ratio in ratios] + ["Macro", "Mean Delay ↓"]
    align = ["---", "---"] + ["---:"] * (len(headers) - 2)
    best = metric_best_methods(rows)
    lines = [
        "# Label-aware Oracle Drift Detection",
        "",
        "All label-aware baselines use ground-truth answer NLL. DDM, EDDM, and HDDM-W monitor a calibration-quantile high-loss event; ADWIN, Page-Hinkley, KSWIN, and OPTWIN monitor the continuous calibration-ECDF loss score.",
        f"Stream layout: {stream_layout}. {delay_description(stream_layout)}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(align) + "|",
    ]
    for row in rows:
        cells = ["5%-20%", str(row["method"])]
        keys = ["fpr"] + [f"r{int(round(ratio * 100)):02d}" for ratio in ratios] + ["mean_accuracy", "mean_delay"]
        for key in keys:
            cells.append(
                format_metric(
                    float(row[f"{key}_mean"]),
                    float(row[f"{key}_std"]),
                    bold=str(row["method"]) in best[key],
                    percent=key != "mean_delay",
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_metric(mean: float, std: float, bold: bool, percent: bool = True) -> str:
    scale = 100.0 if percent else 1.0
    value = f"{mean * scale:.2f}$\\pm${std * scale:.2f}"
    return f"\\textbf{{{value}}}" if bold else value


def write_latex(path: Path, rows: list[dict[str, Any]], stream_layout: str = "shuffled") -> None:
    ratios = [ratio for ratio in DRIFT_RATIOS if ratio > 0.0]
    best = metric_best_methods(rows)
    headers = ["Setting", "Method", "FPR"] + [f"R@{int(round(ratio * 100))}\\%" for ratio in ratios] + ["Macro", "Mean Delay $\\downarrow$"]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Label-aware oracle drift detection results on the self-constructed benchmark under the "
        + stream_layout.replace("_", "-")
        + " stream layout. Lower FPR and mean delay are better; higher recall and macro accuracy are better. "
        + delay_description(stream_layout).replace("%", "\\%")
        + " Error-rate detectors monitor high-loss events, while continuous detectors monitor calibration-ECDF supervised loss; SADA uses only unlabeled forward features.}",
        "\\label{tab:label_aware_oracle_detection}",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * (len(headers) - 2) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    keys = ["fpr"] + [f"r{int(round(ratio * 100)):02d}" for ratio in ratios] + ["mean_accuracy", "mean_delay"]
    for row in rows:
        cells = ["5\\%-20\\%", str(row["method"])]
        for key in keys:
            cells.append(
                latex_metric(
                    float(row[f"{key}_mean"]),
                    float(row[f"{key}_std"]),
                    str(row["method"]) in best[key],
                    percent=key != "mean_delay",
                )
            )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration_losses = load_ordered_losses(args.loss_dir, args.features_dir, "1_Threshold_Cal")
    no_drift_losses = load_ordered_losses(args.loss_dir, args.features_dir, "2_Stream_Sim_NoDrift")
    drift_losses = load_ordered_losses(args.loss_dir, args.features_dir, "2_Stream_Sim_Drift")
    loss_threshold = float(np.quantile(calibration_losses, args.loss_quantile))

    calibration_percentiles = empirical_cdf_transform(calibration_losses, calibration_losses)
    no_drift_percentiles = empirical_cdf_transform(calibration_losses, no_drift_losses)
    drift_percentiles = empirical_cdf_transform(calibration_losses, drift_losses)
    calibration_events = (calibration_percentiles > args.loss_quantile).astype(np.uint8)
    no_drift_events = (no_drift_percentiles > args.loss_quantile).astype(np.uint8)
    drift_events = (drift_percentiles > args.loss_quantile).astype(np.uint8)

    calibration_by_method = {
        method: calibration_events if method in BINARY_METHODS else calibration_percentiles
        for method in METHOD_ORDER
        if method != "SADA"
    }
    calibration_windows_by_method = {
        method: sample_stationary_windows(
            values,
            count=args.calibration_windows,
            window_size=args.window_size,
            seed=args.calibration_seed,
        )
        for method, values in calibration_by_method.items()
    }
    factories, calibration_audit = select_detector_configs(
        calibration_by_method=calibration_by_method,
        calibration_windows_by_method=calibration_windows_by_method,
        target_fpr=args.target_calibration_fpr,
    )

    runs = load_sada_runs(args.sada_runs, args.stream_layout, args.window_size)
    for seed in args.seeds:
        plans = build_sampling_plans(
            no_drift_len=len(no_drift_events),
            drift_len=len(drift_events),
            drift_ratios=DRIFT_RATIOS,
            window_size=args.window_size,
            windows_per_ratio=args.windows_per_ratio,
            seed=seed,
        )
        binary_windows = build_signal_windows(
            no_drift_events,
            drift_events,
            plans,
            stream_layout=args.stream_layout,
        )
        continuous_windows = build_signal_windows(
            no_drift_percentiles,
            drift_percentiles,
            plans,
            stream_layout=args.stream_layout,
        )
        for method in tqdm(factories, desc=f"seed={seed} label-aware methods"):
            calibration_signal = calibration_by_method[method]
            signal_windows = binary_windows if method in BINARY_METHODS else continuous_windows
            runs.append(
                evaluate_detector(
                    method=method,
                    factory=factories[method],
                    calibration_events=calibration_signal,
                    event_windows=signal_windows,
                    plans=plans,
                    stream_layout=args.stream_layout,
                    seed=seed,
                )
            )

    aggregate = aggregate_runs(runs)
    config = {
        "loss_signal": "teacher_forced_mean_answer_nll",
        "loss_quantile": args.loss_quantile,
        "loss_threshold": loss_threshold,
        "calibration_high_loss_rate": float(calibration_events.mean()),
        "no_drift_high_loss_rate": float(no_drift_events.mean()),
        "drift_high_loss_rate": float(drift_events.mean()),
        "calibration_loss_percentile_mean": float(calibration_percentiles.mean()),
        "no_drift_loss_percentile_mean": float(no_drift_percentiles.mean()),
        "drift_loss_percentile_mean": float(drift_percentiles.mean()),
        "method_signals": {
            method: (
                f"binary_high_loss_event_at_q{args.loss_quantile:.2f}"
                if method in BINARY_METHODS
                else "continuous_calibration_ecdf_supervised_loss"
            )
            for method in METHOD_ORDER
            if method != "SADA"
        },
        "window_size": args.window_size,
        "stream_layout": args.stream_layout,
        "windows_per_ratio": args.windows_per_ratio,
        "drift_ratios": DRIFT_RATIOS,
        "seeds": args.seeds,
        "target_calibration_fpr": args.target_calibration_fpr,
        "delay_protocol": {
            "unit": "samples",
            "origin": (
                "true_drift_onset"
                if args.stream_layout == "contiguous_tail"
                else "start_of_detection_window"
            ),
            "miss_censoring": (
                "drift_suffix_length"
                if args.stream_layout == "contiguous_tail"
                else args.window_size
            ),
            "prechange_alarm_handling": (
                "reset_and_exclude_from_recall"
                if args.stream_layout == "contiguous_tail"
                else "not_applicable"
            ),
            "sada_fixed_window_delay": (
                "drift_suffix_length"
                if args.stream_layout == "contiguous_tail"
                else args.window_size
            ),
        },
        "selected_detector_configs": calibration_audit,
    }
    (args.output_dir / "label_aware_run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "label_aware_oracle_runs.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "label_aware_oracle_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "label_aware_oracle_summary.csv", aggregate)
    write_markdown(
        args.output_dir / "label_aware_oracle_table.md",
        aggregate,
        stream_layout=args.stream_layout,
    )
    write_latex(
        args.output_dir / "label_aware_oracle_table.tex",
        aggregate,
        stream_layout=args.stream_layout,
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
