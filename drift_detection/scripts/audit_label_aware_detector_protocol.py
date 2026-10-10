#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_label_aware_oracle_comparison import (  # noqa: E402
    BINARY_METHODS,
    DRIFT_RATIOS,
    build_signal_windows,
    calibration_fpr,
    detector_candidates,
    empirical_cdf_transform,
    evaluate_detector,
    load_ordered_losses,
    sample_stationary_windows,
)
from run_mahalanobis_fusion_search import build_sampling_plans  # noqa: E402
from src.label_aware_detectors import KSWIN, OPTWIN, detects_after_onset, warm_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit custom label-aware detectors against official River implementations."
    )
    parser.add_argument(
        "--loss-dir",
        type=Path,
        default=PROJECT_ROOT / "results_detection" / "label_aware_oracle_20260714" / "supervised_loss",
    )
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "features_pooled_e_fusion")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results_detection" / "label_aware_audit_20260719",
    )
    parser.add_argument("--river-path", type=Path, default=Path("/tmp/label_aware_audit_river"))
    parser.add_argument("--loss-quantile", type=float, default=0.95)
    parser.add_argument("--target-calibration-fpr", type=float, default=0.05)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--windows-per-ratio", type=int, default=300)
    parser.add_argument("--extended-trials", type=int, default=300)
    parser.add_argument("--extended-clean-prefix", type=int, default=200)
    parser.add_argument("--extended-budgets", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--calibration-seed", type=int, default=20260714)
    parser.add_argument(
        "--include-current-all",
        action="store_true",
        help="Also rerun every existing custom detector instead of relying on the July 14 artifacts.",
    )
    return parser.parse_args()


class RiverAdapter:
    """Expose River detectors through the local detector protocol.

    Replaying the post-reset state avoids relying on deepcopy support for
    River's compiled ADWIN object.
    """

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._values: list[float] = []
        self._detector = factory()
        self.drift_detected = False

    def reset(self) -> None:
        self._values = []
        self._detector = self._factory()
        self.drift_detected = False

    def update(self, value: float) -> bool:
        x = float(value)
        self._detector.update(x)
        self._values.append(x)
        self.drift_detected = bool(self._detector.drift_detected)
        return self.drift_detected

    def clone(self) -> "RiverAdapter":
        clone = RiverAdapter(self._factory)
        try:
            clone._detector = copy.deepcopy(self._detector)
            clone._values = list(self._values)
            clone.drift_detected = False
            return clone
        except (TypeError, ValueError, AttributeError):
            pass
        for value in self._values:
            if clone.update(value):
                raise RuntimeError("Cannot replay a River detector state containing an unhandled alarm")
        clone.drift_detected = False
        return clone


Candidate = tuple[dict[str, Any], Callable[[], Any]]


def factory_with_kwargs(cls: type, config: dict[str, Any]) -> Callable[[], RiverAdapter]:
    return lambda: RiverAdapter(lambda: cls(**config))


def river_candidates(river_path: Path) -> dict[str, list[Candidate]]:
    if not river_path.exists():
        raise FileNotFoundError(f"River installation not found: {river_path}")
    sys.path.insert(0, str(river_path))
    from river import drift  # type: ignore
    from river.drift import binary  # type: ignore

    candidates: dict[str, list[Candidate]] = {
        "River DDM": [],
        "River EDDM": [],
        "River ADWIN": [],
        "River HDDM-W one-sided": [],
        "River HDDM-W two-sided": [],
        "River Page-Hinkley": [],
        "River KSWIN": [],
    }
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        config = {"warm_start": 30, "warning_threshold": 2.0, "drift_threshold": threshold}
        candidates["River DDM"].append((config, factory_with_kwargs(binary.DDM, config)))
    for beta in [0.97, 0.95, 0.92, 0.9, 0.85, 0.8, 0.75]:
        config = {"warm_start": 30, "alpha": max(0.98, beta), "beta": beta}
        candidates["River EDDM"].append((config, factory_with_kwargs(binary.EDDM, config)))
    for delta in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        config = {
            "delta": delta,
            "clock": 8,
            "max_buckets": 5,
            "min_window_length": 10,
            "grace_period": 30,
        }
        candidates["River ADWIN"].append((config, factory_with_kwargs(drift.ADWIN, config)))
    for confidence in [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005]:
        base_config = {
            "drift_confidence": confidence,
            "warning_confidence": min(max(confidence * 5.0, confidence), 0.5),
            "lambda_val": 0.05,
        }
        for two_sided, name in [
            (False, "River HDDM-W one-sided"),
            (True, "River HDDM-W two-sided"),
        ]:
            config = {**base_config, "two_sided_test": two_sided}
            candidates[name].append((config, factory_with_kwargs(binary.HDDM_W, config)))
    for threshold in [2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]:
        config = {
            "min_instances": 30,
            "delta": 0.005,
            "threshold": threshold,
            "alpha": 0.9999,
            "mode": "both",
        }
        candidates["River Page-Hinkley"].append(
            (config, factory_with_kwargs(drift.PageHinkley, config))
        )
    for alpha in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]:
        config = {"alpha": alpha, "window_size": 100, "stat_size": 30, "seed": 42}
        candidates["River KSWIN"].append((config, factory_with_kwargs(drift.KSWIN, config)))
    return candidates


def custom_candidates(include_all: bool) -> dict[str, list[Candidate]]:
    candidates: dict[str, list[Candidate]] = {}
    if include_all:
        candidates.update(
            {
                f"Current {method}": rows
                for method, rows in detector_candidates().items()
            }
        )
        candidates["Current KSWIN clock=1"] = []
        for alpha in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]:
            config = {"alpha": alpha, "window_size": 100, "stat_size": 30, "seed": 42, "clock": 1}
            candidates["Current KSWIN clock=1"].append(
                (config, lambda cfg=config: KSWIN(**cfg))
            )
    candidates["Current OPTWIN + variance"] = []
    for delta in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00001, 0.000001]:
        config = {
            "delta": delta,
            "rigor": 0.5,
            "max_window": 200,
            "min_subwindow": 20,
            "clock": 5,
            "cut_step": 5,
            "use_variance_test": True,
            "two_sided": True,
        }
        candidates["Current OPTWIN + variance"].append(
            (config, lambda cfg=config: OPTWIN(**cfg))
        )
    return candidates


def signal_kind(method: str) -> str:
    if any(name in method for name in BINARY_METHODS):
        return "binary"
    return "continuous"


def select_candidates(
    candidates: dict[str, list[Candidate]],
    calibration_binary: np.ndarray,
    calibration_continuous: np.ndarray,
    calibration_windows_binary: np.ndarray,
    calibration_windows_continuous: np.ndarray,
    target_fpr: float,
) -> tuple[dict[str, Callable[[], Any]], dict[str, dict[str, Any]]]:
    selected: dict[str, Callable[[], Any]] = {}
    audit: dict[str, dict[str, Any]] = {}
    for method, rows in candidates.items():
        if signal_kind(method) == "binary":
            calibration = calibration_binary
            windows = calibration_windows_binary
        else:
            calibration = calibration_continuous
            windows = calibration_windows_continuous
        candidate_rows = []
        selected_index: int | None = None
        for index, (config, factory) in enumerate(rows):
            fpr = calibration_fpr(factory, calibration, windows)
            candidate_rows.append({"config": config, "calibration_fpr": fpr})
            if selected_index is None and fpr <= target_fpr:
                selected_index = index
        if selected_index is None:
            selected_index = min(
                range(len(rows)),
                key=lambda index: (candidate_rows[index]["calibration_fpr"], index),
            )
        selected[method] = rows[selected_index][1]
        audit[method] = {
            "selected_index": selected_index,
            "selected": candidate_rows[selected_index],
            "candidates": candidate_rows,
        }
    return selected, audit


def aggregate_runs(runs: list[dict[str, Any]], method_order: list[str]) -> list[dict[str, Any]]:
    metrics = ["fpr"] + [
        f"r{int(round(ratio * 100)):02d}" for ratio in DRIFT_RATIOS if ratio > 0
    ] + ["mean_accuracy", "mean_delay"]
    output = []
    for method in method_order:
        rows = [row for row in runs if row["method"] == method]
        aggregate: dict[str, Any] = {"method": method, "n_runs": len(rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows])
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(aggregate)
    return output


def evaluate_extended_budgets(
    methods: dict[str, Callable[[], Any]],
    calibration_binary: np.ndarray,
    calibration_continuous: np.ndarray,
    no_drift_binary: np.ndarray,
    no_drift_continuous: np.ndarray,
    drift_binary: np.ndarray,
    drift_continuous: np.ndarray,
    seeds: list[int],
    clean_prefix_size: int,
    budgets: list[int],
    trials: int,
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 900_000)
        trial_indices: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for budget in budgets:
            trial_indices[budget] = [
                (
                    rng.choice(
                        len(no_drift_binary),
                        size=clean_prefix_size,
                        replace=False,
                    ),
                    rng.choice(
                        len(drift_binary),
                        size=budget,
                        replace=len(drift_binary) < budget,
                    ),
                )
                for _ in range(trials)
            ]
        for method, factory in methods.items():
            binary = signal_kind(method) == "binary"
            calibration = calibration_binary if binary else calibration_continuous
            no_drift = no_drift_binary if binary else no_drift_continuous
            drift = drift_binary if binary else drift_continuous
            warmed = warm_detector(factory(), calibration)
            for budget in budgets:
                detections = 0
                prechange_alarm_windows = 0
                delays = []
                for clean_indices, drift_indices in trial_indices[budget]:
                    clean = no_drift[clean_indices]
                    changed = drift[drift_indices]
                    detected, delay, prechange_alarms = detects_after_onset(warmed, clean, changed)
                    detections += int(detected)
                    prechange_alarm_windows += int(prechange_alarms > 0)
                    delays.append((int(delay) + 1) if delay is not None else budget)
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "budget": budget,
                        "recall": detections / trials,
                        "prechange_alarm_window_rate": prechange_alarm_windows / trials,
                        "censored_mean_delay": float(np.mean(delays)),
                    }
                )
    return rows


def aggregate_extended(rows: list[dict[str, Any]], method_order: list[str]) -> list[dict[str, Any]]:
    output = []
    budgets = sorted({int(row["budget"]) for row in rows})
    for method in method_order:
        for budget in budgets:
            selected = [
                row for row in rows
                if row["method"] == method and int(row["budget"]) == budget
            ]
            aggregate: dict[str, Any] = {"method": method, "budget": budget}
            for metric in ["recall", "prechange_alarm_window_rate", "censored_mean_delay"]:
                values = np.asarray([float(row[metric]) for row in selected])
                aggregate[f"{metric}_mean"] = float(values.mean())
                aggregate[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
            output.append(aggregate)
    return output


def write_markdown(
    path: Path,
    summary: list[dict[str, Any]],
    extended: list[dict[str, Any]],
    selected_configs: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Label-Aware Detector Audit",
        "",
        "All methods use the same supervised-loss files, calibration ECDF transform, "
        "contiguous-tail windows, seeds, and calibration FPR target.",
        "",
        "## Selected Configurations",
        "",
        "| Method | Calibration FPR | Configuration |",
        "|---|---:|---|",
    ]
    for method, row in selected_configs.items():
        selected = row["selected"]
        lines.append(
            f"| {method} | {100 * float(selected['calibration_fpr']):.2f}% | "
            f"`{json.dumps(selected['config'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## 200-Sample Contiguous Windows",
            "",
            "| Method | FPR | R@5% | R@10% | R@20% | Macro Detection Acc. |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['method']} | {100 * row['fpr_mean']:.2f}% | "
            f"{100 * row['r05_mean']:.2f}% | {100 * row['r10_mean']:.2f}% | "
            f"{100 * row['r20_mean']:.2f}% | {100 * row['mean_accuracy_mean']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Extended Natural-Shift Budgets",
            "",
            "| Method | Post Samples | Recall | Pre-change Alarm Windows | Censored Delay |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in extended:
        lines.append(
            f"| {row['method']} | {row['budget']} | "
            f"{100 * row['recall_mean']:.2f}% | "
            f"{100 * row['prechange_alarm_window_rate_mean']:.2f}% | "
            f"{row['censored_mean_delay_mean']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration_losses = load_ordered_losses(args.loss_dir, args.features_dir, "1_Threshold_Cal")
    no_drift_losses = load_ordered_losses(args.loss_dir, args.features_dir, "2_Stream_Sim_NoDrift")
    drift_losses = load_ordered_losses(args.loss_dir, args.features_dir, "2_Stream_Sim_Drift")
    calibration_continuous = empirical_cdf_transform(calibration_losses, calibration_losses)
    no_drift_continuous = empirical_cdf_transform(calibration_losses, no_drift_losses)
    drift_continuous = empirical_cdf_transform(calibration_losses, drift_losses)
    calibration_binary = (calibration_continuous > args.loss_quantile).astype(np.uint8)
    no_drift_binary = (no_drift_continuous > args.loss_quantile).astype(np.uint8)
    drift_binary = (drift_continuous > args.loss_quantile).astype(np.uint8)

    calibration_windows_binary = sample_stationary_windows(
        calibration_binary,
        count=args.calibration_windows,
        window_size=args.window_size,
        seed=args.calibration_seed,
    )
    calibration_windows_continuous = sample_stationary_windows(
        calibration_continuous,
        count=args.calibration_windows,
        window_size=args.window_size,
        seed=args.calibration_seed,
    )

    all_candidates = custom_candidates(include_all=args.include_current_all)
    all_candidates.update(river_candidates(args.river_path))
    selected, selection_audit = select_candidates(
        candidates=all_candidates,
        calibration_binary=calibration_binary,
        calibration_continuous=calibration_continuous,
        calibration_windows_binary=calibration_windows_binary,
        calibration_windows_continuous=calibration_windows_continuous,
        target_fpr=args.target_calibration_fpr,
    )
    method_order = list(selected)

    runs = []
    for seed in args.seeds:
        plans = build_sampling_plans(
            no_drift_len=len(no_drift_binary),
            drift_len=len(drift_binary),
            drift_ratios=DRIFT_RATIOS,
            window_size=args.window_size,
            windows_per_ratio=args.windows_per_ratio,
            seed=seed,
        )
        binary_windows = build_signal_windows(
            no_drift_binary,
            drift_binary,
            plans,
            stream_layout="contiguous_tail",
        )
        continuous_windows = build_signal_windows(
            no_drift_continuous,
            drift_continuous,
            plans,
            stream_layout="contiguous_tail",
        )
        for method, factory in selected.items():
            binary = signal_kind(method) == "binary"
            runs.append(
                evaluate_detector(
                    method=method,
                    factory=factory,
                    calibration_events=calibration_binary if binary else calibration_continuous,
                    event_windows=binary_windows if binary else continuous_windows,
                    plans=plans,
                    stream_layout="contiguous_tail",
                    seed=seed,
                )
            )

    summary = aggregate_runs(runs, method_order)
    extended_runs = evaluate_extended_budgets(
        methods=selected,
        calibration_binary=calibration_binary,
        calibration_continuous=calibration_continuous,
        no_drift_binary=no_drift_binary,
        no_drift_continuous=no_drift_continuous,
        drift_binary=drift_binary,
        drift_continuous=drift_continuous,
        seeds=args.seeds,
        clean_prefix_size=args.extended_clean_prefix,
        budgets=args.extended_budgets,
        trials=args.extended_trials,
    )
    extended_summary = aggregate_extended(extended_runs, method_order)

    config = {
        "loss_quantile": args.loss_quantile,
        "target_calibration_fpr": args.target_calibration_fpr,
        "calibration_windows": args.calibration_windows,
        "window_size": args.window_size,
        "windows_per_ratio": args.windows_per_ratio,
        "extended_trials": args.extended_trials,
        "extended_clean_prefix": args.extended_clean_prefix,
        "extended_budgets": args.extended_budgets,
        "seeds": args.seeds,
        "river_path": str(args.river_path),
        "include_current_all": args.include_current_all,
        "selected_configs": selection_audit,
    }
    (args.output_dir / "audit_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "audit_runs.json").write_text(
        json.dumps(runs, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "extended_runs.json").write_text(
        json.dumps(extended_runs, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "extended_summary.json").write_text(
        json.dumps(extended_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / "audit_report.md",
        summary=summary,
        extended=extended_summary,
        selected_configs=selection_audit,
    )
    print(f"Audit results written to {args.output_dir}")


if __name__ == "__main__":
    main()
