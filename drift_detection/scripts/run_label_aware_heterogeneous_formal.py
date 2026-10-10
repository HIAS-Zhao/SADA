#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compute_label_aware_supervised_loss import ordered_split_rows  # noqa: E402
from run_label_aware_closed_set_formal import (  # noqa: E402
    BINARY_METHODS,
    CONTINUOUS_METHODS,
    SadaScorer,
    detector_candidates,
    evaluate_sada_plans,
    evaluate_sequential_plans,
    load_reference_features,
    select_methods,
    stable_validation_mask,
    summarize_rows,
    write_json,
)
from src.label_aware_formal import (  # noqa: E402
    answer_format,
    build_window_plans,
    empirical_cdf,
    format_aware_ecdf,
)


RATIOS = [0.0, 0.05, 0.10, 0.20]
BINARY_NATIVE_METHODS = ["DDM", "EDDM", "HDDM-W"]
CONTINUOUS_NATIVE_METHODS = ["ADWIN", "Page-Hinkley", "KSWIN", "OPTWIN"]
VARIANT_ORDER = [
    "mcq",
    "mcq_yesno",
    "open_short",
    "caption",
    "long_report",
]
VARIANTS = {
    "mcq": {
        "calibration_formats": {"mcq"},
        "no_drift_formats": {"mcq"},
        "drift_formats": {"mcq"},
    },
    "mcq_yesno": {
        "calibration_formats": {"mcq"},
        "no_drift_formats": {"mcq"},
        "drift_formats": {"mcq", "yes_no"},
    },
    "open_short": {
        "calibration_formats": {"mcq", "open_short"},
        "no_drift_formats": {"mcq", "open_short"},
        "drift_formats": {"mcq", "yes_no"},
    },
    "caption": {
        "calibration_formats": {"mcq", "open_short", "caption"},
        "no_drift_formats": {"mcq", "open_short", "caption"},
        "drift_formats": {"mcq", "yes_no"},
    },
    "long_report": {
        "calibration_formats": {"mcq", "open_short", "caption"},
        "no_drift_formats": {"mcq", "open_short", "caption"},
        "drift_formats": {"mcq", "yes_no", "long_report"},
    },
}


@dataclass
class HeterogeneousPool:
    split_name: str
    uids: np.ndarray
    task_ids: np.ndarray
    formats: np.ndarray
    nll: np.ndarray
    features: dict[str, np.ndarray]

    def subset(self, mask: np.ndarray) -> "HeterogeneousPool":
        return HeterogeneousPool(
            split_name=self.split_name,
            uids=self.uids[mask],
            task_ids=self.task_ids[mask],
            formats=self.formats[mask],
            nll=self.nll[mask],
            features={
                key: values[mask]
                for key, values in self.features.items()
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal heterogeneous-task label-aware drift experiments."
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "dataset")
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "features_pooled_e_fusion")
    parser.add_argument(
        "--drift-data-json",
        type=Path,
        default=(
            PROJECT_ROOT.parent
            / "dataset_repair_reports"
            / "task01_agromind_mcq_text_merge_20260630_195249"
            / "backup"
            / "2_Stream_Sim_Drift"
            / "data.json"
        ),
    )
    parser.add_argument(
        "--loss-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "supervised_loss_uniform_px1003520"
        ),
    )
    parser.add_argument("--river-path", type=Path, default=Path("/tmp/label_aware_audit_river"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "heterogeneous"
        ),
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--windows-per-setting", type=int, default=1000)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--sada-calibration-windows", type=int, default=10000)
    parser.add_argument("--validation-windows", type=int, default=200)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--event-quantile", type=float, default=0.95)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--calibration-seed", type=int, default=20260719)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--sada-quantile", type=float, default=0.99)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=VARIANT_ORDER,
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        choices=["natural", "task_balanced"],
        default=["natural", "task_balanced"],
    )
    return parser.parse_args()


def read_loss_map(path: Path) -> dict[str, float]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "mean_nll" in row:
            output[str(row["uid"])] = float(row["mean_nll"])
    return output


def load_pool(
    split_name: str,
    dataset_root: Path,
    features_dir: Path,
    drift_data_json: Path,
    loss_dir: Path,
) -> HeterogeneousPool:
    rows = ordered_split_rows(
        dataset_root=dataset_root,
        features_dir=features_dir,
        drift_data_json=drift_data_json,
        split_name=split_name,
    )
    losses = read_loss_map(loss_dir / f"{split_name}.jsonl")
    payload = torch.load(
        features_dir / split_name / "features.pt",
        map_location="cpu",
    )
    uids = np.asarray([str(row["uid"]) for row in rows], dtype=object)
    missing = [uid for uid in uids if uid not in losses]
    if missing:
        raise RuntimeError(f"{split_name}: missing {len(missing)} NLL rows: {missing[:5]}")
    return HeterogeneousPool(
        split_name=split_name,
        uids=uids,
        task_ids=np.asarray([int(row["task_id"]) for row in rows], dtype=np.int64),
        formats=np.asarray([answer_format(row) for row in rows], dtype=object),
        nll=np.asarray([losses[str(uid)] for uid in uids], dtype=np.float64),
        features={
            stream: payload["features"][stream].float().numpy().astype(np.float64)
            for stream in ["A", "E_text"]
        },
    )


def subset_variant(
    calibration: HeterogeneousPool,
    no_drift: HeterogeneousPool,
    drift: HeterogeneousPool,
    variant_name: str,
) -> tuple[HeterogeneousPool, HeterogeneousPool, HeterogeneousPool]:
    variant = VARIANTS[variant_name]
    return (
        calibration.subset(
            np.isin(calibration.formats, sorted(variant["calibration_formats"]))
        ),
        no_drift.subset(
            np.isin(no_drift.formats, sorted(variant["no_drift_formats"]))
        ),
        drift.subset(
            np.isin(drift.formats, sorted(variant["drift_formats"]))
        ),
    )


def signal_bundle(
    calibration: HeterogeneousPool,
    no_drift: HeterogeneousPool,
    drift_validation: HeterogeneousPool,
    drift_test: HeterogeneousPool,
    signal_name: str,
    event_quantile: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if signal_name == "global_nll":
        calibration_scores = empirical_cdf(calibration.nll, calibration.nll)
        no_drift_scores = empirical_cdf(calibration.nll, no_drift.nll)
        validation_scores = empirical_cdf(calibration.nll, drift_validation.nll)
        test_scores = empirical_cdf(calibration.nll, drift_test.nll)
        fallback = {}
    elif signal_name == "format_aware_nll":
        calibration_scores, calibration_missing = format_aware_ecdf(
            calibration.nll,
            calibration.formats,
            calibration.nll,
            calibration.formats,
        )
        no_drift_scores, no_drift_missing = format_aware_ecdf(
            calibration.nll,
            calibration.formats,
            no_drift.nll,
            no_drift.formats,
        )
        validation_scores, validation_missing = format_aware_ecdf(
            calibration.nll,
            calibration.formats,
            drift_validation.nll,
            drift_validation.formats,
        )
        test_scores, test_missing = format_aware_ecdf(
            calibration.nll,
            calibration.formats,
            drift_test.nll,
            drift_test.formats,
        )
        fallback = {
            "calibration": calibration_missing,
            "no_drift": no_drift_missing,
            "drift_validation": validation_missing,
            "drift_test": test_missing,
        }
    else:
        raise ValueError(signal_name)
    return (
        {
            "calibration_continuous": calibration_scores,
            "no_drift_continuous": no_drift_scores,
            "drift_validation_continuous": validation_scores,
            "drift_test_continuous": test_scores,
            "calibration_binary": (calibration_scores > event_quantile).astype(np.float64),
            "no_drift_binary": (no_drift_scores > event_quantile).astype(np.float64),
            "drift_validation_binary": (validation_scores > event_quantile).astype(np.float64),
            "drift_test_binary": (test_scores > event_quantile).astype(np.float64),
        },
        {
            "fallback_counts": fallback,
            "mean_scores": {
                "calibration": float(calibration_scores.mean()),
                "no_drift": float(no_drift_scores.mean()),
                "drift_validation": float(validation_scores.mean()),
                "drift_test": float(test_scores.mean()),
            },
            "high_score_rates": {
                "calibration": float((calibration_scores > event_quantile).mean()),
                "no_drift": float((no_drift_scores > event_quantile).mean()),
                "drift_validation": float((validation_scores > event_quantile).mean()),
                "drift_test": float((test_scores > event_quantile).mean()),
            },
        },
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_all = load_pool(
        "1_Threshold_Cal",
        args.dataset_root,
        args.features_dir,
        args.drift_data_json,
        args.loss_dir,
    )
    no_drift_all = load_pool(
        "2_Stream_Sim_NoDrift",
        args.dataset_root,
        args.features_dir,
        args.drift_data_json,
        args.loss_dir,
    )
    drift_all = load_pool(
        "2_Stream_Sim_Drift",
        args.dataset_root,
        args.features_dir,
        args.drift_data_json,
        args.loss_dir,
    )

    candidates = detector_candidates(args.river_path)
    all_runs = []
    selection_audit: dict[str, Any] = {}
    signal_audit: dict[str, Any] = {}
    sada_threshold_audit: dict[str, Any] = {}
    variant_sizes: dict[str, Any] = {}

    for variant_name in args.variants:
        calibration, no_drift, drift = subset_variant(
            calibration_all,
            no_drift_all,
            drift_all,
            variant_name,
        )
        validation_mask = stable_validation_mask(
            drift.uids,
            args.validation_fraction,
        )
        drift_validation = drift.subset(validation_mask)
        drift_test = drift.subset(~validation_mask)
        reference_tasks = {
            int(task_id)
            for task_id in np.unique(calibration.task_ids)
        }
        scorer = SadaScorer(
            reference_features=load_reference_features(
                args.features_dir,
                reference_tasks,
            ),
            pca_dim=args.pca_dim,
            window_size=args.window_size,
            cov_eps=args.cov_eps,
            seed=args.calibration_seed,
        )
        projected = {
            "calibration": scorer.project(calibration.features),
            "no_drift": scorer.project(no_drift.features),
            "drift_validation": scorer.project(drift_validation.features),
            "drift_test": scorer.project(drift_test.features),
        }
        variant_sizes[variant_name] = {
            "calibration": len(calibration.uids),
            "no_drift": len(no_drift.uids),
            "drift_validation": len(drift_validation.uids),
            "drift_test": len(drift_test.uids),
            "calibration_formats": {
                name: int((calibration.formats == name).sum())
                for name in sorted(set(calibration.formats))
            },
            "no_drift_formats": {
                name: int((no_drift.formats == name).sum())
                for name in sorted(set(no_drift.formats))
            },
            "drift_formats": {
                name: int((drift.formats == name).sum())
                for name in sorted(set(drift.formats))
            },
            "reference_tasks": sorted(reference_tasks),
        }

        for weighting in args.weightings:
            threshold_key = f"{variant_name}/{weighting}"
            sada_thresholds = scorer.calibrate(
                calibration_projected=projected["calibration"],
                calibration_task_ids=calibration.task_ids,
                weighting=weighting,
                windows=args.sada_calibration_windows,
                quantile=args.sada_quantile,
                seed=args.calibration_seed,
            )
            sada_threshold_audit[threshold_key] = sada_thresholds

            for signal_name in ["global_nll", "format_aware_nll"]:
                signals, stats = signal_bundle(
                    calibration,
                    no_drift,
                    drift_validation,
                    drift_test,
                    signal_name,
                    args.event_quantile,
                )
                setting_key = f"{variant_name}/{weighting}/{signal_name}"
                signal_audit[setting_key] = stats
                selected_binary, audit_binary = select_methods(
                    methods=BINARY_NATIVE_METHODS,
                    candidates=candidates,
                    calibration_values=signals["calibration_binary"],
                    calibration_task_ids=calibration.task_ids,
                    validation_clean_values=signals["no_drift_binary"],
                    validation_clean_task_ids=no_drift.task_ids,
                    validation_drift_values=signals["drift_validation_binary"],
                    validation_drift_task_ids=drift_validation.task_ids,
                    weighting=weighting,
                    target_fpr=args.target_fpr,
                    calibration_windows=args.calibration_windows,
                    validation_windows=args.validation_windows,
                    window_size=args.window_size,
                    seed=args.calibration_seed,
                )
                selected_continuous, audit_continuous = select_methods(
                    methods=CONTINUOUS_NATIVE_METHODS,
                    candidates=candidates,
                    calibration_values=signals["calibration_continuous"],
                    calibration_task_ids=calibration.task_ids,
                    validation_clean_values=signals["no_drift_continuous"],
                    validation_clean_task_ids=no_drift.task_ids,
                    validation_drift_values=signals["drift_validation_continuous"],
                    validation_drift_task_ids=drift_validation.task_ids,
                    weighting=weighting,
                    target_fpr=args.target_fpr,
                    calibration_windows=args.calibration_windows,
                    validation_windows=args.validation_windows,
                    window_size=args.window_size,
                    seed=args.calibration_seed,
                )
                selection_audit[setting_key] = {
                    **audit_binary,
                    **audit_continuous,
                }

                for seed in args.seeds:
                    for ratio in RATIOS:
                        n_drift = int(round(args.window_size * ratio))
                        plans = build_window_plans(
                            clean_task_ids=no_drift.task_ids,
                            drift_task_ids=drift_test.task_ids,
                            n_clean=args.window_size - n_drift,
                            n_drift=n_drift,
                            count=args.windows_per_setting,
                            seed=seed + 1000 * VARIANT_ORDER.index(variant_name) + int(10_000 * ratio),
                            weighting=weighting,
                        )
                        sada_row = evaluate_sada_plans(
                            scorer,
                            projected["no_drift"],
                            projected["drift_test"],
                            plans,
                            sada_thresholds,
                            weighting,
                            seed,
                            signal_name,
                            "contiguous_tail",
                        )
                        sada_row.update(
                            {
                                "variant": variant_name,
                                "ratio": ratio,
                            }
                        )
                        all_runs.append(sada_row)
                        for method, selected in selected_binary.items():
                            row = evaluate_sequential_plans(
                                method=method,
                                signal_name=signal_name,
                                selected=selected,
                                calibration_values=signals["calibration_binary"],
                                calibration_task_ids=calibration.task_ids,
                                clean_values=signals["no_drift_binary"],
                                drift_values=signals["drift_test_binary"],
                                plans=plans,
                                weighting=weighting,
                                seed=seed,
                                layout="contiguous_tail",
                            )
                            row.update(
                                {
                                    "variant": variant_name,
                                    "ratio": ratio,
                                }
                            )
                            all_runs.append(row)
                        for method, selected in selected_continuous.items():
                            row = evaluate_sequential_plans(
                                method=method,
                                signal_name=signal_name,
                                selected=selected,
                                calibration_values=signals["calibration_continuous"],
                                calibration_task_ids=calibration.task_ids,
                                clean_values=signals["no_drift_continuous"],
                                drift_values=signals["drift_test_continuous"],
                                plans=plans,
                                weighting=weighting,
                                seed=seed,
                                layout="contiguous_tail",
                            )
                            row.update(
                                {
                                    "variant": variant_name,
                                    "ratio": ratio,
                                }
                            )
                            all_runs.append(row)

        write_json(args.output_dir / "runs_partial.json", all_runs)
        write_json(args.output_dir / "selection_audit_partial.json", selection_audit)
        write_json(args.output_dir / "signal_audit_partial.json", signal_audit)

    config = {
        "protocol": "progressive_format_mixture",
        "loss_dir": str(args.loss_dir),
        "window_size": args.window_size,
        "windows_per_setting": args.windows_per_setting,
        "ratios": RATIOS,
        "seeds": args.seeds,
        "target_fpr": args.target_fpr,
        "event_quantile": args.event_quantile,
        "variants": {
            name: {
                key: sorted(value)
                for key, value in VARIANTS[name].items()
            }
            for name in args.variants
        },
        "variant_sizes": variant_sizes,
        "format_groups": {
            "closed_set": ["mcq", "yes_no"],
            "open_short": ["open_short"],
            "long_generation": ["caption", "long_report"],
        },
        "fallback_policy": "use global calibration ECDF only when a format group has no calibration samples; record every fallback count",
        "optwin_provenance": "local OPTWIN-style reproduction with two-sided mean and variance tests",
    }
    write_json(args.output_dir / "run_config.json", config)
    write_json(args.output_dir / "runs.json", all_runs)
    write_json(
        args.output_dir / "summary.json",
        summarize_rows(
            all_runs,
            ["variant", "weighting", "signal", "method", "ratio"],
        ),
    )
    write_json(args.output_dir / "selection_audit.json", selection_audit)
    write_json(args.output_dir / "signal_audit.json", signal_audit)
    write_json(args.output_dir / "sada_thresholds.json", sada_threshold_audit)
    report = [
        "# Formal Heterogeneous Label-Aware Drift Experiments",
        "",
        "The experiment progressively mixes answer formats and compares a global "
        "teacher-forced NLL ECDF with answer-format-aware ECDF calibration.",
        "",
        "All method configurations are selected by held-out drift validation power "
        "subject to calibration FPR <= 5%. Machine-readable results are stored in "
        "`summary.json` and `runs.json`.",
    ]
    (args.output_dir / "heterogeneous_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(f"Heterogeneous formal results written to {args.output_dir}")


if __name__ == "__main__":
    main()
