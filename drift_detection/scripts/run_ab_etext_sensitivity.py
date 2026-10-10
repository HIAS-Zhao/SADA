#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_mahalanobis_fusion_search import (  # noqa: E402
    build_sampling_plans,
    evaluate_group,
    feature_group_display_name,
    prepare_stream_data,
    precompute_stream_ratio_scores,
    summarize_payload,
)


DEFAULT_GROUPS = [
    "A+B+E_text",
    "A+B",
    "A+E_text",
    "B+E_text",
    "A",
    "B",
    "E_text",
    "A+B+C+D",
    "A+B+C+D+E_text",
]

DEFAULT_DRIFT_RATIOS = [
    0.0,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.30,
    0.50,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sensitivity and ablation analysis for the A+B+E_text drift detector."
    )
    parser.add_argument("--features-dir", type=str, default="features_pooled_e_fusion")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_detection/ab_etext_sensitivity_20260523",
    )
    parser.add_argument("--feature-groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--fusion-rules", nargs="+", default=["max_norm", "mean_norm", "top2_mean_norm"])
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.95, 0.97, 0.99])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[100, 200, 300])
    parser.add_argument("--calibration-windows", type=int, default=10000)
    parser.add_argument("--windows-per-ratio", type=int, default=1000)
    parser.add_argument("--drift-ratios", type=float, nargs="+", default=DEFAULT_DRIFT_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    return parser.parse_args()


def group_kind(group: str, main_group: str = "A+B+E_text") -> str:
    if group == "A+B+C+D":
        return "baseline"
    if group == "A+B+C+D+E_text":
        return "full_plus_baseline"
    streams = group.split("+")
    if group == main_group:
        return "full"
    missing = set(main_group.split("+")) - set(streams)
    if len(missing) == 1:
        return "minus_one"
    if len(missing) == 2:
        return "minus_two"
    return "other"


def write_outputs(output_dir: Path, full_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ab_etext_sensitivity_full.json").write_text(
        json.dumps(full_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ab_etext_sensitivity_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    compact_fields = [
        "rank",
        "feature_group",
        "ablation_type",
        "fusion_rule",
        "window_size",
        "requested_pca_dim",
        "calibration_quantile",
        "fpr",
        "no_drift_accuracy",
        "mean_accuracy",
        "weighted_accuracy",
        "recall_05",
        "recall_10",
        "recall_20",
        "recall_50",
    ]
    with (output_dir / "ab_etext_sensitivity_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=compact_fields)
        writer.writeheader()
        for rank, row in enumerate(summary_rows, start=1):
            item = {
                "rank": rank,
                "feature_group": row["feature_group"],
                "ablation_type": row["ablation_type"],
                "fusion_rule": row["fusion_rule"],
                "window_size": row["window_size"],
                "requested_pca_dim": row["requested_pca_dim"],
                "calibration_quantile": row["calibration_quantile"],
                "fpr": row["fpr"],
                "no_drift_accuracy": row["no_drift_accuracy"],
                "mean_accuracy": row["mean_accuracy"],
                "weighted_accuracy": row["weighted_accuracy"],
                "recall_05": row["recall_by_ratio"].get("0.05"),
                "recall_10": row["recall_by_ratio"].get("0.10"),
                "recall_20": row["recall_by_ratio"].get("0.20"),
                "recall_50": row["recall_by_ratio"].get("0.50"),
            }
            writer.writerow(item)

    lines = [
        "# Feature-Stream Sensitivity and Ablation Summary",
        "",
        "| Rank | Group | Type | Rule | Window | PCA | Quantile | FPR (%) | Macro (%) | Weighted Acc (%) | R@5% (%) | R@10% (%) | R@20% (%) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary_rows[:30], start=1):
        recall = row["recall_by_ratio"]
        lines.append(
            "| {rank} | {group} | {kind} | {rule} | {window} | {pca} | {quantile:.2f} | {fpr:.2f} | {mean:.2f} | {weighted:.2f} | {r05:.2f} | {r10:.2f} | {r20:.2f} |".format(
                rank=rank,
                group=feature_group_display_name(str(row["feature_group"])),
                kind=row["ablation_type"],
                rule=row["fusion_rule"],
                window=int(row["window_size"]),
                pca=int(row["requested_pca_dim"]),
                quantile=float(row["calibration_quantile"]),
                fpr=float(row["fpr"]) * 100.0,
                mean=float(row["mean_accuracy"]) * 100.0,
                weighted=float(row["weighted_accuracy"]) * 100.0,
                r05=float(recall.get("0.05", 0.0)) * 100.0,
                r10=float(recall.get("0.10", 0.0)) * 100.0,
                r20=float(recall.get("0.20", 0.0)) * 100.0,
            )
        )
    (output_dir / "ab_etext_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    features_dir = PROJECT_ROOT / args.features_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stream_names = sorted({stream for group in args.feature_groups for stream in group.split("+")})
    full_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for window_size in args.window_sizes:
        for pca_dim in args.pca_dims:
            stream_data = prepare_stream_data(
                features_dir=features_dir,
                stream_names=stream_names,
                pca_dim=pca_dim,
                window_size=window_size,
                calibration_windows=args.calibration_windows,
                cov_eps=args.cov_eps,
                seed=args.seed,
            )
            reference_stream = stream_names[0]
            plans = build_sampling_plans(
                no_drift_len=int(stream_data[reference_stream]["no_drift_projected"].shape[0]),
                drift_len=int(stream_data[reference_stream]["drift_projected"].shape[0]),
                drift_ratios=args.drift_ratios,
                window_size=window_size,
                windows_per_ratio=args.windows_per_ratio,
                seed=args.seed,
            )
            raw_scores_by_stream = precompute_stream_ratio_scores(stream_data, stream_names, plans)
            thresholds_by_quantile = {
                float(quantile): {
                    stream: float(np.quantile(stream_data[stream]["calibration_scores"], quantile))
                    for stream in stream_names
                }
                for quantile in args.quantiles
            }

            for quantile in args.quantiles:
                thresholds = thresholds_by_quantile[float(quantile)]
                for group in args.feature_groups:
                    streams = group.split("+")
                    rules = ["single"] if len(streams) == 1 else args.fusion_rules
                    for rule in rules:
                        evaluation = evaluate_group(
                            raw_scores_by_stream=raw_scores_by_stream,
                            thresholds_by_stream=thresholds,
                            streams=streams,
                            drift_ratios=args.drift_ratios,
                            plans=plans,
                            fusion_rule=rule,
                        )
                        payload = {
                            "feature_group": group,
                            "fusion_rule": rule,
                            "streams": streams,
                            "window_size": window_size,
                            "requested_pca_dim": pca_dim,
                            "actual_pca_dims": {
                                stream: int(stream_data[stream]["actual_pca_dim"])
                                for stream in streams
                            },
                            "calibration_windows": args.calibration_windows,
                            "calibration_quantile": float(quantile),
                            "stream_thresholds": {stream: float(thresholds[stream]) for stream in streams},
                            "evaluation": evaluation,
                        }
                        full_rows.append(payload)
                        summary = summarize_payload(payload)
                        summary["window_size"] = window_size
                        summary["ablation_type"] = group_kind(group)
                        summary["recall_by_ratio"] = {
                            f"{float(item['drift_ratio']):.2f}": float(item["detection_rate"])
                            for item in evaluation
                            if float(item["drift_ratio"]) > 0.0
                        }
                        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda item: (
            -float(item["mean_accuracy"]),
            -float(item["weighted_accuracy"]),
            float(item["fpr"]),
            len(str(item["feature_group"]).split("+")),
        )
    )
    write_outputs(output_dir, full_rows, summary_rows)

    best = summary_rows[0]
    print(f"Best group: {best['feature_group']}")
    print(f"Best rule: {best['fusion_rule']}")
    print(f"Best window: {best['window_size']}")
    print(f"Best PCA: {best['requested_pca_dim']}")
    print(f"Best quantile: {best['calibration_quantile']:.2f}")
    print(f"Best macro accuracy: {best['mean_accuracy']:.4f}")
    print(f"Best weighted accuracy: {best['weighted_accuracy']:.4f}")
    print(f"Summary JSON: {output_dir / 'ab_etext_sensitivity_summary.json'}")


if __name__ == "__main__":
    main()
