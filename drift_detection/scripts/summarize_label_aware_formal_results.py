#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compute_label_aware_supervised_loss import ordered_split_rows  # noqa: E402
from src.label_aware_formal import answer_format  # noqa: E402


CLOSED_METHODS = [
    "SADA",
    "DDM",
    "EDDM",
    "ADWIN",
    "HDDM-W",
    "Page-Hinkley",
    "KSWIN",
    "OPTWIN",
]
BINARY_NATIVE_METHODS = ["DDM", "EDDM", "HDDM-W"]
CONTINUOUS_NATIVE_METHODS = ["ADWIN", "Page-Hinkley", "KSWIN", "OPTWIN"]
FORMAT_ORDER = [
    "mcq",
    "yes_no",
    "open_short",
    "caption",
    "long_report",
    "unknown",
]
VARIANT_LABELS = {
    "mcq": "MCQ",
    "mcq_yesno": "+ Yes/No",
    "open_short": "+ Open short",
    "caption": "+ Caption",
    "long_report": "+ Long report",
}
METHOD_COLORS = {
    "SADA": "#111827",
    "DDM": "#2563eb",
    "EDDM": "#0d9488",
    "ADWIN": "#dc2626",
    "HDDM-W": "#7c3aed",
    "Page-Hinkley": "#ea580c",
    "KSWIN": "#0891b2",
    "OPTWIN": "#65a30d",
}
FORMAT_COLORS = {
    "mcq": "#2563eb",
    "yes_no": "#0d9488",
    "open_short": "#7c3aed",
    "caption": "#ea580c",
    "long_report": "#dc2626",
    "unknown": "#64748b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-facing artifacts from formal label-aware runs."
    )
    parser.add_argument(
        "--closed-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "closed_set"
        ),
    )
    parser.add_argument(
        "--heterogeneous-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "heterogeneous"
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "paper_artifacts"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT.parent / "dataset",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=PROJECT_ROOT / "features_pooled_e_fusion",
    )
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
        "--skip-heterogeneous",
        action="store_true",
        help="Generate only closed-set artifacts.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None, None
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if array.size > 1 else 0.0,
    )


def format_pm(mean: float | None, std: float | None, scale: float = 100.0) -> str:
    if mean is None:
        return "N/A"
    std_value = 0.0 if std is None else std
    return f"{scale * mean:.2f} +/- {scale * std_value:.2f}"


def grouped_by_seed(
    rows: Iterable[dict[str, Any]],
    metric: str,
) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped[int(row["seed"])].append(float(value))
    return grouped


def aggregate_seed_metric(
    rows: Iterable[dict[str, Any]],
    metric: str,
) -> tuple[float | None, float | None]:
    grouped = grouped_by_seed(rows, metric)
    return mean_std(
        np.mean(values)
        for _, values in sorted(grouped.items())
        if values
    )


def filter_rows(
    rows: Iterable[dict[str, Any]],
    **filters: Any,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(row.get(key) == value for key, value in filters.items())
    ]


def closed_metric_row(
    runs: Sequence[dict[str, Any]],
    method: str,
    signal: str,
    weighting: str,
) -> dict[str, Any]:
    selected = filter_rows(
        runs,
        method=method,
        signal=signal,
        weighting=weighting,
    )
    row: dict[str, Any] = {
        "method": method,
        "signal": signal,
        "weighting": weighting,
    }
    for label, ratio in [("fpr", 0.0), ("r05", 0.05), ("r10", 0.10), ("r20", 0.20)]:
        ratio_rows = [
            item
            for item in selected
            if math.isclose(float(item.get("ratio", -1.0)), ratio)
        ]
        mean, std = aggregate_seed_metric(ratio_rows, "detection_rate")
        row[f"{label}_mean"] = mean
        row[f"{label}_std"] = std
    positive = [
        item
        for item in selected
        if float(item.get("ratio", 0.0)) > 0.0
    ]
    mean, std = aggregate_seed_metric(positive, "detection_rate")
    row["mean_drift_recall_mean"] = mean
    row["mean_drift_recall_std"] = std
    mean, std = aggregate_seed_metric(positive, "conditional_mean_delay")
    row["conditional_delay_mean"] = mean
    row["conditional_delay_std"] = std
    mean, std = aggregate_seed_metric(positive, "censored_mean_delay")
    row["censored_delay_mean"] = mean
    row["censored_delay_std"] = std
    return row


def write_metric_markdown(
    path: Path,
    title: str,
    rows: Sequence[dict[str, Any]],
) -> None:
    lines = [
        f"# {title}",
        "",
        "| Method | Signal | FPR (%) | R@5 (%) | R@10 (%) | R@20 (%) | Mean Drift Recall (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {signal} | {fpr} | {r05} | {r10} | {r20} | {mean} |".format(
                method=row["method"],
                signal=row["signal"],
                fpr=format_pm(row["fpr_mean"], row["fpr_std"]),
                r05=format_pm(row["r05_mean"], row["r05_std"]),
                r10=format_pm(row["r10_mean"], row["r10_std"]),
                r20=format_pm(row["r20_mean"], row["r20_std"]),
                mean=format_pm(
                    row["mean_drift_recall_mean"],
                    row["mean_drift_recall_std"],
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_closed_tables(
    closed_dir: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs = read_json(closed_dir / "main_runs.json")
    binary_rows = [
        closed_metric_row(runs, method, "binary_error", "natural")
        for method in CLOSED_METHODS
    ]
    continuous_rows = [
        closed_metric_row(runs, method, "correct_option_nll", "natural")
        for method in CONTINUOUS_NATIVE_METHODS
    ]
    write_csv(output_dir / "closed_set_binary_main.csv", binary_rows)
    write_csv(output_dir / "closed_set_continuous_supplement.csv", continuous_rows)
    write_metric_markdown(
        output_dir / "closed_set_binary_main.md",
        "Closed-Set Explicit Binary-Error Comparison",
        binary_rows,
    )
    write_metric_markdown(
        output_dir / "closed_set_continuous_supplement.md",
        "Closed-Set Correct-Option-NLL Supplement",
        continuous_rows,
    )
    return binary_rows, continuous_rows


def recommended_signal(method: str) -> str:
    if method == "SADA":
        return "method_native"
    if method in BINARY_NATIVE_METHODS:
        return "binary_error"
    return "correct_option_nll"


def build_budget_artifacts(closed_dir: Path, output_dir: Path) -> None:
    runs = read_json(closed_dir / "budget_runs.json")
    rows = []
    for method in CLOSED_METHODS:
        signal = recommended_signal(method)
        for budget in sorted({int(row["budget"]) for row in runs}):
            selected = filter_rows(
                runs,
                method=method,
                signal=signal,
                budget=budget,
            )
            detection_mean, detection_std = aggregate_seed_metric(
                selected,
                "detection_rate",
            )
            conditional_mean, conditional_std = aggregate_seed_metric(
                selected,
                "conditional_mean_delay",
            )
            censored_mean, censored_std = aggregate_seed_metric(
                selected,
                "censored_mean_delay",
            )
            rows.append(
                {
                    "method": method,
                    "signal": signal,
                    "budget": budget,
                    "recall_mean": detection_mean,
                    "recall_std": detection_std,
                    "conditional_delay_mean": conditional_mean,
                    "conditional_delay_std": conditional_std,
                    "censored_delay_mean": censored_mean,
                    "censored_delay_std": censored_std,
                }
            )
    write_csv(output_dir / "post_change_budget.csv", rows)
    plot_method_curves(
        rows=rows,
        x_key="budget",
        y_key="recall",
        output_base=output_dir / "recall_vs_post_change_budget",
        x_label="Post-change samples in a 200-sample window",
        y_label="Detection rate",
        binary_methods=BINARY_NATIVE_METHODS,
        continuous_methods=CONTINUOUS_NATIVE_METHODS,
    )


def build_task_macro_artifacts(closed_dir: Path, output_dir: Path) -> None:
    runs = read_json(closed_dir / "task_macro_runs.json")
    rows = []
    for method in CLOSED_METHODS:
        signal = recommended_signal(method)
        for ratio in [0.05, 0.10, 0.20]:
            selected = [
                row
                for row in runs
                if row["method"] == method
                and row["signal"] == signal
                and math.isclose(float(row["ratio"]), ratio)
            ]
            by_seed: dict[int, list[float]] = defaultdict(list)
            for row in selected:
                by_seed[int(row["seed"])].append(float(row["detection_rate"]))
            mean, std = mean_std(
                np.mean(values)
                for _, values in sorted(by_seed.items())
                if values
            )
            rows.append(
                {
                    "method": method,
                    "signal": signal,
                    "ratio": ratio,
                    "task_macro_recall_mean": mean,
                    "task_macro_recall_std": std,
                }
            )
    write_csv(output_dir / "closed_set_task_macro.csv", rows)
    lines = [
        "# Closed-Set Task-Macro Recall",
        "",
        "| Method | Signal | Macro R@5 (%) | Macro R@10 (%) | Macro R@20 (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for method in CLOSED_METHODS:
        method_rows = {
            float(row["ratio"]): row
            for row in rows
            if row["method"] == method
        }
        lines.append(
            "| {method} | {signal} | {r05} | {r10} | {r20} |".format(
                method=method,
                signal=recommended_signal(method),
                r05=format_pm(
                    method_rows[0.05]["task_macro_recall_mean"],
                    method_rows[0.05]["task_macro_recall_std"],
                ),
                r10=format_pm(
                    method_rows[0.10]["task_macro_recall_mean"],
                    method_rows[0.10]["task_macro_recall_std"],
                ),
                r20=format_pm(
                    method_rows[0.20]["task_macro_recall_mean"],
                    method_rows[0.20]["task_macro_recall_std"],
                ),
            )
        )
    (output_dir / "closed_set_task_macro.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_ordering_artifacts(closed_dir: Path, output_dir: Path) -> None:
    runs = read_json(closed_dir / "ordering_runs.json")
    ordering = [
        "shuffled",
        "block_1",
        "block_2",
        "block_5",
        "block_10",
        "block_20",
        "contiguous",
    ]
    rows = []
    for method in CLOSED_METHODS:
        signal = recommended_signal(method)
        for label in ordering:
            selected = filter_rows(
                runs,
                method=method,
                signal=signal,
                ordering=label,
            )
            mean, std = aggregate_seed_metric(selected, "detection_rate")
            rows.append(
                {
                    "method": method,
                    "signal": signal,
                    "ordering": label,
                    "ordering_index": ordering.index(label),
                    "window_detection_rate_mean": mean,
                    "window_detection_rate_std": std,
                }
            )
    write_csv(output_dir / "ordering_sensitivity.csv", rows)
    plot_method_curves(
        rows=rows,
        x_key="ordering_index",
        y_key="window_detection_rate",
        output_base=output_dir / "ordering_block_contiguity_sensitivity",
        x_label="Stream layout",
        y_label="Window detection rate",
        binary_methods=BINARY_NATIVE_METHODS,
        continuous_methods=CONTINUOUS_NATIVE_METHODS,
        tick_labels=[
            "Shuffled",
            "Block 1",
            "Block 2",
            "Block 5",
            "Block 10",
            "Block 20",
            "Contiguous",
        ],
    )
    permutation = read_json(closed_dir / "sada_permutation_invariance.json")
    (output_dir / "sada_permutation_invariance.json").write_text(
        json.dumps(permutation, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_method_curves(
    rows: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
    output_base: Path,
    x_label: str,
    y_label: str,
    binary_methods: Sequence[str],
    continuous_methods: Sequence[str],
    tick_labels: Sequence[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    panels = [
        ("Binary error stream", ["SADA", *binary_methods]),
        ("Continuous correct-option NLL", ["SADA", *continuous_methods]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    for ax, (title, methods) in zip(axes, panels):
        for method in methods:
            selected = sorted(
                [row for row in rows if row["method"] == method],
                key=lambda row: float(row[x_key]),
            )
            if not selected:
                continue
            x = np.asarray([float(row[x_key]) for row in selected])
            y = np.asarray(
                [
                    np.nan
                    if row.get(f"{y_key}_mean") is None
                    else float(row[f"{y_key}_mean"])
                    for row in selected
                ]
            )
            std = np.asarray(
                [
                    0.0
                    if row.get(f"{y_key}_std") is None
                    else float(row[f"{y_key}_std"])
                    for row in selected
                ]
            )
            if not np.isfinite(y).any():
                continue
            color = METHOD_COLORS[method]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.4 if method == "SADA" else 1.8,
                markersize=4.8,
                label=method,
                color=color,
            )
            ax.fill_between(
                x,
                np.clip(y - std, 0.0, 1.0),
                np.clip(y + std, 0.0, 1.0),
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylim(0.0, 1.04)
        if tick_labels is not None:
            positions = sorted({int(row[x_key]) for row in rows})
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, rotation=24, ha="right")
        elif x_key == "budget":
            ax.set_xticks(sorted({int(row[x_key]) for row in rows}))
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=240)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def heterogeneous_metric_row(
    runs: Sequence[dict[str, Any]],
    variant: str,
    weighting: str,
    signal: str,
    method: str,
) -> dict[str, Any]:
    selected = filter_rows(
        runs,
        variant=variant,
        weighting=weighting,
        signal=signal,
        method=method,
    )
    row: dict[str, Any] = {
        "variant": variant,
        "weighting": weighting,
        "signal": signal,
        "method": method,
    }
    for label, ratio in [("fpr", 0.0), ("r05", 0.05), ("r10", 0.10), ("r20", 0.20)]:
        ratio_rows = [
            item
            for item in selected
            if math.isclose(float(item.get("ratio", -1.0)), ratio)
        ]
        mean, std = aggregate_seed_metric(ratio_rows, "detection_rate")
        row[f"{label}_mean"] = mean
        row[f"{label}_std"] = std
    positive = [
        item
        for item in selected
        if float(item.get("ratio", 0.0)) > 0.0
    ]
    mean, std = aggregate_seed_metric(positive, "detection_rate")
    row["mean_drift_recall_mean"] = mean
    row["mean_drift_recall_std"] = std
    return row


def build_heterogeneous_tables(
    heterogeneous_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    runs = read_json(heterogeneous_dir / "runs.json")
    variants = [
        name
        for name in VARIANT_LABELS
        if any(row.get("variant") == name for row in runs)
    ]
    rows = []
    for variant in variants:
        for weighting in ["natural", "task_balanced"]:
            for signal in ["global_nll", "format_aware_nll"]:
                methods = [
                    "SADA",
                    "DDM",
                    "EDDM",
                    "ADWIN",
                    "HDDM-W",
                    "Page-Hinkley",
                    "KSWIN",
                    "OPTWIN",
                ]
                for method in methods:
                    metric = heterogeneous_metric_row(
                        runs,
                        variant,
                        weighting,
                        signal,
                        method,
                    )
                    if metric["fpr_mean"] is not None:
                        rows.append(metric)
    write_csv(output_dir / "heterogeneous_global_vs_format_aware_full.csv", rows)

    compact = []
    for variant in variants:
        sada = next(
            row
            for row in rows
            if row["variant"] == variant
            and row["weighting"] == "task_balanced"
            and row["signal"] == "global_nll"
            and row["method"] == "SADA"
        )
        supervised_by_signal: dict[str, list[dict[str, Any]]] = {}
        for signal in ["global_nll", "format_aware_nll"]:
            candidates = [
                row
                for row in rows
                if row["variant"] == variant
                and row["weighting"] == "task_balanced"
                and row["signal"] == signal
                and row["method"] != "SADA"
                and row["mean_drift_recall_mean"] is not None
            ]
            supervised_by_signal[signal] = candidates
        global_rows = supervised_by_signal["global_nll"]
        aware_rows = supervised_by_signal["format_aware_nll"]
        compact.append(
            {
                "variant": variant,
                "sada_mean_recall": sada["mean_drift_recall_mean"],
                "sada_fpr": sada["fpr_mean"],
                "global_calibratable_methods": len(global_rows),
                "global_method_macro_recall": (
                    float(
                        np.mean(
                            [
                                float(row["mean_drift_recall_mean"])
                                for row in global_rows
                            ]
                        )
                    )
                    if global_rows
                    else None
                ),
                "global_method_macro_fpr": (
                    float(np.mean([float(row["fpr_mean"]) for row in global_rows]))
                    if global_rows
                    else None
                ),
                "format_aware_calibratable_methods": len(aware_rows),
                "format_aware_method_macro_recall": (
                    float(
                        np.mean(
                            [
                                float(row["mean_drift_recall_mean"])
                                for row in aware_rows
                            ]
                        )
                    )
                    if aware_rows
                    else None
                ),
                "format_aware_method_macro_fpr": (
                    float(np.mean([float(row["fpr_mean"]) for row in aware_rows]))
                    if aware_rows
                    else None
                ),
            }
        )
    write_csv(output_dir / "heterogeneous_global_vs_format_aware_compact.csv", compact)
    lines = [
        "# Global NLL vs Format-Aware NLL",
        "",
        "Task-balanced results. Supervised columns are method-macro averages over "
        "all calibratable detectors for that signal; the detector count is shown "
        "in parentheses.",
        "",
        "| Format mixture | SADA mean recall / FPR (%) | Global NLL macro recall / FPR (%) | Format-aware macro recall / FPR (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in compact:
        global_recall = row["global_method_macro_recall"]
        global_fpr = row["global_method_macro_fpr"]
        aware_recall = row["format_aware_method_macro_recall"]
        aware_fpr = row["format_aware_method_macro_fpr"]
        lines.append(
            "| {variant} | {sada_recall:.2f} / {sada_fpr:.2f} | "
            "{global_values} ({global_count}) | "
            "{aware_values} ({aware_count}) |".format(
                variant=VARIANT_LABELS[row["variant"]],
                sada_recall=100.0 * float(row["sada_mean_recall"]),
                sada_fpr=100.0 * float(row["sada_fpr"]),
                global_values=(
                    f"{100.0 * float(global_recall):.2f} / "
                    f"{100.0 * float(global_fpr):.2f}"
                    if global_recall is not None and global_fpr is not None
                    else "N/A"
                ),
                global_count=row["global_calibratable_methods"],
                aware_values=(
                    f"{100.0 * float(aware_recall):.2f} / "
                    f"{100.0 * float(aware_fpr):.2f}"
                    if aware_recall is not None and aware_fpr is not None
                    else "N/A"
                ),
                aware_count=row["format_aware_calibratable_methods"],
            )
        )
    (output_dir / "heterogeneous_global_vs_format_aware.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    final_variant = "long_report" if "long_report" in variants else variants[-1]
    per_method = []
    for method in [
        "DDM",
        "EDDM",
        "ADWIN",
        "HDDM-W",
        "Page-Hinkley",
        "KSWIN",
        "OPTWIN",
    ]:
        global_row = next(
            (
                row
                for row in rows
                if row["variant"] == final_variant
                and row["weighting"] == "task_balanced"
                and row["signal"] == "global_nll"
                and row["method"] == method
            ),
            None,
        )
        aware_row = next(
            (
                row
                for row in rows
                if row["variant"] == final_variant
                and row["weighting"] == "task_balanced"
                and row["signal"] == "format_aware_nll"
                and row["method"] == method
            ),
            None,
        )
        per_method.append(
            {
                "variant": final_variant,
                "method": method,
                "global_mean_recall": (
                    global_row["mean_drift_recall_mean"] if global_row else None
                ),
                "global_fpr": global_row["fpr_mean"] if global_row else None,
                "format_aware_mean_recall": (
                    aware_row["mean_drift_recall_mean"] if aware_row else None
                ),
                "format_aware_fpr": aware_row["fpr_mean"] if aware_row else None,
            }
        )
    write_csv(
        output_dir / "heterogeneous_long_report_per_method.csv",
        per_method,
    )
    method_lines = [
        "# Long-Report Mixture: Per-Method Signal Comparison",
        "",
        "Task-balanced results. `N/A` means no configuration satisfied the calibration FPR constraint.",
        "",
        "| Method | Global NLL mean recall / FPR (%) | Format-aware NLL mean recall / FPR (%) |",
        "|---|---:|---:|",
    ]
    for row in per_method:
        global_values = (
            f"{100.0 * float(row['global_mean_recall']):.2f} / "
            f"{100.0 * float(row['global_fpr']):.2f}"
            if row["global_mean_recall"] is not None
            and row["global_fpr"] is not None
            else "N/A"
        )
        aware_values = (
            f"{100.0 * float(row['format_aware_mean_recall']):.2f} / "
            f"{100.0 * float(row['format_aware_fpr']):.2f}"
            if row["format_aware_mean_recall"] is not None
            and row["format_aware_fpr"] is not None
            else "N/A"
        )
        method_lines.append(
            f"| {row['method']} | {global_values} | {aware_values} |"
        )
    (output_dir / "heterogeneous_long_report_per_method.md").write_text(
        "\n".join(method_lines) + "\n",
        encoding="utf-8",
    )
    return rows


def read_loss_map(path: Path) -> dict[str, float]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "mean_nll" in row:
            output[str(row["uid"])] = float(row["mean_nll"])
    return output


def build_loss_ecdf_artifacts(
    loss_dir: Path,
    dataset_root: Path,
    features_dir: Path,
    drift_data_json: Path,
    output_dir: Path,
) -> None:
    split_names = [
        "1_Threshold_Cal",
        "2_Stream_Sim_NoDrift",
        "2_Stream_Sim_Drift",
    ]
    split_labels = {
        "1_Threshold_Cal": "Calibration",
        "2_Stream_Sim_NoDrift": "No drift",
        "2_Stream_Sim_Drift": "Natural drift",
    }
    values_by_split: dict[str, dict[str, np.ndarray]] = {}
    summary_rows = []
    calibration_values = []
    for split_name in split_names:
        rows = ordered_split_rows(
            dataset_root=dataset_root,
            features_dir=features_dir,
            drift_data_json=drift_data_json,
            split_name=split_name,
        )
        losses = read_loss_map(loss_dir / f"{split_name}.jsonl")
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            uid = str(row["uid"])
            if uid not in losses:
                raise RuntimeError(f"{split_name}: missing NLL for {uid}")
            grouped[answer_format(row)].append(losses[uid])
        values_by_split[split_name] = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in grouped.items()
        }
        if split_name == "1_Threshold_Cal":
            calibration_values = [
                value
                for values in grouped.values()
                for value in values
            ]

    global_q95 = float(np.quantile(calibration_values, 0.95))
    for split_name in split_names:
        for format_name in FORMAT_ORDER:
            values = values_by_split[split_name].get(format_name)
            if values is None or values.size == 0:
                continue
            summary_rows.append(
                {
                    "split": split_name,
                    "split_label": split_labels[split_name],
                    "format": format_name,
                    "count": int(values.size),
                    "mean_nll": float(values.mean()),
                    "median_nll": float(np.median(values)),
                    "q95_nll": float(np.quantile(values, 0.95)),
                    "above_global_calibration_q95_rate": float(
                        (values > global_q95).mean()
                    ),
                    "global_calibration_q95": global_q95,
                }
            )
    write_csv(output_dir / "answer_format_loss_summary.csv", summary_rows)

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    all_positive = [
        float(value)
        for split in values_by_split.values()
        for values in split.values()
        for value in values
        if value > 0.0
    ]
    use_log_scale = (
        bool(all_positive)
        and max(all_positive) / max(min(all_positive), 1e-12) > 100.0
    )
    for ax, split_name in zip(axes, split_names):
        for format_name in FORMAT_ORDER:
            values = values_by_split[split_name].get(format_name)
            if values is None or values.size == 0:
                continue
            x = np.sort(values)
            if use_log_scale:
                x = np.maximum(x, np.finfo(np.float64).tiny)
            y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
            ax.plot(
                x,
                y,
                linewidth=2.0,
                color=FORMAT_COLORS[format_name],
                label=f"{format_name} (n={x.size})",
            )
        ax.axvline(
            global_q95,
            color="#111827",
            linestyle="--",
            linewidth=1.6,
            label="Global calibration q95",
        )
        ax.set_title(split_labels[split_name], fontsize=12, fontweight="bold")
        ax.set_xlabel("Teacher-forced mean answer NLL")
        if use_log_scale:
            ax.set_xscale("log")
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("Empirical CDF")
    fig.tight_layout()
    output_base = output_dir / "answer_format_loss_ecdf"
    fig.savefig(output_base.with_suffix(".png"), dpi=240)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def write_index(
    output_dir: Path,
    closed_dir: Path,
    heterogeneous_dir: Path | None,
) -> None:
    lines = [
        "# Formal Label-Aware Paper Artifacts",
        "",
        f"- Closed-set source: `{closed_dir}`",
    ]
    if heterogeneous_dir is not None:
        lines.append(f"- Heterogeneous source: `{heterogeneous_dir}`")
    lines.extend(
        [
            "",
            "## Closed Set",
            "",
            "- `closed_set_binary_main.*`: fair explicit binary-error comparison.",
            "- `closed_set_continuous_supplement.*`: correct-option-NLL variants.",
            "- `closed_set_task_macro.*`: task-macro recall.",
            "- `post_change_budget.csv` and `recall_vs_post_change_budget.*`.",
            "- `ordering_sensitivity.csv` and `ordering_block_contiguity_sensitivity.*`.",
            "- `sada_permutation_invariance.json`.",
        ]
    )
    if heterogeneous_dir is not None:
        lines.extend(
            [
                "",
                "## Heterogeneous Tasks",
                "",
                "- `answer_format_loss_summary.csv` and `answer_format_loss_ecdf.*`.",
                "- `heterogeneous_global_vs_format_aware_full.csv`.",
                "- `heterogeneous_global_vs_format_aware_compact.csv` and `.md`.",
                "- `heterogeneous_long_report_per_method.csv` and `.md`.",
            ]
        )
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    build_closed_tables(args.closed_dir, args.output_dir)
    build_budget_artifacts(args.closed_dir, args.output_dir)
    build_task_macro_artifacts(args.closed_dir, args.output_dir)
    build_ordering_artifacts(args.closed_dir, args.output_dir)

    heterogeneous_source: Path | None = None
    if not args.skip_heterogeneous:
        build_heterogeneous_tables(args.heterogeneous_dir, args.output_dir)
        build_loss_ecdf_artifacts(
            loss_dir=args.loss_dir,
            dataset_root=args.dataset_root,
            features_dir=args.features_dir,
            drift_data_json=args.drift_data_json,
            output_dir=args.output_dir,
        )
        heterogeneous_source = args.heterogeneous_dir

    write_index(
        args.output_dir,
        closed_dir=args.closed_dir,
        heterogeneous_dir=heterogeneous_source,
    )
    print(f"Paper-facing artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
