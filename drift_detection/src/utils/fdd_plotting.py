from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _annotate_line(ax, x_values, y_values, fmt: str = "{:.3f}", y_offset: float = 0.015) -> None:
    for x_value, y_value in zip(x_values, y_values):
        ax.text(x_value, y_value + y_offset, fmt.format(y_value), ha="center", va="bottom", fontsize=9)


def _annotate_bars(ax, bars, fmt: str = "{:.3f}", y_offset: float = 0.01) -> None:
    upper = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + upper * y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_single_stream_plots(payload: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = str(payload["feature_key"])
    evaluation = payload["evaluation"]

    drift_ratios = [float(item["drift_ratio"]) for item in evaluation]
    detection_rates = [float(item["detection_rate"]) for item in evaluation]
    mean_fdd = [float(item["mean_fdd"]) for item in evaluation]
    std_fdd = [float(item["std_fdd"]) for item in evaluation]
    threshold = float(payload["threshold"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(drift_ratios, detection_rates, marker="o", linewidth=2, color="#1f77b4")
    axes[0, 0].set_title(f"Stream {stream}: Detection Rate")
    axes[0, 0].set_xlabel("Drift Ratio")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].grid(True, linestyle="--", alpha=0.35)
    _annotate_line(axes[0, 0], drift_ratios, detection_rates, fmt="{:.3f}")

    axes[0, 1].errorbar(drift_ratios, mean_fdd, yerr=std_fdd, marker="o", linewidth=2, color="#d62728")
    axes[0, 1].axhline(threshold, color="#2ca02c", linestyle="--", linewidth=2, label=f"threshold={threshold:.2f}")
    axes[0, 1].set_title(f"Stream {stream}: Mean FDD")
    axes[0, 1].set_xlabel("Drift Ratio")
    axes[0, 1].set_ylabel("FDD")
    axes[0, 1].grid(True, linestyle="--", alpha=0.35)
    axes[0, 1].legend()
    for x_value, y_value, s_value in zip(drift_ratios, mean_fdd, std_fdd):
        axes[0, 1].text(x_value, y_value + max(std_fdd) * 0.12, f"{y_value:.2f}\n±{s_value:.2f}", ha="center", va="bottom", fontsize=8)

    bar_labels = ["FPR"] + [f"R@{int(round(ratio * 100))}%" for ratio in drift_ratios[1:]]
    bars = axes[1, 0].bar(bar_labels, detection_rates, color=["#ff7f0e"] + ["#1f77b4"] * (len(bar_labels) - 1))
    axes[1, 0].set_title(f"Stream {stream}: FPR and Recall")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(axes[1, 0], bars, fmt="{:.3f}")

    axes[1, 1].axis("off")
    table_rows = [["Metric", "Value"]]
    table_rows.extend(
        [
            ["Stream", stream],
            ["Window size", str(payload["window_size"])],
            ["Requested PCA", str(payload["requested_pca_dim"])],
            ["Actual PCA", str(payload["actual_pca_dim"])],
            ["Calibration windows", str(payload["calibration_windows"])],
            ["Calibration q", f"{payload['calibration_quantile']:.2f}"],
            ["Threshold", f"{threshold:.6f}"],
            ["FPR", f"{detection_rates[0]:.3f}"],
            ["R@20%", f"{detection_rates[3]:.3f}"],
            ["R@30%", f"{detection_rates[4]:.3f}"],
        ]
    )
    table = axes[1, 1].table(cellText=table_rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.35)

    fig.tight_layout()
    fig.savefig(output_dir / f"fdd_dashboard_{stream}.png", dpi=220)
    plt.close(fig)


def save_stream_summary_plots(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = [str(row["stream"]) for row in rows]
    recall_keys = ["recall_05", "recall_10", "recall_20", "recall_30", "recall_50"]
    recall_labels = ["5%", "10%", "20%", "30%", "50%"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for row in rows:
        recalls = [float(row.get(key, 0.0)) for key in recall_keys]
        axes[0].plot(recall_labels, recalls, marker="o", linewidth=2, label=str(row["stream"]))
        _annotate_line(axes[0], recall_labels, recalls, fmt="{:.3f}", y_offset=0.012)
    axes[0].set_title("Recall Comparison")
    axes[0].set_xlabel("Drift Ratio")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    fprs = [float(row.get("fpr", 0.0)) for row in rows]
    bars = axes[1].bar(streams, fprs, color="#ff7f0e")
    axes[1].set_title("FPR Comparison")
    axes[1].set_xlabel("Stream")
    axes[1].set_ylabel("FPR")
    axes[1].set_ylim(0.0, max(0.12, max(fprs) * 1.25 if fprs else 0.12))
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(axes[1], bars, fmt="{:.3f}")

    threshold_values = [float(row.get("threshold", 0.0)) for row in rows]
    bars = axes[2].bar(streams, threshold_values, color="#2ca02c")
    axes[2].set_title("Threshold Comparison")
    axes[2].set_xlabel("Stream")
    axes[2].set_ylabel("Threshold")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(axes[2], bars, fmt="{:.2f}", y_offset=0.005)

    fig.tight_layout()
    fig.savefig(output_dir / "fdd_stream_dashboard.png", dpi=220)
    plt.close(fig)


def save_fusion_plots(results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_names = [str(item["fusion_name"]) for item in results]
    recall_targets = [0.05, 0.10, 0.20, 0.30, 0.50]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for payload in results:
        evaluation = payload["evaluation"]
        x_values = [float(item["drift_ratio"]) for item in evaluation if float(item["drift_ratio"]) > 0.0]
        y_values = [float(item["detection_rate"]) for item in evaluation if float(item["drift_ratio"]) > 0.0]
        axes[0].plot(x_values, y_values, marker="o", linewidth=2, label=str(payload["fusion_name"]))
        _annotate_line(axes[0], x_values, y_values, fmt="{:.3f}", y_offset=0.012)
    axes[0].set_title("Fusion Recall Comparison")
    axes[0].set_xlabel("Drift Ratio")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    fprs = [next(float(item["detection_rate"]) for item in payload["evaluation"] if float(item["drift_ratio"]) == 0.0) for payload in results]
    bars = axes[1].bar(fusion_names, fprs, color="#ff7f0e")
    axes[1].set_title("Fusion FPR Comparison")
    axes[1].set_xlabel("Fusion")
    axes[1].set_ylabel("FPR")
    axes[1].set_ylim(0.0, max(0.14, max(fprs) * 1.25 if fprs else 0.14))
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(axes[1], bars, fmt="{:.3f}")

    width = 0.22
    x = np.arange(len(recall_targets))
    for idx, payload in enumerate(results):
        recall_values = []
        for target in recall_targets:
            matched = next(float(item["detection_rate"]) for item in payload["evaluation"] if abs(float(item["drift_ratio"]) - target) < 1e-9)
            recall_values.append(matched)
        axes[2].bar(x + idx * width, recall_values, width=width, label=str(payload["fusion_name"]))
        for x_value, y_value in zip(x + idx * width, recall_values):
            axes[2].text(x_value, y_value + 0.012, f"{y_value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    axes[2].set_title("Fusion Recall Bars")
    axes[2].set_xlabel("Drift Ratio")
    axes[2].set_ylabel("Recall")
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels([f"{int(r * 100)}%" for r in recall_targets])
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "fdd_fusion_dashboard.png", dpi=220)
    plt.close(fig)


def save_mahalanobis_single_stream_plots(payload: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = str(payload["feature_key"])
    evaluation = payload["evaluation"]

    drift_ratios = [float(item["drift_ratio"]) for item in evaluation]
    detection_rates = [float(item["detection_rate"]) for item in evaluation]
    mean_scores = [float(item["mean_mahalanobis"]) for item in evaluation]
    std_scores = [float(item["std_mahalanobis"]) for item in evaluation]
    threshold = float(payload["threshold"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(drift_ratios, detection_rates, marker="o", linewidth=2, color="#1f77b4")
    axes[0, 0].set_title(f"Stream {stream}: Detection Rate")
    axes[0, 0].set_xlabel("Drift Ratio")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].grid(True, linestyle="--", alpha=0.35)
    _annotate_line(axes[0, 0], drift_ratios, detection_rates, fmt="{:.3f}")

    axes[0, 1].errorbar(drift_ratios, mean_scores, yerr=std_scores, marker="o", linewidth=2, color="#d62728")
    axes[0, 1].axhline(threshold, color="#2ca02c", linestyle="--", linewidth=2, label=f"threshold={threshold:.2f}")
    axes[0, 1].set_title(f"Stream {stream}: Mean Mahalanobis")
    axes[0, 1].set_xlabel("Drift Ratio")
    axes[0, 1].set_ylabel("Mahalanobis")
    axes[0, 1].grid(True, linestyle="--", alpha=0.35)
    axes[0, 1].legend()
    score_offset = max(std_scores) * 0.12 if std_scores and max(std_scores) > 0 else 0.05
    for x_value, y_value, s_value in zip(drift_ratios, mean_scores, std_scores):
        axes[0, 1].text(x_value, y_value + score_offset, f"{y_value:.2f}\n±{s_value:.2f}", ha="center", va="bottom", fontsize=8)

    bar_labels = ["FPR"] + [f"R@{int(round(ratio * 100))}%" for ratio in drift_ratios[1:]]
    bars = axes[1, 0].bar(bar_labels, detection_rates, color=["#ff7f0e"] + ["#1f77b4"] * (len(bar_labels) - 1))
    axes[1, 0].set_title(f"Stream {stream}: FPR and Recall")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(axes[1, 0], bars, fmt="{:.3f}")

    axes[1, 1].axis("off")
    table_rows = [["Metric", "Value"]]
    table_rows.extend(
        [
            ["Stream", stream],
            ["Window size", str(payload["window_size"])],
            ["Requested PCA", str(payload["requested_pca_dim"])],
            ["Actual PCA", str(payload["actual_pca_dim"])],
            ["Calibration windows", str(payload["calibration_windows"])],
            ["Calibration q", f"{payload['calibration_quantile']:.2f}"],
            ["Threshold", f"{threshold:.6f}"],
            ["FPR", f"{detection_rates[0]:.3f}"],
            ["R@20%", f"{detection_rates[3]:.3f}"],
            ["R@30%", f"{detection_rates[4]:.3f}"],
        ]
    )
    table = axes[1, 1].table(cellText=table_rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.35)

    fig.tight_layout()
    fig.savefig(output_dir / f"mahalanobis_dashboard_{stream}.png", dpi=220)
    plt.close(fig)


def save_mahalanobis_summary_plots(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = [str(row["stream"]) for row in rows]
    recall_keys = ["recall_05", "recall_10", "recall_20", "recall_30", "recall_50"]
    recall_labels = ["5%", "10%", "20%", "30%", "50%"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for row in rows:
        recalls = [float(row.get(key, 0.0)) for key in recall_keys]
        axes[0].plot(recall_labels, recalls, marker="o", linewidth=2, label=str(row["stream"]))
        _annotate_line(axes[0], recall_labels, recalls, fmt="{:.3f}", y_offset=0.012)
    axes[0].set_title("Mahalanobis Recall Comparison")
    axes[0].set_xlabel("Drift Ratio")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    fprs = [float(row.get("fpr", 0.0)) for row in rows]
    bars = axes[1].bar(streams, fprs, color="#ff7f0e")
    axes[1].set_title("Mahalanobis FPR Comparison")
    axes[1].set_xlabel("Stream")
    axes[1].set_ylabel("FPR")
    axes[1].set_ylim(0.0, max(0.18, max(fprs) * 1.25 if fprs else 0.18))
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(axes[1], bars, fmt="{:.3f}")

    threshold_values = [float(row.get("threshold", 0.0)) for row in rows]
    bars = axes[2].bar(streams, threshold_values, color="#2ca02c")
    axes[2].set_title("Mahalanobis Threshold Comparison")
    axes[2].set_xlabel("Stream")
    axes[2].set_ylabel("Threshold")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    _annotate_bars(axes[2], bars, fmt="{:.2f}", y_offset=0.005)

    fig.tight_layout()
    fig.savefig(output_dir / "mahalanobis_stream_dashboard.png", dpi=220)
    plt.close(fig)
