#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

for env_name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(env_name, "16")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mahalanobis_detector import MahalanobisDetector

DEFAULT_DRIFT_RATIOS = [0.0, 0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.30, 0.50]


def feature_group_display_name(group: str) -> str:
    return {"A": "V", "E_text": "VT", "A+E_text": "V+VT"}.get(group, group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search strong Mahalanobis fusion configurations on pooled feature streams.")
    parser.add_argument("--features-dir", type=str, default="features_pooled")
    parser.add_argument("--output-dir", type=str, default="results_detection/mahalanobis_fusion_search_20260401")
    parser.add_argument("--feature-groups", nargs="+", default=[
        "A",
        "B",
        "D",
        "E_last",
        "E_text",
        "A+B",
        "A+D",
        "A+E_last",
        "A+E_text",
        "A+B+D",
        "A+B+C+D",
        "A+B+C+D+E_last",
        "A+B+C+D+E_text",
    ])
    parser.add_argument("--fusion-rules", nargs="+", default=["single", "max_norm", "mean_norm", "top2_mean_norm", "two_vote"])
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[192])
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.95, 0.97, 0.99])
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--calibration-windows", type=int, default=10000)
    parser.add_argument("--windows-per-ratio", type=int, default=200)
    parser.add_argument("--drift-ratios", type=float, nargs="+", default=DEFAULT_DRIFT_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    return parser.parse_args()


def load_feature_matrix(features_dir: Path, split_name: str, feature_key: str) -> np.ndarray:
    path = features_dir / split_name / "features.pt"
    payload = torch.load(path, map_location="cpu")
    features = payload["features"][feature_key]
    if isinstance(features, torch.Tensor):
        features = features.float().numpy()
    else:
        features = np.asarray(features, dtype=np.float64)
    return features.astype(np.float64, copy=False)


def apply_fusion_rule(normalized_scores: List[float], fusion_rule: str) -> Tuple[float, bool, int]:
    exceed_count = int(sum(score > 1.0 for score in normalized_scores))

    if fusion_rule == "single":
        fused_score = float(normalized_scores[0])
        return fused_score, fused_score > 1.0, exceed_count
    if fusion_rule == "max_norm":
        fused_score = float(max(normalized_scores))
        return fused_score, fused_score > 1.0, exceed_count
    if fusion_rule == "mean_norm":
        fused_score = float(np.mean(normalized_scores))
        return fused_score, fused_score > 1.0, exceed_count
    if fusion_rule == "top2_mean_norm":
        top_scores = sorted(normalized_scores, reverse=True)[: min(2, len(normalized_scores))]
        fused_score = float(np.mean(top_scores))
        return fused_score, fused_score > 1.0, exceed_count
    if fusion_rule == "two_vote":
        vote_threshold = min(2, len(normalized_scores))
        fused_score = float(exceed_count)
        return fused_score, exceed_count >= vote_threshold, exceed_count
    raise ValueError(f"Unsupported fusion rule: {fusion_rule}")


def apply_fusion_rule_matrix(normalized_score_matrix: np.ndarray, fusion_rule: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    exceed_counts = np.sum(normalized_score_matrix > 1.0, axis=0, dtype=np.int64)

    if fusion_rule == "single":
        fused_scores = normalized_score_matrix[0]
        return fused_scores, fused_scores > 1.0, exceed_counts
    if fusion_rule == "max_norm":
        fused_scores = np.max(normalized_score_matrix, axis=0)
        return fused_scores, fused_scores > 1.0, exceed_counts
    if fusion_rule == "mean_norm":
        fused_scores = np.mean(normalized_score_matrix, axis=0)
        return fused_scores, fused_scores > 1.0, exceed_counts
    if fusion_rule == "top2_mean_norm":
        top_count = min(2, normalized_score_matrix.shape[0])
        top_scores = np.sort(normalized_score_matrix, axis=0)[-top_count:]
        fused_scores = np.mean(top_scores, axis=0)
        return fused_scores, fused_scores > 1.0, exceed_counts
    if fusion_rule == "two_vote":
        vote_threshold = min(2, normalized_score_matrix.shape[0])
        fused_scores = exceed_counts.astype(np.float64)
        return fused_scores, exceed_counts >= vote_threshold, exceed_counts
    raise ValueError(f"Unsupported fusion rule: {fusion_rule}")


def build_sampling_plans(
    no_drift_len: int,
    drift_len: int,
    drift_ratios: List[float],
    window_size: int,
    windows_per_ratio: int,
    seed: int,
) -> Dict[float, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    plans: Dict[float, Dict[str, object]] = {}

    for ratio in drift_ratios:
        n_drift = int(round(window_size * ratio))
        n_clean = window_size - n_drift
        clean_batches = []
        drift_batches = []
        permutations = []

        for _ in range(windows_per_ratio):
            clean_indices = rng.choice(no_drift_len, size=n_clean, replace=no_drift_len < n_clean) if n_clean > 0 else np.array([], dtype=np.int64)
            drift_indices = rng.choice(drift_len, size=n_drift, replace=drift_len < n_drift) if n_drift > 0 else np.array([], dtype=np.int64)
            clean_batches.append(clean_indices)
            drift_batches.append(drift_indices)
            permutations.append(rng.permutation(window_size))

        plans[float(ratio)] = {
            "n_clean": int(n_clean),
            "n_drift": int(n_drift),
            "clean_batches": clean_batches,
            "drift_batches": drift_batches,
            "permutations": permutations,
        }

    return plans


def prepare_stream_data(
    features_dir: Path,
    stream_names: List[str],
    pca_dim: int,
    window_size: int,
    calibration_windows: int,
    cov_eps: float,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    prepared: Dict[str, Dict[str, object]] = {}
    stream_iterator = tqdm(stream_names, total=len(stream_names), desc=f"PCA {pca_dim}: prepare streams", mininterval=1.0)
    for stream in stream_iterator:
        reference = load_feature_matrix(features_dir, "0_Reference_Fit", stream)
        calibration = load_feature_matrix(features_dir, "1_Threshold_Cal", stream)
        no_drift = load_feature_matrix(features_dir, "2_Stream_Sim_NoDrift", stream)
        drift = load_feature_matrix(features_dir, "2_Stream_Sim_Drift", stream)

        detector = MahalanobisDetector(
            pca_dim=pca_dim,
            window_size=window_size,
            calibration_windows=calibration_windows,
            calibration_quantile=0.99,
            cov_eps=cov_eps,
            random_seed=seed,
        )
        detector.fit_reference(reference)
        detector.calibrate_threshold(calibration, progress_label=f"PCA {pca_dim}/{stream}: calibrate")

        prepared[stream] = {
            "detector": detector,
            "calibration_scores": detector.calibration_scores.copy() if detector.calibration_scores is not None else None,
            "actual_pca_dim": int(detector.actual_pca_dim),
            "no_drift_projected": detector.transform(no_drift),
            "drift_projected": detector.transform(drift),
        }
    return prepared


def precompute_stream_ratio_scores(
    stream_data: Dict[str, Dict[str, object]],
    stream_names: List[str],
    plans: Dict[float, Dict[str, object]],
) -> Dict[str, Dict[float, np.ndarray]]:
    raw_scores: Dict[str, Dict[float, np.ndarray]] = {stream: {} for stream in stream_names}

    stream_iterator = tqdm(stream_names, total=len(stream_names), desc="Precompute stream scores", mininterval=1.0)
    for stream in stream_iterator:
        detector = stream_data[stream]["detector"]
        no_drift_projected = stream_data[stream]["no_drift_projected"]
        drift_projected = stream_data[stream]["drift_projected"]

        ratio_iterator = tqdm(plans.items(), total=len(plans), desc=f"{stream}: ratios", leave=False, mininterval=1.0)
        for ratio, plan in ratio_iterator:
            clean_batches = plan["clean_batches"]
            drift_batches = plan["drift_batches"]
            permutations = plan["permutations"]
            scores = np.empty(len(permutations), dtype=np.float64)

            window_iterator = tqdm(
                enumerate(permutations),
                total=len(permutations),
                desc=f"{stream} ratio={float(ratio):.2f}",
                leave=False,
                mininterval=1.0,
            )
            for index, permutation in window_iterator:
                parts = []
                if plan["n_clean"] > 0:
                    parts.append(no_drift_projected[clean_batches[index]])
                if plan["n_drift"] > 0:
                    parts.append(drift_projected[drift_batches[index]])
                window = np.concatenate(parts, axis=0)[permutation]
                scores[index] = detector.score_projected_window(window)

            raw_scores[stream][ratio] = scores

    return raw_scores


def evaluate_group(
    raw_scores_by_stream: Dict[str, Dict[float, np.ndarray]],
    thresholds_by_stream: Dict[str, float],
    streams: List[str],
    drift_ratios: List[float],
    plans: Dict[float, Dict[str, object]],
    fusion_rule: str,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []

    for ratio in drift_ratios:
        normalized_matrix = np.stack(
            [raw_scores_by_stream[stream][ratio] / max(thresholds_by_stream[stream], 1e-12) for stream in streams],
            axis=0,
        )
        fused_scores, detected_mask, exceed_counts = apply_fusion_rule_matrix(normalized_matrix, fusion_rule)
        plan = plans[ratio]
        results.append(
            {
                "drift_ratio": float(ratio),
                "windows": int(fused_scores.shape[0]),
                "n_clean": int(plan["n_clean"]),
                "n_drift": int(plan["n_drift"]),
                "detection_rate": float(np.mean(detected_mask)),
                "accuracy": float(1.0 - np.mean(detected_mask)) if ratio == 0.0 else float(np.mean(detected_mask)),
                "metric_name": "accuracy",
                "mean_rule_score": float(fused_scores.mean()),
                "std_rule_score": float(fused_scores.std(ddof=0)),
                "mean_exceed_count": float(exceed_counts.mean()),
            }
        )

    return results


def summarize_payload(payload: Dict[str, object]) -> Dict[str, object]:
    evaluation = payload["evaluation"]
    fpr = next(float(item["detection_rate"]) for item in evaluation if float(item["drift_ratio"]) == 0.0)
    no_drift_accuracy = 1.0 - fpr
    accuracy_by_ratio = [float(item["accuracy"]) for item in evaluation]
    positive_accuracies = [float(item["accuracy"]) for item in evaluation if float(item["drift_ratio"]) > 0.0]
    positive_ratios = [float(item["drift_ratio"]) for item in evaluation if float(item["drift_ratio"]) > 0.0]
    mean_accuracy = float(np.mean(accuracy_by_ratio)) if accuracy_by_ratio else 0.0
    weighted_accuracy = float(np.average(positive_accuracies, weights=positive_ratios)) if positive_accuracies else no_drift_accuracy
    return {
        "feature_group": payload["feature_group"],
        "fusion_rule": payload["fusion_rule"],
        "requested_pca_dim": payload["requested_pca_dim"],
        "actual_pca_dims": payload["actual_pca_dims"],
        "calibration_quantile": payload["calibration_quantile"],
        "fpr": fpr,
        "no_drift_accuracy": no_drift_accuracy,
        "mean_accuracy": mean_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "score": mean_accuracy,
        "evaluation": evaluation,
    }


def save_dashboard(path: Path, rows: List[Dict[str, object]]) -> None:
    top_rows = rows[:10]
    labels = [f"{feature_group_display_name(str(row['feature_group']))}\n{row['fusion_rule']}\nq={row['calibration_quantile']:.2f}" for row in top_rows]
    scores = [float(row["score"]) for row in top_rows]
    no_drift_accuracies = [float(row["no_drift_accuracy"]) for row in top_rows]
    weighted_accuracies = [float(row["weighted_accuracy"]) for row in top_rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bars = axes[0].bar(labels, scores, color="#1f77b4")
    axes[0].set_title("Macro")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    bars = axes[1].bar(labels, weighted_accuracies, color="#2ca02c")
    axes[1].set_title("Weighted Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, weighted_accuracies):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    bars = axes[2].bar(labels, no_drift_accuracies, color="#ff7f0e")
    axes[2].set_title("Accuracy @ 0% Drift")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, no_drift_accuracies):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    features_dir = PROJECT_ROOT / args.features_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_payloads: List[Dict[str, object]] = []
    stream_names = sorted({stream for group in args.feature_groups for stream in group.split("+")})
    pca_iterator = tqdm(args.pca_dims, total=len(args.pca_dims), desc="PCA sweep", mininterval=1.0)
    for pca_dim in pca_iterator:
        stream_data = prepare_stream_data(
            features_dir=features_dir,
            stream_names=stream_names,
            pca_dim=pca_dim,
            window_size=args.window_size,
            calibration_windows=args.calibration_windows,
            cov_eps=args.cov_eps,
            seed=args.seed,
        )

        reference_stream = stream_names[0]
        plans = build_sampling_plans(
            no_drift_len=int(stream_data[reference_stream]["no_drift_projected"].shape[0]),
            drift_len=int(stream_data[reference_stream]["drift_projected"].shape[0]),
            drift_ratios=args.drift_ratios,
            window_size=args.window_size,
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

        quantile_iterator = tqdm(args.quantiles, total=len(args.quantiles), desc=f"PCA {pca_dim}: quantiles", leave=False, mininterval=1.0)
        for quantile in quantile_iterator:
            group_iterator = tqdm(args.feature_groups, total=len(args.feature_groups), desc=f"PCA {pca_dim} q={quantile:.2f}: groups", leave=False, mininterval=1.0)
            for group in group_iterator:
                streams = group.split("+")
                rules = ["single"] if len(streams) == 1 else args.fusion_rules
                rule_iterator = tqdm(rules, total=len(rules), desc=f"{group}: rules", leave=False, mininterval=1.0)
                for rule in rule_iterator:
                    evaluation = evaluate_group(
                        raw_scores_by_stream=raw_scores_by_stream,
                        thresholds_by_stream=thresholds_by_quantile[float(quantile)],
                        streams=streams,
                        drift_ratios=args.drift_ratios,
                        plans=plans,
                        fusion_rule=rule,
                    )
                    payload = {
                        "feature_group": group,
                        "fusion_rule": rule,
                        "streams": streams,
                        "window_size": args.window_size,
                        "requested_pca_dim": pca_dim,
                        "actual_pca_dims": {stream: int(stream_data[stream]["actual_pca_dim"]) for stream in streams},
                        "calibration_windows": args.calibration_windows,
                        "calibration_quantile": quantile,
                        "stream_thresholds": {stream: float(thresholds_by_quantile[float(quantile)][stream]) for stream in streams},
                        "evaluation": evaluation,
                    }
                    all_payloads.append(payload)

    summary_rows = [summarize_payload(payload) for payload in all_payloads]
    summary_rows.sort(key=lambda item: (-float(item["score"]), -float(item["weighted_accuracy"]), -float(item["no_drift_accuracy"])))

    (output_dir / "mahalanobis_fusion_search_full.json").write_text(
        json.dumps(all_payloads, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "mahalanobis_fusion_search_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "mahalanobis_fusion_search_summary.md", summary_rows)
    save_dashboard(output_dir / "mahalanobis_fusion_search_dashboard.png", summary_rows)

    best = summary_rows[0]
    print(f"Best group: {best['feature_group']}")
    print(f"Best rule: {best['fusion_rule']}")
    print(f"Best quantile: {best['calibration_quantile']:.2f}")
    print(f"Best score: {best['score']:.4f}")
    print(f"Search summary JSON: {output_dir / 'mahalanobis_fusion_search_summary.json'}")
    print(f"Search summary Markdown: {output_dir / 'mahalanobis_fusion_search_summary.md'}")
    print(f"Search dashboard: {output_dir / 'mahalanobis_fusion_search_dashboard.png'}")


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Feature Fusion Sensitivity Summary",
        "",
        "| Rank | Group | Rule | PCA | Quantile | Macro (%) | Weighted Accuracy (%) | Accuracy@0% (%) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:20], start=1):
        lines.append(
            "| {rank} | {group} | {rule} | {pca} | {quantile:.2f} | {mean_accuracy:.2f} | {weighted_accuracy:.2f} | {no_drift_accuracy:.2f} |".format(
                rank=idx,
                group=feature_group_display_name(str(row["feature_group"])),
                rule=row["fusion_rule"],
                pca=int(row["requested_pca_dim"]),
                quantile=float(row["calibration_quantile"]),
                mean_accuracy=float(row["mean_accuracy"]) * 100.0,
                weighted_accuracy=float(row["weighted_accuracy"]) * 100.0,
                no_drift_accuracy=float(row["no_drift_accuracy"]) * 100.0,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
