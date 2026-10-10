#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

for env_name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(env_name, "16")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import cramervonmises_2samp, ks_2samp

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
METHOD_DISPLAY_NAMES = {
    "Mahalanobis": "SADA",
    "DriftLens (Frechet)": "Global Fréchet",
}


def method_display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method)


def feature_group_display_name(group: str) -> str:
    return {"A": "V", "E_text": "VT", "A+E_text": "V+VT"}.get(group, group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SADA feature fusion against statistical drift-detection baselines.")
    parser.add_argument("--features-dir", type=str, default="features_pooled")
    parser.add_argument("--feature-group", type=str, default="A+D")
    parser.add_argument("--fusion-rule", type=str, default="mean_norm", choices=["single", "max_norm", "mean_norm", "top2_mean_norm", "two_vote"])
    parser.add_argument("--pca-dim", type=int, default=192)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--calibration-windows", type=int, default=10000)
    parser.add_argument("--calibration-quantile", type=float, default=0.99)
    parser.add_argument("--windows-per-ratio", type=int, default=200)
    parser.add_argument("--drift-ratios", type=float, nargs="+", default=DEFAULT_DRIFT_RATIOS)
    parser.add_argument("--kernel-ref-size", type=int, default=512)
    parser.add_argument("--lsdd-centers", type=int, default=64)
    parser.add_argument("--lsdd-reg", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=str, default="results_detection/baseline_comparison_best_mahalanobis_20260401")
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


def compute_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    mean_diff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    cov_term = np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(mean_diff + cov_term)


def estimate_rbf_gamma(reference_subset: np.ndarray) -> float:
    sample = reference_subset[: min(256, reference_subset.shape[0])]
    sq_norms = np.sum(sample * sample, axis=1, keepdims=True)
    sq_dists = np.maximum(sq_norms + sq_norms.T - 2.0 * sample @ sample.T, 0.0)
    tri = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    positive = tri[tri > 0]
    median_sq = float(np.median(positive)) if len(positive) > 0 else 1.0
    return 1.0 / max(2.0 * median_sq, 1e-12)


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    sq_dists = np.maximum(x_sq + y_sq - 2.0 * x @ y.T, 0.0)
    return np.exp(-gamma * sq_dists)


class MahalanobisScorer:
    def __init__(self, reference: np.ndarray, cov_eps: float) -> None:
        self.mu = reference.mean(axis=0)
        sigma = np.cov(reference, rowvar=False) + np.eye(reference.shape[1]) * cov_eps
        self.sigma_inv = np.linalg.pinv(sigma)

    def __call__(self, window: np.ndarray) -> float:
        diff = window.mean(axis=0) - self.mu
        sq = float(diff @ self.sigma_inv @ diff)
        return float(np.sqrt(max(sq, 0.0)))


class FrechetScorer:
    def __init__(self, reference: np.ndarray, cov_eps: float) -> None:
        self.mu = reference.mean(axis=0)
        self.sigma = np.cov(reference, rowvar=False) + np.eye(reference.shape[1]) * cov_eps
        self.cov_eps = cov_eps

    def __call__(self, window: np.ndarray) -> float:
        mu_w = window.mean(axis=0)
        sigma_w = np.cov(window, rowvar=False) + np.eye(window.shape[1]) * self.cov_eps
        return compute_frechet_distance(self.mu, self.sigma, mu_w, sigma_w)


class MMDScorer:
    def __init__(self, reference: np.ndarray, gamma: float) -> None:
        self.reference = reference
        self.gamma = gamma
        self.k_xx_mean = float(rbf_kernel(reference, reference, gamma).mean())

    def __call__(self, window: np.ndarray) -> float:
        k_yy = rbf_kernel(window, window, self.gamma)
        k_xy = rbf_kernel(self.reference, window, self.gamma)
        return float(self.k_xx_mean + float(k_yy.mean()) - 2.0 * float(k_xy.mean()))


class KSScorer:
    def __init__(self, reference: np.ndarray) -> None:
        self.reference = reference

    def __call__(self, window: np.ndarray) -> float:
        stats = [ks_2samp(self.reference[:, dim], window[:, dim], method="auto").statistic for dim in range(window.shape[1])]
        return float(max(stats))


class CVMScorer:
    def __init__(self, reference: np.ndarray) -> None:
        self.reference = reference

    def __call__(self, window: np.ndarray) -> float:
        stats = [cramervonmises_2samp(self.reference[:, dim], window[:, dim]).statistic for dim in range(window.shape[1])]
        return float(max(stats))


class LSDDScorer:
    def __init__(self, reference: np.ndarray, gamma: float, centers: np.ndarray, reg: float) -> None:
        self.gamma = gamma
        self.centers = centers
        self.reg = reg
        self.k_ref = rbf_kernel(reference, centers, gamma)
        self.h_ref = self.k_ref.mean(axis=0)
        self.h_x = (self.k_ref.T @ self.k_ref) / reference.shape[0]

    def __call__(self, window: np.ndarray) -> float:
        k_win = rbf_kernel(window, self.centers, self.gamma)
        h = self.h_ref - k_win.mean(axis=0)
        h_y = (k_win.T @ k_win) / window.shape[0]
        h_mat = 0.5 * (self.h_x + h_y)
        alpha = np.linalg.solve(h_mat + self.reg * np.eye(h_mat.shape[0]), h)
        score = float(h @ alpha - 0.5 * alpha @ h_mat @ alpha)
        return max(score, 0.0)


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


def build_calibration_index_batches(pool_size: int, window_size: int, n_windows: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    replace = pool_size < window_size
    return [rng.choice(pool_size, size=window_size, replace=replace) for _ in range(n_windows)]


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


def make_scorers(projected_ref: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> Dict[str, Callable[[np.ndarray], float]]:
    kernel_ref_size = min(args.kernel_ref_size, projected_ref.shape[0])
    kernel_indices = rng.choice(projected_ref.shape[0], size=kernel_ref_size, replace=False)
    kernel_ref = projected_ref[kernel_indices]
    gamma = estimate_rbf_gamma(kernel_ref)
    center_count = min(args.lsdd_centers, kernel_ref.shape[0])
    centers = kernel_ref[rng.choice(kernel_ref.shape[0], size=center_count, replace=False)]

    return {
        "Mahalanobis": MahalanobisScorer(projected_ref, args.cov_eps),
        "DriftLens (Frechet)": FrechetScorer(projected_ref, args.cov_eps),
        "MMD": MMDScorer(kernel_ref, gamma),
        "KS": KSScorer(kernel_ref),
        "LSDD": LSDDScorer(kernel_ref, gamma, centers, args.lsdd_reg),
        "CVM": CVMScorer(kernel_ref),
    }


def build_stream_method_data(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, object]]]:
    features_dir = PROJECT_ROOT / args.features_dir
    rng = np.random.default_rng(args.seed)
    stream_method_data: Dict[str, Dict[str, Dict[str, object]]] = {}

    stream_names = args.feature_group.split("+")
    stream_iterator = tqdm(stream_names, total=len(stream_names), desc="Prepare streams", mininterval=1.0)
    for stream in stream_iterator:
        reference = load_feature_matrix(features_dir, "0_Reference_Fit", stream)
        calibration = load_feature_matrix(features_dir, "1_Threshold_Cal", stream)
        no_drift = load_feature_matrix(features_dir, "2_Stream_Sim_NoDrift", stream)
        drift = load_feature_matrix(features_dir, "2_Stream_Sim_Drift", stream)

        detector = MahalanobisDetector(
            pca_dim=args.pca_dim,
            window_size=args.window_size,
            calibration_windows=args.calibration_windows,
            calibration_quantile=args.calibration_quantile,
            cov_eps=args.cov_eps,
            random_seed=args.seed,
        )
        detector.fit_reference(reference)
        projected_ref = detector.transform(reference)
        projected_cal = detector.transform(calibration)
        projected_nodrift = detector.transform(no_drift)
        projected_drift = detector.transform(drift)
        calibration_batches = build_calibration_index_batches(projected_cal.shape[0], args.window_size, args.calibration_windows, args.seed)

        method_scorers = make_scorers(projected_ref, args, rng)
        stream_method_data[stream] = {}

        method_iterator = tqdm(
            method_scorers.items(),
            total=len(method_scorers),
            desc=f"{stream}: methods",
            leave=False,
            mininterval=1.0,
        )
        for method_name, scorer in method_iterator:
            calibration_scores = np.empty(args.calibration_windows, dtype=np.float64)
            calibration_iterator = tqdm(
                enumerate(calibration_batches),
                total=len(calibration_batches),
                desc=f"{stream}/{method_name}: calibrate",
                leave=False,
                mininterval=1.0,
            )
            for index, batch_indices in calibration_iterator:
                calibration_scores[index] = scorer(projected_cal[batch_indices])

            stream_method_data[stream][method_name] = {
                "threshold": float(np.quantile(calibration_scores, args.calibration_quantile)),
                "scorer": scorer,
                "actual_pca_dim": int(detector.actual_pca_dim),
                "no_drift_projected": projected_nodrift,
                "drift_projected": projected_drift,
            }

    return stream_method_data


def evaluate_method(
    stream_method_data: Dict[str, Dict[str, Dict[str, object]]],
    method_name: str,
    streams: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    evaluation: List[Dict[str, float]] = []
    reference_stream = streams[0]
    no_drift_len = int(stream_method_data[reference_stream][method_name]["no_drift_projected"].shape[0])
    drift_len = int(stream_method_data[reference_stream][method_name]["drift_projected"].shape[0])
    plans = build_sampling_plans(
        no_drift_len=no_drift_len,
        drift_len=drift_len,
        drift_ratios=args.drift_ratios,
        window_size=args.window_size,
        windows_per_ratio=args.windows_per_ratio,
        seed=args.seed,
    )

    raw_scores_by_stream: Dict[str, Dict[float, np.ndarray]] = {stream: {} for stream in streams}
    stream_iterator = tqdm(streams, total=len(streams), desc=f"{method_name}: streams", leave=False, mininterval=1.0)
    for stream in stream_iterator:
        method_data = stream_method_data[stream][method_name]
        ratio_iterator = tqdm(plans.items(), total=len(plans), desc=f"{method_name}/{stream}: ratios", leave=False, mininterval=1.0)
        for ratio, plan in ratio_iterator:
            scores = np.empty(args.windows_per_ratio, dtype=np.float64)
            window_iterator = tqdm(
                enumerate(plan["permutations"]),
                total=len(plan["permutations"]),
                desc=f"{method_name}/{stream} ratio={float(ratio):.2f}",
                leave=False,
                mininterval=1.0,
            )
            for index, permutation in window_iterator:
                parts = []
                if plan["n_clean"] > 0:
                    parts.append(method_data["no_drift_projected"][plan["clean_batches"][index]])
                if plan["n_drift"] > 0:
                    parts.append(method_data["drift_projected"][plan["drift_batches"][index]])
                window = np.concatenate(parts, axis=0)[permutation]
                scores[index] = float(method_data["scorer"](window))
            raw_scores_by_stream[stream][ratio] = scores

    for ratio in args.drift_ratios:
        normalized_matrix = np.stack(
            [
                raw_scores_by_stream[stream][ratio] / max(float(stream_method_data[stream][method_name]["threshold"]), 1e-12)
                for stream in streams
            ],
            axis=0,
        )
        fused_scores, detected_mask, exceed_counts = apply_fusion_rule_matrix(normalized_matrix, args.fusion_rule)
        plan = plans[ratio]
        evaluation.append(
            {
                "drift_ratio": float(ratio),
                "windows": int(args.windows_per_ratio),
                "n_clean": int(plan["n_clean"]),
                "n_drift": int(plan["n_drift"]),
                "detection_rate": float(np.mean(detected_mask)),
                "accuracy": float(1.0 - np.mean(detected_mask)) if ratio == 0.0 else float(np.mean(detected_mask)),
                "metric_name": "accuracy",
                "mean_rule_score": float(fused_scores.mean()),
                "std_rule_score": float(fused_scores.std(ddof=0)),
            }
        )

    fpr = next(float(item["detection_rate"]) for item in evaluation if float(item["drift_ratio"]) == 0.0)
    no_drift_accuracy = 1.0 - fpr
    accuracies = [float(item["accuracy"]) for item in evaluation]
    positive_accuracies = [float(item["accuracy"]) for item in evaluation if float(item["drift_ratio"]) > 0.0]
    positive_ratios = [float(item["drift_ratio"]) for item in evaluation if float(item["drift_ratio"]) > 0.0]
    return {
        "method": method_display_name(method_name),
        "feature_group": args.feature_group,
        "fusion_rule": args.fusion_rule,
        "requested_pca_dim": args.pca_dim,
        "actual_pca_dims": {stream: int(stream_method_data[stream][method_name]["actual_pca_dim"]) for stream in streams},
        "calibration_quantile": args.calibration_quantile,
        "fpr": fpr,
        "no_drift_accuracy": no_drift_accuracy,
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "weighted_accuracy": float(np.average(positive_accuracies, weights=positive_ratios)) if positive_accuracies else no_drift_accuracy,
        "score": float(np.mean(accuracies)) if accuracies else 0.0,
        "stream_thresholds": {stream: float(stream_method_data[stream][method_name]["threshold"]) for stream in streams},
        "evaluation": evaluation,
    }


def save_plots(path: Path, rows: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_ratios = sorted({float(item["drift_ratio"]) for row in rows for item in row["evaluation"]})

    uses_accuracy = any("accuracy" in item for row in rows for item in row["evaluation"] if float(item["drift_ratio"]) > 0.0)

    def map_ratio_to_plot_x(ratio: float) -> float:
        if ratio <= 0.20:
            return ratio * 3.5
        return 0.70 + (ratio - 0.20) * 0.75

    def point_value(item: Dict[str, object]) -> float:
        ratio = float(item["drift_ratio"])
        if ratio == 0.0:
            return float(item["detection_rate"])
        if "accuracy" in item:
            return float(item["accuracy"])
        return float(item["detection_rate"])

    def no_drift_accuracy(row: Dict[str, object]) -> float:
        if "no_drift_accuracy" in row:
            return float(row["no_drift_accuracy"])
        return 1.0 - float(row["fpr"])

    x_ticks = [map_ratio_to_plot_x(ratio) for ratio in all_ratios]
    tick_labels = [f"{ratio:.2f}" if index % 2 == 0 else f"\n{ratio:.2f}" for index, ratio in enumerate(all_ratios)]

    for row in rows:
        method = method_display_name(str(row["method"]))
        ratios = [float(item["drift_ratio"]) for item in row["evaluation"]]
        x_values = [map_ratio_to_plot_x(ratio) for ratio in ratios]
        y_values = [point_value(item) for item in row["evaluation"]]
        axes[0].plot(x_values, y_values, marker="o", linewidth=2, label=method)
        for x_value, y_value in zip(x_values, y_values):
            axes[0].text(x_value, y_value + 0.012, f"{y_value:.3f}", ha="center", va="bottom", fontsize=8)

    labels = [method_display_name(str(row["method"])) for row in rows]
    no_drift_accuracies = [no_drift_accuracy(row) for row in rows]
    bars = axes[1].bar(labels, no_drift_accuracies, color="#ff7f0e")
    for bar, value in zip(bars, no_drift_accuracies):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_xlabel("Drift Injection Ratio")
    axes[0].set_ylabel("FPR @ 0%, Accuracy Otherwise" if uses_accuracy else "FPR @ 0%, Recall Otherwise")
    axes[0].set_title("Baseline Performance by Drift Ratio")
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(tick_labels, fontsize=8)
    axes[0].set_xlim(min(x_ticks) - 0.04, max(x_ticks) + 0.04)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Baseline Accuracy @ 0% Drift")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_markdown(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        f"# {feature_group_display_name(args.feature_group)} Method Comparison",
        "",
        f"Config: group={feature_group_display_name(args.feature_group)}, fusion={args.fusion_rule}, PCA={args.pca_dim}, window={args.window_size}, quantile={args.calibration_quantile}",
        "",
        "Macro is the arithmetic mean over the evaluated drift ratios; accuracy at 0% drift equals 1 minus FPR, and accuracy at positive drift ratios equals recall.",
        "",
        "| Method | Macro (%) | Weighted Accuracy (%) | Accuracy@0% (%) | Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {mean_accuracy:.2f} | {weighted_accuracy:.2f} | {no_drift_accuracy:.2f} | {score:.4f} |".format(
                method=method_display_name(str(row["method"])),
                mean_accuracy=float(row["mean_accuracy"]) * 100.0,
                weighted_accuracy=float(row["weighted_accuracy"]) * 100.0,
                no_drift_accuracy=float(row["no_drift_accuracy"]) * 100.0,
                score=float(row["score"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stream_method_data = build_stream_method_data(args)
    streams = args.feature_group.split("+")
    method_names = ["Mahalanobis", "MMD", "KS", "LSDD", "CVM", "DriftLens (Frechet)"]
    method_iterator = tqdm(method_names, total=len(method_names), desc="Compare methods", mininterval=1.0)
    rows = [evaluate_method(stream_method_data, method_name, streams, args) for method_name in method_iterator]
    rows.sort(key=lambda item: (-float(item["score"]), -float(item["weighted_accuracy"]), -float(item["no_drift_accuracy"])))

    (output_dir / "baseline_comparison_best_mahalanobis.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "baseline_comparison_best_mahalanobis.md", rows, args)
    save_plots(output_dir / "baseline_comparison_best_mahalanobis.png", rows)

    print(f"Comparison JSON: {output_dir / 'baseline_comparison_best_mahalanobis.json'}")
    print(f"Comparison Markdown: {output_dir / 'baseline_comparison_best_mahalanobis.md'}")
    print(f"Comparison Plot: {output_dir / 'baseline_comparison_best_mahalanobis.png'}")


if __name__ == "__main__":
    main()
