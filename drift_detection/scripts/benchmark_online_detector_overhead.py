#!/usr/bin/env python3
"""Benchmark the incremental CPU cost of SADA's online detector head."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
import torch
from sklearn.decomposition import PCA

from run_best_mahalanobis_baseline_comparison import make_scorers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("features_pooled_e_fusion"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results_detection/detector_online_overhead_cpu_20260726/benchmark.json"
        ),
    )
    parser.add_argument("--streams", nargs="+", default=["A", "E_text"])
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--projection-repeats", type=int, default=5000)
    parser.add_argument("--decision-repeats", type=int, default=5000)
    parser.add_argument("--batch-repeats", type=int, default=2000)
    parser.add_argument("--method-repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def latency_summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    reference = torch.load(
        args.features_dir / "0_Reference_Fit" / "features.pt",
        map_location="cpu",
        weights_only=False,
    )
    evaluation = torch.load(
        args.features_dir / "2_Stream_Sim_NoDrift" / "features.pt",
        map_location="cpu",
        weights_only=False,
    )

    models: dict[
        str,
        tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    raw_dim = None
    for stream in args.streams:
        ref = reference["features"][stream].float().numpy().astype(np.float32)
        test = evaluation["features"][stream].float().numpy().astype(np.float32)
        raw_dim = int(ref.shape[1])
        pca = PCA(
            n_components=args.pca_dim,
            svd_solver="randomized",
            random_state=args.seed,
        )
        projected = pca.fit_transform(ref)
        mean = projected.mean(axis=0)
        covariance = np.cov(projected, rowvar=False)
        covariance += np.eye(args.pca_dim, dtype=np.float64) * 1e-6
        models[stream] = (
            pca,
            mean,
            np.linalg.pinv(covariance),
            test,
            projected,
        )

    sample_count = models[args.streams[0]][3].shape[0]

    def project_sample(index: int) -> None:
        for stream in args.streams:
            pca, _, _, test, _ = models[stream]
            pca.transform(test[index : index + 1])

    for _ in range(100):
        project_sample(int(rng.integers(0, sample_count)))
    projection_times = []
    for _ in range(args.projection_repeats):
        index = int(rng.integers(0, sample_count))
        start = time.perf_counter()
        project_sample(index)
        projection_times.append(time.perf_counter() - start)

    projected_test = {
        stream: model[0].transform(model[3]) for stream, model in models.items()
    }
    window_indices = [
        rng.choice(sample_count, size=args.window_size, replace=False)
        for _ in range(max(args.decision_repeats, args.batch_repeats))
    ]

    def score_projected(indices: np.ndarray) -> None:
        scores = []
        for stream in args.streams:
            _, mean, inverse, _, _ = models[stream]
            difference = projected_test[stream][indices].mean(axis=0) - mean
            squared = float(difference @ inverse @ difference)
            scores.append(float(np.sqrt(max(squared, 0.0))))
        max(scores)

    for indices in window_indices[:100]:
        score_projected(indices)
    decision_times = []
    for indices in window_indices[: args.decision_repeats]:
        start = time.perf_counter()
        score_projected(indices)
        decision_times.append(time.perf_counter() - start)

    def score_unprojected(indices: np.ndarray) -> None:
        for stream in args.streams:
            pca, mean, inverse, test, _ = models[stream]
            projected = pca.transform(test[indices])
            difference = projected.mean(axis=0) - mean
            squared = float(difference @ inverse @ difference)
            float(np.sqrt(max(squared, 0.0)))

    for indices in window_indices[:50]:
        score_unprojected(indices)
    batch_times = []
    for indices in window_indices[: args.batch_repeats]:
        start = time.perf_counter()
        score_unprojected(indices)
        batch_times.append(time.perf_counter() - start)

    scorer_args = SimpleNamespace(
        kernel_ref_size=512,
        lsdd_centers=64,
        lsdd_reg=1e-3,
        cov_eps=1e-6,
    )
    scorer_rng = np.random.default_rng(args.seed)
    scorers = {
        stream: make_scorers(models[stream][4], scorer_args, scorer_rng)
        for stream in args.streams
    }
    method_names = [
        "Mahalanobis",
        "DriftLens (Frechet)",
        "LSDD",
        "MMD",
        "KS",
        "CVM",
    ]
    method_labels = {
        "Mahalanobis": "SADA",
        "DriftLens (Frechet)": "Global Fréchet",
        "LSDD": "LSDD",
        "MMD": "MMD",
        "KS": "KS",
        "CVM": "CVM",
    }
    method_windows = {
        stream: [
            projected_test[stream][indices]
            for indices in window_indices[: args.method_repeats]
        ]
        for stream in args.streams
    }
    method_latency = {}
    gc.disable()
    try:
        for method_name in method_names:
            def score_method(window_index: int) -> None:
                scores = [
                    float(scorers[stream][method_name](method_windows[stream][window_index]))
                    for stream in args.streams
                ]
                max(scores)

            for index in range(min(10, args.method_repeats)):
                score_method(index)
            samples = []
            for index in range(args.method_repeats):
                start = time.perf_counter()
                score_method(index)
                samples.append(time.perf_counter() - start)
            summary = latency_summary(samples)
            summary["windows_per_second"] = 1000.0 / summary["mean_ms"]
            method_latency[method_labels[method_name]] = summary
    finally:
        gc.enable()

    result = {
        "date": "2026-07-26",
        "hardware": f"single CPU thread, {cpu_model()}",
        "streams": args.streams,
        "raw_dim": raw_dim,
        "pca_dim_per_stream": args.pca_dim,
        "window_size": args.window_size,
        "repetitions": {
            "per_sample_projection": args.projection_repeats,
            "window_decision_projected": args.decision_repeats,
            "window_batch_path": args.batch_repeats,
            "matched_method_windows": args.method_repeats,
        },
        "per_sample_dual_stream_projection": latency_summary(projection_times),
        "window_end_dual_stream_decision_from_projected_buffer": latency_summary(
            decision_times
        ),
        "window_end_dual_stream_transform_and_decision_batch_path": latency_summary(
            batch_times
        ),
        "matched_projected_window_method_latency": method_latency,
        "matched_method_protocol": {
            "feature_group": "+".join(args.streams),
            "shared_pca_dim_per_stream": args.pca_dim,
            "shared_window_size": args.window_size,
            "shared_window_indices": True,
            "common_cost_excluded": (
                "VLM forward, hook capture, and per-sample PCA projection are "
                "shared by all methods."
            ),
            "method_specific_scope": (
                "Two stream scores plus max fusion, using the exact scorer "
                "classes from run_best_mahalanobis_baseline_comparison.py; "
                "one-time fitting and threshold calibration are excluded."
            ),
        },
        "scope": (
            "Detector-head CPU microbenchmark on cached features; excludes VLM "
            "forward time and does not measure hook-copy/pooling overhead against "
            "an uninstrumented forward."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
