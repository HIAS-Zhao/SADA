#!/usr/bin/env python3
"""Measure detector computation per 200 arrivals for the stride-10 experiment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_label_aware_detector_overhead import cpu_model, latency_summary  # noqa: E402
from run_mahalanobis_fusion_search import load_feature_matrix  # noqa: E402
from run_performance_degrading_early_recall import BINARY_METHODS, METHOD_ORDER, frozen_factories, sha256, write_json  # noqa: E402
from src.label_aware_detectors import warm_detector  # noqa: E402
from src.mahalanobis_detector import MahalanobisDetector  # noqa: E402


def benchmark_label(warmed, calibration_prefixes, traces, repeats):
    samples = []
    alarmed_executions = 0
    for trace_index, (prefix, trace) in enumerate(zip(calibration_prefixes, traces)):
        initial = warmed.clone()
        for value in prefix:
            if initial.update(float(value)):
                initial.reset()
        # One untimed pass per trace also warms the detector's computation path.
        for repeat in range(repeats + 1):
            detector = initial.clone()
            alarmed = False
            start = time.perf_counter()
            for value in trace:
                if detector.update(float(value)):
                    alarmed = True
                    detector.reset()
            elapsed = time.perf_counter() - start
            if repeat > 0:
                samples.append(elapsed)
                alarmed_executions += int(alarmed)
        if (trace_index + 1) % 100 == 0:
            print(f"  {trace_index + 1}/{len(traces)} traces", flush=True)
    return samples, alarmed_executions


def benchmark_sada(streams, thresholds, trial_indices, repeats, stride):
    samples = []
    alarmed_executions = 0
    for indices in trial_indices:
        windows = {
            stream: data["projected"][indices[100:]] for stream, data in streams.items()
        }
        for repeat in range(repeats + 1):
            alarmed = False
            start = time.perf_counter()
            for endpoint in range(200 + stride, 401, stride):
                normalized = [
                    data["detector"].score_projected_window(windows[stream][endpoint - 200:endpoint]) / thresholds[stream]
                    for stream, data in streams.items()
                ]
                alarmed |= max(normalized) > 1.0
            elapsed = time.perf_counter() - start
            if repeat > 0:
                samples.append(elapsed)
                alarmed_executions += int(alarmed)
    return samples, alarmed_executions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "features_pooled_e_fusion")
    parser.add_argument("--trace-count", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    config_path = args.experiment_dir / "early_recall_config.json"
    config = json.loads(config_path.read_text())
    if (config["clean_prefix"], config["sada"]["window_size"], config["sada_stride"]) != (500, 200, 10):
        raise ValueError("This benchmark is for the 500-sample, window-200, stride-10 protocol")
    if args.repeats < 1 or not 1 <= args.trace_count <= config["trials"]:
        raise ValueError("Invalid repeat or trace count")
    output_dir = args.experiment_dir / "runtime"
    output_dir.mkdir(exist_ok=False)
    indices_path = args.experiment_dir / "seed_42/trial_indices.npz"
    indices = np.load(indices_path)["fpr"][:args.trace_count]
    if len(indices) != args.trace_count:
        raise ValueError("Insufficient saved trials")
    signals = np.load(args.experiment_dir / "aligned_signals.npz")
    streams = {}
    for stream in ("A", "E_text"):
        calibration_dir = args.experiment_dir / "seed_42/sada_calibration" / stream
        pca = joblib.load(calibration_dir / "pca_model.joblib")
        with np.load(calibration_dir / "baseline_stats.npz") as stats:
            detector = MahalanobisDetector(128, 200, 10000, 0.99)
            detector.mu_base = stats["mu_base"].copy()
            detector.sigma_inv = stats["sigma_inv"].copy()
        streams[stream] = {"detector": detector, "projected": pca.transform(load_feature_matrix(args.features_dir, "2_Stream_Sim_NoDrift", stream))}
    thresholds = json.loads((args.experiment_dir / "seed_42/sada_thresholds.json").read_text())
    factories = frozen_factories(config)
    results, raw = {}, {}
    gc.disable()
    try:
        for method in METHOD_ORDER:
            print(f"Benchmark {method}: {len(indices)} traces x {args.repeats} repeats", flush=True)
            if method == "SADA":
                samples, alarmed = benchmark_sada(streams, thresholds, indices, args.repeats, config["sada_stride"])
            else:
                calibration = signals["calibration"]
                clean = signals["clean"]
                if method in BINARY_METHODS:
                    calibration = (calibration >= config["degrading_cdf_threshold"]).astype(np.uint8)
                    clean = (clean >= config["degrading_cdf_threshold"]).astype(np.uint8)
                warmed = warm_detector(factories[method](), calibration)
                samples, alarmed = benchmark_label(warmed, clean[indices[:, :300]], clean[indices[:, 300:]], args.repeats)
            results[method] = latency_summary(samples, 200)
            results[method].update(timed_executions=len(samples), executions_with_alarm=alarmed)
            raw[method] = np.asarray(samples)
            print(f"{method}: {results[method]['mean_ms_per_trace']:.6f} ms/200 arrivals", flush=True)
    finally:
        gc.enable()
    np.savez_compressed(output_dir / "timing_seconds.npz", **raw)
    write_json(output_dir / "benchmark.json", {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": f"single CPU thread, {cpu_model()}",
        "protocol": {
            "trace_source": str(indices_path.resolve()), "trace_source_sha256": sha256(indices_path),
            "trace_count": args.trace_count, "repeats_per_trace": args.repeats,
            "clean_prefix": 300, "timed_arrivals": 200, "sada_window": 200, "sada_stride": 10,
            "sada_calls_per_timed_execution": 20,
            "label_aware_updates_per_timed_execution": 200,
            "warmup": "Calibration warmup and each trace's first 300 clean samples; one untimed pass per trace",
            "alarm_handling": "All arrivals processed; label-aware detectors reset after alarms; SADA retains its rolling window",
            "included": "Projected-window scoring, fusion and comparisons for SADA; detector updates and alarm resets for label-aware methods",
            "excluded": "Input inference, labels/NLL/ECDF, PCA projection, trace materialization, calibration, warmup, state cloning",
            "source_sha256": sha256(Path(__file__)), "command": sys.argv,
        },
        "methods": results,
    })


if __name__ == "__main__":
    main()
