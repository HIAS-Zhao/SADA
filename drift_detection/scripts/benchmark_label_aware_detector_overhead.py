#!/usr/bin/env python3
"""Benchmark detector-side CPU updates for the label-aware positive controls."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_label_aware_oracle_comparison import (  # noqa: E402
    empirical_cdf_transform,
    load_ordered_losses,
)
from src.label_aware_detectors import (  # noqa: E402
    ADWIN,
    DDM,
    EDDM,
    HDDMW,
    KSWIN,
    OPTWIN,
    Detector,
    PageHinkley,
    warm_detector,
)


METHODS = ["DDM", "EDDM", "ADWIN", "HDDM-W", "Page-Hinkley", "KSWIN", "OPTWIN"]
BINARY_METHODS = {"DDM", "EDDM", "HDDM-W"}
FACTORIES: dict[str, Callable[..., Detector]] = {
    "DDM": DDM,
    "EDDM": EDDM,
    "ADWIN": ADWIN,
    "HDDM-W": HDDMW,
    "Page-Hinkley": PageHinkley,
    "KSWIN": KSWIN,
    "OPTWIN": OPTWIN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loss-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_oracle_20260714"
            / "supervised_loss"
        ),
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=PROJECT_ROOT / "features_pooled_e_fusion",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "performance_degrading_change_20260714"
            / "performance_degrading_config.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_detector_overhead_cpu_20260726"
            / "benchmark.json"
        ),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--trace-count", type=int, default=300)
    parser.add_argument("--trace-length", type=int, default=200)
    parser.add_argument("--repeats-per-trace", type=int, default=10)
    parser.add_argument("--warmup-traces", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def latency_summary(samples: list[float], trace_length: int) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64) * 1000.0
    mean_ms = float(values.mean())
    return {
        "mean_ms_per_trace": mean_ms,
        "p50_ms_per_trace": float(np.percentile(values, 50)),
        "p95_ms_per_trace": float(np.percentile(values, 95)),
        "p99_ms_per_trace": float(np.percentile(values, 99)),
        "mean_us_per_update": mean_ms * 1000.0 / trace_length,
        "updates_per_second": trace_length * 1000.0 / mean_ms,
    }


def selected_configs(config: dict[str, Any], methods: list[str]) -> dict[str, dict[str, Any]]:
    audit = config["selected_detector_configs"]
    output = {}
    for method in methods:
        if method not in audit:
            raise KeyError(f"Missing selected configuration for {method}")
        method_config = dict(audit[method]["selected"]["config"])
        method_config.pop("implementation", None)
        output[method] = method_config
    return output


def build_traces(
    values: np.ndarray,
    trace_count: int,
    trace_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    replace = len(values) < trace_length
    indices = np.stack(
        [
            rng.choice(len(values), size=trace_length, replace=replace)
            for _ in range(trace_count)
        ],
        axis=0,
    )
    return indices, values[indices]


def benchmark_method(
    warmed: Detector,
    traces: np.ndarray,
    repeats_per_trace: int,
    warmup_traces: int,
) -> tuple[list[float], int]:
    for trace in traces[:warmup_traces]:
        detector = warmed.clone()
        for value in trace:
            detector.update(float(value))

    samples: list[float] = []
    alarmed_executions = 0
    gc.disable()
    try:
        for trace in traces:
            for _ in range(repeats_per_trace):
                detector = warmed.clone()
                alarmed = False
                start = time.perf_counter()
                for value in trace:
                    alarmed |= bool(detector.update(float(value)))
                samples.append(time.perf_counter() - start)
                alarmed_executions += int(alarmed)
    finally:
        gc.enable()
    return samples, alarmed_executions


def main() -> None:
    args = parse_args()
    if args.trace_count <= 0 or args.trace_length <= 0:
        raise ValueError("trace-count and trace-length must be positive")
    if args.repeats_per_trace <= 0:
        raise ValueError("repeats-per-trace must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    failure_threshold = float(config["degrading_cdf_threshold"])
    method_configs = selected_configs(config, args.methods)

    calibration_losses = load_ordered_losses(
        args.loss_dir,
        args.features_dir,
        "1_Threshold_Cal",
    )
    no_drift_losses = load_ordered_losses(
        args.loss_dir,
        args.features_dir,
        "2_Stream_Sim_NoDrift",
    )
    calibration_continuous = empirical_cdf_transform(
        calibration_losses,
        calibration_losses,
    )
    no_drift_continuous = empirical_cdf_transform(
        calibration_losses,
        no_drift_losses,
    )
    calibration_binary = (
        calibration_continuous >= failure_threshold
    ).astype(np.float64)
    no_drift_binary = (
        no_drift_continuous >= failure_threshold
    ).astype(np.float64)

    trace_indices, continuous_traces = build_traces(
        no_drift_continuous,
        trace_count=args.trace_count,
        trace_length=args.trace_length,
        seed=args.seed,
    )
    binary_traces = no_drift_binary[trace_indices]

    method_results = {}
    for method in args.methods:
        detector = FACTORIES[method](**method_configs[method])
        if method in BINARY_METHODS:
            calibration_signal = calibration_binary
            traces = binary_traces
            signal = "binary_high_loss_event"
        else:
            calibration_signal = calibration_continuous
            traces = continuous_traces
            signal = "calibration_ecdf_supervised_nll"
        warmed = warm_detector(detector, calibration_signal)
        samples, alarmed_executions = benchmark_method(
            warmed=warmed,
            traces=traces,
            repeats_per_trace=args.repeats_per_trace,
            warmup_traces=min(args.warmup_traces, len(traces)),
        )
        summary = latency_summary(samples, args.trace_length)
        summary.update(
            {
                "signal": signal,
                "config": method_configs[method],
                "timed_executions": len(samples),
                "executions_with_alarm": alarmed_executions,
                "alarm_execution_rate": alarmed_executions / len(samples),
            }
        )
        method_results[method] = summary

    result = {
        "date": "2026-07-27",
        "hardware": f"single CPU thread, {cpu_model()}",
        "protocol": {
            "source_experiment": str(args.config),
            "trace_pool": "2_Stream_Sim_NoDrift",
            "trace_count": args.trace_count,
            "trace_length": args.trace_length,
            "repeats_per_trace": args.repeats_per_trace,
            "shared_underlying_sample_indices": True,
            "sampling_with_replacement_within_trace": (
                len(no_drift_continuous) < args.trace_length
            ),
            "seed": args.seed,
            "high_loss_event_threshold": failure_threshold,
            "timed_scope": (
                "Detector update and alarm-state computation for all values in "
                "each trace, starting from the same method-specific warmed state."
            ),
            "excluded": [
                "ground-truth label acquisition",
                "supervised answer-NLL computation",
                "calibration-ECDF and high-loss-event construction",
                "detector configuration selection and calibration",
                "warmup and warmed-state cloning",
            ],
        },
        "methods": method_results,
        "scope": (
            "Detector-side CPU microbenchmark after the required supervised "
            "signal is available; not end-to-end label-aware detection latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
