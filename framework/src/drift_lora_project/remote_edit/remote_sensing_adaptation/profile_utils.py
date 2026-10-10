from __future__ import annotations

import ast
import math
import re
from typing import Any


LAMBDA_PROFILE_FIELDS = (
    "lambda_id",
    "image_resolution",
    "batch_size",
    "precision",
    "max_new_tokens",
    "token_retention_ratio",
    "roi_strategy",
    "throughput_samples_per_s",
    "peak_vram_gb",
    "avg_power_w",
    "thermal_level",
    "accuracy",
)

GAMMA_PROFILE_FIELDS = (
    "gamma_id",
    "method",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "learning_rate",
    "batch_size",
    "train_batch_size",
    "grad_accum",
    "grad_accum_steps",
    "train_steps",
    "microprofile_train_steps",
    "target_train_steps",
    "train_data_fraction",
    "train_time_s",
    "train_gpu_seconds",
    "resource_cost",
    "peak_vram_gb",
    "avg_power_w",
    "thermal_level",
    "profile_gain",
    "predicted_final_gain",
    "predicted_strict_acc_gain",
    "expected_acc_gain",
    "quant_bits",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_quant_type",
    "bnb_4bit_use_double_quant",
)


def parse_duration_seconds(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
        number = float(value)
        if unit == "h":
            total += number * 3600
        elif unit == "m":
            total += number * 60
        else:
            total += number
    return total


def resource_cost_from_throughput(stream_rate: float, throughput: float) -> float:
    if stream_rate < 0:
        raise ValueError("stream_rate must be non-negative")
    if throughput <= 0:
        return float("inf")
    return stream_rate / throughput


def lambda_dtype_for_precision(precision: str) -> str:
    lowered = precision.lower()
    if lowered in {"fp32", "float32"}:
        return "float32"
    if lowered in {"bf16", "bfloat16", "fp4"}:
        return "bfloat16"
    raise ValueError(f"Only fp32/bf16/fp4 are allowed in this profiler, got {precision!r}")


def lambda_quant_for_precision(precision: str) -> str:
    return "fp4" if precision.lower() == "fp4" else "none"


def lambda_profile_from_measurement(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "precision" not in normalized and "model_precision" in normalized:
        normalized["precision"] = normalized["model_precision"]
    if "throughput_samples_per_s" not in normalized and "throughput_samples_per_sec" in normalized:
        normalized["throughput_samples_per_s"] = normalized["throughput_samples_per_sec"]
    if "peak_vram_gb" not in normalized and "peak_memory_gib" in normalized:
        normalized["peak_vram_gb"] = normalized["peak_memory_gib"]
    if "avg_power_w" not in normalized:
        if "avg_power_watts" in normalized:
            normalized["avg_power_w"] = normalized["avg_power_watts"]
        elif "power_watts" in normalized:
            normalized["avg_power_w"] = normalized["power_watts"]
    if "thermal_level" not in normalized and "thermal_score" in normalized:
        normalized["thermal_level"] = normalized["thermal_score"]
    if "thermal_level" not in normalized:
        if "peak_temperature_c" in normalized:
            normalized["thermal_level"] = normalized["peak_temperature_c"]
        elif "avg_temperature_c" in normalized:
            normalized["thermal_level"] = normalized["avg_temperature_c"]

    profile: dict[str, Any] = {}
    for key in LAMBDA_PROFILE_FIELDS:
        if key in normalized:
            profile[key] = normalized[key]
    profile.setdefault("token_retention_ratio", 1.0)
    profile.setdefault("roi_strategy", "none")
    profile.setdefault("peak_vram_gb", None)
    profile.setdefault("avg_power_w", None)
    profile.setdefault("thermal_level", None)
    return profile


def gamma_profile_from_measurement(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "train_batch_size" not in normalized and "batch_size" in normalized:
        normalized["train_batch_size"] = normalized["batch_size"]
    if "grad_accum_steps" not in normalized and "grad_accum" in normalized:
        normalized["grad_accum_steps"] = normalized["grad_accum"]
    if "train_time_s" not in normalized and "train_time" in normalized:
        normalized["train_time_s"] = normalized["train_time"]
    if "train_gpu_seconds" not in normalized and "train_time_s" in normalized:
        normalized["train_gpu_seconds"] = normalized["train_time_s"]
    if "peak_vram_gb" not in normalized:
        if "measured_peak_memory_gib" in normalized:
            normalized["peak_vram_gb"] = normalized["measured_peak_memory_gib"]
        elif "peak_memory_gib" in normalized:
            normalized["peak_vram_gb"] = normalized["peak_memory_gib"]
    if "avg_power_w" not in normalized:
        if "avg_power_watts" in normalized:
            normalized["avg_power_w"] = normalized["avg_power_watts"]
        elif "power_watts" in normalized:
            normalized["avg_power_w"] = normalized["power_watts"]
    if "thermal_level" not in normalized and "thermal_score" in normalized:
        normalized["thermal_level"] = normalized["thermal_score"]
    if "thermal_level" not in normalized:
        if "peak_temperature_c" in normalized:
            normalized["thermal_level"] = normalized["peak_temperature_c"]
        elif "avg_temperature_c" in normalized:
            normalized["thermal_level"] = normalized["avg_temperature_c"]
    if "expected_acc_gain" not in normalized:
        if "predicted_final_gain" in normalized:
            normalized["expected_acc_gain"] = normalized["predicted_final_gain"]
        elif "profile_gain" in normalized:
            normalized["expected_acc_gain"] = normalized["profile_gain"]

    profile: dict[str, Any] = {}
    for key in GAMMA_PROFILE_FIELDS:
        if key in normalized:
            profile[key] = normalized[key]
    profile.setdefault("peak_vram_gb", None)
    profile.setdefault("avg_power_w", None)
    profile.setdefault("thermal_level", None)
    return profile


def accuracy_from_report(report: dict[str, Any]) -> float:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("accuracy report must contain a rows list")
    total = 0
    correct = 0
    for row in rows:
        total += int(row.get("total") or 0)
        correct += int(row.get("correct") or 0)
    if total <= 0:
        raise ValueError("accuracy report has no samples")
    return correct / total


def parse_swift_metric_dicts(log_text: str) -> dict[str, float]:
    metric_rows: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            row = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if isinstance(row, dict):
            metric_rows.append(row)

    train_runtime = math.nan
    train_speed = math.nan
    peak_memory = 0.0
    for row in metric_rows:
        memory = row.get("memory(GiB)")
        if memory is not None:
            peak_memory = max(peak_memory, float(memory))
        if row.get("train_runtime") is not None:
            train_runtime = float(row["train_runtime"])
        if row.get("train_speed(s/it)") is not None:
            train_speed = float(row["train_speed(s/it)"])
    return {
        "train_runtime": train_runtime,
        "train_speed_s_per_it": train_speed,
        "peak_memory_gib": peak_memory,
    }
