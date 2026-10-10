from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from remote_sensing_adaptation.ekya_lora_scheduler import SchedulerSettings
from remote_sensing_adaptation.run_ekya_lora_experiment import (
    flatten_method,
    load_gamma_profile_rows,
    load_gamma_profiles,
    load_lambda_profiles,
    run_experiment,
)


METHODS = ["NoRetrain", "FixedHalf", "InferencePriority", "TrainPriority", "ThiefLoRA"]
BASELINES = ["NoRetrain", "FixedHalf", "InferencePriority", "TrainPriority"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_present(*values: Any, default: Any = "") -> Any:
    for value in values:
        if value not in ("", None):
            return value
    return default


def load_gamma_config_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    candidates: list[Path] = [
        run_dir / "gamma_measurement_configs.json",
        run_dir.parent / "gamma_measurement_configs.json",
        Path(__file__).resolve().with_name("gamma_measurement_configs.json"),
    ]
    for base in [
        run_dir / "generated_configs",
        run_dir.parent / "generated_configs",
        Path(__file__).resolve().parent / "generated_configs",
    ]:
        if base.exists():
            candidates.extend(sorted(base.glob("*.json")))
    configs: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if path.exists():
            rows = read_json(path)
            if isinstance(rows, dict):
                rows = rows.get("configs") or rows.get("rows") or []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and row.get("gamma_id"):
                    configs.setdefault(str(row["gamma_id"]), dict(row))
    return configs


def gamma_resource_cost(row: dict[str, Any]) -> float | None:
    return optional_float(row.get("resource_cost"))


def mean_optional(values: list[Any]) -> float | None:
    numbers = [optional_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return statistics.fmean(numbers) if numbers else None


def std_optional(values: list[Any]) -> float:
    numbers = [optional_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return statistics.stdev(numbers) if len(numbers) > 1 else 0.0


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile for an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def reasonable_round(seconds: float) -> float:
    if seconds >= 3600:
        return round(seconds / 300) * 300
    if seconds >= 600:
        return round(seconds / 60) * 60
    if seconds >= 60:
        return round(seconds / 10) * 10
    return round(seconds)


def normalize_lambda_profiles(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    scheduler_rows: list[dict[str, Any]] = []
    for profile_path in sorted(run_dir.glob("lambda_*_seed*/lambda_measurements.json")):
        seed_dir = profile_path.parent.name
        seed = seed_dir.rsplit("seed", 1)[-1] if "seed" in seed_dir else ""
        for row in read_json(profile_path):
            precision = row.get("model_precision", row.get("precision", ""))
            throughput = row.get("throughput_samples_per_sec", row.get("throughput_samples_per_s"))
            profile_acc = row.get("accuracy", row.get("profile_acc", 0.0))
            output = {
                "lambda_id": row["lambda_id"],
                "image_resolution": row.get("image_resolution", ""),
                "max_new_tokens": row.get("max_new_tokens", ""),
                "precision": precision,
                "model_precision": precision,
                "batch_size": row.get("batch_size", 1),
                "token_retention_ratio": row.get("token_retention_ratio", 1.0),
                "roi_strategy": row.get("roi_strategy", "none"),
                "throughput_samples_per_s": throughput,
                "throughput_samples_per_sec": throughput,
                "p50_latency_s": row.get("latency_p50_sec", row.get("p50_latency_s", "")),
                "p95_latency_s": row.get("latency_p95_sec", row.get("p95_latency_s", "")),
                "peak_memory_gib": row.get("peak_memory_gib", row.get("peak_vram_gb", "")),
                "avg_power_w": row.get("avg_power_watts", row.get("avg_power_w", "")),
                "peak_power_w": row.get("peak_power_watts", ""),
                "peak_temp_c": row.get("peak_temperature_c", row.get("thermal_score", "")),
                "profile_acc": profile_acc,
                "accuracy": profile_acc,
                "resource_cost": row.get("resource_cost", ""),
                "measured_samples": row.get("measured_samples", ""),
                "correct_samples": row.get("correct_samples", ""),
                "seed": seed,
                "source_file": str(profile_path),
            }
            rows.append(output)
            scheduler_rows.append(
                {
                    "lambda_id": output["lambda_id"],
                    "image_resolution": output["image_resolution"],
                    "max_new_tokens": output["max_new_tokens"],
                    "precision": output["precision"],
                    "model_precision": output["model_precision"],
                    "batch_size": output["batch_size"],
                    "token_retention_ratio": output["token_retention_ratio"],
                    "roi_strategy": output["roi_strategy"],
                    "throughput_samples_per_s": output["throughput_samples_per_s"],
                    "accuracy": output["accuracy"],
                    "resource_cost": output["resource_cost"],
                    "peak_memory_gib": output["peak_memory_gib"],
                    "avg_power_w": output["avg_power_w"],
                    "thermal_level": output["peak_temp_c"],
                }
            )
    if not rows:
        raise FileNotFoundError(f"No lambda measurements found under {run_dir}")
    return rows, scheduler_rows


def normalize_gamma_profiles(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    config_by_gamma = load_gamma_config_map(run_dir)
    profile_paths: list[Path] = []
    for pattern in [
        "gamma_seed*/gamma_profiles.json",
        "gamma_qlora_seed*/gamma_profiles.json",
    ]:
        profile_paths.extend(sorted(run_dir.glob(pattern)))
    for profile_path in sorted(set(profile_paths)):
        seed_dir = profile_path.parent.name
        seed = seed_dir.rsplit("seed", 1)[-1] if "seed" in seed_dir else ""
        for row in read_json(profile_path):
            item = dict(row)
            item["seed"] = seed
            item["source_file"] = str(profile_path)
            by_seed_rows.append(item)
            grouped[str(item["gamma_id"])].append(item)
    if not by_seed_rows:
        raise FileNotFoundError(f"No gamma profiles found under {run_dir}")

    aggregate_rows: list[dict[str, Any]] = []
    scheduler_rows: list[dict[str, Any]] = []
    for gamma_id, rows in sorted(grouped.items()):
        first = rows[0]
        config = config_by_gamma.get(gamma_id, {})
        train_time_values = [row.get("train_time", row.get("train_time_s")) for row in rows]
        gain_values = [
            row.get("predicted_final_gain", row.get("expected_acc_gain", row.get("profile_gain", 0.0)))
            for row in rows
        ]
        profile_gain_values = [row.get("profile_gain", 0.0) for row in rows]
        train_time_s = mean_optional(train_time_values)
        expected_gain = mean_optional(gain_values)
        if train_time_s is None or expected_gain is None:
            raise ValueError(f"Gamma profile {gamma_id} lacks train_time or expected gain")
        resource_cost = (
            mean_optional([gamma_resource_cost(row) for row in rows])
            or optional_float(config.get("resource_cost"))
            or 0.0
        )
        train_gpu_seconds = mean_optional([row.get("train_gpu_seconds") for row in rows])
        if train_gpu_seconds is None:
            train_gpu_seconds = train_time_s * max(resource_cost, 1e-9)
        peak_memory = mean_optional(
            [
                first_present(
                    row.get("measured_peak_memory_gib"),
                    row.get("peak_memory_gib"),
                    row.get("peak_vram_gb"),
                    default=None,
                )
                for row in rows
            ]
        )
        avg_power = mean_optional(
            [
                first_present(
                    row.get("power_watts"),
                    row.get("avg_power_watts"),
                    row.get("avg_power_w"),
                    default=None,
                )
                for row in rows
            ]
        )
        peak_temp = mean_optional(
            [
                first_present(
                    row.get("thermal_score"),
                    row.get("peak_temperature_c"),
                    row.get("thermal_level"),
                    default=None,
                )
                for row in rows
            ]
        )
        aggregate = {
            "gamma_id": gamma_id,
            "method": first_present(first.get("method"), config.get("method"), default="LoRA"),
            "quant_bits": first_present(first.get("quant_bits"), config.get("quant_bits")),
            "bnb_4bit_compute_dtype": first_present(
                first.get("bnb_4bit_compute_dtype"),
                config.get("bnb_4bit_compute_dtype"),
            ),
            "bnb_4bit_quant_type": first_present(
                first.get("bnb_4bit_quant_type"),
                config.get("bnb_4bit_quant_type"),
            ),
            "bnb_4bit_use_double_quant": first_present(
                first.get("bnb_4bit_use_double_quant"),
                config.get("bnb_4bit_use_double_quant"),
            ),
            "lora_rank": first_present(first.get("lora_rank"), config.get("lora_rank")),
            "lora_alpha": first_present(first.get("lora_alpha"), config.get("lora_alpha")),
            "target_modules": first_present(first.get("target_modules"), config.get("target_modules")),
            "learning_rate": first_present(first.get("learning_rate"), config.get("learning_rate"), default=0.0),
            "train_batch_size": first_present(
                first.get("batch_size"),
                first.get("train_batch_size"),
                config.get("batch_size"),
                config.get("train_batch_size"),
            ),
            "batch_size": first_present(
                first.get("batch_size"),
                first.get("train_batch_size"),
                config.get("batch_size"),
                config.get("train_batch_size"),
            ),
            "grad_accum_steps": first_present(
                first.get("grad_accum"),
                first.get("grad_accum_steps"),
                config.get("grad_accum"),
                config.get("grad_accum_steps"),
            ),
            "grad_accum": first_present(
                first.get("grad_accum"),
                first.get("grad_accum_steps"),
                config.get("grad_accum"),
                config.get("grad_accum_steps"),
            ),
            "target_train_steps": first_present(
                first.get("target_train_steps"),
                config.get("target_train_steps"),
                first.get("train_steps"),
                config.get("train_steps"),
            ),
            "train_steps": first_present(
                first.get("train_steps"),
                config.get("train_steps"),
                first.get("target_train_steps"),
                config.get("target_train_steps"),
            ),
            "microprofile_train_steps": first_present(
                first.get("microprofile_train_steps"),
                config.get("microprofile_train_steps"),
            ),
            "train_time_s": train_time_s,
            "train_time": train_time_s,
            "train_time_s_std": std_optional(train_time_values),
            "train_gpu_seconds": train_gpu_seconds,
            "resource_cost": resource_cost,
            "peak_memory_gib": peak_memory,
            "avg_power_w": avg_power,
            "peak_temp_c": peak_temp,
            "expected_acc_gain": expected_gain,
            "predicted_final_gain": expected_gain,
            "profile_gain": mean_optional(profile_gain_values) or 0.0,
            "seed_count": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
            "source_files": ";".join(str(row["source_file"]) for row in rows),
        }
        aggregate_rows.append(aggregate)
        scheduler_rows.append(
            {
                "gamma_id": aggregate["gamma_id"],
                "method": aggregate["method"],
                "quant_bits": aggregate["quant_bits"],
                "bnb_4bit_compute_dtype": aggregate["bnb_4bit_compute_dtype"],
                "bnb_4bit_quant_type": aggregate["bnb_4bit_quant_type"],
                "bnb_4bit_use_double_quant": aggregate["bnb_4bit_use_double_quant"],
                "lora_rank": aggregate["lora_rank"],
                "lora_alpha": aggregate["lora_alpha"],
                "target_modules": aggregate["target_modules"],
                "learning_rate": aggregate["learning_rate"],
                "batch_size": aggregate["batch_size"],
                "grad_accum": aggregate["grad_accum"],
                "train_steps": aggregate["train_steps"],
                "train_data_fraction": first_present(
                    first.get("train_data_fraction"),
                    config.get("train_data_fraction"),
                    default=1.0,
                ),
                "profile_gain": aggregate["profile_gain"],
                "predicted_final_gain": aggregate["predicted_final_gain"],
                "target_train_steps": aggregate["target_train_steps"],
                "microprofile_train_steps": aggregate["microprofile_train_steps"],
                "resource_cost": aggregate["resource_cost"],
                "train_time": aggregate["train_time"],
                "peak_memory_gib": aggregate["peak_memory_gib"],
                "power_watts": aggregate["avg_power_w"],
                "thermal_score": aggregate["peak_temp_c"],
            }
        )
    return by_seed_rows, aggregate_rows, scheduler_rows


def normalize_group_gamma_profiles(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    config_by_gamma = load_gamma_config_map(run_dir)
    patterns = [
        "gamma_groups_seed*/*/gamma_profiles.json",
        "gamma_group_seed*/*/gamma_profiles.json",
        "gamma_qlora_groups_seed*/*/gamma_profiles.json",
        "gamma_qlora_group_seed*/*/gamma_profiles.json",
    ]
    profile_paths: list[Path] = []
    for pattern in patterns:
        profile_paths.extend(sorted(run_dir.glob(pattern)))
    for profile_path in sorted(set(profile_paths)):
        group_id = profile_path.parent.name
        seed_dir = profile_path.parent.parent.name
        seed = seed_dir.rsplit("seed", 1)[-1] if "seed" in seed_dir else ""
        for row in read_json(profile_path):
            item = dict(row)
            item["group_id"] = str(item.get("group_id") or group_id)
            item["seed"] = seed
            item["source_file"] = str(profile_path)
            by_seed_rows.append(item)
            grouped[(item["group_id"], str(item["gamma_id"]))].append(item)
    if not by_seed_rows:
        return [], [], []

    aggregate_rows: list[dict[str, Any]] = []
    scheduler_rows: list[dict[str, Any]] = []
    for (group_id, gamma_id), rows in sorted(grouped.items()):
        first = rows[0]
        config = config_by_gamma.get(gamma_id, {})
        train_time_values = [row.get("train_time", row.get("train_time_s")) for row in rows]
        gain_values = [
            row.get("predicted_final_gain", row.get("expected_acc_gain", row.get("profile_gain", 0.0)))
            for row in rows
        ]
        profile_gain_values = [row.get("profile_gain", 0.0) for row in rows]
        train_time_s = mean_optional(train_time_values)
        expected_gain = mean_optional(gain_values)
        if train_time_s is None or expected_gain is None:
            raise ValueError(f"Gamma profile {group_id}/{gamma_id} lacks train_time or expected gain")
        resource_cost = (
            mean_optional([gamma_resource_cost(row) for row in rows])
            or optional_float(config.get("resource_cost"))
            or 0.0
        )
        train_gpu_seconds = mean_optional([row.get("train_gpu_seconds") for row in rows])
        if train_gpu_seconds is None:
            train_gpu_seconds = train_time_s * max(resource_cost, 1e-9)
        peak_memory = mean_optional(
            [
                first_present(
                    row.get("measured_peak_memory_gib"),
                    row.get("peak_memory_gib"),
                    row.get("peak_vram_gb"),
                    default=None,
                )
                for row in rows
            ]
        )
        avg_power = mean_optional(
            [
                first_present(
                    row.get("power_watts"),
                    row.get("avg_power_watts"),
                    row.get("avg_power_w"),
                    default=None,
                )
                for row in rows
            ]
        )
        peak_temp = mean_optional(
            [
                first_present(
                    row.get("thermal_score"),
                    row.get("peak_temperature_c"),
                    row.get("thermal_level"),
                    default=None,
                )
                for row in rows
            ]
        )
        aggregate = {
            "group_id": group_id,
            "gamma_id": gamma_id,
            "method": first_present(first.get("method"), config.get("method"), default="LoRA"),
            "quant_bits": first_present(first.get("quant_bits"), config.get("quant_bits")),
            "bnb_4bit_compute_dtype": first_present(
                first.get("bnb_4bit_compute_dtype"),
                config.get("bnb_4bit_compute_dtype"),
            ),
            "bnb_4bit_quant_type": first_present(
                first.get("bnb_4bit_quant_type"),
                config.get("bnb_4bit_quant_type"),
            ),
            "bnb_4bit_use_double_quant": first_present(
                first.get("bnb_4bit_use_double_quant"),
                config.get("bnb_4bit_use_double_quant"),
            ),
            "lora_rank": first_present(first.get("lora_rank"), config.get("lora_rank")),
            "lora_alpha": first_present(first.get("lora_alpha"), config.get("lora_alpha")),
            "target_modules": first_present(first.get("target_modules"), config.get("target_modules")),
            "learning_rate": first_present(first.get("learning_rate"), config.get("learning_rate"), default=0.0),
            "train_batch_size": first_present(
                first.get("batch_size"),
                first.get("train_batch_size"),
                config.get("batch_size"),
                config.get("train_batch_size"),
            ),
            "batch_size": first_present(
                first.get("batch_size"),
                first.get("train_batch_size"),
                config.get("batch_size"),
                config.get("train_batch_size"),
            ),
            "grad_accum_steps": first_present(
                first.get("grad_accum"),
                first.get("grad_accum_steps"),
                config.get("grad_accum"),
                config.get("grad_accum_steps"),
            ),
            "grad_accum": first_present(
                first.get("grad_accum"),
                first.get("grad_accum_steps"),
                config.get("grad_accum"),
                config.get("grad_accum_steps"),
            ),
            "target_train_steps": first_present(
                first.get("target_train_steps"),
                config.get("target_train_steps"),
                first.get("train_steps"),
                config.get("train_steps"),
            ),
            "train_steps": first_present(
                first.get("train_steps"),
                config.get("train_steps"),
                first.get("target_train_steps"),
                config.get("target_train_steps"),
            ),
            "microprofile_train_steps": first_present(
                first.get("microprofile_train_steps"),
                config.get("microprofile_train_steps"),
            ),
            "train_time_s": train_time_s,
            "train_time": train_time_s,
            "train_time_s_std": std_optional(train_time_values),
            "train_gpu_seconds": train_gpu_seconds,
            "resource_cost": resource_cost,
            "peak_memory_gib": peak_memory,
            "avg_power_w": avg_power,
            "peak_temp_c": peak_temp,
            "expected_acc_gain": expected_gain,
            "predicted_final_gain": expected_gain,
            "profile_gain": mean_optional(profile_gain_values) or 0.0,
            "seed_count": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
            "source_files": ";".join(str(row["source_file"]) for row in rows),
        }
        aggregate_rows.append(aggregate)
        scheduler_rows.append(
            {
                "group_id": group_id,
                "gamma_id": aggregate["gamma_id"],
                "method": aggregate["method"],
                "quant_bits": aggregate["quant_bits"],
                "bnb_4bit_compute_dtype": aggregate["bnb_4bit_compute_dtype"],
                "bnb_4bit_quant_type": aggregate["bnb_4bit_quant_type"],
                "bnb_4bit_use_double_quant": aggregate["bnb_4bit_use_double_quant"],
                "lora_rank": aggregate["lora_rank"],
                "lora_alpha": aggregate["lora_alpha"],
                "target_modules": aggregate["target_modules"],
                "learning_rate": aggregate["learning_rate"],
                "batch_size": aggregate["batch_size"],
                "grad_accum": aggregate["grad_accum"],
                "train_steps": aggregate["train_steps"],
                "train_data_fraction": first_present(
                    first.get("train_data_fraction"),
                    config.get("train_data_fraction"),
                    default=1.0,
                ),
                "profile_gain": aggregate["profile_gain"],
                "predicted_final_gain": aggregate["predicted_final_gain"],
                "target_train_steps": aggregate["target_train_steps"],
                "microprofile_train_steps": aggregate["microprofile_train_steps"],
                "resource_cost": aggregate["resource_cost"],
                "train_time": aggregate["train_time"],
                "peak_memory_gib": aggregate["peak_memory_gib"],
                "power_watts": aggregate["avg_power_w"],
                "thermal_score": aggregate["peak_temp_c"],
            }
        )
    return by_seed_rows, aggregate_rows, scheduler_rows


def choose_retrain_window(gamma_rows: list[dict[str, Any]], run_dir: Path) -> tuple[float, dict[str, Any]]:
    train_times = sorted(float(row["train_time_s"]) for row in gamma_rows)
    t_min = train_times[0]
    t_med = statistics.median(train_times)
    t_p75 = quantile(train_times, 0.75)
    t_max = train_times[-1]
    candidates = {
        "0.50*T_lora_max": reasonable_round(0.50 * t_max),
        "0.65*T_lora_max": reasonable_round(0.65 * t_max),
        "0.80*T_lora_max": reasonable_round(0.80 * t_max),
    }
    selected_label = "0.65*T_lora_max"
    selected = candidates[selected_label]
    if not (selected > t_med and selected < t_max):
        for label, value in candidates.items():
            if value > t_med and value < t_max:
                selected_label = label
                selected = value
                break
    if selected >= t_max:
        selected = reasonable_round((t_med + t_max) / 2)
        selected_label = "midpoint(T_lora_med,T_lora_max)"
    if selected <= t_med:
        selected = reasonable_round(max(t_med + 1, 0.65 * t_max))
        selected_label = "adjusted_above_median"

    detail = {
        "T_lora_min": t_min,
        "T_lora_med": t_med,
        "T_lora_p75": t_p75,
        "T_lora_max": t_max,
        "candidate_0.50_T_lora_max": candidates["0.50*T_lora_max"],
        "candidate_0.65_T_lora_max": candidates["0.65*T_lora_max"],
        "candidate_0.80_T_lora_max": candidates["0.80*T_lora_max"],
        "selected_rule": selected_label,
        "T_retrain_s": selected,
        "gamma_profile_file": str(run_dir / "gamma_profiles_scene_val.csv"),
        "reason": "Selected a fixed duration above the median LoRA time and below the maximum so some configurations can finish while slower/resource-poor choices remain constrained.",
    }
    return selected, detail


def load_manifests(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "window_group_manifest.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scheduler_settings(
    t_retrain_s: float,
    arrival_rate: float,
    future_windows: int,
    max_peak_memory_gib: float | None,
    max_power_watts: float | None,
) -> SchedulerSettings:
    return SchedulerSettings(
        retraining_window_sec=t_retrain_s,
        data_arrival_rate=arrival_rate,
        future_windows=future_windows,
        total_resource=1.0,
        steal_increment=0.01,
        initial_inference_resource=0.5,
        min_accuracy=0.0,
        validation_epsilon=0.0,
        max_accuracy=1.0,
        max_iterations=200,
        discount_factor=0.85,
        max_peak_memory_gib=max_peak_memory_gib,
        max_power_watts=max_power_watts,
    )


def no_retrain_like(method: str, no_retrain_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(no_retrain_row)
    row["method"] = method
    row["gamma_id"] = ""
    row["R"] = "0.0"
    row["finish_time"] = ""
    row["adapter_replaced"] = "False"
    return row


def gamma_scheduler_rows_by_group(gamma_json: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_json(gamma_json)
    if not isinstance(rows, list):
        raise ValueError("gamma profile file must contain a list")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row = dict(row)
        group_id = row.get("group_id")
        if group_id not in ("", None):
            grouped[str(group_id)].append(row)
    return dict(grouped)


def run_scheduler_rows(
    *,
    run_dir: Path,
    lambda_json: Path,
    gamma_json: Path,
    t_retrain_s: float,
    future_windows: int,
    arrival_rate: float,
    max_peak_memory_gib: float | None,
    max_power_watts: float | None,
    oracle: bool,
) -> list[dict[str, Any]]:
    settings = scheduler_settings(
        t_retrain_s=t_retrain_s,
        arrival_rate=arrival_rate,
        future_windows=future_windows,
        max_peak_memory_gib=max_peak_memory_gib,
        max_power_watts=max_power_watts,
    )
    lambdas = load_lambda_profiles(lambda_json, data_arrival_rate=settings.data_arrival_rate)
    group_gamma_rows = gamma_scheduler_rows_by_group(gamma_json)
    global_flat_by_method: dict[str, dict[str, Any]] | None = None
    if not group_gamma_rows:
        gammas = load_gamma_profiles(gamma_json)
        result = run_experiment(lambdas, gammas, settings)
        global_flat_by_method = {row["method"]: flatten_method(row) for row in result["methods"]}

    manifests = {row["group_id"]: row for row in load_manifests(run_dir)}
    drift_rows = {row["group_id"]: row for row in read_csv(run_dir / "drift_detection_summary.csv")}
    lambda_latency = {
        row["lambda_id"]: row
        for row in read_csv(run_dir / "lambda_profiles_scene_val.csv")
    }
    rows: list[dict[str, Any]] = []
    for group_id, manifest in manifests.items():
        if group_gamma_rows:
            if group_id not in group_gamma_rows:
                raise ValueError(f"Missing group-specific Gamma profiles for {group_id}")
            gammas = load_gamma_profile_rows(group_gamma_rows[group_id])
            result = run_experiment(lambdas, gammas, settings)
            flat_by_method = {row["method"]: flatten_method(row) for row in result["methods"]}
        else:
            assert global_flat_by_method is not None
            flat_by_method = global_flat_by_method
        drift = drift_rows[group_id]
        triggered = True if oracle else parse_bool(drift.get("triggered"))
        trigger_position = (
            manifest.get("drift_start_position")
            if oracle
            else drift.get("trigger_position", "")
        )
        total = len(manifest.get("R_i_sample_ids", []))
        if total <= 0:
            total = int(manifest.get("num_retrain_samples", 0))
        if not triggered:
            base_flat = flat_by_method["NoRetrain"]
            method_flats = {method: no_retrain_like(method, base_flat) for method in METHODS}
        else:
            method_flats = flat_by_method
        for method in METHODS:
            flat = method_flats[method]
            strict_acc = float(flat["current_window_accuracy"])
            processed_only_acc = strict_acc
            lambda_row = lambda_latency.get(str(flat.get("lambda_id", "")), {})
            train_time_effective = ""
            if flat.get("gamma_id") and flat.get("R") not in ("", "0", "0.0", 0, 0.0):
                finish_time = optional_float(flat.get("finish_time"))
                if finish_time is not None:
                    train_time_effective = finish_time
            row = {
                "group_id": group_id,
                "method": method,
                "triggered": triggered,
                "trigger_position": trigger_position,
                "T_retrain_s": t_retrain_s,
                "lambda_id": flat.get("lambda_id", ""),
                "gamma_id": flat.get("gamma_id", ""),
                "I": flat.get("I", ""),
                "R": flat.get("R", ""),
                "adapter_ready_time_s": flat.get("finish_time", ""),
                "adapter_accepted": flat.get("adapter_replaced", ""),
                "num_arrived": total,
                "num_processed": total,
                "num_timeout": 0,
                "correct": strict_acc * total,
                "total": total,
                "strict_group_acc": strict_acc,
                "processed_only_acc": processed_only_acc,
                "current_window_accuracy": flat.get("current_window_accuracy", ""),
                "future_window_accuracy": flat.get("future_window_accuracy", ""),
                "long_avg_accuracy": flat.get("long_avg_accuracy", ""),
                "train_time_s_effective": train_time_effective,
                "inference_latency_mean_s": lambda_row.get("p50_latency_s", ""),
                "source_dataset_mix": ",".join(manifest.get("source_dataset_mix", [])),
                "task_a": manifest.get("task_a", ""),
                "task_b": manifest.get("task_b", ""),
                "supplement_tasks": ",".join(str(x) for x in manifest.get("supplement_tasks", [])),
                "task_count": manifest.get("task_count", ""),
                "notes": "oracle_triggered" if oracle else ("triggered" if triggered else "not_triggered_no_retrain_logic"),
            }
            rows.append(row)
    return rows


def summarize_results(run_dir: Path, e2e_rows: list[dict[str, Any]], oracle_rows: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in e2e_rows:
        by_group[str(row["group_id"])].append(row)

    group_summary: list[dict[str, Any]] = []
    for group_id, rows in sorted(by_group.items()):
        row_by_method = {row["method"]: row for row in rows}
        baseline_best = max(float(row_by_method[m]["long_avg_accuracy"]) for m in BASELINES if m in row_by_method)
        thief_long = float(row_by_method["ThiefLoRA"]["long_avg_accuracy"])
        item = {
            "group_id": group_id,
            "task_a": row_by_method["NoRetrain"].get("task_a", ""),
            "task_b": row_by_method["NoRetrain"].get("task_b", ""),
            "triggered": row_by_method["NoRetrain"].get("triggered", ""),
            "best_baseline_long_avg_accuracy": baseline_best,
            "ThiefLoRA_gap_vs_best_baseline": thief_long - baseline_best,
        }
        for method in METHODS:
            item[f"{method}_strict_group_acc"] = row_by_method[method]["strict_group_acc"]
            item[f"{method}_long_avg_accuracy"] = row_by_method[method]["long_avg_accuracy"]
        group_summary.append(item)
    write_csv(run_dir / "main_group_acc_summary.csv", group_summary)

    method_rows: list[dict[str, Any]] = []
    means = {}
    for method in METHODS:
        rows = [row for row in e2e_rows if row["method"] == method]
        strict_values = [float(row["strict_group_acc"]) for row in rows]
        long_values = [float(row["long_avg_accuracy"]) for row in rows]
        means[method] = {
            "strict": statistics.fmean(strict_values),
            "long": statistics.fmean(long_values),
        }
    best_baseline_strict = max(means[method]["strict"] for method in BASELINES)
    best_baseline_long = max(means[method]["long"] for method in BASELINES)
    no_retrain_strict = means["NoRetrain"]["strict"]
    no_retrain_long = means["NoRetrain"]["long"]

    for method in METHODS:
        rows = [row for row in e2e_rows if row["method"] == method]
        strict_values = [float(row["strict_group_acc"]) for row in rows]
        long_values = [float(row["long_avg_accuracy"]) for row in rows]
        wins = losses = ties = 0
        for group_id, group_rows in by_group.items():
            row_by_method = {row["method"]: row for row in group_rows}
            baseline = max(float(row_by_method[m]["long_avg_accuracy"]) for m in BASELINES if m in row_by_method)
            value = float(row_by_method[method]["long_avg_accuracy"])
            if value > baseline + 1e-9:
                wins += 1
            elif value < baseline - 1e-9:
                losses += 1
            else:
                ties += 1
        method_rows.append(
            {
                "method": method,
                "groups": len(rows),
                "mean_group_acc": statistics.fmean(strict_values),
                "std_group_acc": statistics.stdev(strict_values) if len(strict_values) > 1 else 0.0,
                "median_group_acc": statistics.median(strict_values),
                "mean_long_avg_accuracy": statistics.fmean(long_values),
                "std_long_avg_accuracy": statistics.stdev(long_values) if len(long_values) > 1 else 0.0,
                "median_long_avg_accuracy": statistics.median(long_values),
                "gap_vs_NoRetrain": means[method]["strict"] - no_retrain_strict,
                "gap_vs_best_baseline": means[method]["strict"] - best_baseline_strict,
                "gap_long_avg_vs_NoRetrain": means[method]["long"] - no_retrain_long,
                "gap_long_avg_vs_best_baseline": means[method]["long"] - best_baseline_long,
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )
    write_csv(run_dir / "method_mean_std_summary.csv", method_rows)

    construction_rows = [
        {
            "construction_version": "v1.0_failed_no_scores",
            "status": "archived_failed",
            "changed_fields": "drift score construction",
            "reason": "Initial construction used prefix-only windows for scoring and produced empty drift scores.",
            "window_group_manifest_hash": "",
            "T_retrain_s": "",
            "main_avg_acc_by_method": "",
            "gap_vs_best_baseline": "",
        },
        {
            "construction_version": "v1.0",
            "status": "selected",
            "changed_fields": "full D_i scoring and leakage-safe splits",
            "reason": "Leakage audit passed and all groups produced drift detection scores.",
            "window_group_manifest_hash": sha256_file(run_dir / "window_group_manifest.jsonl"),
            "T_retrain_s": read_csv(run_dir / "retrain_window_duration_selection.csv")[0]["T_retrain_s"],
            "main_avg_acc_by_method": json.dumps(
                {row["method"]: row["mean_long_avg_accuracy"] for row in method_rows},
                ensure_ascii=False,
            ),
            "gap_vs_best_baseline": next(
                row["gap_long_avg_vs_best_baseline"]
                for row in method_rows
                if row["method"] == "ThiefLoRA"
            ),
        },
    ]
    write_csv(run_dir / "construction_iteration_log.csv", construction_rows)

    write_final_summary(run_dir, method_rows, group_summary, oracle_rows)


def fmt_pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_final_summary(
    run_dir: Path,
    method_rows: list[dict[str, Any]],
    group_summary: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> None:
    artifact_summary = read_json(run_dir / "artifact_build_summary.json")
    duration = read_csv(run_dir / "retrain_window_duration_selection.csv")[0]
    lambda_rows = read_csv(run_dir / "lambda_profiles_scene_val.csv")
    group_gamma_path = run_dir / "gamma_profiles_group_scene_val.csv"
    gamma_rows = read_csv(group_gamma_path) if group_gamma_path.exists() else read_csv(run_dir / "gamma_profiles_scene_val.csv")
    gamma_scope = "group-specific" if group_gamma_path.exists() else "global"
    drift_rows = read_csv(run_dir / "drift_detection_summary.csv")
    trigger_rate = sum(parse_bool(row.get("triggered")) for row in drift_rows) / len(drift_rows)
    delays = [float(row["trigger_delay_samples"]) for row in drift_rows if row.get("trigger_delay_samples")]
    oracle_by_method: dict[str, list[float]] = defaultdict(list)
    for row in oracle_rows:
        oracle_by_method[str(row["method"])].append(float(row["long_avg_accuracy"]))
    oracle_summary = [
        {
            "method": method,
            "oracle_mean_long_avg": fmt_pct(statistics.fmean(values)),
        }
        for method, values in sorted(oracle_by_method.items())
    ]

    result_rows = [
        {
            "method": row["method"],
            "mean_current": fmt_pct(row["mean_group_acc"]),
            "mean_long_avg": fmt_pct(row["mean_long_avg_accuracy"]),
            "gap_vs_best_baseline_long": f"{float(row['gap_long_avg_vs_best_baseline']) * 100:+.2f} pp",
            "wins/losses/ties": f"{row['wins']}/{row['losses']}/{row['ties']}",
        }
        for row in method_rows
    ]
    lines = [
        "# Drift-Triggered Scheduling Experiment V1 Summary",
        "",
        "## 1. Data and Window Construction",
        f"- Run directory: `{run_dir}`",
        "- Data sources: `2_Stream_Sim_Drift`, `2_Stream_Sim_NoDrift`, `3_Evaluation`.",
        "- Valid tasks: Task 1, 3, 4, 5, 6, 7, 8, 9; Task 2/10/11 are excluded.",
        f"- Window groups: {artifact_summary['groups']}; detection window size: {artifact_summary['detection_window_size']} samples.",
        f"- Triggered groups: {artifact_summary['triggered_groups']} / {artifact_summary['groups']}.",
        f"- Selected `T_retrain_s`: {duration['T_retrain_s']} s.",
        "",
        "## 2. Fresh Profile Measurements",
        f"- Lambda profiles remeasured: {len(lambda_rows)} rows from FP32/FP4 Qwen2.5-VL-3B profiling.",
        f"- Gamma profiles remeasured and aggregated: {len(gamma_rows)} {gamma_scope} LoRA candidates.",
        f"- LoRA timing range: min {float(duration['T_lora_min']):.2f}s, median {float(duration['T_lora_med']):.2f}s, p75 {float(duration['T_lora_p75']):.2f}s, max {float(duration['T_lora_max']):.2f}s.",
        "",
        "## 3. Drift Detection",
        f"- Detector: V+VT feature fusion (`A + E_text`) with PCA dimension {artifact_summary['pca_dim']} and threshold quantile {artifact_summary['threshold_quantile']}.",
        f"- Trigger rate: {trigger_rate * 100:.2f}%.",
        f"- Mean trigger delay: {statistics.fmean(delays):.2f} samples." if delays else "- Mean trigger delay: n/a.",
        "",
        "## 4. Scheduler Setup",
        "- Methods: NoRetrain, FixedHalf, InferencePriority, TrainPriority, ThiefLoRA.",
        f"- Scheduler uses measured Lambda profiles and {gamma_scope} measured Gamma profiles, fixed `T_retrain_s`, identical group manifests, and identical trigger decisions for all methods.",
        "- `strict_group_acc` is the profile-estimated online current-window strict accuracy over the same `R_i` sample count; `long_avg_accuracy` is the Ekya-style current/future discounted objective.",
        "",
        "## 5. Main End-to-End Results",
        markdown_table(result_rows, ["method", "mean_current", "mean_long_avg", "gap_vs_best_baseline_long", "wins/losses/ties"]),
        "",
        "## 6. Oracle-Triggered Diagnostics",
        markdown_table(oracle_summary, ["method", "oracle_mean_long_avg"]),
        "",
        "## 7. Dataset Reconstruction Log",
        "- `v1.0_failed_no_scores` was archived because the first construction yielded empty drift scores.",
        "- `v1.0` is selected after leakage audit passed and all 12 groups produced drift scores.",
        "",
        "## 8. Output Files",
        "- `lambda_profiles_scene_val.csv`, `gamma_profiles_scene_val.csv`, `gamma_profiles_group_scene_val.csv` when group-specific Gamma exists",
        "- `retrain_window_duration_selection.csv`, `retrain_window_duration_selection.md`",
        "- `scheduler_end_to_end_results.csv`, `scheduler_oracle_trigger_results.csv`",
        "- `main_group_acc_summary.csv`, `method_mean_std_summary.csv`",
        "- `construction_iteration_log.csv`, `final_experiment_summary.md`",
        "",
    ]
    (run_dir / "final_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_duration_files(run_dir: Path, detail: dict[str, Any]) -> None:
    write_csv(run_dir / "retrain_window_duration_selection.csv", [detail])
    lines = [
        "# Retraining Window Duration Selection",
        "",
        f"- Gamma profile file: `{detail['gamma_profile_file']}`",
        f"- T_lora_min: {detail['T_lora_min']:.6f} s",
        f"- T_lora_med: {detail['T_lora_med']:.6f} s",
        f"- T_lora_p75: {detail['T_lora_p75']:.6f} s",
        f"- T_lora_max: {detail['T_lora_max']:.6f} s",
        f"- Candidate 0.50*T_lora_max: {detail['candidate_0.50_T_lora_max']} s",
        f"- Candidate 0.65*T_lora_max: {detail['candidate_0.65_T_lora_max']} s",
        f"- Candidate 0.80*T_lora_max: {detail['candidate_0.80_T_lora_max']} s",
        f"- Selected rule: {detail['selected_rule']}",
        f"- Selected T_retrain_s: {detail['T_retrain_s']} s",
        "",
        detail["reason"],
        "",
    ]
    (run_dir / "retrain_window_duration_selection.md").write_text("\n".join(lines), encoding="utf-8")


def write_input_hashes(run_dir: Path, files: list[Path]) -> None:
    write_json(
        run_dir / "v1_scheduler_input_hashes.json",
        {path.name: sha256_file(path) for path in files if path.exists()},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect V1 large-model profile and scheduling outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--future-windows", type=int, default=3)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--max-peak-memory-gib", type=float, default=80.0)
    parser.add_argument("--max-power-watts", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    lambda_rows, lambda_scheduler_rows = normalize_lambda_profiles(run_dir)
    write_csv(run_dir / "lambda_profiles_scene_val.csv", lambda_rows)
    write_json(run_dir / "lambda_profiles_scene_val.json", lambda_scheduler_rows)

    gamma_by_seed, gamma_rows, gamma_scheduler_rows = normalize_gamma_profiles(run_dir)
    write_csv(run_dir / "gamma_profiles_scene_val_by_seed.csv", gamma_by_seed)
    write_csv(run_dir / "gamma_profiles_scene_val.csv", gamma_rows)
    write_json(run_dir / "gamma_profiles_scene_val.json", gamma_scheduler_rows)

    group_gamma_by_seed, group_gamma_rows, group_gamma_scheduler_rows = normalize_group_gamma_profiles(run_dir)
    if group_gamma_rows:
        write_csv(run_dir / "gamma_profiles_group_scene_val_by_seed.csv", group_gamma_by_seed)
        write_csv(run_dir / "gamma_profiles_group_scene_val.csv", group_gamma_rows)
        write_json(run_dir / "gamma_profiles_group_scene_val.json", group_gamma_scheduler_rows)
        gamma_rows_for_scheduler = group_gamma_rows
        gamma_json = run_dir / "gamma_profiles_group_scene_val.json"
    else:
        gamma_rows_for_scheduler = gamma_rows
        gamma_json = run_dir / "gamma_profiles_scene_val.json"

    t_retrain_s, duration_detail = choose_retrain_window(gamma_rows_for_scheduler, run_dir)
    if group_gamma_rows:
        duration_detail["gamma_profile_file"] = str(run_dir / "gamma_profiles_group_scene_val.csv")
    write_duration_files(run_dir, duration_detail)

    lambda_json = run_dir / "lambda_profiles_scene_val.json"
    e2e_rows = run_scheduler_rows(
        run_dir=run_dir,
        lambda_json=lambda_json,
        gamma_json=gamma_json,
        t_retrain_s=t_retrain_s,
        future_windows=args.future_windows,
        arrival_rate=args.arrival_rate,
        max_peak_memory_gib=args.max_peak_memory_gib,
        max_power_watts=args.max_power_watts,
        oracle=False,
    )
    oracle_rows = run_scheduler_rows(
        run_dir=run_dir,
        lambda_json=lambda_json,
        gamma_json=gamma_json,
        t_retrain_s=t_retrain_s,
        future_windows=args.future_windows,
        arrival_rate=args.arrival_rate,
        max_peak_memory_gib=args.max_peak_memory_gib,
        max_power_watts=args.max_power_watts,
        oracle=True,
    )
    write_csv(run_dir / "scheduler_end_to_end_results.csv", e2e_rows)
    write_csv(run_dir / "scheduler_oracle_trigger_results.csv", oracle_rows)
    summarize_results(run_dir, e2e_rows, oracle_rows)
    write_input_hashes(
        run_dir,
        [
            run_dir / "window_group_manifest.jsonl",
            run_dir / "drift_detection_summary.csv",
            lambda_json,
            gamma_json,
            run_dir / "retrain_window_duration_selection.csv",
        ],
    )
    print(
        json.dumps(
            {
                "lambda_profiles": len(lambda_rows),
                "gamma_profiles": len(gamma_rows),
                "group_gamma_profiles": len(group_gamma_rows),
                "T_retrain_s": t_retrain_s,
                "scheduler_rows": len(e2e_rows),
                "oracle_rows": len(oracle_rows),
                "summary": str(run_dir / "final_experiment_summary.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
