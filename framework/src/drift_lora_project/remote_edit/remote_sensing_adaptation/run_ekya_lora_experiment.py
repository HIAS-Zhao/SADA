from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from remote_sensing_adaptation.ekya_lora_scheduler import (
    GammaProfile,
    LambdaProfile,
    SchedulerSettings,
    choose_inference_config,
    estimate_accuracy,
    fit_optimus_curve,
    infer_retraining_window,
    pick_configs,
    run_thief_scheduler,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in ("", None):
        return None
    return float(value)


def _scheduler_lambda_row(row: dict[str, Any], data_arrival_rate: float = 1.0) -> dict[str, Any]:
    precision = row.get("model_precision", row.get("precision", ""))
    throughput = row.get("throughput_samples_per_sec", row.get("throughput_samples_per_s"))
    resource_cost = row.get("resource_cost")
    if resource_cost is None:
        if throughput in ("", None):
            raise ValueError(
                "lambda profile rows without resource_cost must provide throughput_samples_per_s"
            )
        throughput_value = float(throughput)
        resource_cost = data_arrival_rate / throughput_value if throughput_value > 0 else float("inf")
    return {
        "lambda_id": str(row["lambda_id"]),
        "image_resolution": int(row["image_resolution"]),
        "max_new_tokens": int(row["max_new_tokens"]),
        "model_precision": str(precision),
        "batch_size": int(row.get("batch_size", 1)),
        "accuracy": float(row["accuracy"]),
        "resource_cost": float(resource_cost),
        "token_retention_ratio": float(row.get("token_retention_ratio", 1.0)),
        "roi_strategy": str(row.get("roi_strategy", "none")),
        "peak_memory_gib": _optional_float(row, "peak_memory_gib")
        if "peak_memory_gib" in row
        else _optional_float(row, "peak_vram_gb"),
        "power_watts": _optional_float(row, "power_watts")
        if "power_watts" in row
        else _optional_float(row, "avg_power_w"),
        "thermal_score": _optional_float(row, "thermal_score")
        if "thermal_score" in row
        else _optional_float(row, "thermal_level"),
    }


def _scheduler_gamma_row(row: dict[str, Any]) -> dict[str, Any]:
    expected_gain = row.get("predicted_final_gain", row.get("expected_acc_gain", row.get("profile_gain", 0.0)))
    train_time = row.get("train_time", row.get("train_time_s"))
    if train_time in ("", None):
        raise ValueError("gamma profile rows must provide train_time or train_time_s")
    return {
        "gamma_id": str(row["gamma_id"]),
        "method": str(row.get("method", "LoRA")),
        "lora_rank": int(row["lora_rank"]),
        "lora_alpha": int(row["lora_alpha"]),
        "target_modules": str(row["target_modules"]),
        "learning_rate": float(row.get("learning_rate", 0.0)),
        "batch_size": int(row.get("batch_size", row.get("train_batch_size", 1))),
        "grad_accum": int(row.get("grad_accum", row.get("grad_accum_steps", 1))),
        "train_steps": int(row.get("train_steps", row.get("epochs", 0))),
        "train_data_fraction": float(row.get("train_data_fraction", 1.0)),
        "profile_gain": float(row.get("profile_gain", expected_gain)),
        "predicted_final_gain": float(expected_gain),
        "resource_cost": float(row.get("resource_cost", 0.0)),
        "train_time": float(train_time),
        "peak_memory_gib": _optional_float(row, "peak_memory_gib")
        if "peak_memory_gib" in row
        else _optional_float(row, "peak_vram_gb"),
        "power_watts": _optional_float(row, "power_watts")
        if "power_watts" in row
        else _optional_float(row, "avg_power_w"),
        "thermal_score": _optional_float(row, "thermal_score")
        if "thermal_score" in row
        else _optional_float(row, "thermal_level"),
    }


def load_lambda_profiles(path: Path, data_arrival_rate: float = 1.0) -> list[LambdaProfile]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("lambda profile file must contain a list")
    return [LambdaProfile(**_scheduler_lambda_row(dict(row), data_arrival_rate)) for row in rows]


def load_gamma_profile_rows(rows: list[dict[str, Any]]) -> list[GammaProfile]:
    if not isinstance(rows, list):
        raise ValueError("gamma profile rows must be a list")
    profiles: list[GammaProfile] = []
    for row in rows:
        row = dict(row)
        row.pop("group_id", None)
        profile_curve = row.pop("profile_curve", None)
        max_accuracy = row.pop("max_accuracy", 1.0)
        if profile_curve and "predicted_final_gain" not in row:
            points = [(float(p["step"]), float(p["accuracy"])) for p in profile_curve]
            curve = fit_optimus_curve(points, max_accuracy=float(max_accuracy))
            start_acc = points[0][1]
            final_acc = curve(float(row.get("train_steps", row.get("epochs", 0))))
            row["predicted_final_gain"] = max(0.0, final_acc - start_acc)
            row.setdefault("profile_gain", max(0.0, points[-1][1] - start_acc))
        profiles.append(GammaProfile(**_scheduler_gamma_row(row)))
    return profiles


def load_gamma_profiles(path: Path) -> list[GammaProfile]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("gamma profile file must contain a list")
    return load_gamma_profile_rows([dict(row) for row in rows])


def build_settings(path: Path) -> SchedulerSettings:
    settings = dict(load_json(path))
    if "retraining_window_sec" not in settings:
        total_resource = float(settings.get("total_resource", 1.0))
        initial_inference_resource = float(settings.get("initial_inference_resource", 0.5))
        initial_training_resource = float(
            settings.pop(
                "initial_training_resource",
                total_resource - initial_inference_resource,
            )
        )
        settings["retraining_window_sec"] = infer_retraining_window(
            train_time_ref=float(settings.pop("train_time_ref")),
            initial_training_resource=initial_training_resource,
            margin=float(settings.pop("window_margin", 1.2)),
        )
    return SchedulerSettings(**settings)


def no_retrain_baseline(lambdas: list[LambdaProfile], settings: SchedulerSettings) -> dict[str, Any]:
    lambda_profile = choose_inference_config(
        lambdas,
        inference_resource=settings.total_resource,
        min_accuracy=settings.min_accuracy,
        settings=settings,
    )
    if lambda_profile is None:
        raise ValueError("No inference profile is feasible for NoRetrain baseline")
    decision = estimate_accuracy(
        lambda_profile=lambda_profile,
        gamma_profile=None,
        inference_resource=settings.total_resource,
        training_resource=0.0,
        settings=settings,
    )
    return {"method": "NoRetrain", **decision.to_dict()}


def fixed_half_baseline(
    lambdas: list[LambdaProfile],
    gammas: list[GammaProfile],
    settings: SchedulerSettings,
) -> dict[str, Any]:
    decision = pick_configs(
        lambdas,
        gammas,
        inference_resource=settings.initial_inference_resource,
        training_resource=settings.total_resource - settings.initial_inference_resource,
        settings=settings,
    )
    return {"method": "FixedHalf", **decision.to_dict()}


def inference_priority_baseline(
    lambdas: list[LambdaProfile],
    gammas: list[GammaProfile],
    settings: SchedulerSettings,
) -> dict[str, Any]:
    lambda_profile = choose_inference_config(
        lambdas,
        inference_resource=settings.total_resource,
        min_accuracy=settings.min_accuracy,
        settings=settings,
    )
    if lambda_profile is None:
        raise ValueError("No inference profile is feasible for InferencePriority baseline")
    inference_resource = min(settings.total_resource, lambda_profile.resource_cost)
    training_resource = settings.total_resource - inference_resource
    decision = estimate_accuracy(
        lambda_profile=lambda_profile,
        gamma_profile=None,
        inference_resource=inference_resource,
        training_resource=training_resource,
        settings=settings,
    )
    for gamma in gammas:
        candidate = estimate_accuracy(
            lambda_profile=lambda_profile,
            gamma_profile=gamma,
            inference_resource=inference_resource,
            training_resource=training_resource,
            settings=settings,
        )
        if candidate.long_avg_accuracy > decision.long_avg_accuracy:
            decision = candidate
    return {"method": "InferencePriority", **decision.to_dict()}


def train_priority_baseline(
    lambdas: list[LambdaProfile],
    gammas: list[GammaProfile],
    settings: SchedulerSettings,
) -> dict[str, Any]:
    candidates = []
    for gamma in gammas:
        finish_resource = gamma.train_time / max(settings.retraining_window_sec, 1e-9)
        training_resource = min(
            settings.total_resource,
            max(gamma.resource_cost, finish_resource),
        )
        inference_resource = settings.total_resource - training_resource
        if inference_resource < 0:
            continue
        try:
            candidates.append(
                pick_configs(
                    lambdas,
                    [gamma],
                    inference_resource=inference_resource,
                    training_resource=training_resource,
                    settings=settings,
                )
            )
        except ValueError:
            continue

    if not candidates:
        lambda_profile = choose_inference_config(
            lambdas,
            inference_resource=settings.total_resource,
            min_accuracy=settings.min_accuracy,
            settings=settings,
        )
        if lambda_profile is None:
            raise ValueError("No inference profile is feasible for TrainPriority baseline")
        decision = estimate_accuracy(
            lambda_profile=lambda_profile,
            gamma_profile=None,
            inference_resource=settings.total_resource,
            training_resource=0.0,
            settings=settings,
        )
        return {"method": "TrainPriority", **decision.to_dict()}

    decision = max(
        candidates,
        key=lambda item: (
            item.gamma_profile is not None,
            item.future_window_accuracy,
            item.long_avg_accuracy,
            item.current_window_accuracy,
            item.training_resource,
        ),
    )
    return {"method": "TrainPriority", **decision.to_dict()}


def run_experiment(
    lambdas: list[LambdaProfile],
    gammas: list[GammaProfile],
    settings: SchedulerSettings,
) -> dict[str, Any]:
    thief = run_thief_scheduler(lambdas, gammas, settings)
    baselines = [
        no_retrain_baseline(lambdas, settings),
        fixed_half_baseline(lambdas, gammas, settings),
        inference_priority_baseline(lambdas, gammas, settings),
        train_priority_baseline(lambdas, gammas, settings),
    ]
    methods = baselines + [{"method": "ThiefLoRA", **thief.to_dict()}]
    return {
        "settings": asdict(settings),
        "lambda_profiles": [asdict(row) for row in lambdas],
        "gamma_profiles": [asdict(row) for row in gammas],
        "methods": methods,
    }


def flatten_method(row: dict[str, Any]) -> dict[str, Any]:
    lambda_profile = row.get("lambda_profile") or {}
    gamma_profile = row.get("gamma_profile") or {}
    return {
        "method": row["method"],
        "lambda_id": lambda_profile.get("lambda_id", ""),
        "token_retention_ratio": lambda_profile.get("token_retention_ratio", ""),
        "roi_strategy": lambda_profile.get("roi_strategy", ""),
        "lambda_peak_memory_gib": lambda_profile.get("peak_memory_gib", ""),
        "lambda_power_watts": lambda_profile.get("power_watts", ""),
        "gamma_id": gamma_profile.get("gamma_id", "") if gamma_profile else "",
        "gamma_peak_memory_gib": gamma_profile.get("peak_memory_gib", "") if gamma_profile else "",
        "gamma_power_watts": gamma_profile.get("power_watts", "") if gamma_profile else "",
        "I": row.get("inference_resource", ""),
        "R": row.get("training_resource", ""),
        "current_window_accuracy": row.get("current_window_accuracy", ""),
        "future_window_accuracy": row.get("future_window_accuracy", ""),
        "long_avg_accuracy": row.get("long_avg_accuracy", ""),
        "finish_time": row.get("finish_time", ""),
        "adapter_replaced": row.get("adapter_replaced", ""),
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ekya_lora_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [flatten_method(row) for row in result["methods"]]
    with (output_dir / "ekya_lora_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| method | lambda | token_retention | roi | gamma | I | R | current_acc | future_acc | long_avg_acc | finish_time | adapter |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {lambda_id} | {token_retention_ratio} | {roi_strategy} | {gamma_id} | {I} | {R} | {current_window_accuracy} | "
            "{future_window_accuracy} | {long_avg_accuracy} | {finish_time} | {adapter_replaced} |".format(**row)
        )
    (output_dir / "ekya_lora_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ekya-style LoRA scheduling experiment.")
    parser.add_argument("--lambda-profiles", type=Path, required=True)
    parser.add_argument("--gamma-profiles", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_settings(args.settings)
    lambdas = load_lambda_profiles(args.lambda_profiles, data_arrival_rate=settings.data_arrival_rate)
    gammas = load_gamma_profiles(args.gamma_profiles)
    result = run_experiment(lambdas, gammas, settings)
    write_outputs(result, args.output_dir)
    print(json.dumps(result["settings"], ensure_ascii=False, indent=2))
    print((args.output_dir / "ekya_lora_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
