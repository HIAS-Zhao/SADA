#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from remote_sensing_adaptation.collect_v1_large_experiment_results import (
    METHODS,
    no_retrain_like,
    optional_float,
    scheduler_settings,
)
from remote_sensing_adaptation.run_ekya_lora_experiment import (
    _scheduler_lambda_row,
    flatten_method,
    load_gamma_profile_rows,
    run_experiment,
)
from remote_sensing_adaptation.ekya_lora_scheduler import LambdaProfile


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
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


def choose_retrain_window(
    gamma_rows: list[dict[str, Any]],
    *,
    fixed: float | None,
    strategy: str,
) -> tuple[float, dict[str, Any]]:
    train_times = sorted(
        float(row.get("train_time", row.get("train_time_s")))
        for row in gamma_rows
        if row.get("train_time", row.get("train_time_s")) not in ("", None)
    )
    if not train_times:
        raise ValueError("Gamma profiles need train_time/train_time_s")
    t_min = train_times[0]
    t_med = statistics.median(train_times)
    t_p75 = quantile(train_times, 0.75)
    t_max = train_times[-1]
    candidates = {
        "0.50*T_lora_max": reasonable_round(0.50 * t_max),
        "0.65*T_lora_max": reasonable_round(0.65 * t_max),
        "0.80*T_lora_max": reasonable_round(0.80 * t_max),
        "T_lora_p75": reasonable_round(t_p75),
        "T_lora_max": reasonable_round(t_max),
    }
    if fixed is not None and fixed > 0:
        selected = float(fixed)
        selected_label = "fixed"
    elif strategy == "max":
        selected = candidates["T_lora_max"]
        selected_label = "T_lora_max"
    elif strategy == "p75":
        selected = candidates["T_lora_p75"]
        selected_label = "T_lora_p75"
    else:
        selected_label = "0.65*T_lora_max"
        selected = candidates[selected_label]
        if not (selected > t_med and selected < t_max):
            for label in ("0.50*T_lora_max", "0.65*T_lora_max", "0.80*T_lora_max"):
                value = candidates[label]
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
        "candidate_T_lora_p75": candidates["T_lora_p75"],
        "candidate_T_lora_max": candidates["T_lora_max"],
        "selected_rule": selected_label,
        "T_retrain_s": selected,
        "reason": "V2 scheduler duration selected from fixed-split Gamma target train_time values.",
    }
    return selected, detail


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(dict(row))
    return dict(grouped)


def group_total(manifest: dict[str, Any], preferred_key: str = "") -> int:
    keys = [preferred_key] if preferred_key else []
    keys.extend(("R_all_sample_ids", "R_test_sample_ids", "R_i_sample_ids", "D_i_drift_sample_ids"))
    for key in keys:
        if not key:
            continue
        values = manifest.get(key)
        if values:
            return len(values)
    return int(manifest.get("task_count") or 0)


def scheduler_lambdas(
    rows: list[dict[str, Any]],
    arrival_rate: float,
    *,
    recompute_resource_cost: bool,
) -> list[LambdaProfile]:
    out: list[LambdaProfile] = []
    for row in rows:
        item = dict(row)
        if recompute_resource_cost:
            throughput = item.get("throughput_samples_per_sec", item.get("throughput_samples_per_s"))
            if throughput not in ("", None):
                item["resource_cost"] = arrival_rate / float(throughput) if float(throughput) > 0 else float("inf")
        out.append(LambdaProfile(**_scheduler_lambda_row(item, data_arrival_rate=arrival_rate)))
    return out


def run_rows(
    *,
    lambda_rows_by_group: dict[str, list[dict[str, Any]]],
    gamma_rows_by_group: dict[str, list[dict[str, Any]]],
    manifests: list[dict[str, Any]],
    t_retrain_s: float,
    future_windows: int,
    arrival_rate: float,
    group_total_key: str,
    recompute_lambda_resource_cost: bool,
    max_peak_memory_gib: float | None,
    max_power_watts: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = scheduler_settings(
        t_retrain_s=t_retrain_s,
        arrival_rate=arrival_rate,
        future_windows=future_windows,
        max_peak_memory_gib=max_peak_memory_gib,
        max_power_watts=max_power_watts,
    )
    all_result_json: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        group_id = str(manifest["group_id"])
        if group_id not in lambda_rows_by_group:
            raise ValueError(f"Missing Lambda rows for {group_id}")
        if group_id not in gamma_rows_by_group:
            raise ValueError(f"Missing Gamma rows for {group_id}")
        lambdas = scheduler_lambdas(
            lambda_rows_by_group[group_id],
            arrival_rate,
            recompute_resource_cost=recompute_lambda_resource_cost,
        )
        gammas = load_gamma_profile_rows(gamma_rows_by_group[group_id])
        result = run_experiment(lambdas, gammas, settings)
        all_result_json[group_id] = result
        flat_by_method = {row["method"]: flatten_method(row) for row in result["methods"]}
        total = group_total(manifest, group_total_key)
        lambda_by_id = {str(row["lambda_id"]): row for row in lambda_rows_by_group[group_id]}
        for method in METHODS:
            flat = flat_by_method.get(method) or no_retrain_like(method, flat_by_method["NoRetrain"])
            lambda_row = lambda_by_id.get(str(flat.get("lambda_id", "")), {})
            train_time_effective = ""
            if flat.get("gamma_id") and flat.get("R") not in ("", "0", "0.0", 0, 0.0):
                finish_time = optional_float(flat.get("finish_time"))
                if finish_time is not None:
                    train_time_effective = finish_time
            rows.append(
                {
                    "group_id": group_id,
                    "method": method,
                    "triggered": True,
                    "trigger_position": manifest.get("drift_start_position", ""),
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
                    "correct": float(flat.get("current_window_accuracy") or 0.0) * total,
                    "total": total,
                    "strict_group_acc": flat.get("current_window_accuracy", ""),
                    "processed_only_acc": flat.get("current_window_accuracy", ""),
                    "current_window_accuracy": flat.get("current_window_accuracy", ""),
                    "future_window_accuracy": flat.get("future_window_accuracy", ""),
                    "long_avg_accuracy": flat.get("long_avg_accuracy", ""),
                    "train_time_s_effective": train_time_effective,
                    "inference_latency_mean_s": lambda_row.get("latency_p50_sec", lambda_row.get("p50_latency_s", "")),
                    "lambda_teacher_accuracy": lambda_row.get("teacher_accuracy", lambda_row.get("accuracy", "")),
                    "lambda_true_accuracy": lambda_row.get("true_accuracy", ""),
                    "lambda_teacher_noise_accuracy_gap": lambda_row.get("teacher_noise_accuracy_gap", ""),
                    "task_a": manifest.get("task_a", ""),
                    "task_b": manifest.get("task_b", ""),
                    "task_count": manifest.get("task_count", ""),
                    "notes": "v2_top80_teacher_profiles",
                }
            )
    return rows, all_result_json


def write_group_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_group = group_by(rows, "group_id")
    summary: list[dict[str, Any]] = []
    for group_id, group_rows in sorted(by_group.items()):
        by_method = {row["method"]: row for row in group_rows}
        baseline_methods = [m for m in ("NoRetrain", "FixedHalf", "InferencePriority", "TrainPriority") if m in by_method]
        best_baseline = max(float(by_method[m]["long_avg_accuracy"]) for m in baseline_methods)
        thief = by_method.get("ThiefLoRA", {})
        summary.append(
            {
                "group_id": group_id,
                "task_a": by_method["NoRetrain"].get("task_a", ""),
                "task_b": by_method["NoRetrain"].get("task_b", ""),
                "best_baseline_long_avg_accuracy": best_baseline,
                "ThiefLoRA_long_avg_accuracy": thief.get("long_avg_accuracy", ""),
                "ThiefLoRA_gap_vs_best_baseline": float(thief.get("long_avg_accuracy") or 0.0) - best_baseline,
                "ThiefLoRA_lambda_id": thief.get("lambda_id", ""),
                "ThiefLoRA_gamma_id": thief.get("gamma_id", ""),
                "ThiefLoRA_I": thief.get("I", ""),
                "ThiefLoRA_R": thief.get("R", ""),
            }
        )
    write_csv(out_dir / "main_group_acc_summary.csv", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V2 scheduler from measured Lambda/Gamma profiles.")
    parser.add_argument("--lambda-profiles", type=Path, required=True)
    parser.add_argument("--gamma-profiles", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--retrain-window-sec", type=float, default=0.0)
    parser.add_argument("--retrain-window-strategy", choices=["v1_065", "max", "p75"], default="v1_065")
    parser.add_argument("--future-windows", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--group-total-key", default="")
    parser.add_argument(
        "--keep-lambda-resource-cost",
        action="store_true",
        help="Keep resource_cost from Lambda profile rows instead of recomputing arrival_rate / throughput.",
    )
    parser.add_argument("--max-peak-memory-gib", type=float, default=80.0)
    parser.add_argument("--max-power-watts", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lambda_rows = read_json(args.lambda_profiles)
    gamma_rows = read_json(args.gamma_profiles)
    manifests = read_jsonl(args.group_manifest)
    if not isinstance(lambda_rows, list) or not isinstance(gamma_rows, list):
        raise TypeError("lambda/gamma profiles must be JSON lists")
    t_retrain_s, duration = choose_retrain_window(
        gamma_rows,
        fixed=args.retrain_window_sec if args.retrain_window_sec > 0 else None,
        strategy=args.retrain_window_strategy,
    )
    rows, raw_results = run_rows(
        lambda_rows_by_group=group_by(lambda_rows, "group_id"),
        gamma_rows_by_group=group_by(gamma_rows, "group_id"),
        manifests=manifests,
        t_retrain_s=t_retrain_s,
        future_windows=args.future_windows,
        arrival_rate=args.arrival_rate,
        group_total_key=args.group_total_key,
        recompute_lambda_resource_cost=not args.keep_lambda_resource_cost,
        max_peak_memory_gib=args.max_peak_memory_gib,
        max_power_watts=args.max_power_watts,
    )
    write_csv(args.out_dir / "scheduler_end_to_end_results.csv", rows)
    write_csv(args.out_dir / "scheduler_oracle_trigger_results.csv", rows)
    write_csv(args.out_dir / "retrain_window_duration_selection.csv", [duration])
    write_json(args.out_dir / "scheduler_raw_results_by_group.json", raw_results)
    write_group_summary(args.out_dir, rows)
    write_json(
        args.out_dir / "v2_scheduler_summary.json",
        {
            "groups": len(manifests),
            "rows": len(rows),
            "T_retrain_s": t_retrain_s,
            "future_windows": args.future_windows,
            "arrival_rate": args.arrival_rate,
            "group_total_key": args.group_total_key,
            "recomputed_lambda_resource_cost": not args.keep_lambda_resource_cost,
            "lambda_profiles": str(args.lambda_profiles),
            "gamma_profiles": str(args.gamma_profiles),
            "group_manifest": str(args.group_manifest),
        },
    )
    print(json.dumps({"rows": len(rows), "T_retrain_s": t_retrain_s, "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
