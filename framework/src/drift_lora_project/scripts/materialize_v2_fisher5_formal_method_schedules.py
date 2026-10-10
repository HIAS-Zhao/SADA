#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ["NoRetrain", "PeriodicFixedRetrain", "StaticSplitContinuous", "Ours"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def fnum(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def by_group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["group_id"])].append(dict(row))
    return dict(out)


def recompute_lambda_cost(row: dict[str, Any], arrival_rate: float) -> float:
    throughput = row.get("throughput_samples_per_sec", row.get("throughput_samples_per_s"))
    if throughput not in ("", None) and fnum(throughput) > 0:
        return arrival_rate / fnum(throughput)
    return fnum(row.get("resource_cost"), 1.0)


def choose_fixed_lambda(lambda_rows: list[dict[str, Any]], preferred_lambda_id: str) -> dict[str, Any]:
    for row in lambda_rows:
        if str(row.get("lambda_id")) == preferred_lambda_id:
            return row
    candidates = [
        row for row in lambda_rows
        if "full" in str(row.get("roi_strategy", row.get("lambda_id", ""))).lower()
        and ("keep100" in str(row.get("lambda_id", "")).lower() or fnum(row.get("token_retention_ratio"), 0.0) >= 0.99)
    ]
    fp32_candidates = [
        row for row in candidates
        if "fp32" in str(row.get("model_precision", row.get("dtype_arg", row.get("lambda_id", "")))).lower()
        or "fp32" in str(row.get("lambda_id", "")).lower()
    ]
    pool = fp32_candidates or candidates or list(lambda_rows)
    if not pool:
        raise ValueError("No lambda rows available for fixed baseline selection")
    return sorted(pool, key=lambda row: str(row.get("lambda_id", "")))[0]


def choose_fixed_gamma(gamma_rows: list[dict[str, Any]], preferred_gamma_id: str) -> dict[str, Any]:
    for row in gamma_rows:
        if str(row.get("gamma_id")) == preferred_gamma_id:
            return row
    for row in gamma_rows:
        if "r16_a32" in str(row.get("gamma_id")) and "target320" in str(row.get("gamma_id")):
            return row
    return sorted(gamma_rows, key=lambda row: str(row.get("gamma_id")))[0]


def group_total(manifest: dict[str, Any], key: str) -> int:
    values = manifest.get(key)
    if values:
        return len(values)
    for fallback in ("R_all_sample_ids", "R_i_sample_ids", "R_test_sample_ids"):
        values = manifest.get(fallback)
        if values:
            return len(values)
    return int(manifest.get("task_count") or 0)


def base_schedule_row(
    *,
    method: str,
    group_id: str,
    manifest: dict[str, Any],
    lambda_row: dict[str, Any],
    gamma_row: dict[str, Any] | None,
    arrival_rate: float,
    total_key: str,
    inference_resource: float,
    training_resource: float,
    adapter_ready_time_s: float | str,
    adapter_accepted: bool,
    note: str,
) -> dict[str, Any]:
    total = group_total(manifest, total_key)
    lambda_acc = fnum(lambda_row.get("accuracy", lambda_row.get("teacher_accuracy")), 0.0)
    gain = fnum(gamma_row.get("predicted_final_gain", gamma_row.get("expected_acc_gain", gamma_row.get("profile_gain"))), 0.0) if gamma_row else 0.0
    if adapter_accepted and adapter_ready_time_s not in ("", None):
        ready = min(total, max(0, math.ceil(arrival_rate * fnum(adapter_ready_time_s))))
    else:
        ready = total
    estimated = ((ready * lambda_acc) + ((total - ready) * min(1.0, lambda_acc + gain))) / total if total else lambda_acc
    return {
        "method": method,
        "group_id": group_id,
        "triggered": False,
        "trigger_position": "",
        "T_retrain_s": total / arrival_rate if arrival_rate > 0 else "",
        "lambda_id": lambda_row.get("lambda_id", ""),
        "gamma_id": gamma_row.get("gamma_id", "") if gamma_row else "",
        "I": round(inference_resource, 6),
        "R": round(training_resource, 6),
        "adapter_ready_time_s": round(fnum(adapter_ready_time_s), 6) if adapter_ready_time_s not in ("", None) else "",
        "adapter_accepted": str(bool(adapter_accepted)),
        "num_arrived": total,
        "num_processed": total,
        "num_timeout": 0,
        "correct": estimated * total,
        "total": total,
        "strict_group_acc": estimated,
        "processed_only_acc": estimated,
        "current_window_accuracy": estimated,
        "future_window_accuracy": min(1.0, lambda_acc + gain) if adapter_accepted else lambda_acc,
        "long_avg_accuracy": estimated,
        "train_time_s_effective": round(fnum(adapter_ready_time_s), 6) if adapter_accepted else "",
        "inference_latency_mean_s": lambda_row.get("latency_p50_sec", lambda_row.get("p50_latency_s", "")),
        "lambda_teacher_accuracy": lambda_row.get("teacher_accuracy", lambda_row.get("accuracy", "")),
        "lambda_true_accuracy": lambda_row.get("true_accuracy", ""),
        "lambda_teacher_noise_accuracy_gap": lambda_row.get("teacher_noise_accuracy_gap", ""),
        "task_a": manifest.get("task_a", ""),
        "task_b": manifest.get("task_b", ""),
        "task_count": manifest.get("task_count", ""),
        "notes": note,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    lambda_rows = read_json(args.lambda_profiles)
    gamma_rows = read_json(args.gamma_profiles)
    baseline_gamma_rows = read_json(args.baseline_gamma_profiles) if args.baseline_gamma_profiles else gamma_rows
    scheduler_rows = read_csv(args.dynamic_scheduler_csv)
    manifests = read_jsonl(args.group_manifest)
    lambdas = by_group(lambda_rows)
    gammas = by_group(gamma_rows)
    baseline_gammas = by_group(baseline_gamma_rows)
    dynamic_by_group = {
        str(row["group_id"]): row
        for row in scheduler_rows
        if str(row.get("method")) == args.dynamic_source_method
    }
    out_counts: dict[str, int] = {}
    all_rows: list[dict[str, Any]] = []

    for method in METHODS:
        method_rows: list[dict[str, Any]] = []
        for manifest in manifests:
            group_id = str(manifest["group_id"])
            group_lambdas = lambdas[group_id]
            group_gammas = gammas[group_id]
            group_baseline_gammas = baseline_gammas[group_id]
            fixed_lambda = choose_fixed_lambda(group_lambdas, args.fixed_lambda_id)
            fixed_lambda_cost = recompute_lambda_cost(fixed_lambda, args.arrival_rate)
            fixed_gamma = choose_fixed_gamma(group_baseline_gammas, args.fixed_gamma_id)
            fixed_train_time = fnum(fixed_gamma.get("train_time", fixed_gamma.get("train_time_s")))
            if method == "NoRetrain":
                row = base_schedule_row(
                    method=method,
                    group_id=group_id,
                    manifest=manifest,
                    lambda_row=fixed_lambda,
                    gamma_row=None,
                    arrival_rate=args.arrival_rate,
                    total_key=args.group_total_key,
                    inference_resource=1.0,
                    training_resource=0.0,
                    adapter_ready_time_s="",
                    adapter_accepted=False,
                    note="formal_no_d_fixed_no_retrain",
                )
            elif method == "PeriodicFixedRetrain":
                train_resource = args.fixed_training_resource
                start_delay = args.period_samples / args.arrival_rate if args.arrival_rate > 0 else 0.0
                ready_time = start_delay + fixed_train_time / max(train_resource, 1e-9)
                row = base_schedule_row(
                    method=method,
                    group_id=group_id,
                    manifest=manifest,
                    lambda_row=fixed_lambda,
                    gamma_row=fixed_gamma,
                    arrival_rate=args.arrival_rate,
                    total_key=args.group_total_key,
                    inference_resource=1.0 - train_resource,
                    training_resource=train_resource,
                    adapter_ready_time_s=ready_time,
                    adapter_accepted=True,
                    note="formal_no_d_periodic_fixed_retrain",
                )
            elif method == "StaticSplitContinuous":
                train_resource = args.fixed_training_resource
                ready_time = fixed_train_time / max(train_resource, 1e-9)
                row = base_schedule_row(
                    method=method,
                    group_id=group_id,
                    manifest=manifest,
                    lambda_row=fixed_lambda,
                    gamma_row=fixed_gamma,
                    arrival_rate=args.arrival_rate,
                    total_key=args.group_total_key,
                    inference_resource=1.0 - train_resource,
                    training_resource=train_resource,
                    adapter_ready_time_s=ready_time,
                    adapter_accepted=True,
                    note="formal_no_d_static_split_continuous",
                )
            elif method == "Ours":
                source = dict(dynamic_by_group[group_id])
                source["method"] = "Ours"
                source["notes"] = "formal_no_d_dynamic_scheduler_from_rwindow_profiles"
                row = source
            method_rows.append(row)
        method_dir = args.out_dir / method
        write_csv(method_dir / "scheduler_end_to_end_results.csv", method_rows)
        write_csv(method_dir / "scheduler_oracle_trigger_results.csv", method_rows)
        write_json(method_dir / "schedule_manifest.json", {
            "method": method,
            "rows": len(method_rows),
            "arrival_rate": args.arrival_rate,
            "comparison_uses_detection_window_D": False,
            "uses_profile_for_selection": method == "Ours",
            "uses_thief_or_dynamic_scheduler": method == "Ours",
            "uses_drift_detection": False,
            "baseline_selection_rule": "fixed_lambda_id_and_fixed_gamma_id_no_profile_no_thief_no_detection"
            if method in {"NoRetrain", "PeriodicFixedRetrain", "StaticSplitContinuous"}
            else "",
            "lambda_profiles": str(args.lambda_profiles),
            "gamma_profiles": str(args.gamma_profiles),
            "baseline_gamma_profiles": str(args.baseline_gamma_profiles or args.gamma_profiles),
            "fixed_lambda_id": args.fixed_lambda_id,
            "fixed_gamma_id": args.fixed_gamma_id,
            "dynamic_scheduler_csv": str(args.dynamic_scheduler_csv) if method == "Ours" else "",
            "offline_joint_oracle": "not_materialized_here; export as posthoc true R_all max-training-gain oracle",
        })
        out_counts[method] = len(method_rows)
        all_rows.extend(method_rows)
    write_csv(args.out_dir / "all_methods_scheduler_end_to_end_results.csv", all_rows)
    write_json(args.out_dir / "formal_method_schedule_summary.json", {
        "methods": METHODS,
        "rows_by_method": out_counts,
        "arrival_rate": args.arrival_rate,
        "comparison_uses_detection_window_D": False,
        "offline_joint_oracle": "posthoc_true_R_all_max_training_gain_oracle_not_profile_or_thief_scheduler",
    })
    return out_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize formal V2 Fisher 5-group scheduler-method schedules without using D.")
    parser.add_argument("--lambda-profiles", type=Path, required=True)
    parser.add_argument("--gamma-profiles", type=Path, required=True)
    parser.add_argument("--dynamic-scheduler-csv", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arrival-rate", type=float, required=True)
    parser.add_argument("--group-total-key", default="R_all_sample_ids")
    parser.add_argument("--dynamic-source-method", default="ThiefLoRA")
    parser.add_argument("--fixed-lambda-id", default="final_fp32_res672_tok16_full_keep100_b1")
    parser.add_argument("--fixed-gamma-id", default="qwen25vl3b_lora_r16_a32_profile80_target320")
    parser.add_argument("--baseline-gamma-profiles", type=Path, default=None)
    parser.add_argument("--fixed-training-resource", type=float, default=0.5)
    parser.add_argument("--period-samples", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = materialize(args)
    print(json.dumps({"rows_by_method": counts, "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
