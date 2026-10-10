#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from argparse import Namespace
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("@@WORKSPACE@@/drift_lora_project")
RESULTS_ROOT = PROJECT_ROOT / "results"

def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


QWEN25_BASE_ROOT = RESULTS_ROOT / "v2_teacher_closed_loop_20260630_181348"
QWEN25_STEP5_DIR = env_path(
    "PRESSURE_QWEN25_STEP5_DIR",
    QWEN25_BASE_ROOT / "step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924",
)
QWEN25_STEP6_DIR = env_path(
    "PRESSURE_QWEN25_STEP6_DIR",
    QWEN25_BASE_ROOT / "step6_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924",
)
QWEN25_TABLE_DIR = env_path(
    "PRESSURE_QWEN25_TABLE_DIR",
    QWEN25_STEP6_DIR / "detailed_result_tables_chinese_recomputed_20260706",
)
QWEN25_EWC_ROOT = env_path(
    "PRESSURE_QWEN25_EWC_ROOT",
    QWEN25_STEP6_DIR / "ewc_student_baseline_20260708",
)
QWEN25_EWC_FISHER_COUNTED_DIR = env_path(
    "PRESSURE_QWEN25_EWC_FISHER_COUNTED_DIR",
    QWEN25_EWC_ROOT / "fisher_counted_ready_time_recompute_20260709",
)
QWEN25_LAMBDA_PROFILES = env_path(
    "PRESSURE_QWEN25_LAMBDA_PROFILES",
    QWEN25_STEP5_DIR / "lambda_analysis_rval" / "lambda_profiles_group_v2.json",
)
QWEN25_GAMMA_PROFILES = env_path(
    "PRESSURE_QWEN25_GAMMA_PROFILES",
    QWEN25_STEP5_DIR / "gamma_rwindow_pseudo" / "gamma_group_profiles.json",
)
QWEN25_GAMMA_PARETO_PROFILES = env_path(
    "PRESSURE_QWEN25_GAMMA_PARETO_PROFILES",
    QWEN25_STEP5_DIR / "gamma_pareto" / "gamma_group_profiles_pareto.json",
)
QWEN25_GROUP_MANIFEST = env_path(
    "PRESSURE_QWEN25_GROUP_MANIFEST",
    QWEN25_STEP5_DIR
    / "inputs"
    / "teacher_pseudo_5group_rwindow_pseudoroot"
    / "window_group_manifest_top80.jsonl",
)
QWEN25_TRUE_ROOT = env_path(
    "PRESSURE_QWEN25_TRUE_ROOT",
    QWEN25_BASE_ROOT
    / "step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924"
    / "inputs"
    / "true_5group_rwindow_root",
)
QWEN25_STATS_SCRIPT = env_path(
    "PRESSURE_QWEN25_STATS_SCRIPT",
    QWEN25_BASE_ROOT
    / "step6_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924"
    / "oracle_grid_completion_20260706"
    / "recompute_final_statistics.py",
)
QWEN35_BASE_ROOT = RESULTS_ROOT / "omniearth_qwen35_framework_20260705"
QWEN35_RUN_DIR = env_path(
    "PRESSURE_QWEN35_RUN_DIR",
    QWEN35_BASE_ROOT / "formal_qwen35_08b_framework_augmented_v2_all_no_pareto",
)
QWEN35_EWC_ROOT = env_path(
    "PRESSURE_QWEN35_EWC_ROOT",
    QWEN35_BASE_ROOT / "formal_qwen35_08b_ewc_student_baseline_20260708",
)
QWEN35_TARGET320_FAIR_DIR = env_path(
    "PRESSURE_QWEN35_TARGET320_FAIR_DIR",
    QWEN35_BASE_ROOT / "formal_qwen35_08b_target320_fair_comparison_20260709",
)
REMOTECLIP_RUN_DIR = env_path(
    "PRESSURE_REMOTECLIP_RUN_DIR",
    RESULTS_ROOT
    / "omniearth_remoteclip_student_vitb32teacher_formal_landcover200_disaster365_q095_r50_20260705_no_pareto",
)
RESNET18_RUN_DIR = env_path(
    "PRESSURE_RESNET18_RUN_DIR",
    RESULTS_ROOT
    / "omniearth_resnet18_student_randomhead_resnet50teacher_ekya_optimus_cap075_period50_20260705_no_pareto",
)
NONQWEN_EWC_SUBDIR = os.environ.get("PRESSURE_NONQWEN_EWC_SUBDIR", "ewc")
REMOTECLIP_EWC_ROOT = env_path(
    "PRESSURE_REMOTECLIP_EWC_ROOT",
    REMOTECLIP_RUN_DIR / NONQWEN_EWC_SUBDIR,
)
RESNET18_EWC_ROOT = env_path(
    "PRESSURE_RESNET18_EWC_ROOT",
    RESNET18_RUN_DIR / NONQWEN_EWC_SUBDIR,
)

METHOD_ORDER = ["NoRetrain", "PeriodicFixedRetrain", "StaticSplitContinuous", "EWC", "Ours", "OfflineJointOracle"]
FAMILY_ORDER = ["qwen25", "qwen35", "remoteclip", "resnet18"]
DEFAULT_ARRIVAL_RATES = "0.025,0.05,0.075,0.1,0.15,0.2,0.5,1,2,4,8,10,12,16,24,32"
DEFAULT_LOAD_FACTORS = "0.25,0.5,0.75,1,1.25,1.5,2,3,4"
MAIN_REPORT = RESULTS_ROOT / "unified_experiment_results_20260706.md"
METHOD_ZH = {
    "NoRetrain": "不重训练",
    "PeriodicFixedRetrain": "固定周期重训练",
    "StaticSplitContinuous": "静态资源持续训练",
    "EWC": "EWC",
    "Ours": "本文方法",
    "OfflineJointOracle": "离线联合Oracle",
}
EVAL_STREAM = os.environ.get("PRESSURE_EVAL_STREAM", "R_all")


def uses_rtest_stream() -> bool:
    return EVAL_STREAM == "R_test"


def eval_group_total_key() -> str:
    return "R_test_sample_ids" if uses_rtest_stream() else "R_all_sample_ids"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_remote_edit_on_path() -> None:
    remote_edit = str(PROJECT_ROOT / "remote_edit")
    if remote_edit not in sys.path:
        sys.path.insert(0, remote_edit)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rate_label(rate: float) -> str:
    return f"ar{rate:g}".replace(".", "p")


def same_rate(value: Any, rate: float) -> bool:
    return math.isclose(as_float(value, default=float("nan")), rate, rel_tol=1e-9, abs_tol=1e-12)


def metric_context(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = {str(row.get("method") or row.get("方法代码")): as_float(row.get("main_metric")) for row in rows}
    return {
        "NoRetrain": values.get("NoRetrain", 0.0),
        "Ours": values.get("Ours", 0.0),
        "OfflineJointOracle": values.get("OfflineJointOracle", 0.0),
    }


def add_relative_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("model_family", "")), str(row.get("pressure_scenario", row.get("arrival_rate", ""))))
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        ctx = metric_context(group_rows)
        for row in group_rows:
            metric = as_float(row.get("main_metric"))
            row["Ours相对NoRetrain提升"] = round(ctx["Ours"] - ctx["NoRetrain"], 6)
            row["Ours距Oracle差距"] = round(ctx["OfflineJointOracle"] - ctx["Ours"], 6)
            if str(row.get("method")) != "Ours":
                row["当前方法相对NoRetrain提升"] = round(metric - ctx["NoRetrain"], 6)
                row["当前方法距Oracle差距"] = round(ctx["OfflineJointOracle"] - metric, 6)
            else:
                row["当前方法相对NoRetrain提升"] = round(ctx["Ours"] - ctx["NoRetrain"], 6)
                row["当前方法距Oracle差距"] = round(ctx["OfflineJointOracle"] - ctx["Ours"], 6)
    return rows


def load_predictions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["uid"])] = str(row.get("prediction", ""))
    return out


def unique_join(values: list[Any]) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.append(text)
    return ";".join(seen)


def pressure_lambda_cost(row: dict[str, Any], rate: float, *, fallback_key: str = "resource_cost") -> float:
    throughput = as_float(row.get("throughput_samples_per_sec", row.get("throughput_samples_per_s")))
    if throughput > 0:
        return rate / throughput
    latency = as_float(row.get("latency_mean_s", row.get("latency_mean_sec")))
    if latency > 0:
        return rate * latency
    return as_float(row.get(fallback_key), default=1.0)


def pressure_adjust_lambda_rows(lambda_rows: list[dict[str, Any]], rate: float) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for row in lambda_rows:
        item = dict(row)
        item["resource_cost_original"] = row.get("resource_cost", "")
        item["resource_cost"] = pressure_lambda_cost(row, rate)
        item["pressure_resource_cost_formula"] = "arrival_rate / throughput_samples_per_sec"
        adjusted.append(item)
    return adjusted


def family_sort_key(family: str) -> int:
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return len(FAMILY_ORDER)


def metric_text(value: Any) -> str:
    if value in ("", None):
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "N/A"


def first_rate_above(rates: list[float], threshold: float) -> str:
    for rate in sorted(rates):
        if rate > threshold:
            return f"{rate:g}"
    return "not_in_sweep"


def median_value(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def lambda_row_throughput(row: dict[str, Any]) -> float:
    throughput = as_float(row.get("throughput_samples_per_sec", row.get("throughput_samples_per_s")))
    if throughput > 0:
        return throughput
    latency = as_float(row.get("latency_mean_s", row.get("latency_mean_sec")))
    if latency > 0:
        return 1.0 / latency
    return 0.0


def unique_throughputs(rows: list[dict[str, Any]]) -> list[float]:
    seen: set[tuple[str, float]] = set()
    values: list[float] = []
    for row in rows:
        throughput = lambda_row_throughput(row)
        if throughput <= 0:
            continue
        key = (str(row.get("lambda_id", row.get("推理配置ID", ""))), round(throughput, 12))
        if key in seen:
            continue
        seen.add(key)
        values.append(throughput)
    return values


def qwen35_lambda_profile_path() -> Path:
    protocol = read_json(QWEN35_RUN_DIR / "framework_results.json")["protocol"]
    return Path(protocol["lambda_root"]) / "lambda_dtop80_by_config.csv"


def source_lambda_rows_by_family() -> dict[str, list[dict[str, Any]]]:
    return {
        "qwen25": read_json(QWEN25_LAMBDA_PROFILES),
        "qwen35": read_csv(qwen35_lambda_profile_path()),
        "remoteclip": read_csv(REMOTECLIP_RUN_DIR / "lambda_profiles" / "lambda_profiles_group_v2.csv"),
        "resnet18": read_csv(RESNET18_RUN_DIR / "lambda_profiles" / "lambda_profiles_group_v2.csv"),
    }


def lambda_throughput_lookup_by_family() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for family, rows in source_lambda_rows_by_family().items():
        by_id: dict[str, float] = {}
        for row in rows:
            lambda_id = str(row.get("lambda_id", row.get("推理配置ID", "")))
            if not lambda_id or lambda_id in by_id:
                continue
            throughput = lambda_row_throughput(row)
            if throughput > 0:
                by_id[lambda_id] = throughput
        out[family] = by_id
    return out


def split_config_ids(text: Any) -> list[str]:
    seen: list[str] = []
    for part in str(text or "").split(";"):
        item = part.strip()
        if item and item not in seen:
            seen.append(item)
    return seen


def selected_noretrain_throughput_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    throughput_by_family = lambda_throughput_lookup_by_family()
    specs: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        no_retrain = next((row for row in rows if row["model_family"] == family and row["method"] == "NoRetrain"), None)
        if no_retrain is None:
            continue
        lambda_ids = split_config_ids(no_retrain.get("selected_inference_configs"))
        throughputs = [
            throughput_by_family.get(family, {}).get(lambda_id, 0.0)
            for lambda_id in lambda_ids
        ]
        throughputs = [value for value in throughputs if value > 0]
        if not throughputs:
            continue
        specs.append(
            {
                "model_family": family,
                "model": no_retrain["model"],
                "selected_noretrain_lambda_ids": ";".join(lambda_ids),
                "reference_throughput_samples_per_sec": min(throughputs),
            }
        )
    return specs


def build_capacity_diagnostics(rates: list[float]) -> list[dict[str, Any]]:
    lambda_rows_by_family = source_lambda_rows_by_family()
    sources: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("qwen25", "Qwen2.5-VL Fisher5", lambda_rows_by_family["qwen25"]),
        ("qwen35", "Qwen3.5-0.8B OmniEarth", lambda_rows_by_family["qwen35"]),
        (
            "remoteclip",
            "RemoteCLIP-ViT-B-32 Teacher + RemoteCLIP-RN50 Student",
            lambda_rows_by_family["remoteclip"],
        ),
        (
            "resnet18",
            "ResNet18 Student + ResNet50 Teacher",
            lambda_rows_by_family["resnet18"],
        ),
    ]
    out: list[dict[str, Any]] = []
    for family, model, rows in sources:
        throughputs = unique_throughputs(rows)
        min_throughput = min(throughputs) if throughputs else 0.0
        median_throughput = median_value(throughputs)
        max_throughput = max(throughputs) if throughputs else 0.0
        if max_throughput >= 8.0:
            note = "快模型：低/中到达率会落在吞吐平台期，需看 8+ 的补压点。"
        elif max_throughput >= 3.0:
            note = "中等吞吐模型：4+ 开始进入明显过载区。"
        else:
            note = "重模型：2+ 已进入明显过载区。"
        out.append(
            {
                "model_family": family,
                "model": model,
                "lambda_config_count": len(throughputs),
                "min_throughput_samples_per_sec": round(min_throughput, 6),
                "median_throughput_samples_per_sec": round(median_throughput, 6),
                "max_throughput_samples_per_sec": round(max_throughput, 6),
                "first_rate_above_min_throughput": first_rate_above(rates, min_throughput),
                "first_rate_above_median_throughput": first_rate_above(rates, median_throughput),
                "first_rate_above_max_throughput": first_rate_above(rates, max_throughput),
                "diagnosis": note,
            }
        )
    return out


def build_spread_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_family"]), as_float(row["arrival_rate"]))].append(row)
    out: list[dict[str, Any]] = []
    for (family, rate), group_rows in sorted(groups.items(), key=lambda item: (family_sort_key(item[0][0]), item[0][1])):
        by_method = {str(row["method"]): row for row in group_rows}
        values = {method: as_float(row.get("main_metric")) for method, row in by_method.items()}
        best_method, best_value = max(values.items(), key=lambda item: item[1])
        min_value = min(values.values())
        max_value = max(values.values())
        ours = values.get("Ours", 0.0)
        no_retrain = values.get("NoRetrain", 0.0)
        oracle = values.get("OfflineJointOracle", 0.0)
        model = str(group_rows[0].get("model", family))
        out.append(
            {
                "model_family": family,
                "model": model,
                "arrival_rate": rate,
                "available_method_count": len(values),
                "method_spread": round(max_value - min_value, 6),
                "Ours_minus_NoRetrain": round(ours - no_retrain, 6),
                "Oracle_minus_Ours": round(oracle - ours, 6),
                "best_method": best_method,
                "best_actual_long": round(best_value, 6),
                "NoRetrain": metric_text(values.get("NoRetrain")),
                "PeriodicFixedRetrain": metric_text(values.get("PeriodicFixedRetrain")),
                "StaticSplitContinuous": metric_text(values.get("StaticSplitContinuous")),
                "EWC": metric_text(values.get("EWC")),
                "Ours": metric_text(values.get("Ours")),
                "OfflineJointOracle": metric_text(values.get("OfflineJointOracle")),
            }
        )
    return out


def summarize_spread_by_family(spread_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in spread_rows:
        by_family[str(row["model_family"])].append(row)
    out: list[dict[str, Any]] = []
    for family, rows in sorted(by_family.items(), key=lambda item: family_sort_key(item[0])):
        max_spread = max(rows, key=lambda row: as_float(row["method_spread"]))
        max_ours_nr = max(rows, key=lambda row: as_float(row["Ours_minus_NoRetrain"]))
        max_oracle_gap = max(rows, key=lambda row: as_float(row["Oracle_minus_Ours"]))
        out.append(
            {
                "model_family": family,
                "model": max_spread["model"],
                "max_method_spread_rate": max_spread["arrival_rate"],
                "max_method_spread": max_spread["method_spread"],
                "max_ours_minus_noretrain_rate": max_ours_nr["arrival_rate"],
                "max_ours_minus_noretrain": max_ours_nr["Ours_minus_NoRetrain"],
                "max_oracle_minus_ours_rate": max_oracle_gap["arrival_rate"],
                "max_oracle_minus_ours": max_oracle_gap["Oracle_minus_Ours"],
            }
        )
    return out


def qwen25_rate_stats_module() -> Any:
    return load_module(QWEN25_STATS_SCRIPT, "pressure_qwen25_stats")


def qwen25_scheduler_module() -> Any:
    ensure_remote_edit_on_path()
    return load_module(
        PROJECT_ROOT / "remote_edit" / "remote_sensing_adaptation" / "run_v2_scheduler_from_profiles.py",
        "pressure_qwen25_scheduler",
    )


def qwen25_eval_uids(stats: Any, group_id: str) -> list[str]:
    if not uses_rtest_stream():
        return stats.group_uids(group_id)
    rows = read_json(QWEN25_TRUE_ROOT / "groups" / group_id / "R_test" / "data.json")
    return [str(row["uid"]) for row in rows]


def run_qwen25_dynamic_scheduler(
    *,
    scheduler: Any,
    rate: float,
    out_dir: Path,
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    group_total_key = eval_group_total_key()
    if uses_rtest_stream():
        totals = {
            len(manifest.get(group_total_key) or [])
            for manifest in manifests
        }
        if len(totals) != 1:
            raise ValueError(f"Qwen2.5 R_test sizes must match for one shared horizon: {sorted(totals)}")
        total = next(iter(totals))
        t_retrain_s = total / rate if rate > 0 else 0.0
        duration = {
            "selected_rule": "R_test_samples/arrival_rate",
            "T_retrain_s": t_retrain_s,
            "R_test_samples": total,
            "arrival_rate": rate,
            "reason": "R_test-only serving horizon.",
        }
    else:
        t_retrain_s, duration = scheduler.choose_retrain_window(
            gamma_rows,
            fixed=None,
            strategy="v1_065",
        )
    rows, raw_results = scheduler.run_rows(
        lambda_rows_by_group=scheduler.group_by(lambda_rows, "group_id"),
        gamma_rows_by_group=scheduler.group_by(gamma_rows, "group_id"),
        manifests=manifests,
        t_retrain_s=t_retrain_s,
        future_windows=0,
        arrival_rate=rate,
        group_total_key=group_total_key,
        recompute_lambda_resource_cost=True,
        max_peak_memory_gib=80.0,
        max_power_watts=None,
    )
    scheduler.write_csv(out_dir / "scheduler_end_to_end_results.csv", rows)
    scheduler.write_csv(out_dir / "scheduler_oracle_trigger_results.csv", rows)
    scheduler.write_csv(out_dir / "retrain_window_duration_selection.csv", [duration])
    scheduler.write_json(out_dir / "scheduler_raw_results_by_group.json", raw_results)
    scheduler.write_group_summary(out_dir, rows)
    scheduler.write_json(
        out_dir / "v2_scheduler_summary.json",
        {
            "groups": len(manifests),
            "rows": len(rows),
            "T_retrain_s": t_retrain_s,
            "future_windows": 0,
            "arrival_rate": rate,
            "group_total_key": group_total_key,
            "eval_stream": EVAL_STREAM,
            "recomputed_lambda_resource_cost": True,
            "lambda_profiles": str(QWEN25_LAMBDA_PROFILES),
            "gamma_profiles": str(QWEN25_GAMMA_PARETO_PROFILES if QWEN25_GAMMA_PARETO_PROFILES.exists() else QWEN25_GAMMA_PROFILES),
            "group_manifest": str(QWEN25_GROUP_MANIFEST),
        },
    )
    return rows


def materialize_qwen25_formal_schedules(
    *,
    materializer: Any,
    rate: float,
    dynamic_csv: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    args = Namespace(
        lambda_profiles=QWEN25_LAMBDA_PROFILES,
        gamma_profiles=QWEN25_GAMMA_PARETO_PROFILES if QWEN25_GAMMA_PARETO_PROFILES.exists() else QWEN25_GAMMA_PROFILES,
        baseline_gamma_profiles=QWEN25_GAMMA_PROFILES,
        dynamic_scheduler_csv=dynamic_csv,
        group_manifest=QWEN25_GROUP_MANIFEST,
        out_dir=out_dir,
        arrival_rate=rate,
        group_total_key=eval_group_total_key(),
        dynamic_source_method="ThiefLoRA",
        fixed_lambda_id="final_fp32_res672_tok16_full_keep100_b1",
        fixed_gamma_id="qwen25vl3b_lora_r16_a32_profile80_target320",
        fixed_training_resource=0.5,
        period_samples=180,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    materializer.materialize(args)
    return read_csv(out_dir / "all_methods_scheduler_end_to_end_results.csv")


def build_qwen25_fallback_ours_rows(
    *,
    materializer: Any,
    rate: float,
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lambdas_by_group = materializer.by_group(lambda_rows)
    gammas_by_group = materializer.by_group(gamma_rows)
    out: list[dict[str, Any]] = []
    for manifest in manifests:
        group_id = str(manifest["group_id"])
        total = materializer.group_total(manifest, eval_group_total_key())
        window_s = total / rate if rate > 0 else 0.0
        best: dict[str, Any] | None = None
        for lambda_row in lambdas_by_group[group_id]:
            lambda_cost = materializer.recompute_lambda_cost(lambda_row, rate)
            lambda_acc = as_float(lambda_row.get("accuracy", lambda_row.get("teacher_accuracy")))
            for gamma_row in [None] + gammas_by_group[group_id]:
                for idx in range(1, 100):
                    inference_resource = idx / 100.0
                    training_resource = 1.0 - inference_resource
                    coverage = min(1.0, inference_resource / max(lambda_cost, 1e-9))
                    base_acc = lambda_acc * coverage
                    adapter_accepted = False
                    ready_time: float | None = None
                    future_acc = base_acc
                    current_acc = base_acc
                    if gamma_row is not None:
                        gamma_cost = as_float(gamma_row.get("resource_cost"))
                        train_time = as_float(gamma_row.get("train_time", gamma_row.get("train_time_s")))
                        gain = as_float(
                            gamma_row.get(
                                "predicted_final_gain",
                                gamma_row.get("expected_acc_gain", gamma_row.get("profile_gain")),
                            )
                        )
                        if training_resource >= gamma_cost and training_resource > 0:
                            ready_time = train_time / training_resource
                            if ready_time <= window_s:
                                adapter_accepted = True
                                updated_acc = min(1.0, lambda_acc + gain) * coverage
                                if updated_acc >= base_acc:
                                    ready_ratio = max(0.0, min(1.0, ready_time / max(window_s, 1e-9)))
                                    current_acc = ready_ratio * base_acc + (1.0 - ready_ratio) * updated_acc
                                    future_acc = updated_acc
                                else:
                                    adapter_accepted = False
                    row = {
                        "method": "ThiefLoRA",
                        "group_id": group_id,
                        "triggered": True,
                        "trigger_position": manifest.get("drift_start_position", ""),
                        "T_retrain_s": window_s,
                        "lambda_id": lambda_row.get("lambda_id", ""),
                        "gamma_id": gamma_row.get("gamma_id", "") if gamma_row and adapter_accepted else "",
                        "I": round(inference_resource, 6),
                        "R": round(training_resource, 6),
                        "adapter_ready_time_s": round(ready_time, 6) if ready_time is not None and adapter_accepted else "",
                        "adapter_accepted": str(adapter_accepted),
                        "num_arrived": total,
                        "num_processed": round(coverage * total, 6),
                        "num_timeout": round(total - coverage * total, 6),
                        "correct": round(current_acc * total, 6),
                        "total": total,
                        "strict_group_acc": round(current_acc, 6),
                        "processed_only_acc": round(current_acc / max(coverage, 1e-9), 6) if coverage > 0 else 0.0,
                        "current_window_accuracy": round(current_acc, 6),
                        "future_window_accuracy": round(future_acc, 6),
                        "long_avg_accuracy": round(current_acc, 6),
                        "train_time_s_effective": round(ready_time, 6) if ready_time is not None and adapter_accepted else "",
                        "inference_latency_mean_s": lambda_row.get("latency_p50_sec", lambda_row.get("p50_latency_s", "")),
                        "lambda_teacher_accuracy": lambda_row.get("teacher_accuracy", lambda_row.get("accuracy", "")),
                        "lambda_true_accuracy": lambda_row.get("true_accuracy", ""),
                        "lambda_teacher_noise_accuracy_gap": lambda_row.get("teacher_noise_accuracy_gap", ""),
                        "task_a": manifest.get("task_a", ""),
                        "task_b": manifest.get("task_b", ""),
                        "task_count": manifest.get("task_count", ""),
                        "notes": "pressure_fallback_dynamic_scheduler_allows_timeout_coverage",
                    }
                    key = (
                        as_float(row["long_avg_accuracy"]),
                        as_float(row["future_window_accuracy"]),
                        as_float(row["current_window_accuracy"]),
                        as_float(row["I"]),
                        -lambda_cost,
                    )
                    best_key = (
                        as_float(best.get("long_avg_accuracy")),
                        as_float(best.get("future_window_accuracy")),
                        as_float(best.get("current_window_accuracy")),
                        as_float(best.get("I")),
                        -materializer.recompute_lambda_cost(
                            next(
                                item
                                for item in lambdas_by_group[group_id]
                                if item.get("lambda_id") == best.get("lambda_id")
                            ),
                            rate,
                        ),
                    ) if best else None
                    if best is None or key > best_key:
                        best = row
        if best is None:
            raise RuntimeError(f"No fallback Ours candidate for {group_id}")
        out.append(best)
    return out


def index_qwen25_report_tree(
    root: Path,
    *,
    base_reports: dict[tuple[str, str], Path] | None = None,
    adapter_reports: dict[tuple[str, str, str], Path] | None = None,
    override: bool = False,
) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str, str], Path]]:
    base_reports = base_reports or {}
    adapter_reports = adapter_reports or {}
    for report in root.rglob("strict_eval_report.json"):
        parts = list(report.parts)
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx] == "base" and idx + 2 < len(parts):
                group_id = parts[idx + 1]
                lambda_id = parts[idx + 2]
                if group_id.startswith("v2_"):
                    key = (group_id, lambda_id)
                    if override or key not in base_reports:
                        base_reports[key] = report
                break
            if parts[idx] == "adapter" and idx + 3 < len(parts):
                group_id = parts[idx + 1]
                gamma_id = parts[idx + 2]
                lambda_id = parts[idx + 3]
                if group_id.startswith("v2_"):
                    key = (group_id, gamma_id, lambda_id)
                    if override or key not in adapter_reports:
                        adapter_reports[key] = report
                break
    return base_reports, adapter_reports


def load_qwen25_strict_report_maps(stats: Any) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str, str], Path]]:
    if uses_rtest_stream():
        base_reports: dict[tuple[str, str], Path] = {}
        adapter_reports: dict[tuple[str, str, str], Path] = {}
        manifest = (
            QWEN25_STEP6_DIR
            / "seed_pareto_grid"
            / "true_accuracy_pareto_grid"
            / "scheduler_true_accuracy_error_rows.csv"
        )
        for row in read_csv(manifest):
            base_path = Path(row["base_report"])
            adapter_path = Path(row["adapter_report"])
            if base_path.exists():
                base_reports[(row["group_id"], row["lambda_id"])] = base_path
            if adapter_path.exists():
                adapter_reports[(row["group_id"], row["gamma_id"], row["lambda_id"])] = adapter_path
        base_reports, adapter_reports = index_qwen25_report_tree(
            QWEN25_STEP6_DIR / "true_accuracy_full_r" / "ar0p05",
            base_reports=base_reports,
            adapter_reports=adapter_reports,
            override=True,
        )
        return index_qwen25_report_tree(
            QWEN25_STEP6_DIR,
            base_reports=base_reports,
            adapter_reports=adapter_reports,
            override=False,
        )
    base_reports, adapter_reports = stats.load_report_maps()
    return index_qwen25_report_tree(
        QWEN25_STEP6_DIR,
        base_reports=base_reports,
        adapter_reports=adapter_reports,
        override=True,
    )


def load_qwen25_ewc_report_maps() -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str, str], Path]]:
    if uses_rtest_stream():
        return index_qwen25_report_tree(
            QWEN25_EWC_ROOT / "true_accuracy_full_r" / "ar0p05" / "shards",
            override=False,
        )
    return index_qwen25_report_tree(QWEN25_EWC_ROOT, override=True)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def compute_qwen25_oracle_from_report_maps(
    *,
    stats: Any,
    labels: list[str],
    lambda_profiles: dict[tuple[str, str], dict[str, Any]],
    oracle_gamma_profiles: dict[tuple[str, str], dict[str, Any]],
    base_reports: dict[tuple[str, str], Path],
    adapter_reports: dict[tuple[str, str, str], Path],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        (group_id, gamma_id, lambda_id, report)
        for (group_id, gamma_id, lambda_id), report in adapter_reports.items()
        if (group_id, gamma_id) in oracle_gamma_profiles
        and (group_id, lambda_id) in lambda_profiles
        and not path_is_within(report, QWEN25_EWC_ROOT)
    ]
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    all_detail: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prediction_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    def predictions(report: Path) -> dict[str, dict[str, Any]]:
        if report not in prediction_cache:
            prediction_cache[report] = stats.prediction_map(report)
        return prediction_cache[report]

    for label in labels:
        rate = as_float(label[2:].replace("p", ".")) if label.startswith("ar") else as_float(label)
        best_by_group: dict[str, tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = {}
        valid_base_candidates = 0
        valid_candidates = 0
        for (group_id, lambda_id), base_report in base_reports.items():
            if (group_id, lambda_id) not in lambda_profiles or not base_report.exists():
                continue
            ordered_uids = qwen25_eval_uids(stats, group_id)
            base_preds = predictions(base_report)
            base_acc, base_correct, base_accepted, base_missing = stats.prediction_accuracy(
                base_preds,
                ordered_uids,
            )
            if base_missing or base_accepted != len(ordered_uids):
                raise RuntimeError(
                    f"Incomplete Qwen2.5 {EVAL_STREAM} base predictions for "
                    f"{group_id}/{lambda_id}"
                )
            valid_base_candidates += 1
            lambda_row = lambda_profiles[(group_id, lambda_id)]
            lambda_cost = pressure_lambda_cost(lambda_row, rate)
            coverage = min(1.0, 1.0 / max(lambda_cost, 1e-9))
            actual_long = coverage * base_acc
            raw = {
                "到达率(samples/s)": rate,
                "组ID": group_id,
                "推理配置ID": lambda_id,
                "训练配置ID": "",
                "推理资源I": 1.0,
                "训练资源R": 0.0,
                "lambda资源成本": round(lambda_cost, 6),
                "coverage": round(coverage, 6),
                "训练时间(s)": "",
                "adapter就绪时间(s)": "",
                "adapter就绪样本数": len(ordered_uids),
                "base全R准确率": round(base_acc, 6),
                "adapter全R准确率": round(base_acc, 6),
                "adapter训练收益": 0.0,
                "before准确率": round(base_acc, 6),
                "before正确数": base_correct,
                "before样本数": len(ordered_uids),
                "after准确率": 0.0,
                "after正确数": 0,
                "after样本数": 0,
                "总样本数": len(ordered_uids),
                "真实端到端processed_acc": round(base_acc, 6),
                "actual_long_avg_accuracy": round(actual_long, 6),
                "实际最终准确率(含coverage)": round(actual_long, 6),
                "base报告": str(base_report),
                "adapter报告": "",
            }
            actual = {
                "group_id": group_id,
                "method": "OfflineJointOracle",
                "lambda_id": lambda_id,
                "gamma_id": "",
                "I": 1.0,
                "R": 0.0,
                "adapter_accepted": False,
                "observed_base_accuracy": base_acc,
                "observed_adapter_accuracy": base_acc,
                "adapter_ready_sample_count": len(ordered_uids),
                "before_adapter_accuracy": base_acc,
                "before_adapter_total": len(ordered_uids),
                "after_adapter_accuracy": 0.0,
                "after_adapter_total": 0,
                "observed_time_weighted_accuracy": base_acc,
                "observed_current_accuracy": actual_long,
                "observed_final_accuracy": actual_long,
                "current_error": 0.0,
                "abs_current_error": 0.0,
                "coverage": coverage,
                "lambda_resource_cost_effective": lambda_cost,
                "base_total": len(ordered_uids),
                "adapter_accepted_count": len(ordered_uids),
                "base_report": str(base_report),
                "adapter_report": "",
            }
            schedule = {
                "method": "OfflineJointOracle",
                "group_id": group_id,
                "lambda_id": lambda_id,
                "gamma_id": "",
                "I": 1.0,
                "R": 0.0,
                "adapter_accepted": "False",
                "current_window_accuracy": actual_long,
                "future_window_accuracy": actual_long,
                "long_avg_accuracy": actual_long,
                "adapter_ready_time_s": "",
                "notes": "seed_specific_base_only_strict_report_oracle",
            }
            key = (
                actual_long,
                base_acc,
                base_acc,
                -len(ordered_uids),
                int("full_keep100" in lambda_id),
                str(base_report),
            )
            current = best_by_group.get(group_id)
            if current is None or key > current[0]:
                best_by_group[group_id] = (
                    key,
                    raw,
                    stats.chinese_detail_row(
                        label,
                        schedule,
                        actual,
                        note="当前seed base-only strict report后验选择",
                    ),
                )
        for group_id, gamma_id, lambda_id, adapter_report in candidates:
            base_report = base_reports.get((group_id, lambda_id))
            if base_report is None or not base_report.exists() or not adapter_report.exists():
                continue
            gamma_row = oracle_gamma_profiles[(group_id, gamma_id)]
            train_time = as_float(gamma_row.get("train_time_s", gamma_row.get("train_time")))
            if train_time <= 0:
                continue
            valid_candidates += 1
            lambda_row = lambda_profiles[(group_id, lambda_id)]
            ordered_uids = qwen25_eval_uids(stats, group_id)
            base_preds = predictions(base_report)
            adapter_preds = predictions(adapter_report)
            base_acc, _, base_accepted, base_missing = stats.prediction_accuracy(base_preds, ordered_uids)
            adapter_acc, _, adapter_accepted, adapter_missing = stats.prediction_accuracy(adapter_preds, ordered_uids)
            if base_missing or adapter_missing or base_accepted != len(ordered_uids) or adapter_accepted != len(ordered_uids):
                raise RuntimeError(
                    f"Incomplete Qwen2.5 {EVAL_STREAM} predictions for "
                    f"{group_id}/{lambda_id}/{gamma_id}"
                )
            lambda_cost = pressure_lambda_cost(lambda_row, rate)
            for index in range(1, 100):
                inference_resource = index / 100.0
                training_resource = 1.0 - inference_resource
                if uses_rtest_stream() and training_resource + 1e-12 < as_float(gamma_row.get("resource_cost")):
                    continue
                ready_time = train_time / training_resource
                ready_count = max(0, min(len(ordered_uids), math.ceil(rate * ready_time)))
                before_uids = ordered_uids[:ready_count]
                after_uids = ordered_uids[ready_count:]
                before_acc, before_correct, before_accepted, _ = stats.prediction_accuracy(base_preds, before_uids)
                after_acc, after_correct, after_accepted, _ = stats.prediction_accuracy(adapter_preds, after_uids)
                accepted = before_accepted + after_accepted
                processed_acc = (before_correct + after_correct) / accepted if accepted else 0.0
                coverage = min(1.0, inference_resource / max(lambda_cost, 1e-9))
                actual_long = coverage * processed_acc
                actual_final = coverage * adapter_acc
                raw = {
                    "到达率(samples/s)": rate,
                    "组ID": group_id,
                    "推理配置ID": lambda_id,
                    "训练配置ID": gamma_id,
                    "推理资源I": round(inference_resource, 6),
                    "训练资源R": round(training_resource, 6),
                    "lambda资源成本": round(lambda_cost, 6),
                    "coverage": round(coverage, 6),
                    "训练时间(s)": round(train_time, 6),
                    "adapter就绪时间(s)": round(ready_time, 6),
                    "adapter就绪样本数": ready_count,
                    "base全R准确率": round(base_acc, 6),
                    "adapter全R准确率": round(adapter_acc, 6),
                    "adapter训练收益": round(adapter_acc - base_acc, 6),
                    "before准确率": round(before_acc, 6),
                    "before正确数": before_correct,
                    "before样本数": len(before_uids),
                    "after准确率": round(after_acc, 6),
                    "after正确数": after_correct,
                    "after样本数": len(after_uids),
                    "总样本数": len(ordered_uids),
                    "真实端到端processed_acc": round(processed_acc, 6),
                    "actual_long_avg_accuracy": round(actual_long, 6),
                    "实际最终准确率(含coverage)": round(actual_final, 6),
                    "base报告": str(base_report),
                    "adapter报告": str(adapter_report),
                }
                actual = {
                    "group_id": group_id,
                    "method": "OfflineJointOracle",
                    "lambda_id": lambda_id,
                    "gamma_id": gamma_id,
                    "I": inference_resource,
                    "R": training_resource,
                    "adapter_accepted": True,
                    "observed_base_accuracy": base_acc,
                    "observed_adapter_accuracy": adapter_acc,
                    "adapter_ready_sample_count": ready_count,
                    "before_adapter_accuracy": before_acc,
                    "before_adapter_total": len(before_uids),
                    "after_adapter_accuracy": after_acc,
                    "after_adapter_total": len(after_uids),
                    "observed_time_weighted_accuracy": processed_acc,
                    "observed_current_accuracy": actual_long,
                    "observed_final_accuracy": actual_final,
                    "current_error": 0.0,
                    "abs_current_error": 0.0,
                    "coverage": coverage,
                    "lambda_resource_cost_effective": lambda_cost,
                    "base_total": len(ordered_uids),
                    "adapter_accepted_count": len(ordered_uids),
                    "base_report": str(base_report),
                    "adapter_report": str(adapter_report),
                }
                schedule = {
                    "method": "OfflineJointOracle",
                    "group_id": group_id,
                    "lambda_id": lambda_id,
                    "gamma_id": gamma_id,
                    "I": inference_resource,
                    "R": training_resource,
                    "adapter_accepted": "True",
                    "current_window_accuracy": actual_long,
                    "future_window_accuracy": actual_final,
                    "long_avg_accuracy": actual_long,
                    "adapter_ready_time_s": ready_time,
                    "notes": "seed_specific_pareto_strict_report_oracle",
                }
                key = (
                    actual_long,
                    processed_acc,
                    adapter_acc,
                    -ready_count,
                    int("full_keep100" in lambda_id),
                    str(adapter_report),
                )
                current = best_by_group.get(group_id)
                if current is None or key > current[0]:
                    best_by_group[group_id] = (key, raw, stats.chinese_detail_row(
                        label,
                        schedule,
                        actual,
                        note="当前seed Pareto strict reports后验I/R枚举选择",
                    ))

        raw_detail = [best_by_group[group_id][1] for group_id in sorted(best_by_group)]
        chinese_detail = [best_by_group[group_id][2] for group_id in sorted(best_by_group)]
        all_detail.extend(raw_detail)
        rows_by_label[label] = chinese_detail
        total = sum(as_int(row["总样本数"]) for row in raw_detail)
        weighted = sum(as_float(row["actual_long_avg_accuracy"]) * as_int(row["总样本数"]) for row in raw_detail)
        summary_rows.append(
            {
                "到达率(samples/s)": rate,
                "组数": len(raw_detail),
                "平均actual_long_avg_accuracy": round(weighted / total, 6) if total else "",
                "候选base数": valid_base_candidates,
                "候选lambda_gamma数": valid_candidates,
                "I/R枚举数": valid_base_candidates + valid_candidates * 99,
                "选择范围": "current-seed base and Pareto strict reports",
            }
        )
    return rows_by_label, all_detail, summary_rows


def build_qwen25_ewc_schedule_rows(
    *,
    stats: Any,
    rate: float,
    lambda_profiles: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    plan_rows = read_csv(QWEN25_EWC_ROOT / "ewc_training_plan_and_results.csv")
    fisher_file = QWEN25_EWC_FISHER_COUNTED_DIR / "qwen25_fisher_runtime_estimates.csv"
    fisher_rows = (
        {row["group_id"]: row for row in read_csv(fisher_file)}
        if fisher_file.exists()
        else {}
    )
    out: list[dict[str, Any]] = []
    for plan in sorted(plan_rows, key=lambda row: row["group_id"]):
        group_id = plan["group_id"]
        lambda_id = plan["lambda_id"]
        gamma_id = plan["gamma_id"]
        training_resource = 0.5
        ewc_train_runtime = as_float(plan.get("ewc_train_runtime_s"))
        fisher_runtime = as_float(
            plan.get(
                "fisher_runtime_s",
                fisher_rows.get(group_id, {}).get("fisher_runtime_s"),
            )
        )
        counted_runtime = ewc_train_runtime + fisher_runtime
        ready_time = counted_runtime / training_resource if training_resource > 0 else math.inf
        group_total = len(qwen25_eval_uids(stats, group_id))
        lambda_row = lambda_profiles[(group_id, lambda_id)]
        out.append(
            {
                "method": "EWC",
                "group_id": group_id,
                "triggered": "False",
                "trigger_position": "",
                "T_retrain_s": group_total / rate if rate > 0 else "",
                "lambda_id": lambda_id,
                "gamma_id": gamma_id,
                "I": 0.5,
                "R": training_resource,
                "adapter_ready_time_s": ready_time,
                "adapter_accepted": "True",
                "num_arrived": group_total,
                "train_time_s_effective": ready_time,
                "lambda_teacher_accuracy": lambda_row.get("teacher_accuracy", lambda_row.get("accuracy", "")),
                "lambda_true_accuracy": lambda_row.get("true_accuracy", ""),
                "notes": (
                    "student_side_ewc_fixed_static_split;"
                    "counted_runtime=Fisher+EWC_train;"
                    f"ewc_train_runtime_s={ewc_train_runtime:.6f};"
                    f"fisher_runtime_s={fisher_runtime:.6f};"
                    "runtime_source="
                    f"{'plan_row' if plan.get('fisher_runtime_s') not in ('', None) else fisher_rows.get(group_id, {}).get('runtime_source', '')}"
                ),
            }
        )
    return out


def compute_qwen25_actual_rtest(
    *,
    stats: Any,
    schedule: dict[str, Any],
    label: str,
    base_report: Path,
    adapter_report: Path | None,
    lambda_profiles: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    group_id = str(schedule["group_id"])
    lambda_id = str(schedule["lambda_id"])
    gamma_id = str(schedule.get("gamma_id") or "")
    rate = as_float(label[2:].replace("p", ".")) if label.startswith("ar") else as_float(label)
    ordered_uids = qwen25_eval_uids(stats, group_id)
    base_preds = stats.prediction_map(base_report)
    base_acc, base_correct, base_accepted, base_missing = stats.prediction_accuracy(base_preds, ordered_uids)
    if base_missing or base_accepted != len(ordered_uids):
        raise RuntimeError(f"Incomplete Qwen2.5 R_test base predictions: {base_report}")

    requested_adapter = bool(gamma_id) and str(schedule.get("adapter_accepted", "")).lower() in {
        "true",
        "1",
        "yes",
    }
    adapter_preds = base_preds
    adapter_acc = base_acc
    adapter_correct = base_correct
    adapter_accepted_count = base_accepted
    adapter_report_text = ""
    adapter_path = ""
    if adapter_report is not None:
        adapter_preds = stats.prediction_map(adapter_report)
        adapter_acc, adapter_correct, adapter_accepted_count, adapter_missing = stats.prediction_accuracy(
            adapter_preds,
            ordered_uids,
        )
        if adapter_missing or adapter_accepted_count != len(ordered_uids):
            raise RuntimeError(f"Incomplete Qwen2.5 R_test adapter predictions: {adapter_report}")
        adapter_report_text = str(adapter_report)
        adapter_path = read_json(adapter_report).get("adapter_path", "")

    lambda_row = lambda_profiles[(group_id, lambda_id)]
    lambda_cost = pressure_lambda_cost(lambda_row, rate)
    inference_resource = as_float(schedule.get("I"))
    coverage = min(1.0, inference_resource / max(lambda_cost, 1e-9))
    finish_time = as_float(
        schedule.get("adapter_ready_time_s"),
        as_float(schedule.get("train_time_s_effective"), math.inf),
    )
    horizon_s = len(ordered_uids) / rate if rate > 0 else 0.0
    adapter_replaced = bool(
        requested_adapter
        and adapter_report is not None
        and math.isfinite(finish_time)
        and finish_time < horizon_s
    )
    ready_count = len(ordered_uids)
    if adapter_replaced:
        ready_count = max(0, min(len(ordered_uids), math.ceil(rate * finish_time)))
    before_uids = ordered_uids[:ready_count]
    after_uids = ordered_uids[ready_count:]
    before_acc, before_correct, before_accepted, before_missing = stats.prediction_accuracy(
        base_preds,
        before_uids,
    )
    after_acc, after_correct, after_accepted, after_missing = stats.prediction_accuracy(
        adapter_preds,
        after_uids,
    )
    if before_missing or after_missing:
        raise RuntimeError(f"Missing Qwen2.5 predictions during R_test replay for {group_id}")
    accepted = before_accepted + after_accepted
    processed_acc = (before_correct + after_correct) / accepted if accepted else 0.0
    observed_current = coverage * processed_acc
    observed_final = coverage * (adapter_acc if adapter_replaced else base_acc)
    estimated_current = as_float(schedule.get("current_window_accuracy"))
    return {
        "group_id": group_id,
        "method": schedule.get("method", ""),
        "lambda_id": lambda_id,
        "gamma_id": gamma_id,
        "I": schedule.get("I", ""),
        "R": schedule.get("R", ""),
        "adapter_accepted": requested_adapter,
        "adapter_replaced": adapter_replaced,
        "estimated_current_accuracy": estimated_current,
        "estimated_long_avg_accuracy": as_float(schedule.get("long_avg_accuracy"), estimated_current),
        "eval_split": "R_test",
        "arrival_rate": rate,
        "lambda_resource_cost_effective": round(lambda_cost, 6),
        "coverage": round(coverage, 6),
        "observed_base_accuracy": round(base_acc, 6),
        "observed_adapter_accuracy": round(adapter_acc, 6),
        "adapter_ready_sample_count": ready_count,
        "before_adapter_accuracy": round(before_acc, 6),
        "before_adapter_correct": before_correct,
        "before_adapter_accepted": before_accepted,
        "before_adapter_total": len(before_uids),
        "before_adapter_missing_predictions": before_missing,
        "after_adapter_accuracy": round(after_acc, 6),
        "after_adapter_correct": after_correct,
        "after_adapter_accepted": after_accepted,
        "after_adapter_total": len(after_uids),
        "after_adapter_missing_predictions": after_missing,
        "observed_time_weighted_accuracy": round(processed_acc, 6),
        "observed_current_accuracy": round(observed_current, 6),
        "observed_final_accuracy": round(observed_final, 6),
        "current_error": round(observed_current - estimated_current, 6),
        "abs_current_error": round(abs(observed_current - estimated_current), 6),
        "final_minus_estimated_current": round(observed_final - estimated_current, 6),
        "lambda_teacher_accuracy": lambda_row.get("teacher_accuracy", lambda_row.get("accuracy", "")),
        "lambda_true_accuracy": lambda_row.get("true_accuracy", ""),
        "base_correct": base_correct,
        "base_accepted": base_accepted,
        "base_total": len(ordered_uids),
        "adapter_correct": adapter_correct,
        "adapter_accepted_count": adapter_accepted_count,
        "adapter_path": adapter_path,
        "base_report": str(base_report),
        "adapter_report": adapter_report_text,
    }


def recompute_qwen25_rate_details(
    *,
    stats: Any,
    label: str,
    schedule_rows: list[dict[str, Any]],
    lambda_profiles: dict[tuple[str, str], dict[str, Any]],
    oracle_rows_by_label: dict[str, list[dict[str, Any]]],
    base_reports: dict[tuple[str, str], Path],
    adapter_reports: dict[tuple[str, str, str], Path],
) -> list[dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    for schedule in sorted(schedule_rows, key=lambda row: (METHOD_ORDER.index(row["method"]), row["group_id"])):
        method = schedule["method"]
        if method == "OfflineJointOracle":
            continue
        group_id = schedule["group_id"]
        lambda_id = schedule["lambda_id"]
        gamma_id = schedule.get("gamma_id", "")
        base_report = base_reports.get((group_id, lambda_id))
        if base_report is None:
            raise FileNotFoundError(f"Qwen2.5 missing base strict report for {group_id}/{lambda_id}")
        adapter_report = adapter_reports.get((group_id, gamma_id, lambda_id)) if gamma_id else None
        if uses_rtest_stream():
            actual = compute_qwen25_actual_rtest(
                stats=stats,
                schedule=schedule,
                label=label,
                base_report=base_report,
                adapter_report=adapter_report,
                lambda_profiles=lambda_profiles,
            )
        else:
            actual = stats.compute_actual_from_reports(
                schedule,
                label=label,
                base_report=base_report,
                adapter_report=adapter_report,
                lambda_profiles=lambda_profiles,
            )
        detail_rows.append(stats.chinese_detail_row(label, schedule, actual, note="pressure sweep strict-report posthoc recompute"))
    detail_rows.extend(oracle_rows_by_label[label])
    return detail_rows


def collect_qwen25_pressure_rates(
    *,
    out_root: Path,
    rates: list[float],
) -> list[dict[str, Any]]:
    stats = qwen25_rate_stats_module()
    scheduler = qwen25_scheduler_module()
    materializer = load_module(
        PROJECT_ROOT / "scripts" / "materialize_v2_fisher5_formal_method_schedules.py",
        "pressure_qwen25_materializer",
    )
    lambda_rows = scheduler.read_json(QWEN25_LAMBDA_PROFILES)
    gamma_rows_for_scheduler = scheduler.read_json(
        QWEN25_GAMMA_PARETO_PROFILES if QWEN25_GAMMA_PARETO_PROFILES.exists() else QWEN25_GAMMA_PROFILES
    )
    manifests = scheduler.read_jsonl(QWEN25_GROUP_MANIFEST)
    lambda_profile_rows = stats.read_json(QWEN25_LAMBDA_PROFILES)
    gamma_profile_rows = stats.read_json(QWEN25_GAMMA_PROFILES)
    lambda_profiles = {(row["group_id"], row["lambda_id"]): row for row in lambda_profile_rows}
    gamma_profiles = {(row["group_id"], row["gamma_id"]): row for row in gamma_profile_rows}
    oracle_gamma_profiles = {
        (row["group_id"], row["gamma_id"]): row
        for row in gamma_rows_for_scheduler
    }
    labels = [rate_label(rate) for rate in rates]
    base_reports, adapter_reports = load_qwen25_strict_report_maps(stats)
    ewc_base_reports, ewc_adapter_reports = load_qwen25_ewc_report_maps()
    oracle_rows_by_label, oracle_detail_rows, oracle_summary_rows = compute_qwen25_oracle_from_report_maps(
        stats=stats,
        labels=labels,
        lambda_profiles=lambda_profiles,
        oracle_gamma_profiles=oracle_gamma_profiles,
        base_reports=base_reports,
        adapter_reports=adapter_reports,
    )
    all_rows: list[dict[str, Any]] = []
    native_root = out_root / "qwen25"
    for rate in rates:
        label = rate_label(rate)
        native_dir = native_root / label
        dynamic_dir = native_dir / "scheduler_dynamic"
        formal_dir = native_dir / "formal_schedules"
        try:
            run_qwen25_dynamic_scheduler(
                scheduler=scheduler,
                rate=rate,
                out_dir=dynamic_dir,
                lambda_rows=lambda_rows,
                gamma_rows=gamma_rows_for_scheduler,
                manifests=manifests,
            )
            qwen25_scheduler_mode = "native_dynamic_scheduler"
        except ValueError as exc:
            dynamic_dir.mkdir(parents=True, exist_ok=True)
            fallback_rows = build_qwen25_fallback_ours_rows(
                materializer=materializer,
                rate=rate,
                lambda_rows=lambda_rows,
                gamma_rows=gamma_rows_for_scheduler,
                manifests=manifests,
            )
            write_csv(dynamic_dir / "scheduler_end_to_end_results.csv", fallback_rows)
            write_csv(dynamic_dir / "scheduler_oracle_trigger_results.csv", fallback_rows)
            write_json(
                dynamic_dir / "v2_scheduler_summary.json",
                {
                    "groups": len(manifests),
                    "rows": len(fallback_rows),
                    "arrival_rate": rate,
                    "group_total_key": eval_group_total_key(),
                    "eval_stream": EVAL_STREAM,
                    "mode": "pressure_fallback_dynamic_scheduler_allows_timeout_coverage",
                    "native_scheduler_error": str(exc),
                },
            )
            qwen25_scheduler_mode = "pressure_fallback_dynamic_scheduler_allows_timeout_coverage"
        schedule_rows = materialize_qwen25_formal_schedules(
            materializer=materializer,
            rate=rate,
            dynamic_csv=dynamic_dir / "scheduler_end_to_end_results.csv",
            out_dir=formal_dir,
        )
        native_detail = recompute_qwen25_rate_details(
            stats=stats,
            label=label,
            schedule_rows=schedule_rows,
            lambda_profiles=lambda_profiles,
            oracle_rows_by_label=oracle_rows_by_label,
            base_reports=base_reports,
            adapter_reports=adapter_reports,
        )
        native_detail.extend(
            recompute_qwen25_rate_details(
                stats=stats,
                label=label,
                schedule_rows=build_qwen25_ewc_schedule_rows(
                    stats=stats,
                    rate=rate,
                    lambda_profiles=lambda_profiles,
                ),
                lambda_profiles=lambda_profiles,
                oracle_rows_by_label={label: []},
                base_reports=ewc_base_reports,
                adapter_reports=ewc_adapter_reports,
            )
        )
        native_summary = stats.summarize(native_detail)
        native_oracle_detail = [row for row in oracle_detail_rows if same_rate(row.get("到达率(samples/s)"), rate)]
        native_oracle_summary = [row for row in oracle_summary_rows if same_rate(row.get("到达率(samples/s)"), rate)]
        write_csv(native_dir / "method_summary_native.csv", native_summary)
        write_csv(native_dir / "group_detail_native.csv", native_detail)
        write_csv(native_dir / "offline_joint_oracle_actual_long_detail.csv", native_oracle_detail)
        write_csv(native_dir / "offline_joint_oracle_actual_long_summary.csv", native_oracle_summary)
        write_json(
            native_dir / "pressure_recompute_manifest.json",
            {
                "model_family": "qwen25",
                "arrival_rate": rate,
                "dynamic_scheduler": str(dynamic_dir / "scheduler_end_to_end_results.csv"),
                "formal_schedules": str(formal_dir / "all_methods_scheduler_end_to_end_results.csv"),
                "lambda_profiles": str(QWEN25_LAMBDA_PROFILES),
                "gamma_profiles_for_scheduler": str(
                    QWEN25_GAMMA_PARETO_PROFILES if QWEN25_GAMMA_PARETO_PROFILES.exists() else QWEN25_GAMMA_PROFILES
                ),
                "baseline_gamma_profiles": str(QWEN25_GAMMA_PROFILES),
                "qwen25_scheduler_mode": qwen25_scheduler_mode,
                "metric": "actual_long_avg_accuracy",
                "actual_long_mapping": (
                    f"Qwen2.5 future_windows=0; actual_long_avg_accuracy equals "
                    f"coverage * {EVAL_STREAM} time-weighted accuracy."
                ),
                "eval_stream": EVAL_STREAM,
                "ewc_source": str(QWEN25_EWC_ROOT),
                "ewc_counted_runtime": "Fisher runtime + EWC train runtime, divided by fixed R=0.5 for adapter ready time.",
            },
        )
        for row in sorted(native_summary, key=lambda item: METHOD_ORDER.index(item["方法代码"])):
            group_rows = [item for item in native_detail if item.get("方法代码") == row["方法代码"]]
            all_rows.append(
                {
                    "model": "Qwen2.5-VL Fisher5",
                    "model_family": "qwen25",
                    "pressure_scenario": label,
                    "arrival_rate": rate,
                    "method": row["方法代码"],
                    "方法": row.get("方法", METHOD_ZH.get(row["方法代码"], row["方法代码"])),
                    "main_metric_name": "actual_long_avg_accuracy",
                    "main_metric": as_float(row.get("平均实际当前准确率")),
                    "time_weighted_metric": as_float(row.get("平均时间加权准确率")),
                    "final_metric": as_float(row.get("平均实际最终准确率")),
                    "inference_config_field": "推理配置ID",
                    "training_config_field": "训练配置ID",
                    "selected_inference_configs": unique_join([item.get("推理配置ID", "") for item in group_rows]),
                    "selected_training_configs": unique_join([item.get("训练配置ID", "") for item in group_rows]),
                    "I_by_group": unique_join([item.get("推理资源I", "") for item in group_rows]),
                    "R_by_group": unique_join([item.get("训练资源R", "") for item in group_rows]),
                    "adapter_ready_samples_avg": as_float(row.get("平均adapter就绪样本数")),
                    "native_summary": str(native_dir / "method_summary_native.csv"),
                    "native_detail": str(native_dir / "group_detail_native.csv"),
                    "source": str(QWEN25_STEP6_DIR),
                }
            )
    return add_relative_metrics(all_rows)


def qwen35_args_for_rate(source_dir: Path, out_dir: Path, rate: float) -> Namespace:
    framework = read_json(source_dir / "framework_results.json")
    protocol = framework["protocol"]
    return Namespace(
        split_root=Path(protocol["split_root"]),
        pseudo_root=Path(protocol["pseudo_root"]),
        lambda_root=Path(protocol["lambda_root"]),
        gamma_root=Path(protocol["gamma_root"]),
        detector_root=Path(protocol["detector_root"]),
        output_dir=out_dir,
        lambda_max_resource_cost=0.65,
        gamma_max_resource_cost=0.35,
        periodic_inference_resource=0.50,
        static_inference_resource=0.70,
        arrival_rate=rate,
        retraining_window_sec=as_float(protocol.get("retraining_window_sec"), 7200.0),
        future_windows=3,
        discount_factor=0.95,
        min_profile_gain=0.0,
    )


def qwen35_ewc_extra_runtime_by_group() -> dict[str, float]:
    detail_path = QWEN35_TARGET320_FAIR_DIR / "qwen35_target320_fair_detail.csv"
    if not detail_path.exists():
        return {}
    out: dict[str, float] = {}
    for row in read_csv(detail_path):
        if row.get("method") == "EWC":
            out[row["group_id"]] = as_float(row.get("extra_retrain_runtime_s"))
    return out


def build_qwen35_ewc_rows(
    *,
    q35: Any,
    args: Namespace,
    group_ids: list[str],
    method_rows: list[dict[str, Any]],
    gamma_detail: list[dict[str, Any]],
    base_eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (QWEN35_EWC_ROOT / "qwen35_ewc_actual_execution.csv").exists():
        return []
    ewc_execution = {
        row["group_id"]: row
        for row in read_csv(QWEN35_EWC_ROOT / "qwen35_ewc_actual_execution.csv")
    }
    extra_runtime_by_group = qwen35_ewc_extra_runtime_by_group()
    no_retrain_by_group = {
        row["group_id"]: row
        for row in method_rows
        if row.get("method") == "NoRetrain"
    }
    gamma_by_key = {(row["group_id"], row["gamma_id"]): row for row in gamma_detail}
    base_eval_by_key = {(row["group_id"], row["lambda_id"]): row for row in base_eval_rows}
    out: list[dict[str, Any]] = []
    for group_id in group_ids:
        ewc = ewc_execution.get(group_id)
        fixed = no_retrain_by_group.get(group_id)
        if not ewc or not fixed:
            continue
        lambda_id = ewc["lambda_id"]
        gamma_id = ewc["gamma_id"]
        fisher_runtime = as_float(
            ewc.get("fisher_runtime_s"),
            extra_runtime_by_group.get(group_id, 0.0),
        )
        counted_runtime = as_float(
            ewc.get("fisher_plus_train_runtime_s"),
            as_float(ewc.get("actual_target_train_runtime_s")) + fisher_runtime,
        )
        gamma_row = dict(gamma_by_key[(group_id, gamma_id)])
        gamma_row["target_train_time_s_est"] = counted_runtime
        gamma_row["resource_cost"] = counted_runtime
        lambda_row = {
            "lambda_id": lambda_id,
            "lambda_resource_cost_effective": fixed["lambda_resource_cost_effective"],
            "dtop80_teacher_accuracy_group": fixed.get("lambda_dtop80_teacher_accuracy_group", ""),
            "dtop80_true_accuracy_group": fixed.get("lambda_dtop80_true_accuracy_group", ""),
            "dtop80_teacher_accuracy": fixed.get("lambda_dtop80_teacher_accuracy_all", ""),
            "dtop80_true_accuracy": fixed.get("lambda_dtop80_true_accuracy_all", ""),
            "latency_mean_s": fixed.get("lambda_latency_mean_s", ""),
        }
        rows = q35.load_group_rows(args.split_root, args.pseudo_root, group_id)
        base_eval = base_eval_by_key[(group_id, lambda_id)]
        base_preds = q35.prediction_map(Path(base_eval["raw_predictions"]))
        adapter_preds = q35.prediction_map(Path(ewc["target_adapter_raw_predictions"]))
        adapter_rval_pseudo = q35.score_rows(rows["R_val_pseudo"], adapter_preds)["accuracy_seen"]
        decision = q35.build_decision(
            method="EWC",
            group_id=group_id,
            drift_task=q35.drift_task_for_group(group_id),
            lambda_row=lambda_row,
            gamma_row=gamma_row,
            rows=rows,
            base_preds=base_preds,
            adapter_preds=adapter_preds,
            base_rval_pseudo_acc=as_float(base_eval.get("rval_pseudo_accuracy")),
            adapter_rval_pseudo_acc_80step=adapter_rval_pseudo,
            inference_resource=as_float(ewc.get("I")),
            training_resource=as_float(ewc.get("R")),
            triggered=True,
            adapter_accepted=True,
            args=args,
        )
        decision["actual_adapter_source"] = "ewc_target320_checkpoint_fisher_counted"
        decision["ewc_train_runtime_s"] = as_float(ewc.get("actual_target_train_runtime_s"))
        decision["ewc_fisher_runtime_s"] = fisher_runtime
        decision["ewc_counted_retrain_runtime_s"] = counted_runtime
        decision["target_adapter_path"] = ewc.get("target_adapter_path", "")
        decision["target_adapter_raw_predictions"] = ewc.get("target_adapter_raw_predictions", "")
        out.append(decision)
    return out


def recompute_qwen35_rates(*, out_root: Path, rates: list[float]) -> list[dict[str, Any]]:
    q35 = load_module(PROJECT_ROOT / "scripts" / "run_qwen35_omniearth_formal_framework_augmented_v2.py", "pressure_qwen35_runner")
    if uses_rtest_stream():
        original_load_group_rows = q35.load_group_rows

        def load_group_rows_rtest(
            split_root: Path,
            pseudo_root: Path,
            group_id: str,
        ) -> dict[str, list[dict[str, Any]]]:
            rows = original_load_group_rows(split_root, pseudo_root, group_id)
            rows["R_all"] = list(rows["R_test"])
            return rows

        q35.load_group_rows = load_group_rows_rtest
    all_rows: list[dict[str, Any]] = []
    for rate in rates:
        label = rate_label(rate)
        out_dir = out_root / "qwen35" / label
        args = qwen35_args_for_rate(QWEN35_RUN_DIR, out_dir, rate)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lambda_rows = q35.read_csv(args.lambda_root / "lambda_dtop80_by_config.csv")
        gamma_rows = q35.read_csv(args.gamma_root / "gamma_smallstep_by_config.csv")
        detector_summary = q35.read_json(args.detector_root / "formal_detector_summary.json")
        group_ids = sorted(path.name for path in (args.split_root / "groups").glob("*") if path.is_dir())
        cache = q35.maybe_load_eval_cache(QWEN35_RUN_DIR)
        if cache is None:
            raise FileNotFoundError(f"Qwen3.5 eval cache not found under {QWEN35_RUN_DIR}")
        base_eval_rows, adapter_eval_rows = cache
        method_rows, overall_rows, lambda_detail, gamma_detail, pair_rows = q35.run_scheduler(
            args=args,
            group_ids=group_ids,
            lambda_rows_all=lambda_rows,
            gamma_rows_all=gamma_rows,
            base_eval_rows=base_eval_rows,
            adapter_eval_rows=adapter_eval_rows,
            detector_summary=detector_summary,
        )
        method_rows.extend(
            build_qwen35_ewc_rows(
                q35=q35,
                args=args,
                group_ids=group_ids,
                method_rows=method_rows,
                gamma_detail=gamma_detail,
                base_eval_rows=base_eval_rows,
            )
        )
        method_rows.sort(key=lambda row: (row["group_id"], METHOD_ORDER.index(row["method"])))
        overall_rows = q35.aggregate_method_rows(method_rows)
        overall_rows.sort(key=lambda row: METHOD_ORDER.index(row["method"]))
        q35.write_csv(out_dir / "scheduler" / "scheduler_end_to_end_results.csv", method_rows)
        q35.write_csv(out_dir / "scheduler" / "scheduler_overall_weighted_results.csv", overall_rows)
        q35.write_csv(out_dir / "true_accuracy" / "scheduler_true_accuracy_error_rows.csv", method_rows)
        q35.write_csv(out_dir / "detailed_result_tables" / "lambda_dtop80_by_group_config.csv", lambda_detail)
        q35.write_csv(out_dir / "detailed_result_tables" / "gamma_smallstep_by_group_config.csv", gamma_detail)
        q35.write_csv(out_dir / "detailed_result_tables" / "lambda_gamma_estimates_all_pairs.csv", pair_rows)
        q35.write_csv(out_dir / "detailed_result_tables" / "selected_config_estimated_vs_actual.csv", method_rows)
        q35.write_json(
            out_dir / "framework_results.json",
            {
                "protocol": {
                    "model_id": q35.MODEL_ID,
                    "teacher_model": "Qwen3.5-27B LoRA checkpoint-90",
                    "student_model": q35.MODEL_ID,
                    "pressure_recompute_from": str(QWEN35_RUN_DIR),
                    "split_root": str(args.split_root),
                    "pseudo_root": str(args.pseudo_root),
                    "lambda_root": str(args.lambda_root),
                    "gamma_root": str(args.gamma_root),
                    "detector_root": str(args.detector_root),
                    "resource_grid": "I=0.01..0.99, R=1-I",
                    "actual_adapter_source": "80-step microprofile checkpoints",
                    "arrival_rate": rate,
                    "retraining_window_sec_reference": args.retraining_window_sec,
                    "eval_stream": EVAL_STREAM,
                },
                "detector": detector_summary,
                "methods": method_rows,
                "overall": overall_rows,
                "lambda_detail": lambda_detail,
                "gamma_detail": gamma_detail,
                "pair_rows": pair_rows,
            },
        )
        q35.write_summary(
            out_dir,
            detector_summary=detector_summary,
            method_rows=method_rows,
            overall_rows=overall_rows,
            lambda_detail=lambda_detail,
            gamma_detail=gamma_detail,
            pair_rows=pair_rows,
        )
        for row in sorted(overall_rows, key=lambda item: METHOD_ORDER.index(item["method"])):
            all_rows.append(
                {
                    "model": "Qwen3.5-0.8B OmniEarth",
                    "model_family": "qwen35",
                    "pressure_scenario": label,
                    "arrival_rate": rate,
                    "method": row["method"],
                    "方法": METHOD_ZH.get(row["method"], row["method"]),
                    "main_metric_name": "actual_long_avg_accuracy",
                    "main_metric": as_float(row.get("actual_long_avg_accuracy")),
                    "time_weighted_metric": as_float(row.get("observed_time_weighted_accuracy")),
                    "final_metric": as_float(row.get("observed_final_accuracy")),
                    "inference_config_field": "lambda_id",
                    "training_config_field": "gamma_id",
                    "selected_inference_configs": row.get("selected_lambda_ids", ""),
                    "selected_training_configs": row.get("selected_gamma_ids", ""),
                    "I_by_group": row.get("I_by_group", ""),
                    "R_by_group": row.get("R_by_group", ""),
                    "adapter_ready_samples_avg": "",
                    "native_summary": str(out_dir / "scheduler" / "scheduler_overall_weighted_results.csv"),
                    "native_detail": str(out_dir / "scheduler" / "scheduler_end_to_end_results.csv"),
                    "source": str(QWEN35_RUN_DIR),
                }
            )
    return add_relative_metrics(all_rows)


def configure_nonqwen_runner(module: Any, family: str) -> None:
    if family == "remoteclip":
        module.MODEL_ID = "RemoteCLIP-RN50"
        module.GROUP_ID = "remoteclip_rn50_landcover_disaster"
        return
    if family == "resnet18":
        module.MODEL_ID = "ResNet18"
        module.GROUP_ID = "resnet18_landcover_disaster"

        def resnet_gamma_configs() -> list[Any]:
            return [module.GammaConfig(gamma_id=f"resnet18_linear_ep{epochs}", epochs=epochs) for epochs in (1, 3, 5)]

        module.gamma_configs = resnet_gamma_configs
        return
    raise ValueError(f"Unknown non-Qwen family: {family}")


def nonqwen_args_for_rate(source_dir: Path, out_dir: Path, rate: float) -> Namespace:
    framework = read_json(source_dir / "framework_results.json")
    protocol = framework["protocol"]
    return Namespace(
        output_dir=out_dir,
        arrival_rate=rate,
        retraining_window_sec=as_float(protocol.get("retraining_window_sec"), 100.0),
        future_windows=3,
        discount_factor=0.95,
        min_profile_gain=0.0,
        periodic_inference_resource=0.50,
        static_inference_resource=0.70,
        scheduler_estimate_mode=protocol.get("scheduler_estimate_mode", "target_rval"),
        gamma_extrapolation_method=protocol.get("gamma_extrapolation_method", "direct_measured"),
        scheduler_optimus_max_accuracy=as_float(protocol.get("scheduler_optimus_max_accuracy"), 1.0),
        periodic_period_samples=as_int(protocol.get("periodic_period_samples"), 0),
    )


def recompute_nonqwen_pair_rows(
    *,
    runner: Any,
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    predictions_by_lambda: dict[str, dict[str, str]],
    adapter_predictions: dict[tuple[str, str], dict[str, str]],
    splits: dict[str, list[dict[str, Any]]],
    pseudo: dict[str, str],
    triggered: bool,
    args: Namespace,
    ours: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lambda_row in lambda_rows:
        lambda_id = lambda_row["lambda_id"]
        base_preds = predictions_by_lambda[lambda_id]
        for gamma_row in [row for row in gamma_rows if row["feature_lambda_id"] == lambda_id]:
            gamma_id = gamma_row["gamma_id"]
            adapter_preds = adapter_predictions[(lambda_id, gamma_id)]
            adapter_accepted = runner.scheduler_projected_gain(gamma_row, args) > args.min_profile_gain
            best: dict[str, Any] | None = None
            for idx in range(1, 100):
                I = idx / 100.0
                row = runner.build_decision(
                    method="Ours",
                    lambda_row=lambda_row,
                    gamma_row=gamma_row,
                    r_all=splits["R_all"],
                    no_drift_rows=splits["R_no_drift"],
                    r_val=splits["R_val"],
                    r_test=splits["R_test"],
                    pseudo=pseudo,
                    base_predictions=base_preds,
                    adapter_predictions=adapter_preds,
                    inference_resource=I,
                    training_resource=1.0 - I,
                    triggered=triggered,
                    adapter_accepted=adapter_accepted,
                    args=args,
                )
                if best is None or as_float(row["estimated_long_avg_accuracy"]) > as_float(best["estimated_long_avg_accuracy"]):
                    best = row
            if best is None:
                continue
            out.append(
                {
                    "group_id": runner.GROUP_ID,
                    "lambda_id": lambda_id,
                    "gamma_id": gamma_id,
                    "is_selected_pair_by_ours": lambda_id == ours["lambda_id"] and gamma_id == ours["gamma_id"],
                    "best_I_for_pair": best["I"],
                    "best_R_for_pair": best["R"],
                    "dtop80_true_accuracy": lambda_row.get("true_accuracy", ""),
                    "dtop80_teacher_accuracy": lambda_row.get("teacher_accuracy", ""),
                    "lambda_resource_cost": lambda_row.get("resource_cost", ""),
                    "small_step_profile_gain": gamma_row.get("profile_gain", ""),
                    "extrapolated_target_gain": runner.scheduler_projected_gain(gamma_row, args),
                    "rval_base_accuracy_before_1epoch": gamma_row.get("rval_base_accuracy_before_1epoch", ""),
                    "rval_adapter_accuracy_after_target": gamma_row.get("rval_adapter_accuracy_after_target", ""),
                    "target_epochs": gamma_row.get("target_epochs", ""),
                    "gamma_resource_cost": gamma_row.get("resource_cost", ""),
                    "estimated_current_accuracy_if_selected": best["estimated_current_accuracy"],
                    "estimated_future_or_final_accuracy_if_selected": best["estimated_future_or_final_accuracy"],
                    "estimated_long_avg_accuracy_if_selected": best["estimated_long_avg_accuracy"],
                    "actual_current_accuracy_if_selected": best["observed_current_accuracy"],
                    "actual_final_accuracy_if_selected": best["observed_final_accuracy"],
                    "actual_long_avg_accuracy_if_selected": best["actual_long_avg_accuracy"],
                    "actual_time_weighted_accuracy_if_selected": best["observed_time_weighted_accuracy"],
                    "current_error_if_selected": best["current_error_actual_minus_estimated"],
                    "final_error_if_selected": best["final_error_actual_minus_estimated_future"],
                }
            )
    return out


def choose_nonqwen_offline_oracle_by_actual_long(
    *,
    runner: Any,
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    predictions_by_lambda: dict[str, dict[str, str]],
    adapter_predictions: dict[tuple[str, str], dict[str, str]],
    splits: dict[str, list[dict[str, Any]]],
    pseudo: dict[str, str],
    args: Namespace,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    gamma_by_lambda = {(row["feature_lambda_id"], row["gamma_id"]): row for row in gamma_rows}
    for lambda_row in lambda_rows:
        lambda_id = lambda_row["lambda_id"]
        base_preds = predictions_by_lambda[lambda_id]
        for gamma_cfg in runner.gamma_configs():
            gamma_row = gamma_by_lambda[(lambda_id, gamma_cfg.gamma_id)]
            adapter_preds = adapter_predictions[(lambda_id, gamma_cfg.gamma_id)]
            for idx in range(1, 100):
                I = idx / 100.0
                candidates.append(
                    runner.build_decision(
                        method="OfflineJointOracle",
                        lambda_row=lambda_row,
                        gamma_row=gamma_row,
                        r_all=splits["R_all"],
                        no_drift_rows=splits["R_no_drift"],
                        r_val=splits["R_val"],
                        r_test=splits["R_test"],
                        pseudo=pseudo,
                        base_predictions=base_preds,
                        adapter_predictions=adapter_preds,
                        inference_resource=I,
                        training_resource=1.0 - I,
                        triggered=True,
                        adapter_accepted=True,
                        args=args,
                    )
                )
    return max(
        candidates,
        key=lambda row: (
            as_float(row.get("actual_long_avg_accuracy")),
            as_float(row.get("observed_final_accuracy")),
            as_float(row.get("observed_time_weighted_accuracy")),
            as_float(row.get("estimated_long_avg_accuracy")),
        ),
    )


def load_nonqwen_ewc_bundle(ewc_dir: Path) -> dict[str, Any] | None:
    method_path = ewc_dir / "ewc_method_row.json"
    gamma_path = ewc_dir / "ewc_gamma_profile.json"
    if not method_path.exists() or not gamma_path.exists():
        return None
    method = read_json(method_path)
    gamma = read_json(gamma_path)
    predictions_path = Path(
        method.get("ewc_adapter_predictions_path")
        or gamma.get("adapter_predictions_path")
        or gamma["adapter_path"]
    )
    return {
        "method": method,
        "gamma": gamma,
        "predictions": load_predictions(predictions_path),
    }


def build_nonqwen_ewc_pressure_row(
    *,
    runner: Any,
    bundle: dict[str, Any],
    lambda_rows: list[dict[str, Any]],
    predictions_by_lambda: dict[str, dict[str, str]],
    splits: dict[str, list[dict[str, Any]]],
    pseudo: dict[str, str],
    args: Namespace,
) -> dict[str, Any]:
    source_method = bundle["method"]
    lambda_row = next(row for row in lambda_rows if row["lambda_id"] == source_method["lambda_id"])
    row = runner.build_decision(
        method="EWC",
        lambda_row=lambda_row,
        gamma_row=bundle["gamma"],
        r_all=splits["R_all"],
        no_drift_rows=splits["R_no_drift"],
        r_val=splits["R_val"],
        r_test=splits["R_test"],
        pseudo=pseudo,
        base_predictions=predictions_by_lambda[lambda_row["lambda_id"]],
        adapter_predictions=bundle["predictions"],
        inference_resource=as_float(source_method["I"]),
        training_resource=as_float(source_method["R"]),
        triggered=True,
        adapter_accepted=True,
        args=args,
        training_start_delay_s=0.0,
    )
    for key in (
        "ewc_lambda",
        "ewc_anchor_path",
        "ewc_fisher_path",
        "ewc_adapter_path",
        "ewc_adapter_predictions_path",
        "ewc_anchor_runtime_s_excluded",
        "ewc_fisher_runtime_s",
        "ewc_target_train_runtime_s",
        "ewc_counted_runtime_s",
        "notes",
    ):
        if key in source_method:
            row[key] = source_method[key]
    return row


def recompute_nonqwen_rates(
    *,
    out_root: Path,
    rates: list[float],
    family: str,
    source_dir: Path,
    display_model: str,
) -> list[dict[str, Any]]:
    runner = load_module(
        PROJECT_ROOT / "scripts" / "run_omniearth_remoteclip_formal_framework.py",
        f"pressure_{family}_runner",
    )
    configure_nonqwen_runner(runner, family)
    framework = read_json(source_dir / "framework_results.json")
    ewc_root = REMOTECLIP_EWC_ROOT if family == "remoteclip" else RESNET18_EWC_ROOT
    ewc_bundle = load_nonqwen_ewc_bundle(ewc_root)
    all_rows: list[dict[str, Any]] = []
    for rate in rates:
        label = rate_label(rate)
        out_dir = out_root / family / label
        out_dir.mkdir(parents=True, exist_ok=True)
        args = nonqwen_args_for_rate(source_dir, out_dir, rate)
        splits = runner.load_existing_splits(source_dir / "splits")
        if uses_rtest_stream():
            splits["R_all"] = list(splits["R_test"])
        pseudo = runner.load_external_pseudo_labels(source_dir / "teacher" / "pseudo_labels")
        lambda_rows = pressure_adjust_lambda_rows(
            read_csv(source_dir / "lambda_profiles" / "lambda_profiles_group_v2.csv"),
            rate,
        )
        gamma_rows = read_csv(source_dir / "gamma_profiles" / "gamma_group_profiles.csv")
        predictions_by_lambda = {
            row["lambda_id"]: load_predictions(Path(row["raw_predictions"])) for row in lambda_rows
        }
        adapter_predictions = {
            (row["feature_lambda_id"], row["gamma_id"]): load_predictions(
                Path(row.get("adapter_predictions_path") or row["adapter_path"])
            )
            for row in gamma_rows
        }
        triggered = bool(framework.get("detection", {}).get("actual_D_window_triggered", True))
        baseline_rows = runner.build_baseline_decisions(
            lambda_rows=lambda_rows,
            gamma_rows=gamma_rows,
            predictions_by_lambda=predictions_by_lambda,
            adapter_predictions=adapter_predictions,
            splits=splits,
            pseudo=pseudo,
            args=args,
        )
        ewc = (
            build_nonqwen_ewc_pressure_row(
                runner=runner,
                bundle=ewc_bundle,
                lambda_rows=lambda_rows,
                predictions_by_lambda=predictions_by_lambda,
                splits=splits,
                pseudo=pseudo,
                args=args,
            )
            if ewc_bundle is not None
            else None
        )
        ours = runner.choose_ours(
            lambda_rows=lambda_rows,
            gamma_rows=gamma_rows,
            predictions_by_lambda=predictions_by_lambda,
            adapter_predictions=adapter_predictions,
            splits=splits,
            pseudo=pseudo,
            triggered=triggered,
            args=args,
        )
        oracle = choose_nonqwen_offline_oracle_by_actual_long(
            runner=runner,
            lambda_rows=lambda_rows,
            gamma_rows=gamma_rows,
            predictions_by_lambda=predictions_by_lambda,
            adapter_predictions=adapter_predictions,
            splits=splits,
            pseudo=pseudo,
            args=args,
        )
        method_rows = baseline_rows + ([ewc] if ewc is not None else []) + [ours, oracle]
        pair_rows = recompute_nonqwen_pair_rows(
            runner=runner,
            lambda_rows=lambda_rows,
            gamma_rows=gamma_rows,
            predictions_by_lambda=predictions_by_lambda,
            adapter_predictions=adapter_predictions,
            splits=splits,
            pseudo=pseudo,
            triggered=triggered,
            args=args,
            ours=ours,
        )
        runner.write_csv(out_dir / "scheduler" / "scheduler_end_to_end_results.csv", method_rows)
        runner.write_csv(out_dir / "method_comparison.csv", method_rows)
        runner.write_json(out_dir / "method_comparison.json", method_rows)
        runner.write_csv(out_dir / "true_accuracy" / "scheduler_true_accuracy_error_rows.csv", method_rows)
        runner.write_csv(out_dir / "detailed_result_tables" / "lambda_gamma_estimates_all_pairs.csv", pair_rows)
        runner.build_detailed_tables(out_dir, lambda_rows, gamma_rows, pair_rows, method_rows)
        write_json(
            out_dir / "framework_results.json",
            {
                "protocol": {
                    **framework.get("protocol", {}),
                    "pressure_recompute_from": str(source_dir),
                    "arrival_rate": rate,
                    "retraining_window_sec_reference": args.retraining_window_sec,
                    "resource_grid": "I=0.01..0.99, R=1-I",
                    "eval_stream": EVAL_STREAM,
                },
                "split_summary": {key: len(value) for key, value in splits.items()},
                "detection": framework.get("detection", {}),
                "teacher_quality": framework.get("teacher_quality", {}),
                "lambda_profiles": lambda_rows,
                "gamma_profiles": gamma_rows,
                "methods": method_rows,
                "pair_rows": pair_rows,
            },
        )
        write_simple_native_summary(out_dir, display_model, rate, method_rows)
        for row in sorted(method_rows, key=lambda item: METHOD_ORDER.index(item["method"])):
            all_rows.append(
                {
                    "model": display_model,
                    "model_family": family,
                    "pressure_scenario": label,
                    "arrival_rate": rate,
                    "method": row["method"],
                    "方法": METHOD_ZH.get(row["method"], row["method"]),
                    "main_metric_name": "actual_long_avg_accuracy",
                    "main_metric": as_float(row.get("actual_long_avg_accuracy")),
                    "time_weighted_metric": as_float(row.get("observed_time_weighted_accuracy")),
                    "final_metric": as_float(row.get("observed_final_accuracy")),
                    "inference_config_field": "lambda_id",
                    "training_config_field": "gamma_id",
                    "selected_inference_configs": row.get("lambda_id", ""),
                    "selected_training_configs": row.get("gamma_id", ""),
                    "I_by_group": row.get("I", ""),
                    "R_by_group": row.get("R", ""),
                    "adapter_ready_samples_avg": row.get("adapter_ready_sample_count", ""),
                    "native_summary": str(out_dir / "method_comparison.csv"),
                    "native_detail": str(out_dir / "detailed_result_tables" / "selected_config_estimated_vs_actual.csv"),
                    "source": str(source_dir),
                }
            )
    return add_relative_metrics(all_rows)


def recompute_family_rates(*, out_root: Path, family: str, rates: list[float]) -> list[dict[str, Any]]:
    if family == "qwen25":
        return collect_qwen25_pressure_rates(out_root=out_root, rates=rates)
    if family == "qwen35":
        return recompute_qwen35_rates(out_root=out_root, rates=rates)
    if family == "remoteclip":
        return recompute_nonqwen_rates(
            out_root=out_root,
            rates=rates,
            family="remoteclip",
            source_dir=REMOTECLIP_RUN_DIR,
            display_model="RemoteCLIP-ViT-B-32 Teacher + RemoteCLIP-RN50 Student",
        )
    if family == "resnet18":
        return recompute_nonqwen_rates(
            out_root=out_root,
            rates=rates,
            family="resnet18",
            source_dir=RESNET18_RUN_DIR,
            display_model="ResNet18 Student + ResNet50 Teacher",
        )
    raise ValueError(f"Unknown family: {family}")


def load_label(factor: float) -> str:
    return f"load{factor:g}x".replace(".", "p")


def annotate_normalized_load_rows(
    *,
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    rate_by_factor: dict[float, float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rate = as_float(row["arrival_rate"])
        factor = min(rate_by_factor, key=lambda item: abs(rate_by_factor[item] - rate))
        item = dict(row)
        item["pressure_scenario"] = load_label(factor)
        item["load_factor_vs_noretrain_throughput"] = factor
        item["reference_noretrain_throughput_samples_per_sec"] = spec["reference_throughput_samples_per_sec"]
        item["reference_noretrain_lambda_ids"] = spec["selected_noretrain_lambda_ids"]
        out.append(item)
    return out


def recompute_normalized_load_scenarios(
    *,
    out_root: Path,
    base_rows: list[dict[str, Any]],
    load_factors: list[float],
) -> list[dict[str, Any]]:
    norm_root = out_root / "normalized_load_runs"
    rows: list[dict[str, Any]] = []
    for spec in selected_noretrain_throughput_specs(base_rows):
        family = str(spec["model_family"])
        reference = as_float(spec["reference_throughput_samples_per_sec"])
        rate_by_factor = {factor: reference * factor for factor in load_factors}
        family_rows = recompute_family_rates(
            out_root=norm_root,
            family=family,
            rates=[rate_by_factor[factor] for factor in load_factors],
        )
        rows.extend(
            annotate_normalized_load_rows(
                rows=family_rows,
                spec=spec,
                rate_by_factor=rate_by_factor,
            )
        )
    rows = add_relative_metrics(rows)
    rows.sort(
        key=lambda row: (
            family_sort_key(str(row["model_family"])),
            as_float(row.get("load_factor_vs_noretrain_throughput")),
            METHOD_ORDER.index(str(row["method"])),
        )
    )
    return rows


def build_no_retrain_load_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method") != "NoRetrain":
            continue
        load = as_float(row.get("load_factor_vs_noretrain_throughput"))
        expected_coverage = min(1.0, 1.0 / max(load, 1e-9))
        out.append(
            {
                "model_family": row["model_family"],
                "model": row["model"],
                "load_factor_vs_noretrain_throughput": load,
                "arrival_rate": row["arrival_rate"],
                "reference_noretrain_throughput_samples_per_sec": row.get("reference_noretrain_throughput_samples_per_sec", ""),
                "reference_noretrain_lambda_ids": row.get("reference_noretrain_lambda_ids", ""),
                "expected_noretrain_coverage": round(expected_coverage, 6),
                "actual_long_avg_accuracy": row["main_metric"],
                "selected_inference_configs": row.get("selected_inference_configs", ""),
                "native_summary": row.get("native_summary", ""),
            }
        )
    return out


def write_normalized_load_report(out_dir: Path, rows: list[dict[str, Any]], load_factors: list[float]) -> list[str]:
    write_csv(out_dir / "normalized_load_method_summary.csv", rows)
    no_retrain_curve = build_no_retrain_load_curve(rows)
    write_csv(out_dir / "normalized_load_no_retrain_curve.csv", no_retrain_curve)
    matrix_rows = build_method_matrix(rows)
    lines = [
        "# 全模型归一化压力场景结果",
        "",
        f"- 负载因子：{', '.join(f'`{factor:g}x`' for factor in load_factors)}。",
        "- 每个模型以自己的 NoRetrain 选中推理配置吞吐为 `1.0x`，因此该表专门用于检查 NoRetrain 是否随压力升高而退化。",
        "- 表中 `arrival_rate` 是按 `load_factor × NoRetrain参考吞吐` 换算得到的模型专用 samples/s。",
        "",
        "## NoRetrain 退化曲线",
        "",
        "| 模型 | load | arrival_rate | 参考吞吐 | 期望coverage | NoRetrain actual_long |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in no_retrain_curve:
        lines.append(
            f"| {row['model']} | {as_float(row['load_factor_vs_noretrain_throughput']):g}x | "
            f"{as_float(row['arrival_rate']):.6f} | "
            f"{as_float(row['reference_noretrain_throughput_samples_per_sec']):.6f} | "
            f"{as_float(row['expected_noretrain_coverage']):.6f} | "
            f"{as_float(row['actual_long_avg_accuracy']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 六方法归一化压力矩阵",
            "",
            "| 模型 | load | arrival_rate | NoRetrain | PeriodicFixedRetrain | StaticSplitContinuous | EWC | Ours | OfflineJointOracle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matrix_rows:
        load = next(
            (
                as_float(source.get("load_factor_vs_noretrain_throughput"))
                for source in rows
                if source["model_family"] == row["model_family"]
                and same_rate(source["arrival_rate"], row["arrival_rate"])
            ),
            0.0,
        )
        lines.append(
            f"| {row['model']} | {load:g}x | {as_float(row['arrival_rate']):.6f} | "
            f"{metric_text(row['NoRetrain'])} | {metric_text(row['PeriodicFixedRetrain'])} | "
            f"{metric_text(row['StaticSplitContinuous'])} | {metric_text(row['EWC'])} | "
            f"{metric_text(row['Ours'])} | {metric_text(row['OfflineJointOracle'])} |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 归一化压力逐方法表：`{out_dir / 'normalized_load_method_summary.csv'}`",
            f"- NoRetrain 退化曲线：`{out_dir / 'normalized_load_no_retrain_curve.csv'}`",
        ]
    )
    (out_dir / "normalized_load_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def build_normalized_load_main_lines(out_dir: Path, rows: list[dict[str, Any]], load_factors: list[float]) -> list[str]:
    no_retrain_curve = build_no_retrain_load_curve(rows)
    lines = [
        "",
        "## NoRetrain 归一化压力诊断",
        "",
        "- 绝对到达率表中 NoRetrain 长时间不变，是因为这些点低于该模型 NoRetrain 选中推理配置的吞吐；下面把每个模型自己的 NoRetrain 吞吐归一为 `1.0x`。",
        f"- 负载因子：{', '.join(f'`{factor:g}x`' for factor in load_factors)}；当 load `>1.0x` 时，理论 coverage 约为 `1/load`。",
        "",
        "| 模型 | load | arrival_rate | 参考NoRetrain吞吐 | 期望coverage | NoRetrain actual_long |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in no_retrain_curve:
        lines.append(
            f"| {row['model']} | {as_float(row['load_factor_vs_noretrain_throughput']):g}x | "
            f"{as_float(row['arrival_rate']):.6f} | "
            f"{as_float(row['reference_noretrain_throughput_samples_per_sec']):.6f} | "
            f"{as_float(row['expected_noretrain_coverage']):.6f} | "
            f"{as_float(row['actual_long_avg_accuracy']):.6f} |"
        )
    lines.extend(
        [
            "",
            "归一化压力完整表见：",
            f"`{out_dir / 'normalized_load_summary.md'}`",
        ]
    )
    return lines


def write_simple_native_summary(out_dir: Path, model: str, rate: float, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {model} Pressure Scenario",
        "",
        f"- arrival_rate: `{rate:g}` samples/s",
        "- Rows are recomputed from saved final-run native profiles and predictions; no adapter retraining is performed.",
        "",
        "| method | lambda_id | gamma_id | I | R | ready_n | actual_long | R_all_time_weighted | current_with_coverage | final |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['lambda_id']} | {row['gamma_id']} | {row['I']} | {row['R']} | "
            f"{row['adapter_ready_sample_count']} | {row['actual_long_avg_accuracy']} | "
            f"{row['observed_time_weighted_accuracy']} | "
            f"{row['observed_current_accuracy']} | {row['observed_final_accuracy']} |"
        )
    (out_dir / "pressure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_method_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model_family"]), str(row["model"]), as_float(row["arrival_rate"]))
        groups.setdefault(key, {})[str(row["method"])] = as_float(row["main_metric"])
    matrix: list[dict[str, Any]] = []
    for (family, model, rate), values in sorted(groups.items(), key=lambda item: (family_sort_key(item[0][0]), item[0][2])):
        item: dict[str, Any] = {
            "model_family": family,
            "model": model,
            "arrival_rate": rate,
        }
        for method in METHOD_ORDER:
            item[method] = values.get(method, "")
        matrix.append(item)
    return matrix


def build_pressure_section_lines(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    rates: list[float],
    capacity_rows: list[dict[str, Any]],
    spread_rows: list[dict[str, Any]],
) -> list[str]:
    spread_summary = summarize_spread_by_family(spread_rows)
    matrix_rows = build_method_matrix(rows)
    lines = [
        "# 全模型压力场景扩展结果",
        "",
        f"更新日期：{date.today().isoformat()}",
        "",
        "新增结果目录：",
        f"`{out_dir}`",
        "",
        "## 口径",
        "",
        f"- 压力场景按样本到达率展开：{', '.join(f'`{rate:g}`' for rate in rates)} samples/s。",
        "- `10/16/24/32` 是针对 RemoteCLIP/ResNet18 这类快模型补充的高压点；继续升高会让所有方法一起被覆盖率压低，因此主要看跨越吞吐阈值附近的变化。",
        "- 每个模型保留原生字段表；跨模型总表统一抽取 `actual_long_avg_accuracy`。",
        "- 压力场景下推理覆盖率按各模型原生吞吐重算为 `arrival_rate / throughput_samples_per_sec`，因此高到达率会体现 NoRetrain 的超载退化。",
        "- Qwen2.5 复用正式 scheduler/materializer 与 full-R strict reports 后处理；Qwen3.5、RemoteCLIP、ResNet18 从最终实验缓存后处理复算；均不重新训练 adapter。",
        "- EWC 使用各模型正式目录中的独立 EWC 运行产物；RemoteCLIP/ResNet18 保留原 StaticSplitContinuous，并额外复算 EWC。",
        "",
        "## 吞吐阈值诊断",
        "",
        "| 模型 | 候选推理吞吐 min/median/max (samples/s) | sweep 首次超过 min/median/max | 说明 |",
        "|---|---:|---|---|",
    ]
    for row in capacity_rows:
        lines.append(
            f"| {row['model']} | {row['min_throughput_samples_per_sec']:.6f} / "
            f"{row['median_throughput_samples_per_sec']:.6f} / {row['max_throughput_samples_per_sec']:.6f} | "
            f"{row['first_rate_above_min_throughput']} / {row['first_rate_above_median_throughput']} / "
            f"{row['first_rate_above_max_throughput']} | {row['diagnosis']} |"
        )
    lines.extend(
        [
            "",
            "## 差异诊断",
            "",
            "| 模型 | 最大方法 spread(rate) | 最大 Ours-NoRetrain(rate) | 最大 Oracle-Ours(rate) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in spread_summary:
        lines.append(
            f"| {row['model']} | {row['max_method_spread']} ({as_float(row['max_method_spread_rate']):g}) | "
            f"{row['max_ours_minus_noretrain']} ({as_float(row['max_ours_minus_noretrain_rate']):g}) | "
            f"{row['max_oracle_minus_ours']} ({as_float(row['max_oracle_minus_ours_rate']):g}) |"
        )
    lines.extend(
        [
            "",
            "## 六方法压力场景矩阵",
            "",
            "| 模型 | 到达率 | NoRetrain | PeriodicFixedRetrain | StaticSplitContinuous | EWC | Ours | OfflineJointOracle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matrix_rows:
        lines.append(
            f"| {row['model']} | {as_float(row['arrival_rate']):g} | "
            f"{metric_text(row['NoRetrain'])} | {metric_text(row['PeriodicFixedRetrain'])} | "
            f"{metric_text(row['StaticSplitContinuous'])} | {metric_text(row['EWC'])} | "
            f"{metric_text(row['Ours'])} | {metric_text(row['OfflineJointOracle'])} |"
        )
    lines.extend(
        [
            "",
            "## 跨模型逐方法明细",
            "",
            "| 模型 | 到达率 | 方法 | actual_long | Ours相对NoRetrain提升 | Ours距Oracle差距 | 原生表 |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['arrival_rate']:g} | {row['方法']} | {row['main_metric']:.6f} | "
            f"{row['Ours相对NoRetrain提升']:.6f} | {row['Ours距Oracle差距']:.6f} | "
            f"`{row['native_summary']}` |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 逐方法总表：`{out_dir / 'pressure_scenario_method_summary.csv'}`",
            f"- 每 rate 差异诊断：`{out_dir / 'pressure_spread_diagnostics.csv'}`",
            f"- 吞吐阈值诊断：`{out_dir / 'pressure_capacity_diagnostics.csv'}`",
            f"- Markdown 汇总：`{out_dir / 'pressure_scenario_summary.md'}`",
            "",
            "## 模型字段说明",
            "",
            "| 模型族 | 推理配置字段 | 训练配置字段 | 说明 |",
            "|---|---|---|---|",
            "| Qwen2.5 | 推理配置ID | 训练配置ID | 中文最终表字段，LoRA 训练配置为 Qwen2.5 adapter 配置。 |",
            "| Qwen3.5 | lambda_id | gamma_id | Qwen3.5 runner schema，gamma 为 LoRA target-step/profile 配置。 |",
            "| RemoteCLIP | lambda_id | gamma_id | `rc_r50_*` 推理配置，`rc_r50_linear_ep*` 线性 head 重训练配置。 |",
            "| ResNet18 | lambda_id | gamma_id | `resnet18_*` 推理配置，`resnet18_linear_ep*` head 训练配置。 |",
        ]
    )
    return lines


def write_unified_report(out_dir: Path, rows: list[dict[str, Any]], rates: list[float]) -> list[str]:
    capacity_rows = build_capacity_diagnostics(rates)
    spread_rows = build_spread_diagnostics(rows)
    write_csv(out_dir / "pressure_scenario_method_summary.csv", rows)
    write_csv(out_dir / "pressure_capacity_diagnostics.csv", capacity_rows)
    write_csv(out_dir / "pressure_spread_diagnostics.csv", spread_rows)
    lines = build_pressure_section_lines(
        out_dir=out_dir,
        rows=rows,
        rates=rates,
        capacity_rows=capacity_rows,
        spread_rows=spread_rows,
    )
    (out_dir / "pressure_scenario_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def update_main_report_section(report_path: Path, section_lines: list[str]) -> None:
    old_text = report_path.read_text(encoding="utf-8")
    report_lines = list(section_lines)
    if report_lines and report_lines[0] == "# 全模型压力场景扩展结果":
        report_lines[0] = "# 5. 全模型压力场景扩展结果"
    section_text = "\n".join(report_lines) + "\n"
    markers = ["# 5. 全模型压力场景扩展结果", "# 全模型压力场景扩展结果"]
    marker_positions = [old_text.find(marker) for marker in markers if old_text.find(marker) >= 0]
    if not marker_positions:
        new_text = old_text.rstrip() + "\n\n" + section_text
    else:
        start = min(marker_positions)
        new_text = old_text[:start].rstrip() + "\n\n" + section_text
    report_path.write_text(new_text, encoding="utf-8")


def parse_rates(text: str) -> list[float]:
    rates = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not rates:
        raise ValueError("At least one arrival rate is required.")
    return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute all-model pressure scenarios from final experiment artifacts.")
    parser.add_argument("--arrival-rates", default=DEFAULT_ARRIVAL_RATES)
    parser.add_argument("--load-factors", default=DEFAULT_LOAD_FACTORS)
    parser.add_argument(
        "--eval-stream",
        choices=["R_all", "R_test"],
        default=EVAL_STREAM,
        help="Serving and ground-truth evaluation stream. Defaults to legacy R_all.",
    )
    parser.add_argument(
        "--skip-normalized-load",
        action="store_true",
        help="Only write the absolute arrival-rate pressure table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "all_model_pressure_scenarios_20260709",
    )
    parser.add_argument(
        "--update-main-report",
        action="store_true",
        help=f"Replace section 5 in {MAIN_REPORT} with the regenerated pressure section.",
    )
    return parser.parse_args()


def main() -> int:
    global EVAL_STREAM
    args = parse_args()
    EVAL_STREAM = args.eval_stream
    rates = parse_rates(args.arrival_rates)
    load_factors = parse_rates(args.load_factors)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_rows.extend(collect_qwen25_pressure_rates(out_root=out_dir, rates=rates))
    all_rows.extend(recompute_qwen35_rates(out_root=out_dir, rates=rates))
    all_rows.extend(
        recompute_nonqwen_rates(
            out_root=out_dir,
            rates=rates,
            family="remoteclip",
            source_dir=REMOTECLIP_RUN_DIR,
            display_model="RemoteCLIP-ViT-B-32 Teacher + RemoteCLIP-RN50 Student",
        )
    )
    all_rows.extend(
        recompute_nonqwen_rates(
            out_root=out_dir,
            rates=rates,
            family="resnet18",
            source_dir=RESNET18_RUN_DIR,
            display_model="ResNet18 Student + ResNet50 Teacher",
        )
    )
    all_rows.sort(key=lambda row: (row["model_family"], float(row["arrival_rate"]), METHOD_ORDER.index(row["method"])))
    section_lines = write_unified_report(out_dir, all_rows, rates)
    if not args.skip_normalized_load:
        normalized_rows = recompute_normalized_load_scenarios(
            out_root=out_dir,
            base_rows=all_rows,
            load_factors=load_factors,
        )
        write_normalized_load_report(out_dir, normalized_rows, load_factors)
        normalized_main_lines = build_normalized_load_main_lines(out_dir, normalized_rows, load_factors)
        section_lines.extend(normalized_main_lines)
        (out_dir / "pressure_scenario_summary.md").write_text("\n".join(section_lines) + "\n", encoding="utf-8")
    if args.update_main_report:
        update_main_report_section(MAIN_REPORT, section_lines)
        print(f"[ok] updated {MAIN_REPORT}")
    print(f"[ok] wrote {out_dir / 'pressure_scenario_summary.md'}")
    print(f"[ok] wrote {out_dir / 'pressure_scenario_method_summary.csv'}")
    print(f"[ok] wrote {out_dir / 'pressure_spread_diagnostics.csv'}")
    print(f"[ok] wrote {out_dir / 'pressure_capacity_diagnostics.csv'}")
    if not args.skip_normalized_load:
        print(f"[ok] wrote {out_dir / 'normalized_load_method_summary.csv'}")
        print(f"[ok] wrote {out_dir / 'normalized_load_no_retrain_curve.csv'}")
        print(f"[ok] wrote {out_dir / 'normalized_load_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
