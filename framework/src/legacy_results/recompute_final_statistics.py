#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


BASE = Path("@@WORKSPACE@@/drift_lora_project/results/v2_teacher_closed_loop_20260630_181348")
STEP5 = BASE / "step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924"
STEP6 = BASE / "step6_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924"
RUN = STEP6 / "oracle_grid_completion_20260706"
PARETO_ROOT = RUN / "true_accuracy_pareto_grid"
COMBINED = PARETO_ROOT / "scheduler_true_accuracy_error_rows_combined.csv"
MISSING_G3_ADAPTER = (
    RUN
    / "baseline_missing_strict_eval_20260706"
    / "adapter"
    / "v2_g3_A9_B5"
    / "qwen25vl3b_lora_r16_a32_profile80_target320"
    / "final_fp32_res672_tok16_full_keep100_b1"
    / "strict_eval_report.json"
)
OUT = STEP6 / "detailed_result_tables_chinese_recomputed_20260706"

METHOD_ZH = {
    "NoRetrain": "不重训练",
    "PeriodicFixedRetrain": "固定周期重训练",
    "StaticSplitContinuous": "静态资源持续训练",
    "Ours": "本文方法",
    "OfflineJointOracle": "离线联合Oracle",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return int(round(number)) if number is not None else None


def round6(value: Any) -> Any:
    number = as_float(value)
    return round(number, 6) if number is not None else ""


def rate_value(label: str) -> str:
    return label[2:].replace("p", ".") if label.startswith("ar") else label


def method_zh(method: str) -> str:
    return METHOD_ZH.get(method, method)


def report_accuracy(path: Path) -> tuple[float, int, int, int]:
    report = read_json(path)
    accepted = int(report.get("accepted") or report.get("total") or 0)
    correct = int(report.get("correct") or 0)
    total = int(report.get("total") or accepted)
    acc = float(report.get("strict_accuracy") if "strict_accuracy" in report else (correct / accepted if accepted else 0.0))
    return acc, correct, accepted, total


def prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    report = read_json(path)
    pred_path = Path(str(report.get("predictions") or ""))
    out: dict[str, dict[str, Any]] = {}
    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid")
            if uid not in ("", None):
                out[str(uid)] = row
    return out


def prediction_accuracy(preds: dict[str, dict[str, Any]], uids: list[str]) -> tuple[float, int, int, int]:
    correct = 0
    accepted = 0
    missing = 0
    for uid in uids:
        row = preds.get(uid)
        if row is None:
            missing += 1
            continue
        if row.get("accepted"):
            accepted += 1
            if row.get("correct"):
                correct += 1
    return (correct / accepted if accepted else 0.0), correct, accepted, missing


def group_uids(group_id: str) -> list[str]:
    rows = read_json(STEP5 / "inputs" / "true_5group_rwindow_root" / "groups" / group_id / "R_all" / "data.json")
    return [str(row["uid"]) for row in rows]


def lambda_resource_cost(lambda_row: dict[str, Any], arrival_rate: float) -> float:
    throughput = as_float(lambda_row.get("throughput_samples_per_sec", lambda_row.get("throughput_samples_per_s")))
    if throughput and throughput > 0:
        return arrival_rate / throughput
    return as_float(lambda_row.get("resource_cost")) or 1.0


def compute_actual_from_reports(
    schedule_row: dict[str, str],
    *,
    label: str,
    base_report: Path,
    adapter_report: Path | None,
    lambda_profiles: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    group_id = str(schedule_row["group_id"])
    lambda_id = str(schedule_row["lambda_id"])
    gamma_id = str(schedule_row.get("gamma_id") or "")
    rate = as_float(rate_value(label)) or 0.0
    ordered_uids = group_uids(group_id)
    base_acc, base_correct, base_accepted, base_total = report_accuracy(base_report)
    base_preds = prediction_map(base_report)
    adapter_active = bool(gamma_id) and str(schedule_row.get("adapter_accepted", "")).lower() in {"true", "1", "yes"}
    if adapter_active:
        if adapter_report is None:
            raise FileNotFoundError(f"missing adapter report for {group_id}/{gamma_id}/{lambda_id}")
        adapter_acc, adapter_correct, adapter_accepted, _ = report_accuracy(adapter_report)
        adapter_preds = prediction_map(adapter_report)
        adapter_report_text = str(adapter_report)
        adapter_path = read_json(adapter_report).get("adapter_path", "")
    else:
        adapter_acc, adapter_correct, adapter_accepted = base_acc, base_correct, base_accepted
        adapter_preds = base_preds
        adapter_report_text = ""
        adapter_path = ""
    lambda_row = lambda_profiles[(group_id, lambda_id)]
    lambda_cost = lambda_resource_cost(lambda_row, rate)
    inference_resource = as_float(schedule_row.get("I")) or 0.0
    coverage = min(1.0, inference_resource / max(lambda_cost, 1e-9))
    finish_time = as_float(schedule_row.get("adapter_ready_time_s")) or as_float(schedule_row.get("train_time_s_effective"))
    ready_count = len(ordered_uids)
    if adapter_active and finish_time is not None:
        ready_count = max(0, min(len(ordered_uids), math.ceil(rate * finish_time)))
    before_uids = ordered_uids[:ready_count]
    after_uids = ordered_uids[ready_count:]
    before_acc, before_correct, before_accepted, before_missing = prediction_accuracy(base_preds, before_uids)
    after_acc, after_correct, after_accepted, after_missing = prediction_accuracy(adapter_preds, after_uids)
    accepted = before_accepted + after_accepted
    processed_acc = (before_correct + after_correct) / accepted if accepted else 0.0
    observed_current = coverage * processed_acc
    observed_final = coverage * (adapter_acc if adapter_active else base_acc)
    estimated_current = as_float(schedule_row.get("current_window_accuracy")) or 0.0
    return {
        "group_id": group_id,
        "method": schedule_row.get("method", ""),
        "lambda_id": lambda_id,
        "gamma_id": gamma_id,
        "I": schedule_row.get("I", ""),
        "R": schedule_row.get("R", ""),
        "adapter_accepted": schedule_row.get("adapter_accepted", ""),
        "estimated_current_accuracy": estimated_current,
        "estimated_long_avg_accuracy": as_float(schedule_row.get("long_avg_accuracy")) or estimated_current,
        "eval_split": "R_all",
        "arrival_rate": rate,
        "lambda_resource_cost_effective": round6(lambda_cost),
        "coverage": round6(coverage),
        "observed_base_accuracy": round6(base_acc),
        "observed_adapter_accuracy": round6(adapter_acc),
        "adapter_ready_sample_count": ready_count if adapter_active else len(ordered_uids),
        "before_adapter_accuracy": round6(before_acc),
        "before_adapter_correct": before_correct,
        "before_adapter_accepted": before_accepted,
        "before_adapter_total": len(before_uids),
        "before_adapter_missing_predictions": before_missing,
        "after_adapter_accuracy": round6(after_acc),
        "after_adapter_correct": after_correct,
        "after_adapter_accepted": after_accepted,
        "after_adapter_total": len(after_uids),
        "after_adapter_missing_predictions": after_missing,
        "observed_time_weighted_accuracy": round6(processed_acc),
        "observed_current_accuracy": round6(observed_current),
        "observed_final_accuracy": round6(observed_final),
        "current_error": round6(observed_current - estimated_current),
        "abs_current_error": round6(abs(observed_current - estimated_current)),
        "final_minus_estimated_current": round6(observed_final - estimated_current),
        "lambda_teacher_accuracy": lambda_row.get("teacher_accuracy", lambda_row.get("accuracy", "")),
        "lambda_true_accuracy": lambda_row.get("true_accuracy", ""),
        "base_correct": base_correct,
        "base_accepted": base_accepted,
        "base_total": base_total,
        "adapter_correct": adapter_correct,
        "adapter_accepted_count": adapter_accepted,
        "adapter_path": adapter_path,
        "base_report": str(base_report),
        "adapter_report": adapter_report_text,
    }


def chinese_detail_row(label: str, schedule_row: dict[str, str], actual: dict[str, Any], *, note: str = "") -> dict[str, Any]:
    method = str(schedule_row.get("method", actual.get("method", "")))
    return {
        "到达率(samples/s)": rate_value(label),
        "方法": method_zh(method),
        "方法代码": method,
        "组ID": actual.get("group_id", schedule_row.get("group_id", "")),
        "推理配置ID": actual.get("lambda_id", schedule_row.get("lambda_id", "")),
        "训练配置ID": actual.get("gamma_id", schedule_row.get("gamma_id", "")),
        "真实评测是否匹配当前配置": "True",
        "真实评测匹配备注": note,
        "推理资源I": round6(schedule_row.get("I", actual.get("I", ""))),
        "训练资源R": round6(schedule_row.get("R", actual.get("R", ""))),
        "是否使用adapter": schedule_row.get("adapter_accepted", actual.get("adapter_accepted", "")),
        "重训练窗口(s)": round6(schedule_row.get("T_retrain_s", "")),
        "adapter就绪时间(s)": round6(schedule_row.get("adapter_ready_time_s", "")),
        "adapter就绪样本数(实际切分)": actual.get("adapter_ready_sample_count", ""),
        "估计当前窗口准确率": round6(schedule_row.get("current_window_accuracy", actual.get("estimated_current_accuracy", ""))),
        "估计adapter后准确率": round6(schedule_row.get("future_window_accuracy", "")),
        "估计长期平均准确率": round6(schedule_row.get("long_avg_accuracy", actual.get("estimated_long_avg_accuracy", ""))),
        "实际base准确率": round6(actual.get("observed_base_accuracy", "")),
        "实际adapter准确率": round6(actual.get("observed_adapter_accuracy", "")),
        "真实训练收益": round6(
            (as_float(actual.get("observed_adapter_accuracy")) or 0.0)
            - (as_float(actual.get("observed_base_accuracy")) or 0.0)
        )
        if as_float(actual.get("observed_adapter_accuracy")) is not None and as_float(actual.get("observed_base_accuracy")) is not None
        else "",
        "adapter前准确率": round6(actual.get("before_adapter_accuracy", "")),
        "adapter前样本数": actual.get("before_adapter_total", ""),
        "adapter后准确率": round6(actual.get("after_adapter_accuracy", "")),
        "adapter后样本数": actual.get("after_adapter_total", ""),
        "时间加权准确率": round6(actual.get("observed_time_weighted_accuracy", "")),
        "实际当前准确率": round6(actual.get("observed_current_accuracy", "")),
        "实际最终准确率": round6(actual.get("observed_final_accuracy", "")),
        "当前误差(实际-估计)": round6(actual.get("current_error", "")),
        "当前绝对误差": round6(actual.get("abs_current_error", "")),
        "最终-估计当前": round6(actual.get("final_minus_estimated_current", "")),
        "coverage": round6(actual.get("coverage", "")),
        "有效lambda资源成本": round6(actual.get("lambda_resource_cost_effective", "")),
        "lambda伪标准确率(R_val)": round6(actual.get("lambda_teacher_accuracy", "")),
        "lambda真实准确率(R_val)": round6(actual.get("lambda_true_accuracy", "")),
        "base正确数": actual.get("base_correct", ""),
        "base总数": actual.get("base_total", ""),
        "adapter正确数": actual.get("adapter_correct", ""),
        "adapter总数": actual.get("adapter_accepted_count", ""),
        "base报告": actual.get("base_report", ""),
        "adapter报告": actual.get("adapter_report", ""),
        "adapter路径": actual.get("adapter_path", ""),
        "调度备注": schedule_row.get("notes", ""),
    }


def load_report_maps() -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str, str], Path]]:
    base_reports: dict[tuple[str, str], Path] = {}
    adapter_reports: dict[tuple[str, str, str], Path] = {}
    for row in read_csv(COMBINED):
        group_id = str(row["group_id"])
        lambda_id = str(row["lambda_id"])
        gamma_id = str(row["gamma_id"])
        base_path = Path(str(row.get("base_report") or ""))
        adapter_path = Path(str(row.get("adapter_report") or ""))
        if base_path.exists():
            base_reports[(group_id, lambda_id)] = base_path
        if adapter_path.exists():
            adapter_reports[(group_id, gamma_id, lambda_id)] = adapter_path
    if MISSING_G3_ADAPTER.exists():
        adapter_reports[
            (
                "v2_g3_A9_B5",
                "qwen25vl3b_lora_r16_a32_profile80_target320",
                "final_fp32_res672_tok16_full_keep100_b1",
            )
        ] = MISSING_G3_ADAPTER
    return base_reports, adapter_reports


def exact_true_map(label: str) -> dict[tuple[str, str, str, str], dict[str, str]]:
    path = STEP6 / "true_accuracy_full_r" / label / "scheduler_true_accuracy_error_rows.csv"
    out = {}
    for row in read_csv(path):
        out[(row["method"], row["group_id"], row["lambda_id"], row.get("gamma_id", ""))] = row
    return out


def compute_oracle_rows(
    labels: list[str],
    lambda_profiles: dict[tuple[str, str], dict[str, Any]],
    gamma_profiles: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    combined_rows = read_csv(COMBINED)
    all_detail: list[dict[str, Any]] = []
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    pred_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    def preds(path: Path) -> dict[str, dict[str, Any]]:
        if path not in pred_cache:
            pred_cache[path] = prediction_map(path)
        return pred_cache[path]

    for label in labels:
        rate = as_float(rate_value(label)) or 0.0
        best_by_group: dict[str, dict[str, Any]] = {}
        for candidate in combined_rows:
            group_id = candidate["group_id"]
            lambda_id = candidate["lambda_id"]
            gamma_id = candidate["gamma_id"]
            lambda_row = lambda_profiles[(group_id, lambda_id)]
            gamma_row = gamma_profiles[(group_id, gamma_id)]
            train_time = as_float(gamma_row.get("train_time_s", gamma_row.get("train_time"))) or 0.0
            if train_time <= 0:
                continue
            base_report = Path(candidate["base_report"])
            adapter_report = Path(candidate["adapter_report"])
            if not base_report.exists() or not adapter_report.exists():
                continue
            ordered_uids = group_uids(group_id)
            base_preds = preds(base_report)
            adapter_preds = preds(adapter_report)
            adapter_acc = as_float(candidate.get("observed_adapter_accuracy")) or report_accuracy(adapter_report)[0]
            base_acc = as_float(candidate.get("observed_base_accuracy")) or report_accuracy(base_report)[0]
            lambda_cost = lambda_resource_cost(lambda_row, rate)
            for idx in range(1, 100):
                inference_resource = idx / 100.0
                training_resource = 1.0 - inference_resource
                ready_time = train_time / training_resource
                ready_count = max(0, min(len(ordered_uids), math.ceil(rate * ready_time)))
                before_uids = ordered_uids[:ready_count]
                after_uids = ordered_uids[ready_count:]
                before_acc, before_correct, before_accepted, _ = prediction_accuracy(base_preds, before_uids)
                after_acc, after_correct, after_accepted, _ = prediction_accuracy(adapter_preds, after_uids)
                accepted = before_accepted + after_accepted
                processed_acc = (before_correct + after_correct) / accepted if accepted else 0.0
                coverage = min(1.0, inference_resource / max(lambda_cost, 1e-9))
                actual_long = coverage * processed_acc
                actual_final = coverage * adapter_acc
                row = {
                    "到达率(samples/s)": rate_value(label),
                    "组ID": group_id,
                    "推理配置ID": lambda_id,
                    "训练配置ID": gamma_id,
                    "推理资源I": round6(inference_resource),
                    "训练资源R": round6(training_resource),
                    "lambda资源成本": round6(lambda_cost),
                    "coverage": round6(coverage),
                    "训练时间(s)": round6(train_time),
                    "adapter就绪时间(s)": round6(ready_time),
                    "adapter就绪样本数": ready_count,
                    "base全R准确率": round6(base_acc),
                    "adapter全R准确率": round6(adapter_acc),
                    "adapter训练收益": round6(adapter_acc - base_acc),
                    "before准确率": round6(before_acc),
                    "before正确数": before_correct,
                    "before样本数": len(before_uids),
                    "after准确率": round6(after_acc),
                    "after正确数": after_correct,
                    "after样本数": len(after_uids),
                    "总样本数": len(ordered_uids),
                    "真实端到端processed_acc": round6(processed_acc),
                    "actual_long_avg_accuracy": round6(actual_long),
                    "实际最终准确率(含coverage)": round6(actual_final),
                    "base报告": str(base_report),
                    "adapter报告": str(adapter_report),
                }
                current = best_by_group.get(group_id)
                key = (
                    as_float(row["actual_long_avg_accuracy"]) or 0.0,
                    as_float(row["真实端到端processed_acc"]) or 0.0,
                    as_float(row["adapter全R准确率"]) or 0.0,
                    -(as_int(row["adapter就绪样本数"]) or 0),
                    int("full_keep100" in lambda_id),
                    str(row["adapter报告"]),
                )
                current_key = (
                    as_float(current.get("actual_long_avg_accuracy")) or 0.0,
                    as_float(current.get("真实端到端processed_acc")) or 0.0,
                    as_float(current.get("adapter全R准确率")) or 0.0,
                    -(as_int(current.get("adapter就绪样本数")) or 0),
                    int("full_keep100" in str(current.get("推理配置ID", ""))),
                    str(current.get("adapter报告", "")),
                ) if current else None
                if current is None or key > current_key:
                    best_by_group[group_id] = row

        detail = [best_by_group[group_id] for group_id in sorted(best_by_group)]
        all_detail.extend(detail)
        total = sum(as_int(row["总样本数"]) or 0 for row in detail)
        actual_sum = sum((as_float(row["actual_long_avg_accuracy"]) or 0.0) * (as_int(row["总样本数"]) or 0) for row in detail)
        processed_sum = sum((as_float(row["真实端到端processed_acc"]) or 0.0) * (as_int(row["总样本数"]) or 0) for row in detail)
        summary_rows.append(
            {
                "到达率(samples/s)": rate_value(label),
                "组数": len(detail),
                "平均actual_long_avg_accuracy": round6(actual_sum / total if total else ""),
                "平均processed_acc(不乘coverage)": round6(processed_sum / total if total else ""),
                "候选lambda_gamma数": len(combined_rows),
                "I/R枚举数": len(combined_rows) * 99,
                "选择范围": "Pareto-complete strict reports only",
            }
        )
        rows_by_label[label] = []
        for row in detail:
            actual = {
                "group_id": row["组ID"],
                "method": "OfflineJointOracle",
                "lambda_id": row["推理配置ID"],
                "gamma_id": row["训练配置ID"],
                "I": row["推理资源I"],
                "R": row["训练资源R"],
                "adapter_accepted": True,
                "observed_base_accuracy": row["base全R准确率"],
                "observed_adapter_accuracy": row["adapter全R准确率"],
                "adapter_ready_sample_count": row["adapter就绪样本数"],
                "before_adapter_accuracy": row["before准确率"],
                "before_adapter_total": row["before样本数"],
                "after_adapter_accuracy": row["after准确率"],
                "after_adapter_total": row["after样本数"],
                "observed_time_weighted_accuracy": row["真实端到端processed_acc"],
                "observed_current_accuracy": row["actual_long_avg_accuracy"],
                "observed_final_accuracy": row["实际最终准确率(含coverage)"],
                "current_error": 0,
                "abs_current_error": 0,
                "coverage": row["coverage"],
                "lambda_resource_cost_effective": row["lambda资源成本"],
                "base_correct": "",
                "base_total": row["总样本数"],
                "adapter_correct": "",
                "adapter_accepted_count": row["总样本数"],
                "base_report": row["base报告"],
                "adapter_report": row["adapter报告"],
            }
            schedule = {
                "method": "OfflineJointOracle",
                "group_id": row["组ID"],
                "lambda_id": row["推理配置ID"],
                "gamma_id": row["训练配置ID"],
                "I": row["推理资源I"],
                "R": row["训练资源R"],
                "adapter_accepted": "True",
                "current_window_accuracy": row["actual_long_avg_accuracy"],
                "future_window_accuracy": row["实际最终准确率(含coverage)"],
                "long_avg_accuracy": row["actual_long_avg_accuracy"],
                "adapter_ready_time_s": row["adapter就绪时间(s)"],
                "notes": "posthoc_pareto_ir_true_oracle",
            }
            rows_by_label[label].append(chinese_detail_row(label, schedule, actual, note="Pareto候选后验I/R枚举选择"))
    return rows_by_label, all_detail, summary_rows


def summarize(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = {
        "平均估计当前窗口准确率": "估计当前窗口准确率",
        "平均实际当前准确率": "实际当前准确率",
        "平均当前误差": "当前误差(实际-估计)",
        "平均当前绝对误差": "当前绝对误差",
        "平均实际base准确率": "实际base准确率",
        "平均实际adapter准确率": "实际adapter准确率",
        "平均真实训练收益": "真实训练收益",
        "平均adapter前准确率": "adapter前准确率",
        "平均adapter后准确率": "adapter后准确率",
        "平均时间加权准确率": "时间加权准确率",
        "平均实际最终准确率": "实际最终准确率",
        "平均adapter就绪样本数": "adapter就绪样本数(实际切分)",
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault((str(row["到达率(samples/s)"]), str(row["方法"]), str(row["方法代码"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (rate, method, method_code), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "到达率(samples/s)": rate,
            "方法": method,
            "方法代码": method_code,
            "组数": len(rows),
            "真实评测匹配组数": sum(1 for row in rows if str(row.get("真实评测是否匹配当前配置")).lower() == "true"),
        }
        for out_key, in_key in fields.items():
            values = [as_float(row.get(in_key)) for row in rows]
            values = [value for value in values if value is not None]
            item[out_key] = round6(statistics.fmean(values)) if values else ""
        out.append(item)
    return out


def main() -> int:
    labels = ["ar0p05", "ar0p1"]
    lambda_profile_rows = read_json(STEP5 / "lambda_analysis_rval" / "lambda_profiles_group_v2.json")
    gamma_profile_rows = read_json(STEP5 / "gamma_rwindow_pseudo" / "gamma_group_profiles.json")
    lambda_profiles = {(row["group_id"], row["lambda_id"]): row for row in lambda_profile_rows}
    gamma_profiles = {(row["group_id"], row["gamma_id"]): row for row in gamma_profile_rows}
    base_reports, adapter_reports = load_report_maps()
    oracle_rows_by_label, oracle_detail, oracle_summary = compute_oracle_rows(labels, lambda_profiles, gamma_profiles)

    detail_rows: list[dict[str, Any]] = []
    supplemented: list[dict[str, Any]] = []
    for label in labels:
        schedules = read_csv(STEP5 / f"formal_schedules_{label}" / "all_methods_scheduler_end_to_end_results.csv")
        true_rows = exact_true_map(label)
        for schedule in sorted(schedules, key=lambda row: (row["method"], row["group_id"])):
            method = schedule["method"]
            if method == "OfflineJointOracle":
                continue
            key = (method, schedule["group_id"], schedule["lambda_id"], schedule.get("gamma_id", ""))
            actual = true_rows.get(key)
            if actual is not None:
                detail_rows.append(chinese_detail_row(label, schedule, actual, note="来自已有true_accuracy_full_r精确匹配行"))
                continue
            group_id = schedule["group_id"]
            lambda_id = schedule["lambda_id"]
            gamma_id = schedule.get("gamma_id", "")
            base_report = base_reports[(group_id, lambda_id)]
            adapter_report = adapter_reports.get((group_id, gamma_id, lambda_id)) if gamma_id else None
            actual = compute_actual_from_reports(
                schedule,
                label=label,
                base_report=base_report,
                adapter_report=adapter_report,
                lambda_profiles=lambda_profiles,
            )
            detail_rows.append(chinese_detail_row(label, schedule, actual, note="由补齐/Pareto strict reports离线复算"))
            supplemented.append(
                {
                    "到达率": rate_value(label),
                    "方法": method,
                    "组ID": group_id,
                    "推理配置ID": lambda_id,
                    "训练配置ID": gamma_id,
                    "base报告": str(base_report),
                    "adapter报告": str(adapter_report or ""),
                }
            )
        detail_rows.extend(oracle_rows_by_label[label])

    summary = summarize(detail_rows)
    write_csv(OUT / "五方法逐组明细.csv", detail_rows)
    write_csv(OUT / "五方法汇总.csv", summary)
    write_csv(OUT / "离线联合Oracle_ParetoIR_逐组明细.csv", oracle_detail)
    write_csv(OUT / "离线联合Oracle_ParetoIR_汇总.csv", oracle_summary)
    write_csv(OUT / "补齐复算行_manifest.csv", supplemented)
    manifest = {
        "out_dir": str(OUT),
        "eval_split": "R_all",
        "group_r_window_samples": 390,
        "arrival_rates": [rate_value(label) for label in labels],
        "offline_joint_oracle_scope": "Pareto-complete scheduler_true_accuracy_error_rows_combined.csv only",
        "offline_joint_oracle_lambda_gamma_rows": len(read_csv(COMBINED)),
        "offline_joint_oracle_ir_grid": "I=0.01..0.99, R=1-I",
        "supplemented_baseline_rows": supplemented,
        "outputs": {
            "五方法逐组明细.csv": str(OUT / "五方法逐组明细.csv"),
            "五方法汇总.csv": str(OUT / "五方法汇总.csv"),
            "离线联合Oracle_ParetoIR_逐组明细.csv": str(OUT / "离线联合Oracle_ParetoIR_逐组明细.csv"),
            "离线联合Oracle_ParetoIR_汇总.csv": str(OUT / "离线联合Oracle_ParetoIR_汇总.csv"),
            "补齐复算行_manifest.csv": str(OUT / "补齐复算行_manifest.csv"),
        },
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"out_dir": str(OUT), "detail_rows": len(detail_rows), "summary_rows": len(summary)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
