#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_PYTHON = Path("@@QWEN25_PYTHON@@")
DEFAULT_MODEL_PATH = Path("@@WORKSPACE@@/qwen_eval/models/Qwen2.5-VL-3B-Instruct")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def lambda_resource_cost(lambda_row: dict[str, Any], arrival_rate: float) -> float:
    throughput = lambda_row.get("throughput_samples_per_sec", lambda_row.get("throughput_samples_per_s"))
    if throughput not in ("", None):
        try:
            throughput_value = float(throughput)
        except (TypeError, ValueError):
            throughput_value = 0.0
        if throughput_value > 0:
            return arrival_rate / throughput_value
    return float(lambda_row.get("resource_cost") or 1.0)


def dtype_from_precision(value: Any, default: str) -> str:
    precision = str(value or default).lower()
    if precision in {"fp32", "float32"}:
        return "float32"
    if precision in {"fp16", "float16"}:
        return "float16"
    return "bfloat16"


def latest_adapter(fulltrain_root: Path, group_id: str, gamma_id: str) -> Path:
    base = fulltrain_root / group_id / gamma_id
    candidates = sorted(
        base.glob("swift_output/*/checkpoint-*/adapter_model.safetensors"),
        key=lambda path: (
            int(path.parent.name.split("-", 1)[1])
            if path.parent.name.startswith("checkpoint-")
            else -1,
            path.stat().st_mtime,
        ),
    )
    if not candidates:
        raise FileNotFoundError(f"No fulltarget adapter found for {group_id}/{gamma_id} under {base}")
    return candidates[-1].parent


def write_uid_filter(path: Path, dataset_dir: Path) -> None:
    rows = read_json(dataset_dir / "data.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"uid": str(row["uid"])}, ensure_ascii=False) + "\n")


def dataset_uids(dataset_dir: Path) -> list[str]:
    rows = read_json(dataset_dir / "data.json")
    if not isinstance(rows, list):
        raise TypeError(f"{dataset_dir / 'data.json'} must contain a list")
    return [str(row["uid"]) for row in rows]


def load_prediction_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pred_path = Path(str(report.get("predictions") or ""))
    if not pred_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid")
            if uid not in (None, ""):
                out[str(uid)] = row
    return out


def prediction_accuracy_for_uids(predictions: dict[str, dict[str, Any]], uids: list[str]) -> tuple[float, int, int, int]:
    correct = 0
    accepted = 0
    missing = 0
    for uid in uids:
        row = predictions.get(uid)
        if row is None:
            missing += 1
            continue
        if row.get("accepted"):
            accepted += 1
            if row.get("correct"):
                correct += 1
    return (correct / accepted if accepted else 0.0), correct, accepted, missing


def run_strict_eval(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    out_dir: Path,
    uid_filter: Path,
    max_new_tokens: int,
    dtype: str,
    adapter_path: Path | None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    report = out_dir / "strict_eval_report.json"
    if report.exists() and not args.force:
        return read_json(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python_bin),
        "-m",
        "remote_sensing_adaptation.eval_qwen_lora_strict",
        "--dataset-dir",
        str(dataset_dir),
        "--model-path",
        str(args.model_path),
        "--out-dir",
        str(out_dir),
        "--uid-filter-jsonl",
        str(uid_filter),
        "--task-ids",
        str(args.task_ids),
        "--max-new-tokens",
        str(max_new_tokens),
        "--dtype",
        dtype,
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--seed",
        str(args.seed),
        "--answer-source",
        str(args.answer_source),
        "--progress-every",
        "40",
        "--batch-size",
        str(batch_size if batch_size is not None else args.strict_eval_batch_size),
    ]
    if adapter_path is not None:
        command.extend(["--adapter-path", str(adapter_path)])
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = f"{package_root}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    log_path = out_dir / "strict_eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    return read_json(report)


def strict_eval_oomed(out_dir: Path) -> bool:
    log_path = out_dir / "strict_eval.log"
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "OutOfMemoryError" in text or "CUDA out of memory" in text


def report_accuracy(report: dict[str, Any]) -> tuple[float, int, int, int]:
    accepted = int(report.get("accepted") or report.get("total") or 0)
    correct = int(report.get("correct") or 0)
    total = int(report.get("total") or accepted)
    accuracy = float(report.get("strict_accuracy") if "strict_accuracy" in report else (correct / accepted if accepted else 0.0))
    return accuracy, correct, accepted, total


def lambda_map(lambda_profiles: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_json(lambda_profiles)
    if not isinstance(rows, list):
        raise TypeError("lambda profiles must be a list")
    return {(str(row["group_id"]), str(row["lambda_id"])): dict(row) for row in rows}


def selected_scheduler_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.scheduler_csv)
    if args.all_methods:
        return rows
    return [row for row in rows if str(row.get("method")) == args.method]


def round6(value: float) -> float:
    return round(float(value), 6)


def materialize_strict_reports(
    args: argparse.Namespace,
    lambdas: dict[tuple[str, str], dict[str, Any]],
    scheduler_rows: list[dict[str, str]],
) -> None:
    uid_filter_dir = args.out_dir / "uid_filters"
    requests: dict[Path, dict[str, Any]] = {}
    for row in scheduler_rows:
        group_id = str(row["group_id"])
        lambda_id = str(row["lambda_id"])
        gamma_id = str(row.get("gamma_id") or "")
        lambda_row = lambdas.get((group_id, lambda_id))
        if lambda_row is None:
            raise KeyError(f"Missing lambda profile for {group_id}/{lambda_id}")
        dataset_dir = args.group_data_dir / "groups" / group_id / args.eval_split
        uid_filter = uid_filter_dir / f"{group_id}.jsonl"
        if not uid_filter.exists() or args.force:
            write_uid_filter(uid_filter, dataset_dir)
        max_new_tokens = int(lambda_row.get("max_new_tokens") or args.default_max_new_tokens)
        dtype = dtype_from_precision(
            lambda_row.get("model_precision", lambda_row.get("precision")),
            args.default_dtype,
        )
        base_out = args.out_dir / "base" / group_id / lambda_id
        requests.setdefault(
            base_out,
            {
                "dataset_dir": dataset_dir,
                "out_dir": base_out,
                "uid_filter": uid_filter,
                "max_new_tokens": max_new_tokens,
                "dtype": dtype,
                "adapter_path": None,
            },
        )
        if gamma_id and parse_bool(row.get("adapter_accepted")):
            adapter_out = args.out_dir / "adapter" / group_id / gamma_id / lambda_id
            requests.setdefault(
                adapter_out,
                {
                    "dataset_dir": dataset_dir,
                    "out_dir": adapter_out,
                    "uid_filter": uid_filter,
                    "max_new_tokens": max_new_tokens,
                    "dtype": dtype,
                    "adapter_path": latest_adapter(args.fulltrain_root, group_id, gamma_id),
                },
            )

    deferred: list[dict[str, Any]] = []
    for request in requests.values():
        try:
            run_strict_eval(
                args=args,
                batch_size=args.strict_eval_batch_size,
                **request,
            )
        except subprocess.CalledProcessError:
            if not strict_eval_oomed(request["out_dir"]):
                raise
            log_path = request["out_dir"] / "strict_eval.log"
            shutil.copy2(
                log_path,
                request["out_dir"] / f"strict_eval_batch{args.strict_eval_batch_size}_oom.log",
            )
            deferred.append(request)
            print(
                f"[strict-eval-deferred] out={request['out_dir']} "
                f"batch={args.strict_eval_batch_size}",
                flush=True,
            )

    write_json(
        args.out_dir / "strict_eval_deferred_oom.json",
        {
            "high_batch_size": args.strict_eval_batch_size,
            "fallback_batch_size": args.strict_eval_fallback_batch_size,
            "deferred_count": len(deferred),
            "deferred_out_dirs": [str(request["out_dir"]) for request in deferred],
        },
    )
    for request in deferred:
        run_strict_eval(
            args=args,
            batch_size=args.strict_eval_fallback_batch_size,
            **request,
        )


def evaluate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    lambdas = lambda_map(args.lambda_profiles)
    scheduler_rows = selected_scheduler_rows(args)
    materialize_strict_reports(args, lambdas, scheduler_rows)
    output_rows: list[dict[str, Any]] = []
    uid_filter_dir = args.out_dir / "uid_filters"
    for row in scheduler_rows:
        group_id = str(row["group_id"])
        lambda_id = str(row["lambda_id"])
        gamma_id = str(row.get("gamma_id") or "")
        lambda_row = lambdas.get((group_id, lambda_id))
        if lambda_row is None:
            raise KeyError(f"Missing lambda profile for {group_id}/{lambda_id}")
        dataset_dir = args.group_data_dir / "groups" / group_id / args.eval_split
        uid_filter = uid_filter_dir / f"{group_id}.jsonl"
        if not uid_filter.exists() or args.force:
            write_uid_filter(uid_filter, dataset_dir)
        ordered_uids = dataset_uids(dataset_dir)
        max_new_tokens = int(lambda_row.get("max_new_tokens") or args.default_max_new_tokens)
        dtype = dtype_from_precision(lambda_row.get("model_precision", lambda_row.get("precision")), args.default_dtype)

        base_report = run_strict_eval(
            args=args,
            dataset_dir=dataset_dir,
            out_dir=args.out_dir / "base" / group_id / lambda_id,
            uid_filter=uid_filter,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            adapter_path=None,
        )
        base_acc, base_correct, base_accepted, base_total = report_accuracy(base_report)
        base_predictions = load_prediction_rows(base_report)

        adapter_active = bool(gamma_id) and parse_bool(row.get("adapter_accepted"))
        adapter_report_path = ""
        adapter_acc = base_acc
        adapter_correct = base_correct
        adapter_accepted = base_accepted
        adapter_predictions = base_predictions
        adapter_path_text = ""
        if adapter_active:
            adapter_path = latest_adapter(args.fulltrain_root, group_id, gamma_id)
            adapter_path_text = str(adapter_path)
            adapter_report = run_strict_eval(
                args=args,
                dataset_dir=dataset_dir,
                out_dir=args.out_dir / "adapter" / group_id / gamma_id / lambda_id,
                uid_filter=uid_filter,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                adapter_path=adapter_path,
            )
            adapter_report_path = str(args.out_dir / "adapter" / group_id / gamma_id / lambda_id / "strict_eval_report.json")
            adapter_acc, adapter_correct, adapter_accepted, _ = report_accuracy(adapter_report)
            adapter_predictions = load_prediction_rows(adapter_report)

        lambda_cost = lambda_resource_cost(lambda_row, args.arrival_rate)
        inference_resource = float(row.get("I") or 0.0)
        coverage = min(1.0, inference_resource / max(lambda_cost, 1e-9))
        finish_time = optional_float(row.get("adapter_ready_time_s")) or optional_float(row.get("train_time_s_effective"))
        retrain_window = float(row.get("T_retrain_s") or 1.0)
        old_ratio = 1.0
        if adapter_active and finish_time is not None:
            old_ratio = max(0.0, min(1.0, finish_time / max(retrain_window, 1e-9)))
        ready_count = len(ordered_uids)
        if adapter_active and finish_time is not None:
            ready_count = max(0, min(len(ordered_uids), math.ceil(args.arrival_rate * finish_time)))
        before_uids = ordered_uids[:ready_count]
        after_uids = ordered_uids[ready_count:]
        before_acc, before_correct, before_accepted, before_missing = prediction_accuracy_for_uids(
            base_predictions,
            before_uids,
        )
        after_acc, after_correct, after_accepted, after_missing = prediction_accuracy_for_uids(
            adapter_predictions if adapter_active else base_predictions,
            after_uids,
        )
        weighted_accepted = before_accepted + after_accepted
        current_processed_acc = (
            (before_correct + after_correct) / weighted_accepted
            if weighted_accepted
            else old_ratio * base_acc + (1.0 - old_ratio) * adapter_acc
        )
        observed_current = coverage * current_processed_acc
        observed_final = coverage * (adapter_acc if adapter_active else base_acc)
        estimated_current = float(row.get("current_window_accuracy") or 0.0)
        estimated_long = float(row.get("long_avg_accuracy") or estimated_current)
        output_rows.append(
            {
                "group_id": group_id,
                "method": row.get("method", ""),
                "lambda_id": lambda_id,
                "gamma_id": gamma_id,
                "I": row.get("I", ""),
                "R": row.get("R", ""),
                "adapter_accepted": row.get("adapter_accepted", ""),
                "estimated_current_accuracy": estimated_current,
                "estimated_long_avg_accuracy": estimated_long,
                "eval_split": args.eval_split,
                "arrival_rate": args.arrival_rate,
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
                "observed_time_weighted_accuracy": round6(current_processed_acc),
                "observed_current_accuracy": round6(observed_current),
                "observed_final_accuracy": round6(observed_final),
                "current_error": round6(observed_current - estimated_current),
                "abs_current_error": round6(abs(observed_current - estimated_current)),
                "final_minus_estimated_current": round6(observed_final - estimated_current),
                "teacher_noise_accuracy_gap": lambda_row.get("teacher_noise_accuracy_gap", ""),
                "lambda_teacher_accuracy": lambda_row.get("teacher_accuracy", lambda_row.get("accuracy", "")),
                "lambda_true_accuracy": lambda_row.get("true_accuracy", ""),
                "base_correct": base_correct,
                "base_accepted": base_accepted,
                "base_total": base_total,
                "adapter_correct": adapter_correct,
                "adapter_accepted_count": adapter_accepted,
                "adapter_path": adapter_path_text,
                "base_report": str(args.out_dir / "base" / group_id / lambda_id / "strict_eval_report.json"),
                "adapter_report": adapter_report_path,
            }
        )
    return output_rows


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_json(out_dir / "scheduler_true_accuracy_error_summary.json", {"rows": 0})
        return
    errors = [float(row["current_error"]) for row in rows]
    abs_errors = [float(row["abs_current_error"]) for row in rows]
    summary = {
        "rows": len(rows),
        "mean_observed_current_accuracy": round6(statistics.fmean(float(row["observed_current_accuracy"]) for row in rows)),
        "mean_estimated_current_accuracy": round6(statistics.fmean(float(row["estimated_current_accuracy"]) for row in rows)),
        "mean_current_error": round6(statistics.fmean(errors)),
        "mean_abs_current_error": round6(statistics.fmean(abs_errors)),
        "current_rmse": round6(math.sqrt(statistics.fmean(value * value for value in errors))),
    }
    write_json(out_dir / "scheduler_true_accuracy_error_summary.json", summary)
    lines = [
        "# V2 Scheduler True Accuracy Error",
        "",
        f"- Rows: {summary['rows']}",
        f"- Mean observed current accuracy: {summary['mean_observed_current_accuracy']:.2%}",
        f"- Mean estimated current accuracy: {summary['mean_estimated_current_accuracy']:.2%}",
        f"- Mean error: {summary['mean_current_error']:+.2%}",
        f"- MAE: {summary['mean_abs_current_error']:.2%}",
        f"- RMSE: {summary['current_rmse']:.2%}",
        "",
    ]
    (out_dir / "scheduler_true_accuracy_error_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2 scheduler selections on true labels.")
    parser.add_argument("--scheduler-csv", type=Path, required=True)
    parser.add_argument("--lambda-profiles", type=Path, required=True)
    parser.add_argument("--group-data-dir", type=Path, required=True)
    parser.add_argument("--fulltrain-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--task-ids", default="1,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="ThiefLoRA")
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--eval-split", default="R_test")
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--default-max-new-tokens", type=int, default=16)
    parser.add_argument("--default-dtype", default="bfloat16")
    parser.add_argument("--answer-source", default="raw")
    parser.add_argument("--strict-eval-batch-size", type=int, default=4)
    parser.add_argument("--strict-eval-fallback-batch-size", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate_rows(args)
    write_csv(args.out_dir / "scheduler_true_accuracy_error_rows.csv", rows)
    write_summary(args.out_dir, rows)
    print(json.dumps({"rows": len(rows), "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
