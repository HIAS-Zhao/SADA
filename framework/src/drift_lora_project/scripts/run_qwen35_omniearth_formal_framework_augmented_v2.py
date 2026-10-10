#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFilter, ImageOps


PROJECT_ROOT = Path("@@WORKSPACE@@/drift_lora_project")
QWEN_PYTHON = Path("@@VLM_PYTHON@@")
VLM_EVAL = Path("@@WORKSPACE@@/qwen_eval/code/candidate_vlm_eval.py")
QWEN08B_MODEL = Path("@@WORKSPACE@@/qwen_eval/models/Qwen3.5-0.8B")
REFERENCE_TASK = "perception_a1_2_land_cover_classification"
MODEL_ID = "Qwen3.5-0.8B"


@dataclass(frozen=True)
class LambdaConfig:
    lambda_id: str
    model_precision: str
    dtype: str
    roi_strategy: str
    token_retention_ratio: float
    target_long_side: int
    batch_size: int
    image_resolution: int
    max_new_tokens: int


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(value: Any) -> str:
    return str(value or "").strip().upper()


def task_key_for(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return str(meta.get("task_key") or item.get("task_key") or item.get("task_id"))


def drift_task_for_group(group_id: str) -> str:
    if "__" not in group_id:
        return group_id
    return group_id.split("__", 1)[1]


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def dtop40_value(row: dict[str, Any], suffix: str, default: Any = "") -> Any:
    return row.get(f"dtop40_{suffix}", row.get(f"dtop80_{suffix}", default))


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if max(width, height) == long_side:
        return image
    scale = long_side / max(width, height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def saliency_mask(image: Image.Image, keep_ratio: float, grid: int = 8) -> Image.Image:
    resized = resize_long_side(image, 672)
    edges = ImageOps.grayscale(resized).filter(ImageFilter.FIND_EDGES)
    width, height = resized.size
    tile_scores = []
    for gy in range(grid):
        for gx in range(grid):
            left = int(round(gx * width / grid))
            right = int(round((gx + 1) * width / grid))
            top = int(round(gy * height / grid))
            bottom = int(round((gy + 1) * height / grid))
            patch = edges.crop((left, top, right, bottom))
            score = sum(patch.histogram()[128:])
            tile_scores.append((score, gx, gy, left, top, right, bottom))
    keep_n = max(1, int(round(len(tile_scores) * keep_ratio)))
    keep = {(gx, gy) for _score, gx, gy, *_box in sorted(tile_scores, reverse=True)[:keep_n]}
    canvas = Image.new("RGB", resized.size, (128, 128, 128))
    for _score, gx, gy, left, top, right, bottom in tile_scores:
        if (gx, gy) in keep:
            canvas.paste(resized.crop((left, top, right, bottom)), (left, top))
    return canvas


def transform_image(src: Path, dst: Path, config: LambdaConfig) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        if config.roi_strategy == "saliency_topk":
            out = saliency_mask(image, config.token_retention_ratio)
        else:
            out = resize_long_side(image, config.target_long_side)
        out.save(dst, quality=92)


def rewrite_image_path(path_value: Any, image_root: Path, config: LambdaConfig) -> str:
    src = Path(str(path_value))
    digest = hashlib.sha256(str(src).encode("utf-8")).hexdigest()[:16]
    suffix = src.suffix.lower() if src.suffix else ".jpg"
    dst = (image_root / config.roi_strategy / f"{digest}{suffix}").resolve()
    transform_image(src, dst, config)
    return str(dst)


def prepare_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    dataset_key: str,
    config: LambdaConfig,
    overwrite: bool,
) -> Path:
    dataset_dir = output_dir / "datasets" / dataset_key / config.lambda_id
    data_path = dataset_dir / "data.json"
    if data_path.exists() and not overwrite:
        return dataset_dir
    if overwrite and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    rows = read_json(source_dir / "data.json")
    image_root = output_dir / "transformed_images"
    transformed = []
    for row in rows:
        new_row = json.loads(json.dumps(row, ensure_ascii=False))
        meta = dict(new_row.get("meta") if isinstance(new_row.get("meta"), dict) else {})
        for key in ("image_path", "pre_image_path", "post_image_path"):
            value = new_row.get(key) or meta.get(key)
            if value:
                new_value = rewrite_image_path(value, image_root, config)
                new_row[key] = new_value
                meta[key] = new_value
        meta.update(
            {
                "lambda_image_transform": config.roi_strategy,
                "lambda_token_retention_ratio": config.token_retention_ratio,
                "lambda_target_long_side": config.target_long_side,
            }
        )
        new_row["meta"] = meta
        transformed.append(new_row)
    write_json(data_path, transformed)
    return dataset_dir


def prediction_map(predictions_path: Path) -> dict[str, str]:
    out = {}
    for row in read_jsonl(predictions_path):
        uid = str(row.get("uid", ""))
        if uid and bool(row.get("accepted")):
            out[uid] = normalize(row.get("pseudo_label"))
    return out


def score_rows(rows: list[dict[str, Any]], preds: dict[str, str]) -> dict[str, Any]:
    seen = 0
    correct = 0
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row.get("uid", ""))
        task_key = task_key_for(row)
        bucket = by_task.setdefault(task_key, {"seen": 0, "correct": 0})
        if uid not in preds:
            continue
        seen += 1
        bucket["seen"] += 1
        if preds[uid] == normalize(row.get("ground_truth") or row.get("gt")):
            correct += 1
            bucket["correct"] += 1
    for bucket in by_task.values():
        bucket["accuracy_seen"] = bucket["correct"] / bucket["seen"] if bucket["seen"] else 0.0
    return {
        "seen": seen,
        "correct": correct,
        "accuracy_seen": correct / seen if seen else 0.0,
        "by_task": by_task,
    }


def mixed_accuracy(
    rows: list[dict[str, Any]],
    base_preds: dict[str, str],
    adapter_preds: dict[str, str] | None,
    *,
    ready_index: int,
) -> dict[str, Any]:
    ready_index = max(0, min(len(rows), ready_index))
    before_rows = rows[:ready_index]
    after_rows = rows[ready_index:] if adapter_preds is not None else []
    before_correct = sum(
        int(base_preds.get(str(row["uid"]), "") == normalize(row.get("ground_truth") or row.get("gt")))
        for row in before_rows
    )
    after_correct = sum(
        int(adapter_preds.get(str(row["uid"]), "") == normalize(row.get("ground_truth") or row.get("gt")))
        for row in after_rows
    ) if adapter_preds is not None else 0
    total_correct = before_correct + after_correct
    return {
        "before_adapter_correct": before_correct,
        "before_adapter_total": len(before_rows),
        "before_adapter_accuracy": before_correct / len(before_rows) if before_rows else 0.0,
        "after_adapter_correct": after_correct,
        "after_adapter_total": len(after_rows),
        "after_adapter_accuracy": after_correct / len(after_rows) if after_rows else 0.0,
        "observed_time_weighted_accuracy": total_correct / len(rows) if rows else 0.0,
        "correct": total_correct,
        "total": len(rows),
    }


def score_by_task_fields(
    *,
    rows: list[dict[str, Any]],
    base_preds: dict[str, str],
    adapter_preds: dict[str, str] | None,
    ready_index: int,
    coverage: float,
    drift_task: str,
) -> dict[str, Any]:
    out = {}
    for prefix, task in (("no_drift", REFERENCE_TASK), ("drift", drift_task)):
        task_rows = [row for row in rows if task_key_for(row) == task]
        if not task_rows:
            continue
        task_correct = 0
        task_final_correct = 0
        task_base_correct = 0
        for idx, row in enumerate(rows):
            if task_key_for(row) != task:
                continue
            uid = str(row["uid"])
            gt = normalize(row.get("ground_truth") or row.get("gt"))
            pred = base_preds
            if adapter_preds is not None and idx >= ready_index:
                pred = adapter_preds
            task_correct += int(pred.get(uid, "") == gt)
            task_base_correct += int(base_preds.get(uid, "") == gt)
            final_pred = adapter_preds if adapter_preds is not None else base_preds
            task_final_correct += int(final_pred.get(uid, "") == gt)
        out[f"{prefix}_total"] = len(task_rows)
        out[f"{prefix}_base_correct"] = task_base_correct
        out[f"{prefix}_base_acc"] = task_base_correct / len(task_rows)
        out[f"{prefix}_observed_acc_processed_only"] = task_correct / len(task_rows)
        out[f"{prefix}_observed_acc_strict_with_coverage"] = coverage * task_correct / len(task_rows)
        out[f"{prefix}_final_acc_processed_only"] = task_final_correct / len(task_rows)
        out[f"{prefix}_final_acc_strict_with_coverage"] = coverage * task_final_correct / len(task_rows)
    return out


def poll_gpu_memory(gpu_id: str) -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", gpu_id, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        values = [float(part.strip()) for part in out.splitlines() if part.strip()]
        return max(values) / 1024.0 if values else 0.0
    except Exception:
        return 0.0


def run_eval(
    *,
    config: LambdaConfig,
    dataset_dir: Path,
    out_dir: Path,
    gpu_id: str,
    task_ids: str,
    adapter_path: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    log_path = out_dir / "eval.log"
    raw_path = out_dir / "raw_predictions.jsonl"
    if raw_path.exists() and not overwrite:
        return {
            "wall_time_s": 0.0,
            "throughput_samples_per_sec": 0.0,
            "latency_mean_s": 0.0,
            "peak_memory_gib": 0.0,
            "raw_predictions": str(raw_path),
            "log_path": str(log_path),
            "cached": True,
        }
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"Qwen3.5-0.8B-{config.lambda_id}"
    if adapter_path is not None:
        model_name += f"-{adapter_path.parent.parent.name}"
    cmd = [
        str(QWEN_PYTHON),
        str(VLM_EVAL),
        "--dataset-dir",
        str(dataset_dir),
        "--model-path",
        str(QWEN08B_MODEL),
        "--model-name",
        model_name,
        "--family",
        "hf_image_text",
        "--out-dir",
        str(out_dir),
        "--task-ids",
        task_ids,
        "--dtype",
        config.dtype,
        "--batch-size",
        str(config.batch_size),
        "--enable-thinking",
        "false",
        "--answer-source",
        "raw",
        "--mcq-max-new-tokens",
        "8",
        "--multi-mcq-max-new-tokens",
        str(config.max_new_tokens),
    ]
    if adapter_path is not None:
        cmd.extend(["--adapter-path", str(adapter_path)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    start = time.perf_counter()
    peak_gib = 0.0
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            peak_gib = max(peak_gib, poll_gpu_memory(gpu_id))
            time.sleep(0.5)
        peak_gib = max(peak_gib, poll_gpu_memory(gpu_id))
    wall_time = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed with code {proc.returncode}; see {log_path}")
    summary = read_json(out_dir / "summary.json")
    seen = int(summary.get("total_seen") or 0)
    return {
        "wall_time_s": wall_time,
        "throughput_samples_per_sec": seen / wall_time if wall_time > 0 else 0.0,
        "latency_mean_s": wall_time / seen if seen else 0.0,
        "peak_memory_gib": peak_gib,
        "raw_predictions": str(raw_path),
        "log_path": str(log_path),
        "cached": False,
    }


def load_lambda_configs(path: Path) -> list[LambdaConfig]:
    configs = []
    for row in read_csv(path):
        configs.append(
            LambdaConfig(
                lambda_id=row["lambda_id"],
                model_precision=row["model_precision"],
                dtype=row["dtype"],
                roi_strategy=row["roi_strategy"],
                token_retention_ratio=float(row["token_retention_ratio"]),
                target_long_side=int(float(row["target_long_side"])),
                batch_size=int(float(row["batch_size"])),
                image_resolution=int(float(row["image_resolution"])),
                max_new_tokens=int(float(row["max_new_tokens"])),
            )
        )
    return configs


def load_group_rows(split_root: Path, pseudo_root: Path, group_id: str) -> dict[str, list[dict[str, Any]]]:
    base = split_root / "groups" / group_id / "part3_retraining"
    pseudo_group = pseudo_root / "groups" / group_id
    return {
        "R_all": read_json(base / "R_all" / "data.json"),
        "R_val": read_json(base / "R_val" / "data.json"),
        "R_test": read_json(base / "R_test" / "data.json"),
        "R_no_drift": read_json(base / "R_no_drift" / "data.json"),
        "R_val_pseudo": read_json(pseudo_group / "R_val" / "data.json"),
    }


def lambda_group_accuracy(lambda_row: dict[str, Any], drift_task: str, score_key: str) -> tuple[float, int, int]:
    scores = json.loads(lambda_row.get(score_key) or "{}")
    bucket = scores.get(drift_task, {})
    seen = int(bucket.get("seen") or 0)
    correct = int(bucket.get("correct") or 0)
    return (correct / seen if seen else 0.0, correct, seen)


def discounted_average(current_acc: float, future_acc: float, future_windows: int, discount: float) -> float:
    weights = [1.0] + [discount**idx for idx in range(1, max(0, future_windows) + 1)]
    return (weights[0] * current_acc + sum(weight * future_acc for weight in weights[1:])) / sum(weights)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def arrival_rate_arg(args: argparse.Namespace) -> float | None:
    value = getattr(args, "arrival_rate", None)
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def effective_window_seconds(total_rows: int, args: argparse.Namespace) -> float:
    rate = arrival_rate_arg(args)
    if rate:
        return total_rows / rate if total_rows > 0 else 0.0
    return float(args.retraining_window_sec)


def ready_index_for_finish(
    *,
    finish_time: float,
    total_rows: int,
    args: argparse.Namespace,
    reference_total_rows: int | None = None,
) -> int:
    if total_rows <= 0:
        return 0
    rate = arrival_rate_arg(args)
    if rate:
        reference_total = max(1, int(reference_total_rows or total_rows))
        arrived_in_reference = max(0.0, rate * finish_time)
        ready_ratio = min(1.0, arrived_in_reference / reference_total)
        return min(total_rows, math.ceil(ready_ratio * total_rows))
    window = max(1e-9, float(args.retraining_window_sec))
    ready_ratio = min(1.0, max(0.0, finish_time / window))
    return min(total_rows, math.ceil(ready_ratio * total_rows))


def lambda_resource_cost_for_pressure(lambda_row: dict[str, Any], args: argparse.Namespace, normalized_cost: float) -> float:
    rate = arrival_rate_arg(args)
    if not rate:
        return normalized_cost
    throughput = as_float(lambda_row, "throughput_samples_per_sec")
    if throughput > 0:
        return rate / throughput
    latency = as_float(lambda_row, "latency_mean_s")
    if latency > 0:
        return rate * latency
    return normalized_cost


def build_decision(
    *,
    method: str,
    group_id: str,
    drift_task: str,
    lambda_row: dict[str, Any],
    gamma_row: dict[str, Any] | None,
    rows: dict[str, list[dict[str, Any]]],
    base_preds: dict[str, str],
    adapter_preds: dict[str, str] | None,
    base_rval_pseudo_acc: float,
    adapter_rval_pseudo_acc_80step: float,
    inference_resource: float,
    training_resource: float,
    triggered: bool,
    adapter_accepted: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    lambda_cost = as_float(lambda_row, "lambda_resource_cost_effective")
    gamma_cost = as_float(gamma_row or {}, "gamma_resource_cost_effective")
    coverage = min(1.0, inference_resource / max(lambda_cost, 1e-9))
    can_train = bool(triggered and adapter_accepted and gamma_row and training_resource >= gamma_cost and training_resource > 0)
    target_time = as_float(gamma_row or {}, "target_train_time_s_est")
    finish_time = target_time / training_resource if can_train else None
    current_window_sec = effective_window_seconds(len(rows["R_all"]), args)
    adapter_replaced = bool(finish_time is not None and finish_time <= current_window_sec)
    ready_index = len(rows["R_all"])
    ready_val_index = len(rows["R_val_pseudo"])
    if adapter_replaced:
        ready_index = ready_index_for_finish(finish_time=finish_time, total_rows=len(rows["R_all"]), args=args)
        ready_val_index = ready_index_for_finish(
            finish_time=finish_time,
            total_rows=len(rows["R_val_pseudo"]),
            args=args,
            reference_total_rows=len(rows["R_all"]),
        )

    actual_adapter_preds = adapter_preds if adapter_replaced else None
    current_actual = mixed_accuracy(rows["R_all"], base_preds, actual_adapter_preds, ready_index=ready_index)
    current_processed = current_actual["observed_time_weighted_accuracy"]
    current_with_coverage = coverage * current_processed
    base_rtest_score = score_rows(rows["R_test"], base_preds)
    adapter_rtest_score = score_rows(rows["R_test"], adapter_preds or {}) if adapter_preds else {"accuracy_seen": 0.0}
    future_true_acc = adapter_rtest_score["accuracy_seen"] if adapter_replaced and adapter_preds else base_rtest_score["accuracy_seen"]
    future_true_with_coverage = coverage * future_true_acc
    actual_long = discounted_average(current_with_coverage, future_true_with_coverage, args.future_windows, args.discount_factor)

    projected_gain = max(0.0, as_float(gamma_row or {}, "extrapolated_target_gain"))
    projected_future_pseudo = clamp01(base_rval_pseudo_acc + projected_gain) if adapter_replaced else base_rval_pseudo_acc
    estimated_ready_ratio = min(1.0, ready_val_index / max(1, len(rows["R_val_pseudo"])))
    estimated_current_processed = (
        estimated_ready_ratio * base_rval_pseudo_acc
        + (1.0 - estimated_ready_ratio) * projected_future_pseudo
        if adapter_replaced
        else base_rval_pseudo_acc
    )
    estimated_current = coverage * estimated_current_processed
    estimated_future = coverage * projected_future_pseudo
    estimated_long = discounted_average(estimated_current, estimated_future, args.future_windows, args.discount_factor)

    row = {
        "group_id": group_id,
        "drift_task": drift_task,
        "model_id": MODEL_ID,
        "method": method,
        "triggered": bool(triggered),
        "lambda_id": lambda_row["lambda_id"],
        "gamma_id": gamma_row["gamma_id"] if gamma_row and can_train else "",
        "I": round(inference_resource, 6),
        "R": round(training_resource, 6),
        "adapter_accepted": bool(adapter_accepted and gamma_row),
        "adapter_replaced": adapter_replaced,
        "coverage": round(coverage, 6),
        "estimated_current_accuracy": round(estimated_current, 6),
        "estimated_future_or_final_accuracy": round(estimated_future, 6),
        "estimated_long_avg_accuracy": round(estimated_long, 6),
        "actual_long_avg_accuracy": round(actual_long, 6),
        "observed_current_accuracy": round(current_with_coverage, 6),
        "observed_final_accuracy": round(future_true_with_coverage, 6),
        "observed_time_weighted_accuracy": round(current_processed, 6),
        "adapter_ready_sample_count": ready_index,
        "adapter_ready_time_s": round(finish_time, 6) if finish_time is not None else "",
        "train_time_s_effective": round(finish_time, 6) if finish_time is not None else "",
        "T_retrain_s": round(current_window_sec, 6),
        "arrival_rate": round(arrival_rate_arg(args), 6) if arrival_rate_arg(args) else "",
        "lambda_resource_cost_effective": round(lambda_cost, 6),
        "gamma_resource_cost_effective": round(gamma_cost, 6),
        "lambda_dtop40_teacher_accuracy_group": dtop40_value(lambda_row, "teacher_accuracy_group"),
        "lambda_dtop40_true_accuracy_group": dtop40_value(lambda_row, "true_accuracy_group"),
        "lambda_dtop40_teacher_accuracy_all": dtop40_value(lambda_row, "teacher_accuracy"),
        "lambda_dtop40_true_accuracy_all": dtop40_value(lambda_row, "true_accuracy"),
        "lambda_latency_mean_s": lambda_row.get("latency_mean_s", ""),
        "rval_base_pseudo_accuracy_before_adapter": round(base_rval_pseudo_acc, 6),
        "rval_adapter_pseudo_accuracy_after_80step": round(adapter_rval_pseudo_acc_80step, 6),
        "small_step_profile_gain": gamma_row.get("small_step_profile_gain", "") if gamma_row else "",
        "extrapolated_target_gain": gamma_row.get("extrapolated_target_gain", "") if gamma_row else "",
        "target_train_time_s_est": gamma_row.get("target_train_time_s_est", "") if gamma_row else "",
        "microprofile_train_time_s": gamma_row.get("microprofile_train_time_s", "") if gamma_row else "",
        "actual_adapter_source": "microprofile_80step_checkpoint" if adapter_preds else "",
        "actual_rtest_base_accuracy_before_adapter": round(base_rtest_score["accuracy_seen"], 6),
        "actual_rtest_adapter_accuracy_after_adapter": round(adapter_rtest_score["accuracy_seen"], 6) if adapter_preds else "",
    }
    row.update({key: round(value, 6) if isinstance(value, float) else value for key, value in current_actual.items()})
    row.update(
        score_by_task_fields(
            rows=rows["R_all"],
            base_preds=base_preds,
            adapter_preds=actual_adapter_preds,
            ready_index=ready_index,
            coverage=coverage,
            drift_task=drift_task,
        )
    )
    row["num_arrived"] = len(rows["R_all"])
    row["num_processed"] = round(coverage * len(rows["R_all"]), 6)
    row["num_timeout"] = round(len(rows["R_all"]) - coverage * len(rows["R_all"]), 6)
    row["base_correct"] = score_rows(rows["R_all"], base_preds)["correct"]
    row["base_total"] = len(rows["R_all"])
    row["adapter_correct"] = score_rows(rows["R_all"], adapter_preds or {})["correct"] if adapter_preds else ""
    row["adapter_total"] = len(rows["R_all"]) if adapter_preds else ""
    row["current_error_actual_minus_estimated"] = round(row["observed_current_accuracy"] - row["estimated_current_accuracy"], 6)
    row["abs_current_error"] = abs(row["current_error_actual_minus_estimated"])
    row["final_error_actual_minus_estimated_future"] = round(
        row["observed_final_accuracy"] - row["estimated_future_or_final_accuracy"], 6
    )
    return row


def choose_fixed_lambda(lambda_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in lambda_rows:
        if (
            row.get("model_precision") == "bf16"
            and row.get("roi_strategy") == "full"
            and abs(as_float(row, "token_retention_ratio") - 1.0) < 1e-9
        ):
            return row
    return max(
        lambda_rows,
        key=lambda row: (
            float(dtop40_value(row, "teacher_accuracy_group", 0.0)),
            -as_float(row, "lambda_resource_cost_effective"),
        ),
    )


def choose_fixed_gamma(gamma_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in gamma_rows:
        if row["gamma_id"] == "qwen35_lora_r16_a32_profile80_target320":
            return row
    return max(gamma_rows, key=lambda row: (as_float(row, "extrapolated_target_gain"), -as_float(row, "gamma_resource_cost_effective")))


def aggregate_method_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    methods = []
    for row in method_rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    weighted_fields = [
        "estimated_current_accuracy",
        "estimated_future_or_final_accuracy",
        "estimated_long_avg_accuracy",
        "actual_long_avg_accuracy",
        "observed_current_accuracy",
        "observed_final_accuracy",
        "observed_time_weighted_accuracy",
        "actual_rtest_base_accuracy_before_adapter",
        "actual_rtest_adapter_accuracy_after_adapter",
    ]
    for method in methods:
        rows = [row for row in method_rows if row["method"] == method]
        weight = sum(as_float(row, "num_arrived") for row in rows)
        agg = {
            "group_id": "ALL_GROUPS_WEIGHTED",
            "model_id": MODEL_ID,
            "method": method,
            "groups": len(rows),
            "num_arrived": weight,
        }
        for field in weighted_fields:
            values = [as_float(row, field, float("nan")) for row in rows if row.get(field) != ""]
            if not values:
                agg[field] = ""
                continue
            agg[field] = round(
                sum(as_float(row, field) * as_float(row, "num_arrived") for row in rows if row.get(field) != "") / weight,
                6,
            ) if weight else 0.0
        agg["selected_lambda_ids"] = ";".join(str(row["lambda_id"]) for row in rows)
        agg["selected_gamma_ids"] = ";".join(str(row.get("gamma_id", "")) for row in rows)
        agg["I_by_group"] = ";".join(str(row["I"]) for row in rows)
        agg["R_by_group"] = ";".join(str(row["R"]) for row in rows)
        out.append(agg)
    return out


def run_eval_cache(
    *,
    args: argparse.Namespace,
    lambda_configs: list[LambdaConfig],
    canonical_timing_by_lambda: dict[str, dict[str, Any]],
    canonical_timing_source: Path,
    gamma_rows: list[dict[str, Any]],
    group_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_group = {group_id: load_group_rows(args.split_root, args.pseudo_root, group_id) for group_id in group_ids}
    datasets: dict[tuple[str, str], Path] = {}
    for group_id in group_ids:
        source = args.split_root / "groups" / group_id / "part3_retraining" / "R_all"
        for config in lambda_configs:
            datasets[(group_id, config.lambda_id)] = prepare_dataset(
                source_dir=source,
                output_dir=args.output_dir / "eval_cache",
                dataset_key=f"{group_id}/R_all",
                config=config,
                overwrite=args.overwrite_datasets,
            )

    gpu_ids = [part.strip() for part in args.gpu_ids.split(",") if part.strip()]
    jobs: list[dict[str, Any]] = []
    by_lambda_id = {config.lambda_id: config for config in lambda_configs}
    for group_id in group_ids:
        group_gamma = [row for row in gamma_rows if row["group_id"] == group_id]
        for config in lambda_configs:
            jobs.append(
                {
                    "kind": "base",
                    "group_id": group_id,
                    "lambda_id": config.lambda_id,
                    "gamma_id": "",
                    "adapter_path": None,
                    "out_dir": args.output_dir / "eval_cache" / "base" / group_id / config.lambda_id,
                }
            )
            for gamma in group_gamma:
                jobs.append(
                    {
                        "kind": "adapter",
                        "group_id": group_id,
                        "lambda_id": config.lambda_id,
                        "gamma_id": gamma["gamma_id"],
                        "adapter_path": Path(gamma["microprofile_adapter_path"]),
                        "out_dir": args.output_dir
                        / "eval_cache"
                        / "adapter"
                        / group_id
                        / config.lambda_id
                        / gamma["gamma_id"],
                    }
                )
    if args.max_eval_jobs:
        jobs = jobs[: args.max_eval_jobs]

    completed: list[dict[str, Any]] = []

    def worker(index: int, job: dict[str, Any]) -> dict[str, Any]:
        config = by_lambda_id[job["lambda_id"]]
        dataset_dir = datasets[(job["group_id"], job["lambda_id"])]
        gpu_id = gpu_ids[index % len(gpu_ids)]
        stats = run_eval(
            config=config,
            dataset_dir=dataset_dir,
            out_dir=job["out_dir"],
            gpu_id=gpu_id,
            task_ids=args.task_ids,
            adapter_path=job["adapter_path"],
            overwrite=args.overwrite_evals,
        )
        preds = prediction_map(Path(stats["raw_predictions"]))
        group_rows = rows_by_group[job["group_id"]]
        scores = {
            "R_all_true": score_rows(group_rows["R_all"], preds),
            "R_val_true": score_rows(group_rows["R_val"], preds),
            "R_test_true": score_rows(group_rows["R_test"], preds),
            "R_val_pseudo": score_rows(group_rows["R_val_pseudo"], preds),
        }
        canonical_timing = canonical_timing_by_lambda[job["lambda_id"]]
        out = {
            "kind": job["kind"],
            "group_id": job["group_id"],
            "lambda_id": job["lambda_id"],
            "gamma_id": job["gamma_id"],
            "dataset_dir": str(dataset_dir),
            "raw_predictions": stats["raw_predictions"],
            "log_path": stats["log_path"],
            "cached": stats["cached"],
            "wall_time_s": canonical_timing.get("wall_time_s", ""),
            "latency_mean_s": canonical_timing["latency_mean_s"],
            "throughput_samples_per_sec": canonical_timing["throughput_samples_per_sec"],
            "timing_source": str(canonical_timing_source),
            "timing_reused": True,
            "observed_wall_time_s": stats["wall_time_s"],
            "observed_latency_mean_s": stats["latency_mean_s"],
            "observed_throughput_samples_per_sec": stats["throughput_samples_per_sec"],
            "peak_memory_gib": stats["peak_memory_gib"],
            "rall_true_accuracy": scores["R_all_true"]["accuracy_seen"],
            "rval_true_accuracy": scores["R_val_true"]["accuracy_seen"],
            "rtest_true_accuracy": scores["R_test_true"]["accuracy_seen"],
            "rval_pseudo_accuracy": scores["R_val_pseudo"]["accuracy_seen"],
            "rall_correct": scores["R_all_true"]["correct"],
            "rall_total": scores["R_all_true"]["seen"],
            "rtest_correct": scores["R_test_true"]["correct"],
            "rtest_total": scores["R_test_true"]["seen"],
            "rall_score_by_task": json.dumps(scores["R_all_true"]["by_task"], ensure_ascii=False, sort_keys=True),
            "rval_score_by_task": json.dumps(scores["R_val_true"]["by_task"], ensure_ascii=False, sort_keys=True),
            "rtest_score_by_task": json.dumps(scores["R_test_true"]["by_task"], ensure_ascii=False, sort_keys=True),
        }
        return out

    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(gpu_ids)))) as executor:
        futures = {executor.submit(worker, idx, job): job for idx, job in enumerate(jobs)}
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            completed.sort(key=lambda item: (item["kind"], item["group_id"], item["lambda_id"], item["gamma_id"]))
            write_csv(args.output_dir / "eval_cache" / "r_window_eval_cache.partial.csv", completed)
            print(
                f"[eval-cache] {row['kind']} {row['group_id']} {row['lambda_id']} {row['gamma_id']} "
                f"R_all={row['rall_true_accuracy']:.4f} R_test={row['rtest_true_accuracy']:.4f}",
                flush=True,
            )

    write_csv(args.output_dir / "eval_cache" / "r_window_eval_cache.csv", completed)
    base_rows = [row for row in completed if row["kind"] == "base"]
    adapter_rows = [row for row in completed if row["kind"] == "adapter"]
    write_csv(args.output_dir / "eval_cache" / "base_r_window_by_group_lambda.csv", base_rows)
    write_csv(args.output_dir / "eval_cache" / "adapter_r_window_by_group_lambda_gamma.csv", adapter_rows)
    return base_rows, adapter_rows


def maybe_load_eval_cache(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    cache_path = out_dir / "eval_cache" / "r_window_eval_cache.csv"
    if not cache_path.exists():
        return None
    rows = read_csv(cache_path)
    return [row for row in rows if row["kind"] == "base"], [row for row in rows if row["kind"] == "adapter"]


def run_scheduler(
    *,
    args: argparse.Namespace,
    group_ids: list[str],
    lambda_rows_all: list[dict[str, Any]],
    gamma_rows_all: list[dict[str, Any]],
    base_eval_rows: list[dict[str, Any]],
    adapter_eval_rows: list[dict[str, Any]],
    detector_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    max_latency = max(as_float(row, "latency_mean_s") for row in lambda_rows_all)
    max_target_time = max(as_float(row, "target_train_time_s_est") for row in gamma_rows_all)
    detector_by_group = {row["group_id"]: row for row in detector_summary["groups"]}
    base_eval_by_key = {(row["group_id"], row["lambda_id"]): row for row in base_eval_rows}
    adapter_eval_by_key = {
        (row["group_id"], row["lambda_id"], row["gamma_id"]): row for row in adapter_eval_rows
    }

    method_rows: list[dict[str, Any]] = []
    lambda_detail: list[dict[str, Any]] = []
    gamma_detail: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for group_id in group_ids:
        drift_task = drift_task_for_group(group_id)
        rows = load_group_rows(args.split_root, args.pseudo_root, group_id)
        lambda_rows = []
        for base in lambda_rows_all:
            row = dict(base)
            true_acc, true_correct, true_total = lambda_group_accuracy(row, drift_task, "true_score_by_task")
            teacher_acc, teacher_correct, teacher_total = lambda_group_accuracy(row, drift_task, "teacher_score_by_task")
            row["group_id"] = group_id
            row["drift_task"] = drift_task
            row["dtop40_true_accuracy_group"] = true_acc
            row["dtop40_teacher_accuracy_group"] = teacher_acc
            row["dtop40_correct_group"] = true_correct
            row["dtop40_teacher_correct_group"] = teacher_correct
            row["dtop40_total_group"] = true_total or teacher_total
            row["dtop80_true_accuracy_group"] = true_acc
            row["dtop80_teacher_accuracy_group"] = teacher_acc
            row["dtop80_correct_group"] = true_correct
            row["dtop80_teacher_correct_group"] = teacher_correct
            row["dtop80_total_group"] = true_total or teacher_total
            normalized_cost = max(
                0.01,
                min(args.lambda_max_resource_cost, as_float(row, "latency_mean_s") / max_latency * args.lambda_max_resource_cost),
            )
            row["lambda_resource_cost_effective"] = lambda_resource_cost_for_pressure(row, args, normalized_cost)
            lambda_rows.append(row)
        gamma_rows = []
        for base in gamma_rows_all:
            if base["group_id"] != group_id:
                continue
            row = dict(base)
            row["gamma_resource_cost_effective"] = max(
                0.01,
                min(args.gamma_max_resource_cost, as_float(row, "target_train_time_s_est") / max_target_time * args.gamma_max_resource_cost),
            )
            gamma_rows.append(row)

        fixed_lambda = choose_fixed_lambda(lambda_rows)
        fixed_gamma = choose_fixed_gamma(gamma_rows)
        triggered = bool(detector_by_group[group_id].get("actual_D_window_triggered"))
        candidates = []
        oracle_candidates = []
        pair_best: dict[tuple[str, str], dict[str, Any]] = {}
        for lambda_row in lambda_rows:
            base_eval = base_eval_by_key[(group_id, lambda_row["lambda_id"])]
            base_preds = prediction_map(Path(base_eval["raw_predictions"]))
            base_rval_pseudo = as_float(base_eval, "rval_pseudo_accuracy")
            for gamma_row in gamma_rows:
                adapter_eval = adapter_eval_by_key[(group_id, lambda_row["lambda_id"], gamma_row["gamma_id"])]
                adapter_preds = prediction_map(Path(adapter_eval["raw_predictions"]))
                adapter_rval_pseudo = as_float(adapter_eval, "rval_pseudo_accuracy")
                accepted = as_float(gamma_row, "extrapolated_target_gain") > args.min_profile_gain
                best_for_pair: dict[str, Any] | None = None
                for idx in range(1, 100):
                    I = idx / 100.0
                    decision = build_decision(
                        method="Ours",
                        group_id=group_id,
                        drift_task=drift_task,
                        lambda_row=lambda_row,
                        gamma_row=gamma_row,
                        rows=rows,
                        base_preds=base_preds,
                        adapter_preds=adapter_preds,
                        base_rval_pseudo_acc=base_rval_pseudo,
                        adapter_rval_pseudo_acc_80step=adapter_rval_pseudo,
                        inference_resource=I,
                        training_resource=1.0 - I,
                        triggered=triggered,
                        adapter_accepted=accepted,
                        args=args,
                    )
                    candidates.append(decision)
                    oracle_decision = dict(decision)
                    oracle_decision["method"] = "OfflineJointOracle"
                    oracle_decision["triggered"] = True
                    oracle_candidates.append(oracle_decision)
                    if best_for_pair is None or float(decision["estimated_long_avg_accuracy"]) > float(
                        best_for_pair["estimated_long_avg_accuracy"]
                    ):
                        best_for_pair = decision
                if best_for_pair is not None:
                    pair_best[(lambda_row["lambda_id"], gamma_row["gamma_id"])] = best_for_pair

        ours = max(
            candidates,
            key=lambda row: (
                as_float(row, "estimated_long_avg_accuracy"),
                as_float(row, "estimated_future_or_final_accuracy"),
                as_float(row, "estimated_current_accuracy"),
                -as_float(row, "lambda_resource_cost_effective"),
            ),
        )
        oracle = max(
            oracle_candidates,
            key=lambda row: (
                as_float(row, "actual_long_avg_accuracy"),
                as_float(row, "observed_final_accuracy"),
                as_float(row, "observed_current_accuracy"),
            ),
        )

        fixed_base_eval = base_eval_by_key[(group_id, fixed_lambda["lambda_id"])]
        fixed_base_preds = prediction_map(Path(fixed_base_eval["raw_predictions"]))
        fixed_adapter_eval = adapter_eval_by_key[(group_id, fixed_lambda["lambda_id"], fixed_gamma["gamma_id"])]
        fixed_adapter_preds = prediction_map(Path(fixed_adapter_eval["raw_predictions"]))
        fixed_base_rval_pseudo = as_float(fixed_base_eval, "rval_pseudo_accuracy")
        fixed_adapter_rval_pseudo = as_float(fixed_adapter_eval, "rval_pseudo_accuracy")
        no_retrain = build_decision(
            method="NoRetrain",
            group_id=group_id,
            drift_task=drift_task,
            lambda_row=fixed_lambda,
            gamma_row=None,
            rows=rows,
            base_preds=fixed_base_preds,
            adapter_preds=None,
            base_rval_pseudo_acc=fixed_base_rval_pseudo,
            adapter_rval_pseudo_acc_80step=fixed_base_rval_pseudo,
            inference_resource=1.0,
            training_resource=0.0,
            triggered=False,
            adapter_accepted=False,
            args=args,
        )
        periodic = build_decision(
            method="PeriodicFixedRetrain",
            group_id=group_id,
            drift_task=drift_task,
            lambda_row=fixed_lambda,
            gamma_row=fixed_gamma,
            rows=rows,
            base_preds=fixed_base_preds,
            adapter_preds=fixed_adapter_preds,
            base_rval_pseudo_acc=fixed_base_rval_pseudo,
            adapter_rval_pseudo_acc_80step=fixed_adapter_rval_pseudo,
            inference_resource=args.periodic_inference_resource,
            training_resource=1.0 - args.periodic_inference_resource,
            triggered=True,
            adapter_accepted=True,
            args=args,
        )
        static = build_decision(
            method="StaticSplitContinuous",
            group_id=group_id,
            drift_task=drift_task,
            lambda_row=fixed_lambda,
            gamma_row=fixed_gamma,
            rows=rows,
            base_preds=fixed_base_preds,
            adapter_preds=fixed_adapter_preds,
            base_rval_pseudo_acc=fixed_base_rval_pseudo,
            adapter_rval_pseudo_acc_80step=fixed_adapter_rval_pseudo,
            inference_resource=args.static_inference_resource,
            training_resource=1.0 - args.static_inference_resource,
            triggered=True,
            adapter_accepted=True,
            args=args,
        )
        group_methods = [no_retrain, periodic, static, ours, oracle]
        method_rows.extend(group_methods)

        selected_pairs = {(row["lambda_id"], row["gamma_id"]) for row in group_methods if row.get("gamma_id")}
        for row in lambda_rows:
            detail = {
                "group_id": group_id,
                "drift_task": drift_task,
                "lambda_id": row["lambda_id"],
                "selected_by_ours": row["lambda_id"] == ours["lambda_id"],
                "dtop40_true_accuracy": row["dtop40_true_accuracy_group"],
                "dtop40_teacher_accuracy": row["dtop40_teacher_accuracy_group"],
                "dtop40_correct": row["dtop40_correct_group"],
                "dtop40_teacher_correct": row["dtop40_teacher_correct_group"],
                "dtop40_total": row["dtop40_total_group"],
                "dtop80_true_accuracy": row["dtop40_true_accuracy_group"],
                "dtop80_teacher_accuracy": row["dtop40_teacher_accuracy_group"],
                "dtop80_correct": row["dtop40_correct_group"],
                "dtop80_teacher_correct": row["dtop40_teacher_correct_group"],
                "dtop80_total": row["dtop40_total_group"],
                "model_precision": row["model_precision"],
                "roi_strategy": row["roi_strategy"],
                "token_retention_ratio": row["token_retention_ratio"],
                "resource_cost": row["lambda_resource_cost_effective"],
                "latency_mean_s": row["latency_mean_s"],
                "raw_predictions": row["raw_predictions"],
            }
            lambda_detail.append(detail)
        for row in gamma_rows:
            detail = dict(row)
            detail["selected_by_ours"] = (ours["lambda_id"], row["gamma_id"]) in selected_pairs
            gamma_detail.append(detail)
        for (lambda_id, gamma_id), row in pair_best.items():
            pair_rows.append(
                {
                    "group_id": group_id,
                    "drift_task": drift_task,
                    "lambda_id": lambda_id,
                    "gamma_id": gamma_id,
                    "is_selected_pair_by_ours": lambda_id == ours["lambda_id"] and gamma_id == ours["gamma_id"],
                    "best_I_for_pair": row["I"],
                    "best_R_for_pair": row["R"],
                    "dtop40_teacher_accuracy": row["lambda_dtop40_teacher_accuracy_group"],
                    "dtop40_true_accuracy": row["lambda_dtop40_true_accuracy_group"],
                    "dtop80_teacher_accuracy": row["lambda_dtop40_teacher_accuracy_group"],
                    "dtop80_true_accuracy": row["lambda_dtop40_true_accuracy_group"],
                    "lambda_resource_cost": row["lambda_resource_cost_effective"],
                    "gamma_resource_cost": row["gamma_resource_cost_effective"],
                    "small_step_profile_gain": row["small_step_profile_gain"],
                    "extrapolated_target_gain": row["extrapolated_target_gain"],
                    "rval_base_accuracy_before_adapter": row["rval_base_pseudo_accuracy_before_adapter"],
                    "rval_adapter_accuracy_after_80step": row["rval_adapter_pseudo_accuracy_after_80step"],
                    "estimated_current_accuracy_if_selected": row["estimated_current_accuracy"],
                    "estimated_future_or_final_accuracy_if_selected": row["estimated_future_or_final_accuracy"],
                    "estimated_long_avg_accuracy_if_selected": row["estimated_long_avg_accuracy"],
                    "actual_current_accuracy_if_selected": row["observed_current_accuracy"],
                    "actual_final_accuracy_if_selected": row["observed_final_accuracy"],
                    "actual_long_avg_accuracy_if_selected": row["actual_long_avg_accuracy"],
                    "current_error_if_selected": row["current_error_actual_minus_estimated"],
                    "final_error_if_selected": row["final_error_actual_minus_estimated_future"],
                }
            )

    return method_rows, aggregate_method_rows(method_rows), lambda_detail, gamma_detail, pair_rows


def write_summary(
    out_dir: Path,
    *,
    detector_summary: dict[str, Any],
    method_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    lambda_detail: list[dict[str, Any]],
    gamma_detail: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Qwen3.5-0.8B OmniEarth Augmented v2 Formal Framework",
        "",
        "## Protocol",
        "",
        "- Split: augmented v2 from base, preserving Part0 and 27B teacher train/test UIDs.",
        "- Lambda selection: all 6 configs, no Pareto filter.",
        "- Gamma selection: 5 LoRA configs per group, 80-step measured profile plus target-step extrapolation.",
        "- Actual adapter accuracy below uses the cached 80-step adapter checkpoint for each gamma config.",
        "- Scheduler grid: I=0.01..0.99, R=1-I.",
        "",
        "## Detector",
        "",
        "| group | D_top40 | FPR@0 | R@50 | triggered | trigger_start |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in detector_summary["groups"]:
        lines.append(
            f"| {row['group_id']} | {row.get('D_top40', row.get('D_top80', ''))} | {row['fpr_at_0']:.3f} | "
            f"{row['recall_at_50']:.3f} | {row['actual_D_window_triggered']} | "
            f"{row['actual_D_window_trigger_start']} |"
        )
    lines.extend(
        [
            "",
            "## Overall Method Comparison",
            "",
            "| method | estimated_long | actual_long | current | final | before/after R_test |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in overall_rows:
        lines.append(
            f"| {row['method']} | {row['estimated_long_avg_accuracy']} | {row['actual_long_avg_accuracy']} | "
            f"{row['observed_current_accuracy']} | {row['observed_final_accuracy']} | "
            f"{row['actual_rtest_base_accuracy_before_adapter']} -> {row['actual_rtest_adapter_accuracy_after_adapter']} |"
        )
    lines.extend(
        [
            "",
            "## Selected Configs By Group",
            "",
            "| group | method | lambda | gamma | I | R | ready_n | estimated_long | actual_long | R_test before->after |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in method_rows:
        lines.append(
            f"| {row['group_id']} | {row['method']} | {row['lambda_id']} | {row['gamma_id']} | "
            f"{row['I']} | {row['R']} | {row['adapter_ready_sample_count']} | "
            f"{row['estimated_long_avg_accuracy']} | {row['actual_long_avg_accuracy']} | "
            f"{row['actual_rtest_base_accuracy_before_adapter']} -> {row['actual_rtest_adapter_accuracy_after_adapter']} |"
        )
    lines.extend(
        [
            "",
            "## Table Counts",
            "",
            f"- lambda_dtop40_by_group_config rows: {len(lambda_detail)}",
            f"- gamma_smallstep_by_group_config rows: {len(gamma_detail)}",
            f"- lambda_gamma_estimates_all_pairs rows: {len(pair_rows)}",
        ]
    )
    (out_dir / "framework_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    result_root = PROJECT_ROOT / "results" / "omniearth_qwen35_framework_20260705"
    parser = argparse.ArgumentParser(description="Run Qwen3.5-0.8B OmniEarth augmented-v2 formal scheduler.")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=result_root / "formal_qwen35_framework_splits_augmented_v2_from_base",
    )
    parser.add_argument(
        "--pseudo-root",
        type=Path,
        default=result_root / "formal_teacher_27b_lora_r8a16_s90_augmented_v2" / "pseudo_labels",
    )
    parser.add_argument(
        "--lambda-root",
        type=Path,
        default=result_root / "formal_qwen35_08b_lambda_profiles_augmented_v2_all",
    )
    parser.add_argument(
        "--gamma-root",
        type=Path,
        default=result_root / "formal_qwen35_08b_gamma_profiles_augmented_v2",
    )
    parser.add_argument(
        "--detector-root",
        type=Path,
        default=result_root / "formal_qwen35_detector_augmented_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=result_root / "formal_qwen35_08b_framework_augmented_v2_all_no_pareto",
    )
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-ids", default="930,931,932")
    parser.add_argument("--lambda-max-resource-cost", type=float, default=0.65)
    parser.add_argument("--gamma-max-resource-cost", type=float, default=0.35)
    parser.add_argument("--periodic-inference-resource", type=float, default=0.50)
    parser.add_argument("--static-inference-resource", type=float, default=0.70)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Optional samples/s pressure setting. When set, adapter ready samples use ceil(arrival_rate * finish_time).",
    )
    parser.add_argument("--retraining-window-sec", type=float, default=7200.0)
    parser.add_argument("--future-windows", type=int, default=3)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--min-profile-gain", type=float, default=0.0)
    parser.add_argument("--overwrite-evals", action="store_true")
    parser.add_argument("--overwrite-datasets", action="store_true")
    parser.add_argument("--skip-evals", action="store_true")
    parser.add_argument("--max-eval-jobs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.split_root = args.split_root.resolve()
    args.pseudo_root = args.pseudo_root.resolve()
    args.lambda_root = args.lambda_root.resolve()
    args.gamma_root = args.gamma_root.resolve()
    args.detector_root = args.detector_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lambda_configs = load_lambda_configs(args.lambda_root / "lambda_config_table.csv")
    canonical_timing_source = args.lambda_root / "lambda_dtop40_by_config.csv"
    if not canonical_timing_source.exists():
        canonical_timing_source = args.lambda_root / "lambda_dtop80_by_config.csv"
    lambda_rows = read_csv(canonical_timing_source)
    canonical_timing_by_lambda = {str(row["lambda_id"]): row for row in lambda_rows}
    for row in lambda_rows:
        row["timing_source"] = str(canonical_timing_source)
        row["timing_reused"] = True
    gamma_rows = read_csv(args.gamma_root / "gamma_smallstep_by_config.csv")
    detector_summary = read_json(args.detector_root / "formal_detector_summary.json")
    group_ids = sorted(path.name for path in (args.split_root / "groups").glob("*") if path.is_dir())

    cache = maybe_load_eval_cache(args.output_dir) if args.skip_evals else None
    if cache is None:
        base_eval_rows, adapter_eval_rows = run_eval_cache(
            args=args,
            lambda_configs=lambda_configs,
            canonical_timing_by_lambda=canonical_timing_by_lambda,
            canonical_timing_source=canonical_timing_source,
            gamma_rows=gamma_rows,
            group_ids=group_ids,
        )
    else:
        base_eval_rows, adapter_eval_rows = cache
    expected_base = len(group_ids) * len(lambda_configs)
    expected_adapter = len(group_ids) * len(lambda_configs) * 5
    if len(base_eval_rows) < expected_base or len(adapter_eval_rows) < expected_adapter:
        raise RuntimeError(
            f"Eval cache incomplete: base {len(base_eval_rows)}/{expected_base}, "
            f"adapter {len(adapter_eval_rows)}/{expected_adapter}"
        )

    method_rows, overall_rows, lambda_detail, gamma_detail, pair_rows = run_scheduler(
        args=args,
        group_ids=group_ids,
        lambda_rows_all=lambda_rows,
        gamma_rows_all=gamma_rows,
        base_eval_rows=base_eval_rows,
        adapter_eval_rows=adapter_eval_rows,
        detector_summary=detector_summary,
    )
    scheduler_dir = args.output_dir / "scheduler"
    detail_dir = args.output_dir / "detailed_result_tables"
    true_dir = args.output_dir / "true_accuracy"
    write_csv(scheduler_dir / "scheduler_end_to_end_results.csv", method_rows)
    write_csv(scheduler_dir / "scheduler_overall_weighted_results.csv", overall_rows)
    write_csv(true_dir / "scheduler_true_accuracy_error_rows.csv", method_rows)
    write_csv(detail_dir / "lambda_dtop40_by_group_config.csv", lambda_detail)
    write_csv(detail_dir / "lambda_dtop80_by_group_config.csv", lambda_detail)
    write_csv(detail_dir / "gamma_smallstep_by_group_config.csv", gamma_detail)
    write_csv(detail_dir / "lambda_gamma_estimates_all_pairs.csv", pair_rows)
    write_csv(detail_dir / "selected_config_estimated_vs_actual.csv", method_rows)
    write_json(
        args.output_dir / "framework_results.json",
        {
            "protocol": {
                "model_id": MODEL_ID,
                "teacher_model": "Qwen3.5-27B LoRA checkpoint-90",
                "student_model": MODEL_ID,
                "split_root": str(args.split_root),
                "pseudo_root": str(args.pseudo_root),
                "lambda_root": str(args.lambda_root),
                "canonical_lambda_timing_source": str(canonical_timing_source),
                "timing_policy": "canonical_reuse_observed_runtime_non_authoritative",
                "gamma_root": str(args.gamma_root),
                "detector_root": str(args.detector_root),
                "lambda_selection_policy": "all_configs_no_pareto",
                "resource_grid": "I=0.01..0.99, R=1-I",
                "actual_adapter_source": "80-step microprofile checkpoints",
                "experiment_seed": args.seed,
                "retraining_window_sec": args.retraining_window_sec,
                "arrival_rate": args.arrival_rate if args.arrival_rate is not None else "",
            },
            "detector": detector_summary,
            "methods": method_rows,
            "overall": overall_rows,
            "lambda_detail": lambda_detail,
            "gamma_detail": gamma_detail,
            "pair_rows": pair_rows,
        },
    )
    write_summary(
        args.output_dir,
        detector_summary=detector_summary,
        method_rows=method_rows,
        overall_rows=overall_rows,
        lambda_detail=lambda_detail,
        gamma_detail=gamma_detail,
        pair_rows=pair_rows,
    )
    print((args.output_dir / "framework_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
