#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("@@WORKSPACE@@/drift_lora_project")
QWEN25_PYTHON = Path("@@QWEN25_PYTHON@@")
QWEN25_MODEL = Path("@@WORKSPACE@@/qwen_eval/models/Qwen2.5-VL-3B-Instruct")
QWEN35_PYTHON = Path("@@VLM_PYTHON@@")
QWEN35_MODEL = Path("@@WORKSPACE@@/qwen_eval/models/Qwen3.5-0.8B")
QWEN35_EVAL = Path("@@WORKSPACE@@/qwen_eval/code/candidate_vlm_eval.py")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize(value: Any) -> str:
    return str(value or "").strip().upper()


def latest_checkpoint(swift_output: Path, target_steps: int | None = None) -> Path | None:
    checkpoints = []
    for path in list(swift_output.glob("v*/checkpoint-*")) + list(swift_output.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            step = -1
        if target_steps is None or step >= target_steps:
            checkpoints.append((step, path.stat().st_mtime, path))
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][2]


def parse_train_runtime(swift_output: Path) -> float:
    runtime = 0.0
    for path in sorted(swift_output.glob("v*/logging.jsonl")):
        for row in read_jsonl(path):
            if "train_runtime" in row:
                runtime = float(row["train_runtime"])
    if runtime:
        return runtime
    states = sorted(swift_output.glob("v*/checkpoint-*/trainer_state.json"))
    for state_path in states:
        state = read_json(state_path)
        for row in state.get("log_history", []):
            if "train_runtime" in row:
                runtime = float(row["train_runtime"])
    return runtime


def run_command(cmd: list[str], *, env: dict[str, str], log_path: Path, cwd: Path = PROJECT_ROOT) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")
    return elapsed


def ewc_env(
    base: dict[str, str],
    args: argparse.Namespace,
    *,
    gpu_id: str,
    anchor: Path | None = None,
    fisher: Path | None = None,
) -> dict[str, str]:
    env = dict(base)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["CUBLAS_WORKSPACE_CONFIG"] = env.get("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    for key in (
        "EWC_MODE",
        "EWC_FISHER_STATE",
        "EWC_ANCHOR_MODE",
        "EWC_ANCHOR_ADAPTER",
        "EWC_LAMBDA",
        "EWC_BA_TARGET_MODULES",
        "EWC_NORMALIZE",
        "EWC_REDUCTION",
        "EWC_APPLY_HALF",
        "EWC_CACHE_TENSORS",
    ):
        env.pop(key, None)
    if fisher is None:
        env["EWC_MODE"] = "none"
    else:
        if args.ewc_anchor_mode == "adapter" and anchor is None:
            raise ValueError("EWC anchor mode 'adapter' requires a trained anchor adapter")
        env["EWC_MODE"] = args.ewc_mode
        env["EWC_FISHER_STATE"] = str(fisher)
        env["EWC_ANCHOR_MODE"] = args.ewc_anchor_mode
        if anchor is not None:
            env["EWC_ANCHOR_ADAPTER"] = str(anchor)
        env["EWC_LAMBDA"] = str(args.ewc_lambda)
        env["EWC_BA_TARGET_MODULES"] = args.ewc_ba_target_modules
        env["EWC_NORMALIZE"] = args.ewc_normalize
        env["EWC_REDUCTION"] = args.ewc_reduction
        env["EWC_APPLY_HALF"] = "1"
        env["EWC_CACHE_TENSORS"] = "1"
    return env


def common_ewc_script() -> Path:
    return PROJECT_ROOT / "scripts" / "run_swift_sft_ewc.py"


def fisher_script() -> Path:
    return PROJECT_ROOT / "scripts" / "compute_lora_ba_fisher.py"


def train_swift(
    *,
    python_bin: Path,
    model: Path,
    model_type: str,
    train_jsonl: Path,
    val_jsonl: Path,
    out_dir: Path,
    log_path: Path,
    gpu_id: str,
    args: argparse.Namespace,
    lora_rank: int,
    lora_alpha: int,
    target_modules: str,
    learning_rate: float,
    target_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    torch_dtype: str,
    anchor: Path | None = None,
    fisher: Path | None = None,
    max_pixels: int | None = None,
    qwen25_freeze_flags: bool = False,
) -> Path:
    ckpt = latest_checkpoint(out_dir, target_steps)
    if ckpt is not None and not args.force:
        return ckpt
    cmd = [
        str(python_bin),
        str(common_ewc_script()),
        "--model",
        str(model),
        "--model_type",
        model_type,
        "--dataset",
        str(train_jsonl),
        "--val_dataset",
        str(val_jsonl),
        "--output_dir",
        str(out_dir),
        "--tuner_type",
        "lora",
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        "--target_modules",
        target_modules,
        "--learning_rate",
        str(learning_rate),
        "--max_steps",
        str(target_steps),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--torch_dtype",
        torch_dtype,
        "--gradient_checkpointing",
        "true",
        "--eval_steps",
        str(min(80, target_steps)),
        "--save_steps",
        str(target_steps),
        "--save_total_limit",
        "1",
        "--logging_steps",
        "10",
        "--warmup_ratio",
        "0.05",
        "--lr_scheduler_type",
        "cosine",
        "--weight_decay",
        "0.01",
        "--dataloader_num_workers",
        "1",
        "--report_to",
        "none",
        "--seed",
        str(args.seed),
    ]
    if qwen25_freeze_flags:
        cmd.extend(["--freeze_llm", "false", "--freeze_vit", "true", "--freeze_aligner", "true"])
    if model_type == "qwen3_5":
        cmd.extend(["--bf16", "true"])
    if max_pixels is not None:
        cmd.extend(["--max_pixels", str(max_pixels)])
    if anchor is not None:
        cmd.extend(["--adapters", str(anchor)])
    env = ewc_env(os.environ.copy(), args, gpu_id=gpu_id, anchor=anchor if fisher is not None else None, fisher=fisher)
    if max_pixels is not None:
        env["MAX_PIXELS"] = str(max_pixels)
    package_root = str(PROJECT_ROOT / "remote_edit")
    env["PYTHONPATH"] = f"{package_root}:{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    run_command(cmd, env=env, log_path=log_path)
    ckpt = latest_checkpoint(out_dir, target_steps)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint after training under {out_dir}")
    return ckpt


def compute_ba_fisher(
    *,
    python_bin: Path,
    model: Path,
    model_type: str,
    train_jsonl: Path,
    out_file: Path,
    work_dir: Path,
    log_path: Path,
    gpu_id: str,
    args: argparse.Namespace,
    adapter: Path | None,
    lora_rank: int,
    lora_alpha: int,
    target_modules: str,
    torch_dtype: str,
    per_device_train_batch_size: int,
    max_pixels: int | None = None,
    qwen25_freeze_flags: bool = False,
) -> tuple[Path, float]:
    runtime_path = out_file.with_suffix(".runtime.json")
    if out_file.exists() and out_file.stat().st_size > 0 and not args.force:
        runtime = 0.0
        if runtime_path.is_file():
            runtime = float(read_json(runtime_path).get("wall_runtime_s") or 0.0)
        return out_file, runtime
    cmd = [
        str(python_bin),
        str(fisher_script()),
        "--fisher-output",
        str(out_file),
        "--fisher-dtype",
        args.fisher_dtype,
        "--fisher-accumulation",
        args.fisher_accumulation,
        "--fisher-max-batches",
        str(args.fisher_max_batches),
        "--ba-target-modules",
        args.ewc_ba_target_modules,
        "--model",
        str(model),
        "--model_type",
        model_type,
        "--dataset",
        str(train_jsonl),
        "--output_dir",
        str(work_dir),
        "--tuner_type",
        "lora",
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        "--target_modules",
        target_modules,
        "--learning_rate",
        "1e-5",
        "--max_steps",
        "1",
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--gradient_accumulation_steps",
        "1",
        "--torch_dtype",
        torch_dtype,
        "--gradient_checkpointing",
        "true",
        "--eval_steps",
        "999999",
        "--save_steps",
        "999999",
        "--logging_steps",
        "10",
        "--report_to",
        "none",
    ]
    if adapter is not None:
        cmd.extend(["--adapters", str(adapter)])
    if qwen25_freeze_flags:
        cmd.extend(["--freeze_llm", "false", "--freeze_vit", "true", "--freeze_aligner", "true"])
    if model_type == "qwen3_5":
        cmd.extend(["--bf16", "true"])
    if max_pixels is not None:
        cmd.extend(["--max_pixels", str(max_pixels)])
    env = ewc_env(os.environ.copy(), args, gpu_id=gpu_id)
    if max_pixels is not None:
        env["MAX_PIXELS"] = str(max_pixels)
    package_root = str(PROJECT_ROOT / "remote_edit")
    env["PYTHONPATH"] = f"{package_root}:{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    elapsed = run_command(cmd, env=env, log_path=log_path)
    write_json(
        runtime_path,
        {
            "fisher_path": str(out_file),
            "wall_runtime_s": elapsed,
            "adapter_path": str(adapter) if adapter is not None else "",
            "anchor_mode": args.ewc_anchor_mode,
            "optimizer_steps": 0,
        },
    )
    return out_file, elapsed


def selected_static_rows(csv_path: Path, detection_csv: Path | None = None) -> list[dict[str, str]]:
    rows = [dict(row) for row in read_csv(csv_path) if row.get("method") == "StaticSplitContinuous"]
    if not rows:
        raise ValueError(f"No StaticSplitContinuous rows in {csv_path}")
    detection_by_group: dict[str, dict[str, str]] = {}
    if detection_csv is not None:
        detection_by_group = {row["group_id"]: row for row in read_csv(detection_csv)}
    for row in rows:
        detection = detection_by_group.get(row["group_id"])
        if detection is not None:
            row["triggered"] = detection.get("triggered", row.get("triggered", ""))
            row["trigger_position"] = detection.get("trigger_position", row.get("trigger_position", ""))
        row["method"] = "EWC"
        row["notes"] = (
            "student_side_ewc_fixed_static_split;"
            + ("formal_detection_window_reused" if detection is not None else "source_trigger_fields_reused")
        )
    return rows


def write_task_filtered_jsonl(source: Path, destination: Path, task_ids: set[int], *, force: bool) -> int:
    if destination.is_file() and not force:
        return len(read_jsonl(destination))
    rows = [row for row in read_jsonl(source) if int(row.get("task_id", -1)) in task_ids]
    if not rows:
        raise ValueError(f"No rows with task_ids={sorted(task_ids)} in {source}")
    write_jsonl(destination, rows)
    return len(rows)


def qwen25_batch_size(group_id: str, args: argparse.Namespace) -> int:
    if group_id.endswith("_B4") or "_B4" in group_id:
        return int(args.qwen25_b4_batch_size)
    return int(args.qwen25_b1_batch_size if group_id.endswith("_B1") or "_B1" in group_id else args.qwen25_other_batch_size)


def qwen25_gpu_for_group(group_id: str, gpu_ids: list[str], index: int, args: argparse.Namespace) -> str:
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")
    if group_id.endswith("_B4") or "_B4" in group_id:
        return args.qwen25_b4_gpu_ids
    if group_id.endswith("_B1") or "_B1" in group_id:
        if "2" in gpu_ids:
            return "2"
        return gpu_ids[index % len(gpu_ids)]
    roomy = [gpu for gpu in gpu_ids if gpu != "2"]
    candidates = roomy or gpu_ids
    return candidates[index % len(candidates)]


def write_true_eval_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_json(out_dir / "scheduler_true_accuracy_error_summary.json", {"rows": 0})
        return
    observed = [float(row.get("observed_current_accuracy") or 0.0) for row in rows]
    estimated = [float(row.get("estimated_current_accuracy") or 0.0) for row in rows]
    errors = [obs - est for obs, est in zip(observed, estimated)]
    summary = {
        "rows": len(rows),
        "mean_observed_current_accuracy": sum(observed) / len(observed),
        "mean_estimated_current_accuracy": sum(estimated) / len(estimated),
        "mean_current_error": sum(errors) / len(errors),
        "mean_abs_current_error": sum(abs(value) for value in errors) / len(errors),
        "current_rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
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


def run_qwen25(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "remote_edit"))
    train_mod = load_module(
        PROJECT_ROOT / "remote_edit" / "remote_sensing_adaptation" / "train_v2_selected_fulltarget_adapters.py",
        "train_v2_selected_fulltarget_adapters_ewc",
    )
    profile_mod = load_module(
        PROJECT_ROOT / "remote_edit" / "remote_sensing_adaptation" / "profile_v2_qwen_lora_gamma_fixed.py",
        "profile_v2_qwen_lora_gamma_fixed_ewc",
    )
    out_root = args.qwen25_out
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        out_root / "protocol_manifest.json",
        {
            "family": "qwen25",
            "seed": args.seed,
            "timeline": ["detection_window", "fisher_without_optimizer_step", "task_b_retraining"],
            "ewc_anchor_mode": args.ewc_anchor_mode,
            "task_b_only_training": True,
            "fisher_time_counted": True,
            "trained_anchor_stage": args.ewc_anchor_mode == "adapter",
            "gpu_ids": args.gpu_ids,
        },
    )
    gamma_rows = read_json(args.qwen25_gamma_json)
    gamma_by_key = {(str(row.get("group_id") or ""), str(row["gamma_id"])): row for row in gamma_rows}
    gpu_ids = [part.strip() for part in args.gpu_ids.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must not be empty")
    plan_rows: list[dict[str, Any]] = []
    schedule_specs = (
        ("ar0p1", args.qwen25_schedule_ar0p1, args.qwen25_detection_schedule_ar0p1),
        ("ar0p05", args.qwen25_schedule_ar0p05, args.qwen25_detection_schedule_ar0p05),
    )
    for tag, schedule_csv, detection_csv in schedule_specs:
        rows = selected_static_rows(schedule_csv, detection_csv)
        schedule_out = out_root / "schedules" / tag / "ewc_scheduler_end_to_end_results.csv"
        write_csv(schedule_out, rows)
    base_rows = selected_static_rows(args.qwen25_schedule_ar0p1, args.qwen25_detection_schedule_ar0p1)
    if args.qwen25_only_groups:
        only_groups = {part.strip() for part in args.qwen25_only_groups.split(",") if part.strip()}
        base_rows = [row for row in base_rows if row.get("group_id") in only_groups]
        if not base_rows:
            raise ValueError(f"No Qwen2.5 rows matched --qwen25-only-groups={args.qwen25_only_groups}")

    def train_worker(idx: int, row: dict[str, str]) -> dict[str, Any]:
        group_id = row["group_id"]
        gamma_id = row["gamma_id"]
        gamma = gamma_by_key[(group_id, gamma_id)]
        config = train_mod.fulltarget_config(gamma)
        fixed = out_root / "prepared" / group_id
        reference_root = fixed / "reference_task_a"
        drift_root = fixed / "drift"
        old_data = args.qwen25_pseudo_root / "groups" / group_id / "D_i_A" / "data.json"
        train_data = args.qwen25_pseudo_root / "groups" / group_id / "R_train" / "data.json"
        val_data = args.qwen25_pseudo_root / "groups" / group_id / "R_val" / "data.json"
        old_prepared = profile_mod.prepare_fixed_root(
            train_data=old_data,
            val_data=old_data,
            out_dir=reference_root,
            task_ids=profile_mod.parse_task_ids(args.qwen25_task_ids),
            pixel_percentile=95,
            seed=args.seed,
            force=args.force,
        )
        drift_prepared = profile_mod.prepare_fixed_root(
            train_data=train_data,
            val_data=val_data,
            out_dir=drift_root,
            task_ids=profile_mod.parse_task_ids(args.qwen25_task_ids),
            pixel_percentile=95,
            seed=args.seed,
            force=args.force,
        )
        max_pixels = int(read_json(old_prepared / "experiment_metadata.json")["recommended_max_pixels"])
        target_steps = int(args.target_steps_override or config["target_train_steps"])
        physical_batch_size = qwen25_batch_size(group_id, args)
        gpu_id = qwen25_gpu_for_group(group_id, gpu_ids, idx, args)
        anchor_ckpt: Path | None = None
        if args.ewc_anchor_mode == "adapter":
            anchor_ckpt = train_swift(
                python_bin=args.qwen25_python,
                model=args.qwen25_model,
                model_type="qwen2_5_vl",
                train_jsonl=old_prepared / "swift_train.jsonl",
                val_jsonl=old_prepared / "swift_val.jsonl",
                out_dir=out_root / "anchors" / group_id / gamma_id / "swift_output",
                log_path=out_root / "logs" / f"qwen25_anchor_{group_id}.log",
                gpu_id=gpu_id,
                args=args,
                lora_rank=int(config["lora_rank"]),
                lora_alpha=int(config["lora_alpha"]),
                target_modules=str(config["target_modules"]),
                learning_rate=float(config["learning_rate"]),
                target_steps=target_steps,
                per_device_train_batch_size=physical_batch_size,
                gradient_accumulation_steps=int(config["grad_accum"]),
                torch_dtype="bfloat16",
                max_pixels=max_pixels,
                qwen25_freeze_flags=True,
            )
        fisher, fisher_runtime = compute_ba_fisher(
            python_bin=args.qwen25_python,
            model=args.qwen25_model,
            model_type="qwen2_5_vl",
            train_jsonl=old_prepared / "swift_train.jsonl",
            out_file=out_root / "fisher" / group_id / gamma_id / "fisher_ba_kv.safetensors",
            work_dir=out_root / "fisher_work" / group_id / gamma_id,
            log_path=out_root / "logs" / f"qwen25_fisher_{group_id}.log",
            gpu_id=gpu_id,
            args=args,
            adapter=anchor_ckpt,
            lora_rank=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            target_modules=str(config["target_modules"]),
            torch_dtype="bfloat16",
            per_device_train_batch_size=physical_batch_size,
            max_pixels=max_pixels,
            qwen25_freeze_flags=True,
        )
        ewc_swift = out_root / "fulltarget_adapters" / group_id / gamma_id / "swift_output"
        ewc_ckpt = train_swift(
            python_bin=args.qwen25_python,
            model=args.qwen25_model,
            model_type="qwen2_5_vl",
            train_jsonl=drift_prepared / "swift_train.jsonl",
            val_jsonl=drift_prepared / "swift_val.jsonl",
            out_dir=ewc_swift,
            log_path=out_root / "logs" / f"qwen25_ewc_train_{group_id}.log",
            gpu_id=gpu_id,
            args=args,
            lora_rank=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            target_modules=str(config["target_modules"]),
            learning_rate=float(config["learning_rate"]),
            target_steps=target_steps,
            per_device_train_batch_size=physical_batch_size,
            gradient_accumulation_steps=int(config["grad_accum"]),
            torch_dtype="bfloat16",
            anchor=anchor_ckpt,
            fisher=fisher,
            max_pixels=max_pixels,
            qwen25_freeze_flags=True,
        )
        runtime = parse_train_runtime(ewc_swift)
        total_runtime = fisher_runtime + runtime
        return {
            "family": "qwen25",
            "group_id": group_id,
            "gamma_id": gamma_id,
            "lambda_id": row["lambda_id"],
            "triggered": row.get("triggered", ""),
            "trigger_position": row.get("trigger_position", ""),
            "anchor_mode": args.ewc_anchor_mode,
            "anchor_adapter": str(anchor_ckpt) if anchor_ckpt is not None else "",
            "anchor_state": "trained_adapter" if anchor_ckpt is not None else "initial_zero_BA",
            "fisher": str(fisher),
            "fisher_runtime_s": fisher_runtime,
            "ewc_adapter": str(ewc_ckpt),
            "ewc_train_runtime_s": runtime,
            "fisher_plus_train_runtime_s": total_runtime,
            "qwen25_physical_batch_size": physical_batch_size,
            "gpu_id": gpu_id,
        }

    max_workers = max(1, min(args.max_workers, len(gpu_ids), len(base_rows)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_worker, idx, row): row for idx, row in enumerate(base_rows)}
        for future in as_completed(futures):
            plan_rows.append(future.result())
            plan_rows.sort(key=lambda item: str(item["group_id"]))
            write_csv(out_root / "ewc_training_plan_and_results.partial.csv", plan_rows)
    write_csv(out_root / "ewc_training_plan_and_results.csv", plan_rows)
    if args.qwen25_train_only:
        return

    plan_by_pair = {(str(row["group_id"]), str(row["gamma_id"])): row for row in plan_rows}
    for tag in ("ar0p1", "ar0p05"):
        schedule_path = out_root / "schedules" / tag / "ewc_scheduler_end_to_end_results.csv"
        updated = []
        for schedule_row in read_csv(schedule_path):
            key = (schedule_row["group_id"], schedule_row["gamma_id"])
            plan = plan_by_pair.get(key)
            if plan is not None:
                runtime = float(plan["fisher_plus_train_runtime_s"] or 0.0)
                r_share = float(schedule_row.get("R") or 1.0)
                ready_time = runtime / max(r_share, 1e-9)
                schedule_row["adapter_ready_time_s"] = ready_time
                schedule_row["train_time_s_effective"] = ready_time
                schedule_row["notes"] = (
                    "student_side_ewc_detection_then_task_b;"
                    f"anchor_mode={plan['anchor_mode']};anchor={plan['anchor_adapter']};"
                    f"fisher={plan['fisher']};fisher_time_counted=true;"
                    f"batch={plan['qwen25_physical_batch_size']};gpu={plan['gpu_id']}"
                )
            updated.append(schedule_row)
        write_csv(schedule_path, updated)

    for tag, arrival in (("ar0p1", 0.1), ("ar0p05", 0.05)):
        out_dir = out_root / "true_accuracy_full_r" / tag
        if (out_dir / "scheduler_true_accuracy_error_rows.csv").exists() and not args.force:
            continue
        schedule_rows = read_csv(out_root / "schedules" / tag / "ewc_scheduler_end_to_end_results.csv")

        def eval_worker(idx: int, schedule_row: dict[str, str]) -> list[dict[str, str]]:
            group_id = schedule_row["group_id"]
            gpu_id = qwen25_gpu_for_group(group_id, gpu_ids, idx, args)
            shard_schedule = out_root / "schedules" / tag / "shards" / f"{group_id}.csv"
            shard_out = out_dir / "shards" / group_id
            write_csv(shard_schedule, [schedule_row])
            cmd = [
                str(args.qwen25_python),
                "-m",
                "remote_sensing_adaptation.evaluate_v2_scheduler_true_accuracy",
                "--scheduler-csv",
                str(shard_schedule),
                "--lambda-profiles",
                str(args.qwen25_lambda_profiles),
                "--group-data-dir",
                str(args.qwen25_true_root),
                "--fulltrain-root",
                str(out_root / "fulltarget_adapters"),
                "--out-dir",
                str(shard_out),
                "--model-path",
                str(args.qwen25_model),
                "--cuda-visible-devices",
                gpu_id,
                "--task-ids",
                args.qwen25_task_ids,
                "--method",
                "EWC",
                "--eval-split",
                "R_all",
                "--arrival-rate",
                str(arrival),
                "--default-max-new-tokens",
                "16",
                "--default-dtype",
                "bfloat16",
                "--strict-eval-batch-size",
                str(qwen25_batch_size(group_id, args)),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONPATH"] = f"{PROJECT_ROOT / 'remote_edit'}:{env.get('PYTHONPATH', '')}"
            run_command(cmd, env=env, log_path=out_root / "logs" / f"qwen25_eval_{tag}_{group_id}.log")
            return read_csv(shard_out / "scheduler_true_accuracy_error_rows.csv")

        combined_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(eval_worker, idx, row): row for idx, row in enumerate(schedule_rows)}
            for future in as_completed(futures):
                combined_rows.extend(future.result())
                combined_rows.sort(key=lambda item: str(item.get("group_id", "")))
                write_csv(out_dir / "scheduler_true_accuracy_error_rows.partial.csv", combined_rows)
        combined_rows.sort(key=lambda item: str(item.get("group_id", "")))
        write_csv(out_dir / "scheduler_true_accuracy_error_rows.csv", combined_rows)
        write_true_eval_summary(out_dir, combined_rows)


def qwen35_swift_row(split_mod: Any, row: dict[str, Any], split_root: Path) -> dict[str, Any] | None:
    converted, _missing = split_mod.swift_row(row, split_root)
    return converted


def write_qwen35_swift(path: Path, rows: list[dict[str, Any]], split_mod: Any, split_root: Path) -> None:
    swift_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        converted, missing = split_mod.swift_row(row, split_root)
        if converted is None:
            skipped.append({"uid": row.get("uid", ""), "missing_images": missing})
        else:
            swift_rows.append(converted)
    write_jsonl(path / "data.jsonl", swift_rows)
    write_jsonl(path / "skipped_missing_images.jsonl", skipped)


def prediction_map(predictions_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_jsonl(predictions_path):
        uid = str(row.get("uid", ""))
        if uid and bool(row.get("accepted")):
            out[uid] = normalize(row.get("pseudo_label"))
    return out


def score_rows(rows: list[dict[str, Any]], preds: dict[str, str]) -> dict[str, Any]:
    correct = 0
    seen = 0
    for row in rows:
        uid = str(row["uid"])
        if uid not in preds:
            continue
        seen += 1
        correct += int(preds[uid] == normalize(row.get("ground_truth") or row.get("gt")))
    return {"seen": seen, "correct": correct, "accuracy": correct / seen if seen else 0.0}


def mixed_accuracy(rows: list[dict[str, Any]], base: dict[str, str], adapter: dict[str, str], ready_index: int) -> float:
    correct = 0
    for idx, row in enumerate(rows):
        preds = base if idx < ready_index else adapter
        correct += int(preds.get(str(row["uid"]), "") == normalize(row.get("ground_truth") or row.get("gt")))
    return correct / len(rows) if rows else 0.0


def eval_qwen35_adapter(args: argparse.Namespace, *, dataset_dir: Path, adapter_path: Path, out_dir: Path, gpu_id: str, lambda_id: str) -> Path:
    raw_path = out_dir / "raw_predictions.jsonl"
    if raw_path.exists() and not args.force:
        return raw_path
    cmd = [
        str(args.qwen35_python),
        str(QWEN35_EVAL),
        "--dataset-dir",
        str(dataset_dir),
        "--model-path",
        str(args.qwen35_model),
        "--adapter-path",
        str(adapter_path),
        "--model-name",
        f"Qwen3.5-0.8B-EWC-{lambda_id}",
        "--family",
        "hf_image_text",
        "--out-dir",
        str(out_dir),
        "--task-ids",
        args.qwen35_task_ids,
        "--dtype",
        "bfloat16",
        "--batch-size",
        "1",
        "--enable-thinking",
        "false",
        "--answer-source",
        "raw",
        "--mcq-max-new-tokens",
        "8",
        "--multi-mcq-max-new-tokens",
        "16",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    run_command(cmd, env=env, log_path=out_dir / "eval.log")
    return raw_path


def run_qwen35(args: argparse.Namespace) -> None:
    split_mod = load_module(PROJECT_ROOT / "scripts" / "build_qwen35_omniearth_formal_splits.py", "qwen35_splits_for_ewc")
    out_root = args.qwen35_out
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        out_root / "protocol_manifest.json",
        {
            "family": "qwen35",
            "seed": args.seed,
            "timeline": ["detection_window", "fisher_without_optimizer_step", "task_b_retraining"],
            "ewc_anchor_mode": args.ewc_anchor_mode,
            "task_b_only_training": args.qwen35_target_task_only,
            "per_device_train_batch_size": args.qwen35_batch_size,
            "gradient_accumulation_steps": args.qwen35_gradient_accumulation_steps,
            "effective_train_batch_size": (
                args.qwen35_batch_size * args.qwen35_gradient_accumulation_steps
            ),
            "fisher_time_counted": True,
            "trained_anchor_stage": args.ewc_anchor_mode == "adapter",
            "gpu_ids": args.gpu_ids,
            "continue_on_error": args.continue_on_error,
        },
    )
    static_rows = selected_static_rows(args.qwen35_scheduler_csv)
    fixed_rows = [row for row in static_rows if row.get("gamma_id") == args.qwen35_fixed_gamma]
    if not fixed_rows:
        raise ValueError(f"No fixed gamma rows found for {args.qwen35_fixed_gamma}")
    if args.qwen35_only_groups:
        only_groups = {part.strip() for part in args.qwen35_only_groups.split(",") if part.strip()}
        fixed_rows = [row for row in fixed_rows if row.get("group_id") in only_groups]
        if not fixed_rows:
            raise ValueError(f"No Qwen3.5 rows matched --qwen35-only-groups={args.qwen35_only_groups}")
    scheduler_path = out_root / "scheduler" / "ewc_scheduler_end_to_end_results.csv"

    def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(row.get("group_id") or ""),
            str(row.get("gamma_id") or ""),
            str(row.get("lambda_id") or ""),
        )

    scheduler_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if scheduler_path.exists() and not args.force:
        scheduler_by_key.update({row_key(row): row for row in read_csv(scheduler_path)})
    scheduler_by_key.update({row_key(row): row for row in fixed_rows})
    write_csv(
        scheduler_path,
        sorted(scheduler_by_key.values(), key=lambda row: row_key(row)),
    )
    gamma_rows = read_csv(args.qwen35_gamma_csv)
    gamma_by_key = {(row["group_id"], row["gamma_id"]): row for row in gamma_rows}
    eval_cache = read_csv(args.qwen35_framework_root / "eval_cache" / "r_window_eval_cache.csv")
    base_eval = {(row["group_id"], row["lambda_id"]): row for row in eval_cache if row["kind"] == "base"}
    gpu_ids = [part.strip() for part in args.gpu_ids.split(",") if part.strip()]
    reference_fit = read_json(args.qwen35_split_root / "part0_reference" / "reference_fit" / "data.json")
    threshold_cal = read_json(args.qwen35_split_root / "part0_reference" / "threshold_cal" / "data.json")
    result_csv = out_root / "qwen35_ewc_actual_execution.csv"
    partial_csv = out_root / "qwen35_ewc_actual_execution.partial.csv"
    result_json = out_root / "qwen35_ewc_actual_execution.json"
    failure_json = out_root / "qwen35_ewc_failures.json"
    result_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not args.force:
        for existing_path in (result_csv, partial_csv):
            if existing_path.exists():
                result_by_key.update(
                    {row_key(row): row for row in read_csv(existing_path)}
                )
    failure_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if failure_json.exists() and not args.force:
        failure_by_key.update(
            {row_key(row): row for row in read_json(failure_json)}
        )
    results: list[dict[str, Any]] = sorted(
        result_by_key.values(),
        key=lambda row: row_key(row),
    )

    def worker(index: int, row: dict[str, str]) -> dict[str, Any]:
        group_id = row["group_id"]
        gamma_id = row["gamma_id"]
        lambda_id = row["lambda_id"]
        gamma = gamma_by_key[(group_id, gamma_id)]
        gpu_id = gpu_ids[index % len(gpu_ids)]
        group_split = args.qwen35_split_root / "groups" / group_id / "part3_retraining"
        detection_split = args.qwen35_split_root / "groups" / group_id / "part2_detection"
        if args.ewc_anchor_mode == "initial_zero":
            old_train_rows = reference_fit + read_json(detection_split / "D_no_drift" / "data.json")
        else:
            old_train_rows = reference_fit + read_json(group_split / "R_no_drift" / "data.json")
        old_val_rows = threshold_cal
        old_dir = out_root / "prepared" / group_id / "old_reference"
        if args.force or not (old_dir / "train" / "data.jsonl").exists():
            write_qwen35_swift(old_dir / "train", old_train_rows, split_mod, args.qwen35_split_root)
            write_qwen35_swift(old_dir / "val", old_val_rows, split_mod, args.qwen35_split_root)
        target_train_jsonl = args.qwen35_pseudo_root / "groups" / group_id / "R_train" / "swift_pseudo" / "data.jsonl"
        target_val_jsonl = args.qwen35_pseudo_root / "groups" / group_id / "R_val" / "swift_pseudo" / "data.jsonl"
        drift_task_ids = {
            int(item["task_id"]) for item in read_json(group_split / "R_drift" / "data.json") if item.get("task_id") is not None
        }
        if len(drift_task_ids) != 1:
            raise ValueError(f"Expected exactly one drift task for {group_id}, got {sorted(drift_task_ids)}")
        target_train_count = len(read_jsonl(target_train_jsonl))
        target_val_count = len(read_jsonl(target_val_jsonl))
        if args.qwen35_target_task_only:
            target_dir = out_root / "prepared" / group_id / "target_task_b_only"
            target_train_jsonl = target_dir / "train.jsonl"
            target_val_jsonl = target_dir / "val.jsonl"
            target_train_count = write_task_filtered_jsonl(
                args.qwen35_pseudo_root / "groups" / group_id / "R_train" / "swift_pseudo" / "data.jsonl",
                target_train_jsonl,
                drift_task_ids,
                force=args.force,
            )
            target_val_count = write_task_filtered_jsonl(
                args.qwen35_pseudo_root / "groups" / group_id / "R_val" / "swift_pseudo" / "data.jsonl",
                target_val_jsonl,
                drift_task_ids,
                force=args.force,
            )
            write_json(
                target_dir / "manifest.json",
                {
                    "group_id": group_id,
                    "task_ids": sorted(drift_task_ids),
                    "train_rows": target_train_count,
                    "val_rows": target_val_count,
                    "source": "existing teacher-pseudo R splits filtered to task B",
                },
            )
        target_steps = int(args.target_steps_override or float(gamma["target_train_steps"]))
        lora_rank = int(float(gamma["lora_rank"]))
        lora_alpha = int(float(gamma["lora_alpha"]))
        per_device_train_batch_size = args.qwen35_batch_size
        grad_accum = args.qwen35_gradient_accumulation_steps
        anchor_ckpt: Path | None = None
        if args.ewc_anchor_mode == "adapter":
            anchor_ckpt = train_swift(
                python_bin=args.qwen35_python,
                model=args.qwen35_model,
                model_type="qwen3_5",
                train_jsonl=old_dir / "train" / "data.jsonl",
                val_jsonl=old_dir / "val" / "data.jsonl",
                out_dir=out_root / "anchors" / group_id / gamma_id / "swift_output",
                log_path=out_root / "logs" / f"qwen35_anchor_{group_id}.log",
                gpu_id=gpu_id,
                args=args,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=gamma["target_modules"],
                learning_rate=float(gamma["learning_rate"]),
                target_steps=target_steps,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=grad_accum,
                torch_dtype="bfloat16",
                max_pixels=1003520,
            )
        fisher, fisher_runtime = compute_ba_fisher(
            python_bin=args.qwen35_python,
            model=args.qwen35_model,
            model_type="qwen3_5",
            train_jsonl=old_dir / "train" / "data.jsonl",
            out_file=out_root / "fisher" / group_id / gamma_id / "fisher_ba_kv.safetensors",
            work_dir=out_root / "fisher_work" / group_id / gamma_id,
            log_path=out_root / "logs" / f"qwen35_fisher_{group_id}.log",
            gpu_id=gpu_id,
            args=args,
            adapter=anchor_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=gamma["target_modules"],
            torch_dtype="bfloat16",
            per_device_train_batch_size=1,
            max_pixels=1003520,
        )
        ewc_swift = out_root / "runs" / group_id / gamma_id / "swift_output"
        ewc_ckpt = train_swift(
            python_bin=args.qwen35_python,
            model=args.qwen35_model,
            model_type="qwen3_5",
            train_jsonl=target_train_jsonl,
            val_jsonl=target_val_jsonl,
            out_dir=ewc_swift,
            log_path=out_root / "logs" / f"qwen35_ewc_train_{group_id}.log",
            gpu_id=gpu_id,
            args=args,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=gamma["target_modules"],
            learning_rate=float(gamma["learning_rate"]),
            target_steps=target_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=grad_accum,
            torch_dtype="bfloat16",
            anchor=anchor_ckpt,
            fisher=fisher,
            max_pixels=1003520,
        )
        runtime = parse_train_runtime(ewc_swift)
        fisher_plus_train_runtime = fisher_runtime + runtime
        if args.qwen35_train_only:
            return {
                "group_id": group_id,
                "model_id": "Qwen3.5-0.8B",
                "method": "EWC",
                "lambda_id": lambda_id,
                "gamma_id": gamma_id,
                "I": row["I"],
                "R": row["R"],
                "target_train_steps": target_steps,
                "target_task_ids": ",".join(str(value) for value in sorted(drift_task_ids)),
                "target_train_rows": target_train_count,
                "target_val_rows": target_val_count,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": grad_accum,
                "effective_train_batch_size": per_device_train_batch_size * grad_accum,
                "ewc_lambda": args.ewc_lambda,
                "ewc_mode": args.ewc_mode,
                "ewc_anchor_mode": args.ewc_anchor_mode,
                "fisher_runtime_s": fisher_runtime,
                "actual_target_train_runtime_s": runtime,
                "fisher_plus_train_runtime_s": fisher_plus_train_runtime,
                "anchor_adapter_path": str(anchor_ckpt) if anchor_ckpt is not None else "",
                "anchor_state": "trained_adapter" if anchor_ckpt is not None else "initial_zero_BA",
                "fisher_path": str(fisher),
                "target_adapter_path": str(ewc_ckpt),
            }
        dataset_dir = args.qwen35_framework_root / "eval_cache" / "datasets" / group_id / "R_all" / lambda_id
        raw_path = eval_qwen35_adapter(
            args,
            dataset_dir=dataset_dir,
            adapter_path=ewc_ckpt,
            out_dir=out_root / "eval" / group_id / lambda_id / gamma_id,
            gpu_id=gpu_id,
            lambda_id=lambda_id,
        )
        r_all = read_json(group_split / "R_all" / "data.json")
        r_test = read_json(group_split / "R_test" / "data.json")
        base_preds = prediction_map(Path(base_eval[(group_id, lambda_id)]["raw_predictions"]))
        adapter_preds = prediction_map(raw_path)
        r_share = float(row.get("R") or 1.0)
        finish_time = fisher_plus_train_runtime / max(r_share, 1e-9)
        ready_index = min(len(r_all), math.ceil(min(1.0, finish_time / args.qwen35_retraining_window_sec) * len(r_all)))
        current = mixed_accuracy(r_all, base_preds, adapter_preds, ready_index)
        before = score_rows(r_test, base_preds)["accuracy"]
        after = score_rows(r_test, adapter_preds)["accuracy"]
        weights = [1.0] + [args.discount_factor**idx for idx in range(1, args.future_windows + 1)]
        actual_long = (weights[0] * current + sum(weight * after for weight in weights[1:])) / sum(weights)
        return {
            "group_id": group_id,
            "model_id": "Qwen3.5-0.8B",
            "method": "EWC",
            "lambda_id": lambda_id,
            "gamma_id": gamma_id,
            "I": row["I"],
            "R": row["R"],
            "target_train_steps": target_steps,
            "target_task_ids": ",".join(str(value) for value in sorted(drift_task_ids)),
            "target_train_rows": target_train_count,
            "target_val_rows": target_val_count,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_train_batch_size": per_device_train_batch_size * grad_accum,
            "ewc_lambda": args.ewc_lambda,
            "ewc_mode": args.ewc_mode,
            "ewc_anchor_mode": args.ewc_anchor_mode,
            "fisher_runtime_s": fisher_runtime,
            "actual_target_train_runtime_s": runtime,
            "fisher_plus_train_runtime_s": fisher_plus_train_runtime,
            "adapter_ready_time_s": finish_time,
            "adapter_ready_sample_count": ready_index,
            "rall_current_time_weighted_acc": current,
            "rtest_base_acc_before_adapter": before,
            "rtest_target_adapter_acc_after_adapter": after,
            "actual_long_avg_accuracy_target_adapter": actual_long,
            "anchor_adapter_path": str(anchor_ckpt) if anchor_ckpt is not None else "",
            "anchor_state": "trained_adapter" if anchor_ckpt is not None else "initial_zero_BA",
            "fisher_path": str(fisher),
            "target_adapter_path": str(ewc_ckpt),
            "target_adapter_raw_predictions": str(raw_path),
        }

    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(gpu_ids)))) as executor:
        futures = {executor.submit(worker, idx, row): row for idx, row in enumerate(fixed_rows)}
        for future in as_completed(futures):
            source_row = futures[future]
            key = row_key(source_row)
            try:
                result = future.result()
            except Exception as exc:
                failure_by_key[key] = {
                    "group_id": source_row.get("group_id", ""),
                    "gamma_id": source_row.get("gamma_id", ""),
                    "lambda_id": source_row.get("lambda_id", ""),
                    "seed": args.seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                write_json(
                    failure_json,
                    sorted(failure_by_key.values(), key=lambda row: row_key(row)),
                )
                if not args.continue_on_error:
                    raise
                print(
                    "[qwen35] continuing after failure for "
                    f"{source_row.get('group_id', '')}: {exc}",
                    flush=True,
                )
                continue
            result_by_key[key] = result
            failure_by_key.pop(key, None)
            results = sorted(result_by_key.values(), key=lambda row: row_key(row))
            write_csv(partial_csv, results)
            write_json(
                failure_json,
                sorted(failure_by_key.values(), key=lambda row: row_key(row)),
            )
    results = sorted(result_by_key.values(), key=lambda row: row_key(row))
    write_csv(result_csv, results)
    write_json(result_json, results)
    write_json(
        failure_json,
        sorted(failure_by_key.values(), key=lambda row: row_key(row)),
    )
    if args.qwen35_train_only or not results:
        return
    total = sum(len(read_json(args.qwen35_split_root / "groups" / row["group_id"] / "part3_retraining" / "R_all" / "data.json")) for row in results)
    overall = {}
    for key in ["rall_current_time_weighted_acc", "rtest_base_acc_before_adapter", "rtest_target_adapter_acc_after_adapter", "actual_long_avg_accuracy_target_adapter"]:
        overall[key] = sum(
            float(row[key]) * len(read_json(args.qwen35_split_root / "groups" / row["group_id"] / "part3_retraining" / "R_all" / "data.json"))
            for row in results
        ) / total
    write_json(out_root / "qwen35_ewc_actual_execution_overall.json", overall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run student-side EWC baselines for final Qwen2.5/Qwen3.5 experiments.")
    parser.add_argument("--families", default="qwen25,qwen35")
    parser.add_argument("--gpu-ids", default="4")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-steps-override", type=int, default=0)
    parser.add_argument("--ewc-mode", default="ba_kv")
    parser.add_argument("--ewc-anchor-mode", choices=["adapter", "initial_zero"], default="adapter")
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--ewc-ba-target-modules", default="k_proj,v_proj")
    parser.add_argument("--ewc-normalize", default="mean")
    parser.add_argument("--ewc-reduction", default="mean")
    parser.add_argument("--fisher-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--fisher-accumulation", default="gpu_local", choices=["cpu", "gpu_local"])
    parser.add_argument("--fisher-max-batches", type=int, default=0)

    q25_root = PROJECT_ROOT / "results" / "v2_teacher_closed_loop_20260630_181348"
    q25_step5 = q25_root / "step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924"
    parser.add_argument("--qwen25-out", type=Path, default=q25_root / "step6_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924" / "ewc_student_baseline_20260708")
    parser.add_argument("--qwen25-python", type=Path, default=QWEN25_PYTHON)
    parser.add_argument("--qwen25-model", type=Path, default=QWEN25_MODEL)
    parser.add_argument("--qwen25-pseudo-root", type=Path, default=q25_step5 / "inputs" / "teacher_pseudo_5group_rwindow_pseudoroot")
    parser.add_argument("--qwen25-true-root", type=Path, default=q25_step5 / "inputs" / "true_5group_rwindow_root")
    parser.add_argument("--qwen25-lambda-profiles", type=Path, default=q25_step5 / "lambda_analysis_rval" / "lambda_profiles_group_v2.json")
    parser.add_argument("--qwen25-gamma-json", type=Path, default=q25_step5 / "gamma_rwindow_pseudo" / "gamma_group_profiles.json")
    parser.add_argument("--qwen25-schedule-ar0p1", type=Path, default=q25_step5 / "formal_schedules_ar0p1" / "StaticSplitContinuous" / "scheduler_end_to_end_results.csv")
    parser.add_argument("--qwen25-schedule-ar0p05", type=Path, default=q25_step5 / "formal_schedules_ar0p05" / "StaticSplitContinuous" / "scheduler_end_to_end_results.csv")
    parser.add_argument("--qwen25-detection-schedule-ar0p1", type=Path, default=None)
    parser.add_argument("--qwen25-detection-schedule-ar0p05", type=Path, default=None)
    parser.add_argument("--qwen25-task-ids", default="1,3,4,5,6,7,8,9")
    parser.add_argument("--qwen25-b1-batch-size", type=int, default=4)
    parser.add_argument("--qwen25-other-batch-size", type=int, default=32)
    parser.add_argument("--qwen25-b4-batch-size", type=int, default=8)
    parser.add_argument("--qwen25-b4-gpu-ids", default="0,1")
    parser.add_argument("--qwen25-only-groups", default="")
    parser.add_argument("--qwen25-train-only", action="store_true")

    q35_root = PROJECT_ROOT / "results" / "omniearth_qwen35_framework_20260705"
    parser.add_argument("--qwen35-out", type=Path, default=q35_root / "formal_qwen35_08b_ewc_student_baseline_20260708")
    parser.add_argument("--qwen35-python", type=Path, default=QWEN35_PYTHON)
    parser.add_argument("--qwen35-model", type=Path, default=QWEN35_MODEL)
    parser.add_argument("--qwen35-framework-root", type=Path, default=q35_root / "formal_qwen35_08b_framework_augmented_v2_all_no_pareto")
    parser.add_argument("--qwen35-gamma-csv", type=Path, default=q35_root / "formal_qwen35_08b_gamma_profiles_augmented_v2" / "gamma_smallstep_by_config.csv")
    parser.add_argument("--qwen35-pseudo-root", type=Path, default=q35_root / "formal_teacher_27b_lora_r8a16_s90_augmented_v2" / "pseudo_labels")
    parser.add_argument("--qwen35-split-root", type=Path, default=q35_root / "formal_qwen35_framework_splits_augmented_v2_from_base")
    parser.add_argument("--qwen35-scheduler-csv", type=Path, default=q35_root / "formal_qwen35_08b_framework_augmented_v2_all_no_pareto" / "scheduler" / "scheduler_end_to_end_results.csv")
    parser.add_argument("--qwen35-fixed-gamma", default="qwen35_lora_r16_a32_profile80_target320")
    parser.add_argument("--qwen35-task-ids", default="930,931,932")
    parser.add_argument("--qwen35-retraining-window-sec", type=float, default=7200.0)
    parser.add_argument("--qwen35-target-task-only", action="store_true")
    parser.add_argument("--qwen35-batch-size", type=int, default=32)
    parser.add_argument("--qwen35-gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--qwen35-only-groups", default="")
    parser.add_argument("--qwen35-train-only", action="store_true")
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--future-windows", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ewc_anchor_mode == "initial_zero" and args.ewc_mode != "ba_kv":
        raise ValueError("--ewc-anchor-mode initial_zero currently requires --ewc-mode ba_kv")
    families = {part.strip() for part in args.families.split(",") if part.strip()}
    if "qwen25" in families:
        run_qwen25(args)
    if "qwen35" in families:
        run_qwen35(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
