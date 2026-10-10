#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
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


def int_field(row: dict[str, Any], *keys: str, default: int | None = None) -> int:
    for key in keys:
        value = row.get(key)
        if value not in ("", None):
            return int(float(value))
    if default is None:
        raise KeyError(f"Missing integer field among {keys}")
    return default


def float_field(row: dict[str, Any], *keys: str, default: float | None = None) -> float:
    for key in keys:
        value = row.get(key)
        if value not in ("", None):
            return float(value)
    if default is None:
        raise KeyError(f"Missing float field among {keys}")
    return default


def maybe_copy(src: dict[str, Any], dst: dict[str, Any], key: str) -> None:
    value = src.get(key)
    if value not in ("", None):
        dst[key] = value


def fulltarget_config(gamma_row: dict[str, Any]) -> dict[str, Any]:
    target_steps = int_field(gamma_row, "target_train_steps", "train_steps")
    config: dict[str, Any] = {
        "gamma_id": str(gamma_row["gamma_id"]),
        "method": str(gamma_row.get("method") or "LoRA"),
        "lora_rank": int_field(gamma_row, "lora_rank"),
        "lora_alpha": int_field(gamma_row, "lora_alpha"),
        "target_modules": str(gamma_row["target_modules"]),
        "learning_rate": float_field(gamma_row, "learning_rate"),
        "batch_size": int_field(gamma_row, "batch_size", "train_batch_size", default=1),
        "grad_accum": int_field(gamma_row, "grad_accum", "grad_accum_steps", default=1),
        "train_steps": target_steps,
        "microprofile_train_steps": target_steps,
        "target_train_steps": target_steps,
        "train_data_fraction": float_field(gamma_row, "train_data_fraction", default=1.0),
        "eval_steps": min(20, target_steps),
        "save_steps": target_steps,
        "logging_steps": min(10, target_steps),
    }
    for key in [
        "lora_dropout",
        "gradient_checkpointing",
        "warmup_ratio",
        "lr_scheduler_type",
        "weight_decay",
        "max_grad_norm",
        "max_length",
        "attn_impl",
        "dataloader_num_workers",
        "dataset_num_proc",
        "quant_bits",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
    ]:
        maybe_copy(gamma_row, config, key)
    return config


def latest_adapter(run_root: Path, group_id: str, gamma_id: str) -> Path | None:
    base = run_root / group_id / gamma_id
    candidates = sorted(
        base.glob("swift_output/*/checkpoint-*/adapter_model.safetensors"),
        key=lambda path: (
            int(path.parent.name.split("-", 1)[1])
            if path.parent.name.startswith("checkpoint-")
            else -1,
            path.stat().st_mtime,
        ),
    )
    return candidates[-1].parent if candidates else None


def selected_jobs(
    scheduler_rows: list[dict[str, str]],
    *,
    method: str,
    require_adapter_accepted: bool,
) -> list[tuple[str, str]]:
    jobs: set[tuple[str, str]] = set()
    for row in scheduler_rows:
        if str(row.get("method")) != method:
            continue
        gamma_id = str(row.get("gamma_id") or "")
        if not gamma_id:
            continue
        if require_adapter_accepted and not parse_bool(row.get("adapter_accepted")):
            continue
        jobs.add((str(row["group_id"]), gamma_id))
    return sorted(jobs)


def run_job(
    *,
    args: argparse.Namespace,
    group_id: str,
    gamma_id: str,
    config: dict[str, Any],
) -> None:
    job_dir = args.out_dir / group_id / gamma_id
    job_dir.mkdir(parents=True, exist_ok=True)
    lock_path = job_dir / ".fulltarget_train.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if latest_adapter(args.out_dir, group_id, gamma_id) is not None and not args.force:
            return
        config_path = args.config_dir / f"{group_id}__{gamma_id}.json"
        write_json(config_path, [config])
        log_dir = args.out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{group_id}__{gamma_id}.log"
        command = [
            str(args.python_bin),
            "-m",
            "remote_sensing_adaptation.profile_v2_qwen_lora_gamma_fixed",
            "--configs",
            str(config_path),
            "--pseudo-root",
            str(args.pseudo_root),
            "--out-dir",
            str(args.out_dir),
            "--model-path",
            str(args.model_path),
            "--cuda-visible-devices",
            str(args.cuda_visible_devices),
            "--task-ids",
            str(args.task_ids),
            "--seed",
            str(args.seed),
            "--group-ids",
            group_id,
            "--no-strict-gain-eval",
        ]
        env = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = f"{package_root}:{env.get('PYTHONPATH', '')}"
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("[command] " + " ".join(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V2 scheduler-selected adapters to target steps.")
    parser.add_argument("--scheduler-csv", type=Path, required=True)
    parser.add_argument("--gamma-json", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--task-ids", default="1,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="ThiefLoRA")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-unaccepted-adapter", action="store_true")
    parser.add_argument("--plan-csv", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir = args.config_dir or (args.out_dir / "configs")
    args.plan_csv = args.plan_csv or (args.out_dir / "true_selected_fulltarget_plan.csv")
    gamma_rows = read_json(args.gamma_json)
    gamma_by_pair = {
        (str(row.get("group_id") or ""), str(row["gamma_id"])): dict(row)
        for row in gamma_rows
    }
    jobs = selected_jobs(
        read_csv(args.scheduler_csv),
        method=args.method,
        require_adapter_accepted=not args.allow_unaccepted_adapter,
    )
    plan_rows: list[dict[str, Any]] = []
    configs: dict[tuple[str, str], dict[str, Any]] = {}
    for group_id, gamma_id in jobs:
        gamma_row = gamma_by_pair.get((group_id, gamma_id))
        if gamma_row is None:
            raise KeyError(f"Missing Gamma profile for {group_id}/{gamma_id}")
        config = fulltarget_config(gamma_row)
        configs[(group_id, gamma_id)] = config
        adapter = latest_adapter(args.out_dir, group_id, gamma_id)
        plan_rows.append(
            {
                "group_id": group_id,
                "gamma_id": gamma_id,
                "method": config["method"],
                "target_steps": config["target_train_steps"],
                "adapter_exists": bool(adapter),
                "adapter_path": str(adapter or ""),
            }
        )
    write_csv(args.plan_csv, plan_rows)
    if args.execute:
        for group_id, gamma_id in jobs:
            run_job(args=args, group_id=group_id, gamma_id=gamma_id, config=configs[(group_id, gamma_id)])
    print(json.dumps({"jobs": len(plan_rows), "plan": str(args.plan_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
