#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from remote_sensing_adaptation.ekya_lora_scheduler import fit_optimus_curve
from remote_sensing_adaptation.profile_utils import gamma_profile_from_measurement, parse_swift_metric_dicts


PROJECT_ROOT = Path("@@WORKSPACE@@")
QWEN_EVAL = PROJECT_ROOT / "qwen_eval"
DEFAULT_MODEL_PATH = QWEN_EVAL / "models" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_DATASET_DIR = QWEN_EVAL / "runs" / "qwen72b_drift_pseudolabel_5gpu" / "pseudo_dataset"
DEFAULT_CONFIGS = Path(__file__).with_name("gamma_measurement_configs.json")
DEFAULT_PREP_SCRIPT = QWEN_EVAL / "code" / "llm_lora_tools" / "prepare_qwen25vl_llm_lora_drift.py"
DEFAULT_STRICT_EVAL_SCRIPT = Path(__file__).with_name("eval_qwen_lora_strict.py")
PYTHON = Path("@@HOME@@/software/anaconda3/envs/qwen25vl/bin/python")
SWIFT = Path("@@HOME@@/software/anaconda3/envs/qwen25vl/bin/swift")
DEFAULT_BNB_CUDA13_LIB = Path(
    "@@HOME@@/software/anaconda3/pkgs/libnvjitlink-13.1.115-h7354ed3_0/lib"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def ensure_bnb_cuda13_library_path(env: dict[str, str]) -> dict[str, str]:
    updated = dict(env)
    lib_dir = Path(updated.get("BNB_CUDA13_LIB") or DEFAULT_BNB_CUDA13_LIB)
    if lib_dir.exists():
        current = updated.get("LD_LIBRARY_PATH", "")
        parts = [part for part in current.split(os.pathsep) if part]
        lib_text = str(lib_dir)
        if lib_text not in parts:
            parts.insert(0, lib_text)
        updated["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
        updated["BNB_CUDA13_LIB"] = lib_text
    return updated


def latest_trainer_state(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("trainer_state.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def latest_adapter_checkpoint(output_dir: Path) -> Path | None:
    def checkpoint_key(path: Path) -> tuple[int, float]:
        parent = path.parent
        step = -1
        if parent.name.startswith("checkpoint-"):
            try:
                step = int(parent.name.split("-", 1)[1])
            except ValueError:
                step = -1
        return step, path.stat().st_mtime

    candidates = sorted(output_dir.rglob("adapter_model.safetensors"), key=checkpoint_key)
    return candidates[-1].parent if candidates else None


def extract_curve(trainer_state_path: Path) -> list[tuple[float, float]]:
    state = read_json(trainer_state_path)
    points: list[tuple[float, float]] = []
    for row in state.get("log_history", []):
        if "eval_token_acc" in row and "step" in row:
            points.append((float(row["step"]), float(row["eval_token_acc"])))
    if len(points) >= 2:
        return points
    points = []
    for row in state.get("log_history", []):
        if "token_acc" in row and "step" in row:
            points.append((float(row["step"]), float(row["token_acc"])))
    return points


def estimate_profile_gains(curve_points: list[tuple[float, float]], train_steps: int) -> tuple[float, float]:
    if len(curve_points) < 2:
        return 0.0, 0.0
    start_acc = curve_points[0][1]
    last_acc = curve_points[-1][1]
    profile_gain = max(0.0, last_acc - start_acc)
    if profile_gain <= 0:
        return 0.0, 0.0
    curve = fit_optimus_curve(curve_points, max_accuracy=1.0)
    predicted_final_gain = max(0.0, curve(float(train_steps)) - start_acc)
    return profile_gain, predicted_final_gain


def microprofile_train_steps(config: dict[str, Any]) -> int:
    return int(config.get("microprofile_train_steps", config["train_steps"]))


def target_train_steps(config: dict[str, Any]) -> int:
    return int(config.get("target_train_steps", config["train_steps"]))


def extrapolate_train_time(
    measured_train_time_s: float,
    measured_train_steps: int,
    target_train_steps: int,
) -> float:
    if measured_train_steps <= 0:
        return measured_train_time_s
    if target_train_steps <= measured_train_steps:
        return measured_train_time_s
    return measured_train_time_s * (target_train_steps / measured_train_steps)


def strict_accuracy_gain(base_accuracy: float | None, updated_accuracy: float | None) -> float | None:
    if base_accuracy is None or updated_accuracy is None:
        return None
    return max(0.0, updated_accuracy - base_accuracy)


def estimate_strict_profile_gains(
    base_accuracy: float,
    microprofile_accuracy: float,
    measured_steps: int,
    target_steps: int,
    max_accuracy: float = 1.0,
) -> tuple[float, float, list[dict[str, float]]]:
    measured_step = float(max(1, measured_steps))
    target_step = float(max(measured_step, target_steps))
    base = float(base_accuracy)
    micro = float(microprofile_accuracy)
    curve_points = [
        {"step": 0.0, "accuracy": base},
        {"step": measured_step, "accuracy": micro},
    ]
    profile_gain = max(0.0, micro - base)
    if profile_gain <= 0:
        return 0.0, 0.0, curve_points
    curve = fit_optimus_curve(
        [(point["step"], point["accuracy"]) for point in curve_points],
        max_accuracy=max_accuracy,
    )
    predicted_final_gain = max(0.0, float(curve(target_step)) - base)
    return profile_gain, predicted_final_gain, curve_points


def strict_accuracy_from_report(report: dict[str, Any]) -> float:
    if "strict_accuracy" in report:
        return float(report["strict_accuracy"])
    accepted = int(report.get("accepted") or report.get("total") or 0)
    correct = int(report.get("correct") or 0)
    if accepted <= 0:
        return 0.0
    return correct / accepted


def apply_strict_profile_gain(
    measurement: dict[str, Any],
    base_report: dict[str, Any],
    adapter_report: dict[str, Any],
    measured_steps: int,
    target_steps: int,
) -> dict[str, Any]:
    base_accuracy = strict_accuracy_from_report(base_report)
    adapter_accuracy = strict_accuracy_from_report(adapter_report)
    profile_gain, predicted_gain, strict_curve = estimate_strict_profile_gains(
        base_accuracy=base_accuracy,
        microprofile_accuracy=adapter_accuracy,
        measured_steps=measured_steps,
        target_steps=target_steps,
    )
    updated = dict(measurement)
    updated["profile_gain"] = profile_gain
    updated["predicted_final_gain"] = predicted_gain
    updated["predicted_strict_acc_gain"] = predicted_gain
    updated["strict_base_accuracy"] = base_accuracy
    updated["strict_adapter_accuracy"] = adapter_accuracy
    updated["strict_profile_curve"] = strict_curve
    return updated


def is_qlora_config(config: dict[str, Any]) -> bool:
    method = str(config.get("method", "")).strip().lower()
    return method == "qlora" or config.get("quant_bits") not in (None, "", 0, "0")


def build_swift_sft_command(
    config: dict[str, Any],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    swift_output: Path,
) -> list[str]:
    train_jsonl = str(metadata["train_jsonl"])
    val_jsonl = str(metadata["val_jsonl"])
    command = [
        str(args.swift_bin),
        "sft",
        "--model",
        str(args.model_path),
        "--model_type",
        "qwen2_5_vl",
        "--dataset",
        train_jsonl,
        "--val_dataset",
        val_jsonl,
        "--tuner_type",
        "lora",
        "--torch_dtype",
        "bfloat16",
        "--freeze_llm",
        "false",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--target_modules",
        str(config["target_modules"]),
        "--lora_rank",
        str(config["lora_rank"]),
        "--lora_alpha",
        str(config["lora_alpha"]),
        "--lora_dropout",
        str(config.get("lora_dropout", 0.05)),
        "--learning_rate",
        str(config["learning_rate"]),
        "--per_device_train_batch_size",
        str(config["batch_size"]),
        "--per_device_eval_batch_size",
        str(config.get("eval_batch_size", config["batch_size"])),
        "--gradient_accumulation_steps",
        str(config["grad_accum"]),
        "--gradient_checkpointing",
        str(config.get("gradient_checkpointing", "false")).lower(),
        "--warmup_ratio",
        str(config.get("warmup_ratio", 0.05)),
        "--lr_scheduler_type",
        str(config.get("lr_scheduler_type", "cosine")),
        "--weight_decay",
        str(config.get("weight_decay", 0.01)),
        "--max_grad_norm",
        str(config.get("max_grad_norm", 1.0)),
        "--max_steps",
        str(microprofile_train_steps(config)),
        "--eval_steps",
        str(config.get("eval_steps", 20)),
        "--save_steps",
        str(config.get("save_steps", microprofile_train_steps(config))),
        "--logging_steps",
        str(config.get("logging_steps", 10)),
        "--save_total_limit",
        "2",
        "--seed",
        str(args.seed),
        "--max_length",
        str(config.get("max_length", 4096)),
        "--attn_impl",
        str(config.get("attn_impl", args.attn_impl)),
        "--dataloader_num_workers",
        str(config.get("dataloader_num_workers", 4)),
        "--dataset_num_proc",
        str(config.get("dataset_num_proc", 4)),
        "--output_dir",
        str(swift_output),
    ]
    if is_qlora_config(config):
        command.extend(["--quant_bits", str(config.get("quant_bits", 4))])
        if config.get("bnb_4bit_compute_dtype"):
            command.extend(["--bnb_4bit_compute_dtype", str(config["bnb_4bit_compute_dtype"])])
        if config.get("bnb_4bit_quant_type"):
            command.extend(["--bnb_4bit_quant_type", str(config["bnb_4bit_quant_type"])])
        if config.get("bnb_4bit_use_double_quant") not in (None, ""):
            command.extend(
                [
                    "--bnb_4bit_use_double_quant",
                    str(config["bnb_4bit_use_double_quant"]).lower(),
                ]
            )
    for extra_arg in config.get("extra_swift_args", []):
        command.append(str(extra_arg))
    return command


def build_strict_eval_command(
    args: argparse.Namespace,
    out_dir: Path,
    uid_filter_jsonl: Path,
    adapter_path: Path | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        str(args.strict_eval_script),
        "--dataset-dir",
        str(args.dataset_dir),
        "--model-path",
        str(args.model_path),
        "--out-dir",
        str(out_dir),
        "--uid-filter-jsonl",
        str(uid_filter_jsonl),
        "--task-ids",
        str(args.task_ids),
        "--max-samples",
        str(args.strict_eval_max_samples),
        "--max-new-tokens",
        str(args.strict_eval_max_new_tokens),
        "--dtype",
        str(args.strict_eval_dtype),
        "--answer-source",
        str(args.strict_eval_answer_source),
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--seed",
        str(args.seed),
    ]
    if adapter_path is not None:
        command.extend(["--adapter-path", str(adapter_path)])
    return command


def run_strict_eval(
    args: argparse.Namespace,
    out_dir: Path,
    uid_filter_jsonl: Path,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    report_path = out_dir / "strict_eval_report.json"
    if report_path.exists():
        return read_json(report_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_strict_eval_command(
        args=args,
        out_dir=out_dir,
        uid_filter_jsonl=uid_filter_jsonl,
        adapter_path=adapter_path,
    )
    env = ensure_bnb_cuda13_library_path(os.environ.copy())
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = f"{package_root}:{env.get('PYTHONPATH', '')}"
    log_path = out_dir / "strict_eval.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("[command] " + " ".join(command) + "\n")
        log_handle.flush()
        subprocess.run(command, check=True, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    return read_json(report_path)


def gpu_total_memory_gib(gpu_index: str) -> float:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                gpu_index.split(",")[0],
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        mib = float(result.stdout.strip().splitlines()[0])
        return mib / 1024.0
    except Exception:
        return 80.0


def read_power_watts(gpu_index: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout.strip().splitlines()
    if not text:
        return None
    try:
        return float(text[0].strip())
    except ValueError:
        return None


def read_temperature_c(gpu_index: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout.strip().splitlines()
    if not text:
        return None
    try:
        return float(text[0].strip())
    except ValueError:
        return None


class PowerSampler:
    def __init__(self, gpu_index: str, interval_sec: float = 0.5) -> None:
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self.samples: list[float] = []
        self.temperature_samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "PowerSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_sec * 4))

    def _run(self) -> None:
        while not self._stop.is_set():
            power_value = read_power_watts(self.gpu_index)
            if power_value is not None:
                self.samples.append(power_value)
            temperature_value = read_temperature_c(self.gpu_index)
            if temperature_value is not None:
                self.temperature_samples.append(temperature_value)
            self._stop.wait(self.interval_sec)


def run_prepare(args: argparse.Namespace) -> Path:
    run_name = "ekya_lora_gamma_profile_prepare"
    cmd = [
        str(PYTHON if PYTHON.exists() else "python"),
        str(args.prepare_script),
        "--dataset-dir",
        str(args.dataset_dir),
        "--out-root",
        str(args.out_dir),
        "--run-name",
        run_name,
        "--project-root",
        str(PROJECT_ROOT),
        "--seed",
        str(args.seed),
        "--val-ratio",
        str(args.val_ratio),
        "--percentile",
        str(args.pixel_percentile),
        "--task-ids",
        args.task_ids,
    ]
    subprocess.run(cmd, check=True)
    return args.out_dir / run_name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_group_seed(seed: int, group_id: str) -> int:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) + int(seed)


def write_group_prepared_root(
    *,
    base_prepared_root: Path,
    group: dict[str, Any],
    out_dir: Path,
    seed: int,
    val_ratio: float,
) -> Path:
    metadata = read_json(base_prepared_root / "experiment_metadata.json")
    wanted_ids = {str(uid) for uid in group.get("R_i_sample_ids", [])}
    if not wanted_ids:
        raise ValueError(f"Group {group.get('group_id')} has no R_i_sample_ids")
    base_rows = read_jsonl(Path(metadata["train_jsonl"])) + read_jsonl(Path(metadata["val_jsonl"]))
    selected = [row for row in base_rows if str(row.get("uid")) in wanted_ids]
    if not selected:
        raise ValueError(f"Group {group.get('group_id')} has no rows after filtering prepared JSONL")
    selected_ids = {str(row.get("uid")) for row in selected}
    missing = sorted(wanted_ids - selected_ids)
    if missing:
        raise ValueError(
            f"Group {group.get('group_id')} missing {len(missing)} prepared rows; first missing: {missing[:5]}"
        )

    group_id = str(group["group_id"])
    rng = random.Random(stable_group_seed(seed, group_id))
    selected = list(selected)
    rng.shuffle(selected)
    if len(selected) == 1:
        val_count = 0
    else:
        val_count = max(1, min(len(selected) - 1, round(len(selected) * val_ratio)))
    val_rows = selected[:val_count]
    train_rows = selected[val_count:]
    if not train_rows:
        train_rows, val_rows = selected, []

    group_root = out_dir / "prepared"
    train_jsonl = group_root / "swift_train.jsonl"
    val_jsonl = group_root / "swift_val.jsonl"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    group_metadata = dict(metadata)
    split_summary = dict(group_metadata.get("split_summary", {}))
    split_summary.update(
        {
            "group_id": group_id,
            "group_rows": len(selected),
            "group_train_rows": len(train_rows),
            "group_val_rows": len(val_rows),
        }
    )
    group_metadata.update(
        {
            "group_id": group_id,
            "source_prepared_root": str(base_prepared_root),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "split_summary": split_summary,
        }
    )
    write_json(out_dir / "experiment_metadata.json", group_metadata)
    return out_dir


def read_group_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def selected_groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.group_manifest:
        return []
    groups = read_group_manifest(args.group_manifest)
    if args.group_ids:
        wanted = {item.strip() for item in args.group_ids.split(",") if item.strip()}
        groups = [group for group in groups if str(group["group_id"]) in wanted]
    if not groups:
        raise ValueError("No groups selected for group-specific Gamma profiling")
    return groups


def run_swift_config(
    config: dict[str, Any],
    prepared_root: Path,
    args: argparse.Namespace,
    total_memory_gib: float,
) -> dict[str, Any]:
    metadata = read_json(prepared_root / "experiment_metadata.json")
    max_pixels = str(metadata["recommended_max_pixels"])
    run_dir = args.out_dir / str(config["gamma_id"])
    swift_output = run_dir / "swift_output"
    log_path = run_dir / "swift_sft.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = ensure_bnb_cuda13_library_path(os.environ.copy())
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["MAX_PIXELS"] = max_pixels
    env["PATH"] = f"{SWIFT.parent}:{env.get('PATH', '')}"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    command = build_swift_sft_command(config, metadata, args, swift_output)

    started = time.perf_counter()
    power_sampler = PowerSampler(
        args.cuda_visible_devices.split(",", 1)[0].strip(),
        interval_sec=args.power_sample_interval,
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("[command] " + " ".join(command) + "\n")
        log_handle.flush()
        with power_sampler:
            subprocess.run(command, check=True, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    wall_time = time.perf_counter() - started

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_swift_metric_dicts(log_text)
    trainer_state = latest_trainer_state(swift_output)
    curve_points = extract_curve(trainer_state) if trainer_state else []
    measured_steps = microprofile_train_steps(config)
    target_steps = target_train_steps(config)
    profile_gain, predicted_final_gain = estimate_profile_gains(
        curve_points,
        train_steps=target_steps,
    )
    predicted_token_acc_gain = predicted_final_gain
    strict_gain = strict_accuracy_gain(
        config.get("base_strict_accuracy", args.base_strict_accuracy),
        config.get("updated_strict_accuracy", args.updated_strict_accuracy),
    )
    if strict_gain is not None:
        predicted_final_gain = strict_gain

    train_runtime = metrics["train_runtime"]
    train_time = wall_time if math.isnan(train_runtime) else train_runtime
    predicted_train_time = extrapolate_train_time(
        measured_train_time_s=train_time,
        measured_train_steps=measured_steps,
        target_train_steps=target_steps,
    )
    peak_memory_gib = metrics["peak_memory_gib"]
    avg_power_watts = (
        sum(power_sampler.samples) / len(power_sampler.samples) if power_sampler.samples else None
    )
    peak_power_watts = max(power_sampler.samples) if power_sampler.samples else None
    avg_temperature_c = (
        sum(power_sampler.temperature_samples) / len(power_sampler.temperature_samples)
        if power_sampler.temperature_samples
        else None
    )
    peak_temperature_c = (
        max(power_sampler.temperature_samples) if power_sampler.temperature_samples else None
    )
    resource_cost = peak_memory_gib / total_memory_gib if peak_memory_gib > 0 else 1.0
    measurement = {
        "gamma_id": str(config["gamma_id"]),
        "method": str(config.get("method", "LoRA")),
        "lora_rank": int(config["lora_rank"]),
        "lora_alpha": int(config["lora_alpha"]),
        "target_modules": str(config["target_modules"]),
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "grad_accum": int(config["grad_accum"]),
        "train_steps": target_steps,
        "microprofile_train_steps": measured_steps,
        "target_train_steps": target_steps,
        "train_data_fraction": float(config.get("train_data_fraction", 1.0)),
        "profile_gain": profile_gain,
        "predicted_final_gain": predicted_final_gain,
        "predicted_token_acc_gain": predicted_token_acc_gain,
        "resource_cost": resource_cost,
        "train_time": predicted_train_time,
        "profile_train_time": train_time,
        "profile_curve": [{"step": step, "accuracy": acc} for step, acc in curve_points],
        "measured_peak_memory_gib": peak_memory_gib,
        "avg_power_watts": avg_power_watts,
        "peak_power_watts": peak_power_watts,
        "power_watts": avg_power_watts,
        "power_sample_count": len(power_sampler.samples),
        "avg_temperature_c": avg_temperature_c,
        "peak_temperature_c": peak_temperature_c,
        "thermal_score": peak_temperature_c,
        "temperature_sample_count": len(power_sampler.temperature_samples),
        "measured_wall_time_sec": wall_time,
        "trainer_state": str(trainer_state) if trainer_state else "",
        "log_path": str(log_path),
        "prepared_root": str(prepared_root),
    }
    for key in [
        "quant_bits",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
    ]:
        if config.get(key) not in (None, ""):
            measurement[key] = config[key]
    if args.strict_gain_eval:
        adapter_path = latest_adapter_checkpoint(swift_output)
        if adapter_path is None:
            raise ValueError(f"No LoRA adapter checkpoint found under {swift_output}")
        base_report = run_strict_eval(
            args=args,
            out_dir=prepared_root / "strict_eval_base",
            uid_filter_jsonl=Path(metadata["val_jsonl"]),
        )
        adapter_report = run_strict_eval(
            args=args,
            out_dir=run_dir / "strict_eval_adapter",
            uid_filter_jsonl=Path(metadata["val_jsonl"]),
            adapter_path=adapter_path,
        )
        measurement = apply_strict_profile_gain(
            measurement,
            base_report=base_report,
            adapter_report=adapter_report,
            measured_steps=measured_steps,
            target_steps=target_steps,
        )
        measurement["strict_eval_base_report"] = str(prepared_root / "strict_eval_base" / "strict_eval_report.json")
        measurement["strict_eval_adapter_report"] = str(run_dir / "strict_eval_adapter" / "strict_eval_report.json")
        measurement["adapter_path"] = str(adapter_path)
    return measurement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Qwen2.5-VL-3B LoRA Gamma profiles with SWIFT.")
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prepare-script", type=Path, default=DEFAULT_PREP_SCRIPT)
    parser.add_argument("--swift-bin", type=Path, default=SWIFT)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--task-ids", default="1,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pixel-percentile", type=int, default=95)
    parser.add_argument("--attn-impl", default="flash_attn")
    parser.add_argument("--power-sample-interval", type=float, default=0.5)
    parser.add_argument("--base-strict-accuracy", type=float, default=None)
    parser.add_argument("--updated-strict-accuracy", type=float, default=None)
    parser.add_argument("--group-manifest", type=Path, default=None)
    parser.add_argument("--group-ids", default="")
    parser.add_argument("--strict-gain-eval", action="store_true")
    parser.add_argument("--strict-eval-script", type=Path, default=DEFAULT_STRICT_EVAL_SCRIPT)
    parser.add_argument("--strict-eval-max-samples", type=int, default=0)
    parser.add_argument("--strict-eval-max-new-tokens", type=int, default=16)
    parser.add_argument("--strict-eval-dtype", default="bfloat16")
    parser.add_argument("--strict-eval-answer-source", default="raw")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = read_json(args.configs)
    prepared_root = run_prepare(args)
    total_memory_gib = gpu_total_memory_gib(args.cuda_visible_devices)
    groups = selected_groups(args)
    if groups:
        all_measurements: list[dict[str, Any]] = []
        all_profiles: list[dict[str, Any]] = []
        for group in groups:
            group_id = str(group["group_id"])
            group_dir = args.out_dir / group_id
            group_prepared_root = write_group_prepared_root(
                base_prepared_root=prepared_root,
                group=group,
                out_dir=group_dir / "ekya_lora_gamma_profile_prepare",
                seed=args.seed,
                val_ratio=args.val_ratio,
            )
            group_args = copy.copy(args)
            group_args.out_dir = group_dir
            measurements = [
                {**run_swift_config(config, group_prepared_root, group_args, total_memory_gib), "group_id": group_id}
                for config in configs
            ]
            profiles = [
                {**gamma_profile_from_measurement(row), "group_id": group_id}
                for row in measurements
            ]
            write_json(group_dir / "gamma_measurements.json", measurements)
            write_json(group_dir / "gamma_profiles.json", profiles)
            all_measurements.extend(measurements)
            all_profiles.extend(profiles)
        write_json(args.out_dir / "gamma_group_measurements.json", all_measurements)
        write_json(args.out_dir / "gamma_group_profiles.json", all_profiles)
        print(json.dumps({"gamma_profiles": all_profiles, "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2))
        return 0

    measurements = [
        run_swift_config(config, prepared_root, args, total_memory_gib)
        for config in configs
    ]
    profiles = [gamma_profile_from_measurement(row) for row in measurements]
    write_json(args.out_dir / "gamma_measurements.json", measurements)
    write_json(args.out_dir / "gamma_profiles.json", profiles)
    print(json.dumps({"gamma_profiles": profiles, "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
