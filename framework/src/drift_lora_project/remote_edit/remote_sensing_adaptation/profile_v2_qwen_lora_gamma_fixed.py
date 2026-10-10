#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from remote_sensing_adaptation.profile_qwen_lora_gamma import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STRICT_EVAL_SCRIPT,
    PYTHON,
    SWIFT,
    estimate_profile_gains,
    estimate_strict_profile_gains,
    extrapolate_train_time,
    gpu_total_memory_gib,
    run_swift_config,
)
from remote_sensing_adaptation.profile_utils import gamma_profile_from_measurement


PROJECT_ROOT = Path("@@WORKSPACE@@")
QWEN_EVAL = PROJECT_ROOT / "qwen_eval"
DEFAULT_PSEUDO_ROOT = (
    PROJECT_ROOT
    / "drift_lora_project/results/v2_teacher_closed_loop_20260630_181348/step4_teacher_pseudolabels"
)
DEFAULT_LORA_CONFIGS = (
    PROJECT_ROOT
    / "drift_lora_project/ekya-sparse/remote_sensing_adaptation/gamma_measurement_configs.json"
)
DEFAULT_QLORA_CONFIGS = (
    PROJECT_ROOT
    / "drift_lora_project/remote_edit/remote_sensing_adaptation/generated_configs/v1_qlora_gamma_measurement_configs.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_task_ids(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def load_prepare_helpers() -> dict[str, Any]:
    sys.path.insert(0, str(QWEN_EVAL / "code" / "llm_lora_tools"))
    from prepare_qwen25vl_llm_lora_drift import (  # type: ignore
        build_image_statistics,
        convert_item_to_swift,
        load_prompt_builder,
    )

    return {
        "build_image_statistics": build_image_statistics,
        "convert_item_to_swift": convert_item_to_swift,
        "load_prompt_builder": load_prompt_builder,
    }


def convert_fixed_split(
    *,
    items: list[dict[str, Any]],
    dataset_dir: Path,
    prompt_builder: Any,
    convert_item_to_swift: Any,
    task_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in items:
        if task_ids and int(item.get("task_id", 0)) not in task_ids:
            continue
        row, missing = convert_item_to_swift(item, dataset_dir, prompt_builder)
        if row is None:
            skipped.append(
                {
                    "uid": str(item.get("uid", "")),
                    "task_id": item.get("task_id"),
                    "reason": "missing_images",
                    "missing_images": missing,
                }
            )
        else:
            rows.append(row)
    return rows, skipped


def prepare_fixed_root(
    *,
    train_data: Path,
    val_data: Path,
    out_dir: Path,
    task_ids: set[int],
    pixel_percentile: int,
    seed: int,
    force: bool,
) -> Path:
    metadata_path = out_dir / "experiment_metadata.json"
    if metadata_path.exists() and not force:
        return out_dir

    helpers = load_prepare_helpers()
    train_items = read_json(train_data)
    val_items = read_json(val_data)
    if not isinstance(train_items, list) or not isinstance(val_items, list):
        raise TypeError("train/val data.json must each contain a list")

    dataset_dir = train_data.parent
    prompt_builder = helpers["load_prompt_builder"](PROJECT_ROOT)
    train_rows, skipped_train = convert_fixed_split(
        items=train_items,
        dataset_dir=dataset_dir,
        prompt_builder=prompt_builder,
        convert_item_to_swift=helpers["convert_item_to_swift"],
        task_ids=task_ids,
    )
    val_rows, skipped_val = convert_fixed_split(
        items=val_items,
        dataset_dir=dataset_dir,
        prompt_builder=prompt_builder,
        convert_item_to_swift=helpers["convert_item_to_swift"],
        task_ids=task_ids,
    )
    if not train_rows:
        raise ValueError(f"No train rows converted from {train_data}")
    if not val_rows:
        raise ValueError(f"No val rows converted from {val_data}")

    filtered_all = [
        item
        for item in train_items + val_items
        if not task_ids or int(item.get("task_id", 0)) in task_ids
    ]
    stats = helpers["build_image_statistics"](
        filtered_all,
        dataset_dir,
        percentile=pixel_percentile,
    )

    write_jsonl(out_dir / "swift_train.jsonl", train_rows)
    write_jsonl(out_dir / "swift_val.jsonl", val_rows)
    write_jsonl(out_dir / "skipped_missing_images.jsonl", skipped_train + skipped_val)
    write_json(out_dir / "image_stats.json", stats)

    split_summary = {
        "source_train_samples": len(train_items),
        "source_val_samples": len(val_items),
        "task_ids": sorted(task_ids),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "skipped_train_missing_images": len(skipped_train),
        "skipped_val_missing_images": len(skipped_val),
        "task_counts_train": dict(sorted(Counter(str(row["task_id"]) for row in train_rows).items())),
        "task_counts_val": dict(sorted(Counter(str(row["task_id"]) for row in val_rows).items())),
    }
    metadata = {
        "run_name": out_dir.name,
        "out_dir": str(out_dir),
        "dataset_dir": str(dataset_dir),
        "train_data": str(train_data),
        "val_data": str(val_data),
        "train_jsonl": str(out_dir / "swift_train.jsonl"),
        "val_jsonl": str(out_dir / "swift_val.jsonl"),
        "recommended_max_pixels": stats["recommended_max_pixels"],
        "task_ids": sorted(task_ids),
        "seed": seed,
        "val_ratio": None,
        "split_summary": split_summary,
        "fixed_split": True,
    }
    write_json(metadata_path, metadata)
    return out_dir


def load_configs(paths: list[Path], limit: int) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if isinstance(payload, dict):
            payload = payload.get("configs") or payload.get("rows") or []
        if not isinstance(payload, list):
            raise TypeError(f"Config file must contain a list: {path}")
        configs.extend(dict(row) for row in payload)
    if limit > 0:
        configs = configs[:limit]
    seen: set[str] = set()
    duplicates: list[str] = []
    for config in configs:
        gamma_id = str(config["gamma_id"])
        if gamma_id in seen:
            duplicates.append(gamma_id)
        seen.add(gamma_id)
    if duplicates:
        raise ValueError(f"Duplicate gamma_id values: {duplicates[:5]}")
    return configs


def microprofile_cache_key(config: dict[str, Any]) -> str:
    """Identity for configs that run the same small-step SWIFT training."""

    normalized = {
        key: value
        for key, value in config.items()
        if key not in {"gamma_id", "train_steps", "target_train_steps"}
    }
    normalized["microprofile_train_steps"] = int(
        config.get("microprofile_train_steps", config.get("train_steps", 0))
    )
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, default=str)


def curve_points_from_measurement(measurement: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in measurement.get("profile_curve", []):
        if not isinstance(point, dict):
            continue
        if "step" not in point:
            continue
        value = point.get("accuracy", point.get("acc"))
        if value in (None, ""):
            continue
        points.append((float(point["step"]), float(value)))
    return points


def retarget_measurement(base: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(base)
    source_gamma_id = str(updated.get("gamma_id", ""))
    updated.update(config)

    measured_steps = int(config.get("microprofile_train_steps", config.get("train_steps", 0)))
    target_steps = int(config.get("target_train_steps", config.get("train_steps", measured_steps)))
    updated["train_steps"] = target_steps
    updated["microprofile_train_steps"] = measured_steps
    updated["target_train_steps"] = target_steps
    updated["dedup_source_gamma_id"] = source_gamma_id
    updated["dedup_reused_microprofile"] = str(config["gamma_id"]) != source_gamma_id

    profile_train_time = float(
        updated.get("profile_train_time")
        or updated.get("measured_wall_time_sec")
        or updated.get("train_time")
        or 0.0
    )
    if profile_train_time > 0:
        target_train_time = extrapolate_train_time(
            measured_train_time_s=profile_train_time,
            measured_train_steps=measured_steps,
            target_train_steps=target_steps,
        )
        updated["train_time"] = target_train_time
        updated["train_time_s"] = target_train_time
        updated["train_gpu_seconds"] = target_train_time

    token_curve = curve_points_from_measurement(updated)
    if len(token_curve) >= 2:
        profile_gain, predicted_token_gain = estimate_profile_gains(
            token_curve,
            train_steps=target_steps,
        )
        updated["profile_gain"] = profile_gain
        updated["predicted_token_acc_gain"] = predicted_token_gain
        if "strict_base_accuracy" not in updated or "strict_adapter_accuracy" not in updated:
            updated["predicted_final_gain"] = predicted_token_gain

    if "strict_base_accuracy" in updated and "strict_adapter_accuracy" in updated:
        strict_gain, strict_predicted_gain, strict_curve = estimate_strict_profile_gains(
            base_accuracy=float(updated["strict_base_accuracy"]),
            microprofile_accuracy=float(updated["strict_adapter_accuracy"]),
            measured_steps=measured_steps,
            target_steps=target_steps,
        )
        updated["profile_gain"] = strict_gain
        updated["predicted_final_gain"] = strict_predicted_gain
        updated["predicted_strict_acc_gain"] = strict_predicted_gain
        updated["strict_profile_curve"] = strict_curve

    if "predicted_final_gain" in updated:
        updated["expected_acc_gain"] = updated["predicted_final_gain"]
    elif "profile_gain" in updated:
        updated["expected_acc_gain"] = updated["profile_gain"]
    return updated


def group_ids_from_pseudo_root(pseudo_root: Path, requested: str) -> list[str]:
    if requested.strip():
        return [item.strip() for item in requested.split(",") if item.strip()]
    groups_root = pseudo_root / "groups"
    if not groups_root.exists():
        raise FileNotFoundError(f"Missing assembled pseudo groups: {groups_root}")
    return sorted(path.name for path in groups_root.iterdir() if path.is_dir())


def group_split_data(pseudo_root: Path, group_id: str, split: str) -> Path:
    path = pseudo_root / "groups" / group_id / split / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {split} pseudo data for {group_id}: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure V2 Qwen2.5-VL-3B Gamma profiles using fixed R_train/R_val pseudo-label splits."
    )
    parser.add_argument("--configs", type=Path, nargs="+", default=[DEFAULT_LORA_CONFIGS, DEFAULT_QLORA_CONFIGS])
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--swift-bin", type=Path, default=SWIFT)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--task-ids", default="1,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-ids", default="")
    parser.add_argument("--config-limit", type=int, default=0)
    parser.add_argument("--pixel-percentile", type=int, default=95)
    parser.add_argument("--attn-impl", default="flash_attn")
    parser.add_argument("--power-sample-interval", type=float, default=0.5)
    parser.add_argument("--strict-gain-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-eval-script", type=Path, default=DEFAULT_STRICT_EVAL_SCRIPT)
    parser.add_argument("--strict-eval-max-samples", type=int, default=0)
    parser.add_argument("--strict-eval-max-new-tokens", type=int, default=16)
    parser.add_argument("--strict-eval-dtype", default="bfloat16")
    parser.add_argument("--strict-eval-answer-source", default="raw")
    parser.add_argument("--base-strict-accuracy", type=float, default=None)
    parser.add_argument("--updated-strict-accuracy", type=float, default=None)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = parse_task_ids(args.task_ids)
    configs = load_configs(args.configs, args.config_limit)
    total_memory_gib = gpu_total_memory_gib(args.cuda_visible_devices)
    group_ids = group_ids_from_pseudo_root(args.pseudo_root, args.group_ids)

    all_measurements: list[dict[str, Any]] = []
    all_profiles: list[dict[str, Any]] = []
    for group_id in group_ids:
        group_dir = args.out_dir / group_id
        prepared_root = prepare_fixed_root(
            train_data=group_split_data(args.pseudo_root, group_id, "R_train"),
            val_data=group_split_data(args.pseudo_root, group_id, "R_val"),
            out_dir=group_dir / "fixed_prepare",
            task_ids=task_ids,
            pixel_percentile=args.pixel_percentile,
            seed=args.seed,
            force=args.force_prepare,
        )
        group_args = copy.copy(args)
        group_args.out_dir = group_dir
        group_args.dataset_dir = group_split_data(args.pseudo_root, group_id, "R_val").parent
        microprofile_cache: dict[str, dict[str, Any]] = {}
        measurements = []
        for config in configs:
            cache_key = microprofile_cache_key(config)
            if cache_key in microprofile_cache:
                measurement = retarget_measurement(microprofile_cache[cache_key], config)
            else:
                measurement = run_swift_config(config, prepared_root, group_args, total_memory_gib)
                microprofile_cache[cache_key] = measurement
            measurements.append({**measurement, "group_id": group_id})
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
    write_json(
        args.out_dir / "gamma_v2_fixed_profile_summary.json",
        {
            "groups": group_ids,
            "configs": [str(path) for path in args.configs],
            "gamma_profiles": len(all_profiles),
            "strict_gain_eval": args.strict_gain_eval,
            "out_dir": str(args.out_dir),
        },
    )
    print(
        json.dumps(
            {"gamma_profiles": len(all_profiles), "out_dir": str(args.out_dir)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
