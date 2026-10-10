#!/usr/bin/env python3
"""Prepare drift-only Qwen2.5-VL LLM-LoRA data for SWIFT.

This script is intentionally read-only with respect to the source dataset. It
copies data.json into the experiment backup directory, writes an audit manifest,
converts usable samples to SWIFT multimodal SFT JSONL, and stratifies train/val
by task_id.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path("@@WORKSPACE@@")
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "2_Stream_Sim_Drift"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "qwen_eval" / "runs"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PIXEL_ALIGNMENT = 28 * 28
DEFAULT_TASK_IDS = {1, 3, 4, 5, 6, 7}


PromptBuilder = Callable[[int, str, dict[str, Any] | None, str | None, Any], str]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_path(path_value: str | None, dataset_dir: Path) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    return str((dataset_dir / str(path_value).lstrip("./")).resolve())


def looks_like_image_path(value: Any) -> bool:
    return isinstance(value, str) and Path(value).suffix.lower() in IMG_EXTS


def task_question(item: dict[str, Any]) -> str:
    if int(item.get("task_id", 0)) == 2:
        prompts = item.get("prompts")
        if isinstance(prompts, list) and prompts:
            return str(prompts[0])
    return str(item.get("question", ""))


def image_paths_for_item(item: dict[str, Any], dataset_dir: Path) -> list[str]:
    task_id = int(item.get("task_id", 0))
    if task_id == 2:
        return [
            p
            for p in (
                norm_path(item.get("pre_image_path"), dataset_dir),
                norm_path(item.get("post_image_path"), dataset_dir),
            )
            if p
        ]

    options = item.get("options")
    if task_id == 1 and isinstance(options, dict) and any(looks_like_image_path(v) for v in options.values()):
        return [norm_path(options[k], dataset_dir) for k in sorted(options) if looks_like_image_path(options[k])]

    image_path = norm_path(item.get("image_path"), dataset_dir)
    return [image_path] if image_path else []


def prompt_options_for_item(item: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    options = item.get("options")
    if not isinstance(options, dict):
        return {}
    if int(item.get("task_id", 0)) == 1 and any(looks_like_image_path(v) for v in options.values()):
        return {k: norm_path(v, dataset_dir) if looks_like_image_path(v) else v for k, v in options.items()}
    return options


def sample_type_for_item(item: dict[str, Any]) -> str:
    options = item.get("options")
    if int(item.get("task_id", 0)) == 1 and isinstance(options, dict) and any(looks_like_image_path(v) for v in options.values()):
        return "image_options"
    return str(item.get("type", ""))


def answer_for_item(item: dict[str, Any]) -> str:
    answer = item.get("ground_truth", "")
    if int(item.get("task_id", 0)) == 1 and (item.get("type_id") == 3 or item.get("type") == "multi_mcq_opt_text"):
        if isinstance(answer, list):
            return ",".join(str(value).strip().upper() for value in answer if str(value).strip())
        return ",".join(
            part.strip().upper()
            for part in str(answer).replace(";", ",").split(",")
            if part.strip()
        )
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


def default_prompt_builder(task_id: int, question: str, options: dict[str, Any] | None = None, sample_type: str | None = None, type_id: Any = None) -> str:
    """Fallback only used when the project prompt helper is not importable."""
    if task_id == 1 and (type_id == 1 or sample_type == "image_options") and options:
        labels = [str(k).upper() for k in sorted(options)]
        ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
        image_options = "\n".join(
            f"- {label}: image {idx + 1} ({ordinals[idx] if idx < len(ordinals) else f'#{idx + 1}'} attached image)"
            for idx, label in enumerate(labels)
        )
        label_examples = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 1 else labels[0]
        return (
            f"{question}\n\n"
            f"You are given {len(labels)} attached candidate images in order. "
            "Each candidate image corresponds to exactly one option letter:\n"
            f"{image_options}\n\n"
            "Choose the option whose image best answers the question. "
            "Do not answer with the crop name, pest name, image filename, or option text.\n"
            f"Respond with ONLY one capital letter: {label_examples}.\n"
            "No explanation, punctuation, or extra text."
        )
    if task_id == 1 and type_id == 4:
        return f"{question}\n\nAnswer only with 'Yes' or 'No'."
    if task_id == 1 and type_id == 3 and options:
        opts = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
        return (
            f"{question}\n\n"
            f"Options:\n{opts}\n\n"
            "Based on the provided image, respond with ONLY the capital letters of all correct options "
            "as a comma-separated list (e.g., B,C). Do not include any explanation."
        )
    if task_id == 1 and (type_id == 2 or sample_type == "text_options") and options:
        labels = [str(k).upper() for k in sorted(options)]
        label_examples = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 1 else labels[0]
        opts = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
        return (
            f"{question}\n\n"
            f"Options:\n{opts}\n\n"
            "Based on the provided image, please respond with ONLY the capital letter of the correct "
            f"option (e.g., {label_examples}). Do not include any explanation."
        )
    if options:
        opts = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
        return f"{question}\n\nOptions:\n{opts}\n\nAnswer with only the required label or short answer."
    return f"{question}\n\nAnswer according to the task instruction."


def load_prompt_builder(project_root: Path = PROJECT_ROOT) -> PromptBuilder:
    code_dir = project_root / "code"
    if code_dir.exists():
        sys.path.insert(0, str(code_dir))
    try:
        from generic_inference import build_prompt  # type: ignore

        return build_prompt
    except Exception:
        return default_prompt_builder


def convert_item_to_swift(
    item: dict[str, Any],
    dataset_dir: Path,
    prompt_builder: PromptBuilder,
) -> tuple[dict[str, Any] | None, list[str]]:
    images = image_paths_for_item(item, dataset_dir)
    bad_images = bad_image_paths(images)
    if bad_images:
        return None, bad_images

    task_id = int(item.get("task_id", 0))
    options = prompt_options_for_item(item, dataset_dir)
    sample_type = sample_type_for_item(item)
    prompt = prompt_builder(task_id, task_question(item), options, sample_type, item.get("type_id"))
    image_tokens = "<image>" * len(images)
    content = f"{image_tokens}\n{prompt}" if image_tokens else prompt
    answer = answer_for_item(item)

    return (
        {
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": answer},
            ],
            "images": images,
            "uid": str(item.get("uid", "")),
            "task_id": task_id,
        },
        [],
    )


def stratified_split(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row.get("task_id", 0))].append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for task_id in sorted(groups):
        group = groups[task_id]
        rng.shuffle(group)
        val_count = 0 if len(group) <= 1 else max(1, math.ceil(len(group) * val_ratio))
        val.extend(group[:val_count])
        train.extend(group[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def prepare_splits(
    items: list[dict[str, Any]],
    dataset_dir: Path,
    out_dir: Path,
    prompt_builder: PromptBuilder,
    seed: int = 42,
    val_ratio: float = 0.1,
    task_ids: set[int] | None = None,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    filtered_items = [
        item for item in items if task_ids is None or int(item.get("task_id", 0)) in task_ids
    ]

    for item in filtered_items:
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
            accepted.append(row)

    train, val = stratified_split(accepted, val_ratio, seed)
    write_jsonl(out_dir / "swift_train.jsonl", train)
    write_jsonl(out_dir / "swift_val.jsonl", val)
    write_jsonl(out_dir / "skipped_missing_images.jsonl", skipped)

    summary = {
        "source_samples": len(items),
        "filtered_samples": len(filtered_items),
        "task_ids": sorted(task_ids) if task_ids is not None else None,
        "accepted": len(accepted),
        "skipped_missing_images": len(skipped),
        "train_rows": len(train),
        "val_rows": len(val),
        "task_counts_accepted": dict(sorted(Counter(str(r["task_id"]) for r in accepted).items())),
        "task_counts_train": dict(sorted(Counter(str(r["task_id"]) for r in train).items())),
        "task_counts_val": dict(sorted(Counter(str(r["task_id"]) for r in val).items())),
    }
    write_json(out_dir / "prepare_summary.json", summary)
    return summary


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def referenced_images_for_audit(items: list[dict[str, Any]], dataset_dir: Path) -> list[str]:
    refs: set[str] = set()
    for item in items:
        refs.update(image_paths_for_item(item, dataset_dir))
    return sorted(refs)


def build_image_manifest(images_dir: Path, dataset_dir: Path) -> list[dict[str, Any]]:
    if not images_dir.exists():
        return []
    manifest: list[dict[str, Any]] = []
    for path in sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
        manifest.append(
            {
                "relative_path": path.relative_to(dataset_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return manifest


def build_dataset_audit(dataset_dir: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    refs = referenced_images_for_audit(items, dataset_dir)
    missing = [p for p in refs if not Path(p).exists()]
    return {
        "dataset_dir": str(dataset_dir),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_count": len(items),
        "task_counts": dict(sorted(Counter(str(item.get("task_id", "")) for item in items).items())),
        "image_reference_count": len(refs),
        "missing_image_reference_count": len(missing),
        "missing_image_references": missing,
        "image_manifest": build_image_manifest(dataset_dir / "images", dataset_dir),
    }


def percentile_nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]


def align_pixels_down(value: int, alignment: int = PIXEL_ALIGNMENT) -> int:
    if value <= 0:
        return 0
    return max(alignment, (value // alignment) * alignment)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def bad_image_paths(paths: list[str]) -> list[str]:
    bad: list[str] = []
    for image_path in paths:
        path = Path(image_path)
        if not path.exists():
            bad.append(image_path)
            continue
        try:
            image_size(path)
        except Exception:
            bad.append(image_path)
    return bad


def build_image_statistics(items: list[dict[str, Any]], dataset_dir: Path, percentile: int = 95) -> dict[str, Any]:
    image_cache: dict[str, dict[str, Any]] = {}
    sample_rows: list[dict[str, Any]] = []
    missing: set[str] = set()
    unreadable: set[str] = set()

    for item in items:
        paths = image_paths_for_item(item, dataset_dir)
        sample_pixels = 0
        existing_count = 0
        for image_path in paths:
            path = Path(image_path)
            if not path.exists():
                missing.add(image_path)
                continue
            if image_path not in image_cache:
                try:
                    width, height = image_size(path)
                except Exception:
                    unreadable.add(image_path)
                    continue
                image_cache[image_path] = {
                    "path": image_path,
                    "width": width,
                    "height": height,
                    "pixels": width * height,
                }
            sample_pixels += int(image_cache[image_path]["pixels"])
            existing_count += 1
        sample_rows.append(
            {
                "uid": str(item.get("uid", "")),
                "task_id": item.get("task_id"),
                "image_count": len(paths),
                "existing_image_count": existing_count,
                "total_pixels": sample_pixels,
            }
        )

    single_pixels = [int(row["pixels"]) for row in image_cache.values()]
    p_value = percentile_nearest_rank(single_pixels, percentile)
    max_pixels = align_pixels_down(p_value)
    multi_image_count = sum(1 for row in sample_rows if int(row["image_count"]) > 1)
    return {
        "image_count": len(single_pixels),
        "sample_count": len(items),
        "multi_image_sample_count": multi_image_count,
        "multi_image_sample_ratio": multi_image_count / len(items) if items else 0,
        "missing_image_count": len(missing),
        "unreadable_image_count": len(unreadable),
        "single_image_pixels_p95": p_value if percentile == 95 else None,
        f"single_image_pixels_p{percentile}": p_value,
        "recommended_max_pixels": max_pixels,
        "pixel_alignment": PIXEL_ALIGNMENT,
        "images": sorted(image_cache.values(), key=lambda x: x["path"]),
        "samples": sample_rows,
        "missing_images": sorted(missing),
        "unreadable_images": sorted(unreadable),
    }


def copy_data_backup(dataset_dir: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_dir / "data.json", backup_dir / "original_data.json")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    run_name = args.run_name or f"qwen25vl3b_llm_lora_drift_{dt.datetime.now().strftime('%Y%m%d')}"
    out_dir = Path(args.out_root) / run_name
    backup_dir = out_dir / "backup"
    prepared_dir = out_dir / "prepared"

    items = read_json(dataset_dir / "data.json")
    if not isinstance(items, list):
        raise TypeError(f"{dataset_dir / 'data.json'} must contain a list of samples")

    copy_data_backup(dataset_dir, backup_dir)
    audit = build_dataset_audit(dataset_dir, items)
    write_json(backup_dir / "dataset_audit_before.json", audit)

    task_ids = parse_task_ids(args.task_ids)
    filtered_items = [item for item in items if int(item.get("task_id", 0)) in task_ids]

    stats = build_image_statistics(filtered_items, dataset_dir, percentile=args.percentile)
    write_json(prepared_dir / "image_stats.json", stats)

    prompt_builder = load_prompt_builder(Path(args.project_root))
    split_summary = prepare_splits(
        items,
        dataset_dir,
        prepared_dir,
        prompt_builder,
        seed=args.seed,
        val_ratio=args.val_ratio,
        task_ids=task_ids,
    )
    metadata = {
        "run_name": run_name,
        "out_dir": str(out_dir),
        "dataset_dir": str(dataset_dir),
        "train_jsonl": str(prepared_dir / "swift_train.jsonl"),
        "val_jsonl": str(prepared_dir / "swift_val.jsonl"),
        "recommended_max_pixels": stats["recommended_max_pixels"],
        "task_ids": sorted(task_ids),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "split_summary": split_summary,
    }
    write_json(out_dir / "experiment_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--percentile", type=int, default=95)
    parser.add_argument("--task-ids", default="1,3,4,5,6,7")
    return parser.parse_args()


def parse_task_ids(value: str) -> set[int]:
    if not value.strip():
        return set(DEFAULT_TASK_IDS)
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def main() -> int:
    metadata = run(parse_args())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
