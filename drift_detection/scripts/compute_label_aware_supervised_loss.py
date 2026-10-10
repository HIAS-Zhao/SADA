#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_DIR  # noqa: E402

COMMON_CODE_ROOT = PROJECT_ROOT.parent / "code"
if str(COMMON_CODE_ROOT) not in sys.path:
    sys.path.append(str(COMMON_CODE_ROOT))

from generic_inference import build_prompt  # noqa: E402

LOGGER = logging.getLogger("label_aware_loss")
DEFAULT_SPLITS = ["1_Threshold_Cal", "2_Stream_Sim_NoDrift", "2_Stream_Sim_Drift"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-sample teacher-forced supervised NLL for label-aware drift detectors."
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "dataset")
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "features_pooled_e_fusion")
    parser.add_argument(
        "--drift-data-json",
        type=Path,
        default=(
            PROJECT_ROOT.parent
            / "dataset_repair_reports"
            / "task01_agromind_mcq_text_merge_20260630_195249"
            / "backup"
            / "2_Stream_Sim_Drift"
            / "data.json"
        ),
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results_detection" / "label_aware_oracle_20260714" / "supervised_loss",
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-image-pixels", type=int, default=0)
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_feature_metadata(features_dir: Path, split_name: str) -> dict[str, Any]:
    path = features_dir / split_name / "features.pt"
    payload = torch.load(path, map_location="cpu")
    metadata = payload["metadata"]
    return {
        "uids": [str(uid) for uid in metadata["uids"]],
        "task_ids": [int(task_id) for task_id in metadata["task_ids"]],
        "ground_truths": [str(value) for value in metadata["ground_truths"]],
    }


def ordered_split_rows(
    dataset_root: Path,
    features_dir: Path,
    drift_data_json: Path,
    split_name: str,
) -> list[dict[str, Any]]:
    metadata = load_feature_metadata(features_dir, split_name)
    data_path = drift_data_json if split_name == "2_Stream_Sim_Drift" else dataset_root / split_name / "data.json"
    rows_by_uid = {str(row["uid"]): row for row in load_json(data_path)}

    missing = [uid for uid in metadata["uids"] if uid not in rows_by_uid]
    if missing:
        raise RuntimeError(f"{split_name}: {len(missing)} feature UIDs are missing from {data_path}: {missing[:5]}")

    ordered: list[dict[str, Any]] = []
    for uid, task_id, feature_ground_truth in zip(
        metadata["uids"],
        metadata["task_ids"],
        metadata["ground_truths"],
    ):
        row = dict(rows_by_uid[uid])
        row["uid"] = uid
        row["task_id"] = task_id
        data_ground_truth = str(row.get("ground_truth", ""))
        if data_ground_truth != feature_ground_truth:
            raise RuntimeError(
                f"{split_name}/{uid}: recovered ground truth does not match feature metadata "
                f"({data_ground_truth!r} != {feature_ground_truth!r})"
            )
        ordered.append(row)
    return ordered


def resolve_image_paths(row: dict[str, Any], split_dir: Path) -> list[Path]:
    task_id = int(row["task_id"])
    options = row.get("options")
    paths: list[str] = []

    if task_id == 1 and isinstance(options, dict):
        first_option = next(iter(options.values()), "")
        if isinstance(first_option, str) and (
            first_option.startswith("./images/")
            or first_option.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"))
        ):
            paths.extend(str(options[key]) for key in sorted(options))
    elif task_id == 2:
        paths.extend([str(row.get("pre_image_path", "")), str(row.get("post_image_path", ""))])
    else:
        paths.append(str(row.get("image_path", "")))

    resolved: list[Path] = []
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            normalized = raw_path[2:] if raw_path.startswith("./") else raw_path
            path = split_dir / normalized
        resolved.append(path)
    if not resolved:
        raise RuntimeError(f"{row['uid']}: no image paths found")
    return resolved


def load_images(paths: list[Path], max_image_pixels: int = 0) -> list[Image.Image]:
    images: list[Image.Image] = []
    resampling = getattr(Image, "Resampling", Image)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            converted = image.convert("RGB")
            if max_image_pixels > 0 and converted.width * converted.height > max_image_pixels:
                scale = math.sqrt(max_image_pixels / float(converted.width * converted.height))
                converted = converted.resize(
                    (
                        max(1, int(converted.width * scale)),
                        max(1, int(converted.height * scale)),
                    ),
                    resampling.LANCZOS,
                )
            images.append(converted)
    return images


def build_messages(row: dict[str, Any], images: list[Image.Image], include_answer: bool) -> list[dict[str, Any]]:
    options = row.get("options") if isinstance(row.get("options"), dict) else None
    sample_type = str(row.get("type", ""))
    if int(row["task_id"]) == 1:
        if isinstance(options, dict) and options:
            first_option = next(iter(options.values()), "")
            if isinstance(first_option, str) and (
                first_option.startswith("./images/")
                or first_option.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"))
            ):
                sample_type = "image_options"
            elif int(row.get("type_id") or 0) == 3:
                sample_type = "multi_mcq_opt_text"
            else:
                sample_type = "text_options"
        else:
            sample_type = "yesno"

    prompt = build_prompt(
        task_id=int(row["task_id"]),
        question=str(row.get("question", "")),
        options=options,
        sample_type=sample_type,
        type_id=row.get("type_id"),
    )
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(row.get("ground_truth", ""))}],
            }
        )
    return messages


def processor_inputs(
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    images: list[Image.Image],
    add_generation_prompt: bool,
    device: str,
) -> dict[str, torch.Tensor]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in inputs.items()}


def supervised_loss_for_row(
    row: dict[str, Any],
    split_dir: Path,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_image_pixels: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    images = load_images(resolve_image_paths(row, split_dir), max_image_pixels=max_image_pixels)
    prompt_messages = build_messages(row, images, include_answer=False)
    full_messages = build_messages(row, images, include_answer=True)

    prompt_inputs = processor_inputs(
        processor,
        prompt_messages,
        images,
        add_generation_prompt=True,
        device=device,
    )
    full_inputs = processor_inputs(
        processor,
        full_messages,
        images,
        add_generation_prompt=False,
        device=device,
    )

    prompt_ids = prompt_inputs["input_ids"]
    full_ids = full_inputs["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    if full_ids.shape[1] <= prompt_len:
        raise RuntimeError(
            f"{row['uid']}: answer added no tokens (prompt={prompt_len}, full={int(full_ids.shape[1])})"
        )
    if not torch.equal(prompt_ids[0], full_ids[0, :prompt_len]):
        mismatch = (prompt_ids[0] != full_ids[0, :prompt_len]).nonzero(as_tuple=True)[0]
        raise RuntimeError(f"{row['uid']}: prompt/full token prefix mismatch at {mismatch[:5].tolist()}")

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    special_ids = set(getattr(processor.tokenizer, "all_special_ids", []))
    if special_ids:
        answer_ids = labels[:, prompt_len:]
        for token_id in special_ids:
            answer_ids[answer_ids == int(token_id)] = -100
    answer_tokens = int((labels != -100).sum().item())
    if answer_tokens <= 0:
        raise RuntimeError(f"{row['uid']}: no supervised answer tokens remain after masking")

    model_inputs = {key: value for key, value in full_inputs.items() if key != "input_ids"}
    with torch.inference_mode():
        outputs = model(input_ids=full_ids, labels=labels, **model_inputs)
    loss = float(outputs.loss.detach().float().cpu().item())

    del outputs, labels, prompt_inputs, full_inputs, full_ids, prompt_ids, images
    return {
        "uid": str(row["uid"]),
        "task_id": int(row["task_id"]),
        "mean_nll": loss,
        "perplexity": float(torch.exp(torch.tensor(min(loss, 20.0))).item()),
        "answer_tokens": answer_tokens,
        "elapsed_s": float(time.perf_counter() - started),
    }


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "mean_nll" in row:
                completed.add(str(row["uid"]))
    return completed


def select_rows(rows: list[dict[str, Any]], limit_per_task: int, max_samples: int) -> list[dict[str, Any]]:
    if limit_per_task > 0:
        counts: Counter[int] = Counter()
        selected: list[dict[str, Any]] = []
        for row in rows:
            task_id = int(row["task_id"])
            if counts[task_id] >= limit_per_task:
                continue
            selected.append(row)
            counts[task_id] += 1
        rows = selected
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    task_rows: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        task_rows.setdefault(int(row["task_id"]), []).append(row)
    summary = {
        "samples": len(rows),
        "mean_elapsed_s": (
            sum(float(row["elapsed_s"]) for row in rows) / len(rows) if rows else None
        ),
        "tasks": {
            str(task_id): {
                "samples": len(values),
                "mean_nll": sum(float(row["mean_nll"]) for row in values) / len(values),
                "mean_answer_tokens": (
                    sum(int(row["answer_tokens"]) for row in values) / len(values)
                ),
            }
            for task_id, values in sorted(task_rows.items())
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s on %s", args.model_dir, args.device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(args.model_dir))

    invocation = {
        "dataset_root": str(args.dataset_root),
        "features_dir": str(args.features_dir),
        "drift_data_json": str(args.drift_data_json),
        "model_dir": str(args.model_dir),
        "splits": args.splits,
        "device": args.device,
        "max_image_pixels": args.max_image_pixels,
        "limit_per_task": args.limit_per_task,
        "max_samples": args.max_samples,
    }
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        existing = load_json(config_path)
        invocations = list(existing.get("invocations", [existing]))
        invocations.append(invocation)
    else:
        invocations = [invocation]
    run_config = {
        "signal": "teacher_forced_mean_answer_nll",
        "invocations": invocations,
    }
    config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name in args.splits:
        output_path = args.output_dir / f"{split_name}.jsonl"
        if args.overwrite and output_path.exists():
            output_path.unlink()
        completed = load_completed(output_path)
        rows = ordered_split_rows(
            dataset_root=args.dataset_root,
            features_dir=args.features_dir,
            drift_data_json=args.drift_data_json,
            split_name=split_name,
        )
        rows = select_rows(rows, args.limit_per_task, args.max_samples)
        pending = [row for row in rows if str(row["uid"]) not in completed]
        split_dir = args.dataset_root / split_name

        LOGGER.info(
            "%s: target=%d completed=%d pending=%d",
            split_name,
            len(rows),
            len(completed),
            len(pending),
        )
        with output_path.open("a", encoding="utf-8") as output_file:
            for row in tqdm(pending, desc=split_name):
                try:
                    result = supervised_loss_for_row(
                        row=row,
                        split_dir=split_dir,
                        model=model,
                        processor=processor,
                        device=args.device,
                        max_image_pixels=args.max_image_pixels,
                    )
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                except Exception as error:
                    error_row = {
                        "uid": str(row["uid"]),
                        "task_id": int(row["task_id"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                    output_file.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                    output_file.flush()
                    LOGGER.exception("%s/%s failed", split_name, row["uid"])
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        parsed_rows = []
        with output_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                row = json.loads(line)
                if "mean_nll" in row:
                    parsed_rows.append(row)
        write_summary(args.output_dir / f"{split_name}_summary.json", parsed_rows)

    LOGGER.info("Supervised-loss extraction complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
