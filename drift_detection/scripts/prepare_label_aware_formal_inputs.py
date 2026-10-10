#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.label_aware_formal import (  # noqa: E402
    absolutize_row_paths,
    answer_format,
    task_counts,
)


TASKS_BY_SPLIT = {
    "1_Threshold_Cal": {8, 9},
    "2_Stream_Sim_NoDrift": {8, 9},
    "2_Stream_Sim_Drift": {1, 3, 5, 6, 7},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare feature-aligned closed-set inputs for strict Qwen evaluation."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "strict_inputs"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_metadata(features_dir: Path, split_name: str) -> dict[str, list[Any]]:
    payload = torch.load(
        features_dir / split_name / "features.pt",
        map_location="cpu",
    )
    metadata = payload["metadata"]
    return {
        "uids": [str(value) for value in metadata["uids"]],
        "task_ids": [int(value) for value in metadata["task_ids"]],
        "ground_truths": [value for value in metadata["ground_truths"]],
    }


def verify_images(row: dict[str, Any]) -> None:
    paths = []
    for key in ["image_path", "pre_image_path", "post_image_path"]:
        value = row.get(key)
        if isinstance(value, str) and value:
            paths.append(Path(value))
    options = row.get("options")
    if isinstance(options, dict):
        for value in options.values():
            if isinstance(value, str) and Path(value).is_absolute():
                paths.append(Path(value))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{row['uid']}: missing image paths: {missing[:5]}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "features_dir": str(args.features_dir),
        "drift_data_json": str(args.drift_data_json),
        "tasks_by_split": {
            split_name: sorted(tasks)
            for split_name, tasks in TASKS_BY_SPLIT.items()
        },
        "splits": {},
    }

    for split_name, allowed_tasks in TASKS_BY_SPLIT.items():
        metadata = feature_metadata(args.features_dir, split_name)
        data_path = (
            args.drift_data_json
            if split_name == "2_Stream_Sim_Drift"
            else args.dataset_root / split_name / "data.json"
        )
        rows_by_uid = {
            str(row["uid"]): row
            for row in read_json(data_path)
        }
        selected = []
        manifest = []
        image_root = args.dataset_root / split_name

        for feature_index, (uid, task_id, ground_truth) in enumerate(
            zip(
                metadata["uids"],
                metadata["task_ids"],
                metadata["ground_truths"],
            )
        ):
            if task_id not in allowed_tasks:
                continue
            if uid not in rows_by_uid:
                raise KeyError(f"{split_name}: feature UID {uid} is missing from {data_path}")
            row = absolutize_row_paths(rows_by_uid[uid], image_root=image_root)
            if str(row.get("ground_truth", "")) != str(ground_truth):
                raise ValueError(
                    f"{split_name}/{uid}: ground truth mismatch "
                    f"{row.get('ground_truth')!r} != {ground_truth!r}"
                )
            row["task_id"] = task_id
            verify_images(row)
            selected.append(row)
            manifest.append(
                {
                    "uid": uid,
                    "task_id": task_id,
                    "answer_format": answer_format(row),
                    "feature_index": feature_index,
                }
            )

        split_dir = args.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "data.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (split_dir / "uid_manifest.jsonl").open("w", encoding="utf-8") as file:
            for row in manifest:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        config["splits"][split_name] = {
            "source_data": str(data_path),
            "samples": len(selected),
            "task_counts": task_counts(row["task_id"] for row in selected),
            "format_counts": {
                format_name: sum(
                    answer_format(row) == format_name
                    for row in selected
                )
                for format_name in sorted(
                    {answer_format(row) for row in selected}
                )
            },
        }

    (args.output_dir / "prepare_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
