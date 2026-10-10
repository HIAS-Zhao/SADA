#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REFERENCE_TASK = "perception_a1_2_land_cover_classification"
DRIFT_TASKS = ",".join(
    [
        "reasoning_b1_1_spatial_relationship_reasoning",
        "reasoning_b3_2_disaster_cause_inference",
    ]
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


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def stable_order(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (stable_int(f"{seed}:{row['uid']}"), str(row["uid"])))


def task_key_for(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return str(meta.get("task_key") or item.get("task_key") or item.get("task_id"))


def label_for(item: dict[str, Any]) -> str:
    return str(item.get("ground_truth") or item.get("gt") or item.get("answer") or "")


def grouped_by_task(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[task_key_for(item)].append(item)
    return dict(grouped)


def allocate_stratified(rows: list[dict[str, Any]], count: int, seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count == 0:
        return [], list(rows)
    if count >= len(rows):
        return stable_order(rows, f"{seed}:all"), []

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[label_for(row)].append(row)
    labels = sorted(by_label)
    for label in labels:
        by_label[label] = stable_order(by_label[label], f"{seed}:label:{label}")

    allocations = {label: int(math.floor(count * len(by_label[label]) / len(rows))) for label in labels}
    remaining = count - sum(allocations.values())
    remainders = sorted(
        (
            count * len(by_label[label]) / len(rows) - allocations[label],
            stable_int(f"{seed}:rem:{label}"),
            label,
        )
        for label in labels
    )
    for _fraction, _tie, label in reversed(remainders):
        if remaining <= 0:
            break
        if allocations[label] < len(by_label[label]):
            allocations[label] += 1
            remaining -= 1

    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label in labels:
        for row in by_label[label][: allocations[label]]:
            picked.append(row)
            seen.add(str(row["uid"]))

    if len(picked) < count:
        for row in stable_order(rows, f"{seed}:fill"):
            uid = str(row["uid"])
            if uid in seen:
                continue
            picked.append(row)
            seen.add(uid)
            if len(picked) == count:
                break

    rest = [row for row in rows if str(row["uid"]) not in seen]
    return stable_order(picked, f"{seed}:picked"), stable_order(rest, f"{seed}:rest")


def annotate(item: dict[str, Any], split_name: str, *, group_id: str = "", role: str = "") -> dict[str, Any]:
    row = dict(item)
    meta = dict(row.get("meta") if isinstance(row.get("meta"), dict) else {})
    meta.update(
        {
            "formal_split": split_name,
            "formal_group_id": group_id,
            "formal_role": role,
        }
    )
    row["meta"] = meta
    row["task_key"] = task_key_for(item)
    return row


def norm_path(path_value: Any, dataset_dir: Path) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    return str((dataset_dir / str(path_value).lstrip("./")).resolve())


def prompt_for_item(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    question = str(item.get("question") or meta.get("question") or "")
    options = item.get("options")
    if not isinstance(options, dict):
        options = meta.get("options") if isinstance(meta.get("options"), dict) else {}
    if not options:
        return f"{question}\n\nProvide a concise answer based on the image."
    rendered = "\n".join(f"{key}: {value}" for key, value in sorted(options.items()))
    return f"{question}\n\nOptions:\n{rendered}\n\nAnswer with just the letter."


def image_paths_for_item(item: dict[str, Any], dataset_dir: Path) -> list[str]:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    out = []
    for key in ("pre_image_path", "post_image_path", "image_path"):
        value = item.get(key) or meta.get(key)
        path = norm_path(value, dataset_dir)
        if path:
            out.append(path)
    seen = set()
    unique = []
    for path in out:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def swift_row(item: dict[str, Any], dataset_dir: Path, *, label_key: str = "ground_truth") -> tuple[dict[str, Any] | None, list[str]]:
    images = image_paths_for_item(item, dataset_dir)
    missing = [path for path in images if not Path(path).exists()]
    if missing:
        return None, missing
    image_tokens = "<image>" * len(images)
    prompt = prompt_for_item(item)
    answer = str(item.get(label_key) or item.get("ground_truth") or item.get("gt") or "").strip()
    return (
        {
            "messages": [
                {"role": "user", "content": f"{image_tokens}\n{prompt}" if image_tokens else prompt},
                {"role": "assistant", "content": answer},
            ],
            "images": images,
            "uid": str(item.get("uid", "")),
            "task_id": int(item.get("task_id", 0)),
            "task_key": task_key_for(item),
            "formal_split": item.get("meta", {}).get("formal_split", ""),
            "formal_group_id": item.get("meta", {}).get("formal_group_id", ""),
        },
        [],
    )


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(path / "data.json", rows)


def write_swift(path: Path, rows: list[dict[str, Any]], dataset_dir: Path) -> list[dict[str, Any]]:
    swift_rows = []
    skipped = []
    for row in rows:
        converted, missing = swift_row(row, dataset_dir)
        if converted is None:
            skipped.append({"uid": row.get("uid", ""), "task_key": task_key_for(row), "missing_images": missing})
        else:
            swift_rows.append(converted)
    write_jsonl(path / "data.jsonl", swift_rows)
    write_jsonl(path / "skipped_missing_images.jsonl", skipped)
    return skipped


def split_mixed(no_rows: list[dict[str, Any]], drift_rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    if len(no_rows) != len(drift_rows):
        raise ValueError("mixed windows require equal no-drift and drift counts")
    rng = random.Random(seed)
    indexes = list(range(len(no_rows)))
    rng.shuffle(indexes)
    out: list[dict[str, Any]] = []
    for index in indexes:
        pair = [no_rows[index], drift_rows[index]]
        if rng.random() < 0.5:
            pair.reverse()
        out.extend(pair)
    return out


def split_r_window(rows: list[dict[str, Any]], train_weight: int, val_weight: int, test_weight: int) -> dict[str, list[dict[str, Any]]]:
    total_weight = train_weight + val_weight + test_weight
    train_n = int(round(len(rows) * train_weight / total_weight))
    val_n = int(round(len(rows) * val_weight / total_weight))
    if train_n + val_n >= len(rows):
        train_n = max(1, len(rows) - 2)
        val_n = 1
    return {
        "R_train": rows[:train_n],
        "R_val": rows[train_n : train_n + val_n],
        "R_test": rows[train_n + val_n :],
    }


def uid_rows(split_name: str, rows: list[dict[str, Any]], *, group_id: str = "", role: str = "") -> list[dict[str, Any]]:
    return [
        {
            "uid": str(row["uid"]),
            "task_key": task_key_for(row),
            "split": split_name,
            "group_id": group_id,
            "role": role,
            "ground_truth": label_for(row),
        }
        for row in rows
    ]


def assert_no_duplicate_leaf_uids(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["uid"] for row in rows)
    duplicates = sorted(uid for uid, count in counts.items() if count > 1)
    return {
        "leaf_rows": len(rows),
        "unique_leaf_uids": len(counts),
        "duplicate_leaf_uids": duplicates,
        "passed": not duplicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build formal Qwen3.5 OmniEarth framework splits.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-task", default=REFERENCE_TASK)
    parser.add_argument("--drift-tasks", default=DRIFT_TASKS)
    parser.add_argument("--part0-fit-count", type=int, default=60)
    parser.add_argument("--part0-threshold-count", type=int, default=40)
    parser.add_argument("--teacher-train-per-drift", type=int, default=120)
    parser.add_argument("--teacher-test-per-drift", type=int, default=40)
    parser.add_argument("--d-per-group", type=int, default=25)
    parser.add_argument("--r-per-group", type=int, default=25)
    parser.add_argument("--r-train-weight", type=int, default=180)
    parser.add_argument("--r-val-weight", type=int, default=100)
    parser.add_argument("--r-test-weight", type=int, default=110)
    parser.add_argument("--d-top-fraction", type=float, default=0.40)
    parser.add_argument("--seed", default="qwen35_omniearth_formal_v1")
    args = parser.parse_args()

    items = read_json(args.dataset_dir / "data.json")
    if not isinstance(items, list):
        raise TypeError(f"{args.dataset_dir / 'data.json'} must contain a list")
    grouped = grouped_by_task(items)
    drift_tasks = parse_csv(args.drift_tasks)
    missing = [key for key in [args.reference_task, *drift_tasks] if key not in grouped]
    if missing:
        raise KeyError(f"Dataset missing required tasks: {missing}")

    reference_rows = stable_order(grouped[args.reference_task], f"{args.seed}:reference")
    no_needed = args.part0_fit_count + args.part0_threshold_count + len(drift_tasks) * (
        args.d_per_group + args.r_per_group
    )
    if no_needed > len(reference_rows):
        raise ValueError(f"Need {no_needed} no-drift rows but only found {len(reference_rows)}")

    part0_fit, rest_no = allocate_stratified(reference_rows, args.part0_fit_count, f"{args.seed}:part0:fit")
    part0_threshold, rest_no = allocate_stratified(rest_no, args.part0_threshold_count, f"{args.seed}:part0:threshold")

    part0_fit = [annotate(row, "part0_reference_fit", role="no_drift_reference_fit") for row in part0_fit]
    part0_threshold = [annotate(row, "part0_threshold_cal", role="no_drift_threshold_cal") for row in part0_threshold]

    teacher_train_all: list[dict[str, Any]] = []
    teacher_test_all: list[dict[str, Any]] = []
    leaf_membership = []
    leaf_membership.extend(uid_rows("part0_reference_fit", part0_fit, role="no_drift_reference_fit"))
    leaf_membership.extend(uid_rows("part0_threshold_cal", part0_threshold, role="no_drift_threshold_cal"))

    groups_summary = []
    for drift_task in drift_tasks:
        group_id = f"{args.reference_task}__{drift_task}"
        drift_rows = stable_order(grouped[drift_task], f"{args.seed}:drift:{drift_task}")
        drift_needed = (
            args.teacher_train_per_drift
            + args.teacher_test_per_drift
            + args.d_per_group
            + args.r_per_group
        )
        if drift_needed > len(drift_rows):
            raise ValueError(f"Need {drift_needed} rows for {drift_task} but only found {len(drift_rows)}")

        d_no, rest_no = allocate_stratified(rest_no, args.d_per_group, f"{args.seed}:{group_id}:D:no")
        r_no, rest_no = allocate_stratified(rest_no, args.r_per_group, f"{args.seed}:{group_id}:R:no")
        teacher_train, rest_drift = allocate_stratified(
            drift_rows,
            args.teacher_train_per_drift,
            f"{args.seed}:{group_id}:teacher_train",
        )
        teacher_test, rest_drift = allocate_stratified(
            rest_drift,
            args.teacher_test_per_drift,
            f"{args.seed}:{group_id}:teacher_test",
        )
        d_drift, rest_drift = allocate_stratified(rest_drift, args.d_per_group, f"{args.seed}:{group_id}:D:drift")
        r_drift, _unused_drift = allocate_stratified(rest_drift, args.r_per_group, f"{args.seed}:{group_id}:R:drift")

        teacher_train = [annotate(row, "part1_teacher_ft_train", group_id=group_id, role="teacher_train_drift") for row in teacher_train]
        teacher_test = [annotate(row, "part1_teacher_ft_test", group_id=group_id, role="teacher_test_drift") for row in teacher_test]
        d_no = [annotate(row, "part2_D_no_drift", group_id=group_id, role="D_no_drift") for row in d_no]
        d_drift = [annotate(row, "part2_D_drift", group_id=group_id, role="D_drift") for row in d_drift]
        r_no = [annotate(row, "part3_R_no_drift", group_id=group_id, role="R_no_drift") for row in r_no]
        r_drift = [annotate(row, "part3_R_drift", group_id=group_id, role="R_drift") for row in r_drift]

        d_window = split_mixed(d_no, d_drift, f"{args.seed}:{group_id}:D_window")
        r_all = split_mixed(r_no, r_drift, f"{args.seed}:{group_id}:R_all")
        r_splits = split_r_window(r_all, args.r_train_weight, args.r_val_weight, args.r_test_weight)

        group_dir = args.output_dir / "groups" / group_id
        write_dataset(group_dir / "part2_detection" / "D_no_drift", d_no)
        write_dataset(group_dir / "part2_detection" / "D_drift", d_drift)
        write_dataset(group_dir / "part2_detection" / "D_window", d_window)
        write_dataset(group_dir / "part3_retraining" / "R_no_drift", r_no)
        write_dataset(group_dir / "part3_retraining" / "R_drift", r_drift)
        write_dataset(group_dir / "part3_retraining" / "R_all", r_all)
        for name, rows in r_splits.items():
            write_dataset(group_dir / "part3_retraining" / name, rows)

        teacher_train_all.extend(teacher_train)
        teacher_test_all.extend(teacher_test)
        leaf_membership.extend(uid_rows("part1_teacher_ft_train", teacher_train, group_id=group_id, role="teacher_train_drift"))
        leaf_membership.extend(uid_rows("part1_teacher_ft_test", teacher_test, group_id=group_id, role="teacher_test_drift"))
        leaf_membership.extend(uid_rows("part2_D_no_drift", d_no, group_id=group_id, role="D_no_drift"))
        leaf_membership.extend(uid_rows("part2_D_drift", d_drift, group_id=group_id, role="D_drift"))
        leaf_membership.extend(uid_rows("part3_R_no_drift", r_no, group_id=group_id, role="R_no_drift"))
        leaf_membership.extend(uid_rows("part3_R_drift", r_drift, group_id=group_id, role="R_drift"))

        groups_summary.append(
            {
                "group_id": group_id,
                "reference_task": args.reference_task,
                "drift_task": drift_task,
                "teacher_train": len(teacher_train),
                "teacher_test": len(teacher_test),
                "D_no_drift": len(d_no),
                "D_drift": len(d_drift),
                "D_window": len(d_window),
                "D_top40_planned": int(math.ceil(len(d_window) * args.d_top_fraction)),
                "R_no_drift": len(r_no),
                "R_drift": len(r_drift),
                "R_all": len(r_all),
                "R_train": len(r_splits["R_train"]),
                "R_val": len(r_splits["R_val"]),
                "R_test": len(r_splits["R_test"]),
            }
        )

    write_dataset(args.output_dir / "part0_reference" / "reference_fit", part0_fit)
    write_dataset(args.output_dir / "part0_reference" / "threshold_cal", part0_threshold)
    write_dataset(args.output_dir / "part1_teacher_ft" / "train", teacher_train_all)
    write_dataset(args.output_dir / "part1_teacher_ft" / "teacher_test", teacher_test_all)
    teacher_skipped = write_swift(args.output_dir / "part1_teacher_ft" / "swift_train", teacher_train_all, args.dataset_dir)
    teacher_test_skipped = write_swift(args.output_dir / "part1_teacher_ft" / "swift_teacher_test", teacher_test_all, args.dataset_dir)

    leakage = assert_no_duplicate_leaf_uids(leaf_membership)
    write_csv(args.output_dir / "leaf_membership.csv", leaf_membership)
    manifest = {
        "dataset_dir": args.dataset_dir.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "reference_task": args.reference_task,
        "drift_tasks": drift_tasks,
        "seed": args.seed,
        "counts": {
            "part0_reference_fit": len(part0_fit),
            "part0_threshold_cal": len(part0_threshold),
            "teacher_train_total": len(teacher_train_all),
            "teacher_test_total": len(teacher_test_all),
            "teacher_swift_skipped": len(teacher_skipped),
            "teacher_test_swift_skipped": len(teacher_test_skipped),
            "unused_no_drift_rows": len(rest_no),
        },
        "groups": groups_summary,
        "D_top40": {
            "status": "pending_detector_scores",
            "fraction": args.d_top_fraction,
            "candidate_universe": "complete_D_window",
            "note": "D_top40 is selected after fitting the Part0 detector, scoring every sample in each complete D window, and retaining the highest-scoring 40%.",
        },
        "leakage_check": leakage,
    }
    write_json(args.output_dir / "split_manifest.json", manifest)

    lines = [
        "# Qwen3.5 OmniEarth Formal Framework Splits",
        "",
        f"- dataset_dir: `{args.dataset_dir}`",
        f"- reference_task: `{args.reference_task}`",
        f"- drift_tasks: `{', '.join(drift_tasks)}`",
        f"- leakage_check: `{'pass' if leakage['passed'] else 'fail'}`",
        f"- D_top40: pending detector scores over each complete D window, fraction `{args.d_top_fraction}`",
        "",
        "| part/group | train/test/D/R counts |",
        "|---|---|",
        f"| Part0 reference | fit={len(part0_fit)}, threshold_cal={len(part0_threshold)} |",
        f"| Part1 teacher | train={len(teacher_train_all)}, teacher_test={len(teacher_test_all)} |",
    ]
    for row in groups_summary:
        lines.append(
            "| {group_id} | D={D_no_drift}+{D_drift}, planned D_top40={D_top40_planned}, R={R_train}/{R_val}/{R_test}, teacher={teacher_train}/{teacher_test} |".format(
                **row
            )
        )
    (args.output_dir / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "split_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
