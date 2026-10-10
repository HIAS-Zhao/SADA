from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def answer_format(row: dict[str, Any]) -> str:
    task_id = int(row["task_id"])
    sample_type = str(row.get("type") or "").strip().lower()
    type_id = row.get("type_id")
    if task_id == 1 and (int(type_id or 0) == 4 or sample_type in {"yesno", "yn_opt_none"}):
        return "yes_no"
    if task_id == 4:
        return "yes_no"
    if task_id in {1, 3, 5, 6, 7, 8, 9}:
        return "mcq"
    if task_id == 10:
        return "caption"
    if task_id == 11:
        return "open_short"
    if task_id == 2:
        return "long_report"
    return "unknown"


def format_calibration_group(format_name: str) -> str:
    if format_name in {"mcq", "yes_no"}:
        return "closed_set"
    if format_name == "open_short":
        return "open_short"
    if format_name in {"caption", "long_report"}:
        return "long_generation"
    return format_name


def normalize_label(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item).strip().upper() for item in value)
    return str(value).strip().casefold()


def parse_option_scores(raw_output: str) -> dict[str, float]:
    payload = json.loads(raw_output)
    if payload.get("method") != "logprob" or not isinstance(payload.get("scores"), dict):
        raise ValueError("strict prediction does not contain option log-probability scores")
    scores = {
        str(label).strip(): float(value)
        for label, value in payload["scores"].items()
    }
    if not scores or not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("strict prediction contains empty or non-finite option scores")
    return scores


def correct_option_nll(scores: dict[str, float], target: Any) -> float:
    normalized_target = normalize_label(target)
    matched = [
        score
        for label, score in scores.items()
        if normalize_label(label) == normalized_target
    ]
    if len(matched) != 1:
        raise ValueError(
            f"target {target!r} matched {len(matched)} labels in {sorted(scores)}"
        )
    values = np.asarray(list(scores.values()), dtype=np.float64)
    maximum = float(values.max())
    log_partition = maximum + math.log(float(np.exp(values - maximum).sum()))
    return float(log_partition - matched[0])


def supervised_margin_loss(scores: dict[str, float], target: Any) -> float:
    normalized_target = normalize_label(target)
    target_scores = [
        score
        for label, score in scores.items()
        if normalize_label(label) == normalized_target
    ]
    wrong_scores = [
        score
        for label, score in scores.items()
        if normalize_label(label) != normalized_target
    ]
    if len(target_scores) != 1 or not wrong_scores:
        raise ValueError(f"cannot compute supervised margin for target {target!r}")
    return float(max(wrong_scores) - target_scores[0])


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if reference.size == 0:
        raise ValueError("ECDF reference must be non-empty")
    return np.searchsorted(reference, values, side="right") / reference.size


def format_aware_ecdf(
    calibration_values: np.ndarray,
    calibration_formats: Sequence[str],
    values: np.ndarray,
    formats: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    calibration_values = np.asarray(calibration_values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    calibration_groups = np.asarray(
        [format_calibration_group(value) for value in calibration_formats],
        dtype=object,
    )
    value_groups = np.asarray(
        [format_calibration_group(value) for value in formats],
        dtype=object,
    )
    references = {
        str(group): calibration_values[calibration_groups == group]
        for group in np.unique(calibration_groups)
    }
    missing: dict[str, int] = {}
    output = np.empty(values.shape, dtype=np.float64)
    global_reference = calibration_values
    for group in np.unique(value_groups):
        mask = value_groups == group
        reference = references.get(str(group))
        if reference is None or reference.size == 0:
            missing[str(group)] = int(mask.sum())
            reference = global_reference
        output[mask] = empirical_cdf(reference, values[mask])
    return output, missing


def _sample_natural(
    pool_size: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if pool_size <= 0:
        raise ValueError("cannot sample from an empty pool")
    return rng.choice(pool_size, size=count, replace=pool_size < count)


def _sample_task_balanced(
    task_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    by_task: dict[int, np.ndarray] = {
        int(task_id): np.flatnonzero(task_ids == task_id)
        for task_id in np.unique(task_ids)
    }
    if not by_task:
        raise ValueError("cannot task-balance an empty pool")
    tasks = np.asarray(sorted(by_task), dtype=np.int64)
    selected_tasks = rng.choice(tasks, size=count, replace=True)
    output = np.empty(count, dtype=np.int64)
    for task_id in tasks:
        positions = np.flatnonzero(selected_tasks == task_id)
        if positions.size:
            choices = by_task[int(task_id)]
            output[positions] = rng.choice(
                choices,
                size=positions.size,
                replace=choices.size < positions.size,
            )
    return output


def sample_indices(
    task_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
    weighting: str,
) -> np.ndarray:
    task_ids = np.asarray(task_ids, dtype=np.int64)
    if weighting == "natural":
        return _sample_natural(len(task_ids), count, rng)
    if weighting == "task_balanced":
        return _sample_task_balanced(task_ids, count, rng)
    raise ValueError(f"unsupported weighting: {weighting}")


@dataclass(frozen=True)
class WindowPlan:
    clean_indices: np.ndarray
    drift_indices: np.ndarray
    order: np.ndarray
    n_clean: int
    n_drift: int


def build_window_plans(
    clean_task_ids: np.ndarray,
    drift_task_ids: np.ndarray,
    n_clean: int,
    n_drift: int,
    count: int,
    seed: int,
    weighting: str,
    layout: str = "contiguous_tail",
    block_size: int | None = None,
) -> list[WindowPlan]:
    rng = np.random.default_rng(seed)
    plans: list[WindowPlan] = []
    for _ in range(count):
        clean_indices = sample_indices(clean_task_ids, n_clean, rng, weighting)
        drift_indices = sample_indices(drift_task_ids, n_drift, rng, weighting)
        order = build_layout_order(
            n_clean=n_clean,
            n_drift=n_drift,
            layout=layout,
            rng=rng,
            block_size=block_size,
        )
        plans.append(
            WindowPlan(
                clean_indices=clean_indices,
                drift_indices=drift_indices,
                order=order,
                n_clean=n_clean,
                n_drift=n_drift,
            )
        )
    return plans


def build_layout_order(
    n_clean: int,
    n_drift: int,
    layout: str,
    rng: np.random.Generator,
    block_size: int | None = None,
) -> np.ndarray:
    total = n_clean + n_drift
    if layout == "contiguous_tail":
        return np.arange(total, dtype=np.int64)
    if layout == "shuffled":
        return rng.permutation(total)
    if layout != "block":
        raise ValueError(f"unsupported layout: {layout}")
    if block_size is None or block_size <= 0:
        raise ValueError("block layout requires a positive block_size")
    if n_drift == 0:
        return np.arange(n_clean, dtype=np.int64)

    clean_positions = list(range(n_clean))
    drift_positions = list(range(n_clean, total))
    drift_blocks = [
        drift_positions[start : start + block_size]
        for start in range(0, n_drift, block_size)
    ]
    clean_blocks = np.array_split(
        np.asarray(clean_positions, dtype=np.int64),
        len(drift_blocks) + 1,
    )
    order: list[int] = []
    for index, clean_block in enumerate(clean_blocks):
        order.extend(int(value) for value in clean_block)
        if index < len(drift_blocks):
            order.extend(drift_blocks[index])
    return np.asarray(order, dtype=np.int64)


def materialize_signal_window(
    clean_values: np.ndarray,
    drift_values: np.ndarray,
    plan: WindowPlan,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if plan.n_clean:
        parts.append(np.asarray(clean_values)[plan.clean_indices])
    if plan.n_drift:
        parts.append(np.asarray(drift_values)[plan.drift_indices])
    combined = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    return combined[plan.order]


def reorder_window_plans(
    plans: Sequence[WindowPlan],
    layout: str,
    seed: int,
    block_size: int | None = None,
) -> list[WindowPlan]:
    rng = np.random.default_rng(seed)
    return [
        WindowPlan(
            clean_indices=plan.clean_indices,
            drift_indices=plan.drift_indices,
            order=build_layout_order(
                n_clean=plan.n_clean,
                n_drift=plan.n_drift,
                layout=layout,
                rng=rng,
                block_size=block_size,
            ),
            n_clean=plan.n_clean,
            n_drift=plan.n_drift,
        )
        for plan in plans
    ]


def task_counts(task_ids: Iterable[int]) -> dict[str, int]:
    counts: dict[int, int] = defaultdict(int)
    for task_id in task_ids:
        counts[int(task_id)] += 1
    return {str(task_id): counts[task_id] for task_id in sorted(counts)}


def absolutize_row_paths(row: dict[str, Any], image_root: Path) -> dict[str, Any]:
    output = dict(row)

    def resolve(raw: Any) -> Any:
        if not isinstance(raw, str) or not raw:
            return raw
        path = Path(raw)
        if path.is_absolute():
            return str(path)
        normalized = raw[2:] if raw.startswith("./") else raw
        return str((image_root / normalized).resolve())

    for key in ["image_path", "pre_image_path", "post_image_path"]:
        if key in output:
            output[key] = resolve(output[key])
    options = output.get("options")
    if isinstance(options, dict):
        resolved_options = {}
        for label, value in options.items():
            if isinstance(value, str) and (
                value.startswith("./images/")
                or Path(value).suffix.lower() in IMAGE_SUFFIXES
            ):
                resolved_options[label] = resolve(value)
            else:
                resolved_options[label] = value
        output["options"] = resolved_options
    return output
