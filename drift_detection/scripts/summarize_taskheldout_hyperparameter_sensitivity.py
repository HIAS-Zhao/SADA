#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("@@WORKSPACE@@/drift_detection_exp")
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results_detection"
    / "task_heldout_detector_20260726"
    / "candidate_fold_scores.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results_detection"
    / "taskheldout_hyperparameter_sensitivity_20260726"
)
AXES = {
    "pca_dim": [64, 128, 192, 256],
    "window_size": [100, 150, 200, 300],
    "quantile": [0.95, 0.97, 0.99, 0.995],
}
MAIN = {
    "feature_group": "A+E_text",
    "pca_dim": 128,
    "window_size": 200,
    "quantile": 0.99,
}


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
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def equal_value(row: dict[str, str], key: str, expected: Any) -> bool:
    if key == "feature_group":
        return row[key] == expected
    return abs(float(row[key]) - float(expected)) <= 1e-12


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for axis, values in AXES.items():
        for value in values:
            selected = []
            for row in rows:
                if row["feature_group"] != MAIN["feature_group"]:
                    continue
                if not equal_value(row, axis, value):
                    continue
                if any(
                    not equal_value(row, key, expected)
                    for key, expected in MAIN.items()
                    if key not in {"feature_group", axis}
                ):
                    continue
                selected.append(row)
            heldout_tasks = sorted(int(row["heldout_task"]) for row in selected)
            if heldout_tasks != list(range(1, 8)):
                raise RuntimeError(
                    f"Incomplete one-axis slice for {axis}={value}: "
                    f"held-out tasks={heldout_tasks}"
                )

            def values_for(field: str) -> list[float]:
                return [float(row[field]) for row in selected]

            macro = values_for("test_macro_mean")
            fpr = values_for("test_fpr_mean")
            recall5 = values_for("test_recall_05_mean")
            recall10 = values_for("test_recall_10_mean")
            worst_macro_index = min(range(len(selected)), key=lambda i: macro[i])
            worst_r5_index = min(range(len(selected)), key=lambda i: recall5[i])
            out.append(
                {
                    "axis": axis,
                    "value": value,
                    "is_main_value": value == MAIN[axis],
                    "heldout_tasks": len(selected),
                    "macro_mean": statistics.mean(macro),
                    "macro_across_task_std": statistics.stdev(macro),
                    "fpr_mean": statistics.mean(fpr),
                    "recall_05_mean": statistics.mean(recall5),
                    "recall_10_mean": statistics.mean(recall10),
                    "worst_task_macro": macro[worst_macro_index],
                    "worst_task_macro_id": int(
                        selected[worst_macro_index]["heldout_task"]
                    ),
                    "worst_task_recall_05": recall5[worst_r5_index],
                    "worst_task_recall_05_id": int(
                        selected[worst_r5_index]["heldout_task"]
                    ),
                }
            )
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = {
        "pca_dim": "PCA dim.",
        "window_size": "Window",
        "quantile": "Quantile",
    }
    lines = [
        "# Task-Held-Out Detector Hyperparameter Sensitivity",
        "",
        "This is a one-at-a-time diagnostic around the fixed main setting "
        "`V+VT / PCA=128 / K=200 / q=0.99`. Each cell averages the seven "
        "held-out-task test folds; no setting in this table is used to retune "
        "the reported main detector.",
        "",
    ]
    for axis in AXES:
        lines.extend(
            [
                f"## {labels[axis]}",
                "",
                "| Value | Macro | FPR | R@5% | R@10% | Worst-task Macro | "
                "Worst-task R@5% |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [item for item in rows if item["axis"] == axis]:
            value = f"**{row['value']}**" if row["is_main_value"] else row["value"]
            lines.append(
                "| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} (T{}) | "
                "{:.2f} (T{}) |".format(
                    value,
                    100 * row["macro_mean"],
                    100 * row["fpr_mean"],
                    100 * row["recall_05_mean"],
                    100 * row["recall_10_mean"],
                    100 * row["worst_task_macro"],
                    row["worst_task_macro_id"],
                    100 * row["worst_task_recall_05"],
                    row["worst_task_recall_05_id"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    rows = summarize(read_csv(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "local_one_axis_sensitivity.csv", rows)
    write_json(
        args.output_dir / "protocol.json",
        {
            "input": str(args.input.resolve()),
            "main_setting": MAIN,
            "axes": AXES,
            "aggregation": (
                "Fixed one-at-a-time slices; arithmetic mean across seven "
                "held-out-task final-test folds."
            ),
            "selection_use": (
                "Diagnostic only. The held-out sensitivity table is not used "
                "to retune the fixed main detector."
            ),
        },
    )
    write_markdown(args.output_dir / "summary.md", rows)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
