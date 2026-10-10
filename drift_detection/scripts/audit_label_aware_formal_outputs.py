#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS_METHODS = {"ADWIN", "Page-Hinkley", "KSWIN", "OPTWIN"}
DRIFT_TASKS = [1, 3, 5, 6, 7]
TASK_MACRO_RATIOS = [0.05, 0.10, 0.20]
ORDERINGS = [
    "shuffled",
    "block_1",
    "block_2",
    "block_5",
    "block_10",
    "block_20",
    "contiguous",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit completeness and invariants of formal label-aware outputs."
    )
    parser.add_argument(
        "--closed-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "closed_set"
        ),
    )
    parser.add_argument(
        "--heterogeneous-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "heterogeneous"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "audit"
        ),
    )
    parser.add_argument(
        "--skip-heterogeneous",
        action="store_true",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def row_key(row: dict[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def check_unique(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    errors: list[str],
    label: str,
) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = row_key(row, fields)
        if key in seen:
            errors.append(f"{label}: duplicate run key {key}")
        seen.add(key)


def check_rates_and_delays(
    rows: Iterable[dict[str, Any]],
    errors: list[str],
    label: str,
) -> None:
    for row in rows:
        key = (
            row.get("method"),
            row.get("signal"),
            row.get("seed"),
            row.get("ratio", row.get("budget", row.get("ordering"))),
        )
        for name in ["detection_rate", "prechange_alarm_window_rate"]:
            value = row.get(name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                errors.append(f"{label}: {name} out of range for {key}: {value}")
        n_drift = int(row.get("n_drift") or 0)
        conditional = row.get("conditional_mean_delay")
        censored = row.get("censored_mean_delay")
        if (
            conditional is not None
            and n_drift > 0
            and not 0.0 < float(conditional) <= max(n_drift, 200)
        ):
            errors.append(
                f"{label}: conditional delay out of range for {key}: {conditional}"
            )
        if (
            censored is not None
            and n_drift > 0
            and not 0.0 < float(censored) <= max(n_drift, 200)
        ):
            errors.append(
                f"{label}: censored delay out of range for {key}: {censored}"
            )


def selected_methods(
    audit: dict[str, Any],
    target_fpr: float,
    errors: list[str],
    label: str,
) -> set[str]:
    selected = set()
    for method, row in audit.items():
        status = row.get("status")
        if status == "selected":
            selected_row = row.get("selected") or {}
            fpr = selected_row.get("calibration_fpr")
            if fpr is None or float(fpr) > target_fpr + 1e-12:
                errors.append(
                    f"{label}/{method}: selected calibration FPR {fpr} "
                    f"exceeds target {target_fpr}"
                )
            selected.add(method)
        elif status != "not_calibratable":
            errors.append(f"{label}/{method}: invalid status {status!r}")
    return selected


def require_run_keys(
    rows: Sequence[dict[str, Any]],
    expected: set[tuple[Any, ...]],
    fields: Sequence[str],
    errors: list[str],
    label: str,
) -> None:
    observed = {row_key(row, fields) for row in rows}
    missing = sorted(expected - observed, key=str)
    extra = sorted(observed - expected, key=str)
    if missing:
        errors.append(f"{label}: missing {len(missing)} keys; first={missing[:5]}")
    if extra:
        errors.append(f"{label}: unexpected {len(extra)} keys; first={extra[:5]}")


def audit_closed(closed_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    config = read_json(closed_dir / "run_config.json")
    main_runs = read_json(closed_dir / "main_runs.json")
    selection_audit = read_json(closed_dir / "selection_audit.json")
    seeds = [int(value) for value in config["seeds"]]
    ratios = [float(value) for value in config["main_ratios"]]
    target_fpr = float(config["target_fpr"])

    check_unique(
        main_runs,
        ["weighting", "signal", "method", "seed", "ratio"],
        errors,
        "closed/main",
    )
    check_rates_and_delays(main_runs, errors, "closed/main")
    expected_main: set[tuple[Any, ...]] = set()
    selected_by_setting: dict[str, set[str]] = {}
    for setting, audit in selection_audit.items():
        weighting, signal = setting.split("/", maxsplit=1)
        methods = selected_methods(
            audit,
            target_fpr,
            errors,
            f"closed/{setting}",
        )
        selected_by_setting[setting] = methods
        expected_methods = set(methods)
        if signal == "binary_error":
            expected_methods.add("SADA")
        for method in expected_methods:
            for seed in seeds:
                for ratio in ratios:
                    expected_main.add(
                        (weighting, signal, method, seed, ratio)
                    )
    require_run_keys(
        main_runs,
        expected_main,
        ["weighting", "signal", "method", "seed", "ratio"],
        errors,
        "closed/main",
    )

    budget_runs = read_json(closed_dir / "budget_runs.json")
    budgets = [int(value) for value in config["budgets"]]
    check_unique(
        budget_runs,
        ["signal", "method", "seed", "budget"],
        errors,
        "closed/budget",
    )
    check_rates_and_delays(budget_runs, errors, "closed/budget")
    expected_budget_methods = {"SADA"}
    expected_budget_methods.update(
        method
        for method in selected_by_setting["natural/binary_error"]
        if method not in CONTINUOUS_METHODS
    )
    expected_budget_methods.update(
        selected_by_setting["natural/correct_option_nll"]
    )
    expected_budget = {
        (
            "method_native"
            if method == "SADA"
            else (
                "binary_error"
                if method not in CONTINUOUS_METHODS
                else "correct_option_nll"
            ),
            method,
            seed,
            budget,
        )
        for method in expected_budget_methods
        for seed in seeds
        for budget in budgets
    }
    require_run_keys(
        budget_runs,
        expected_budget,
        ["signal", "method", "seed", "budget"],
        errors,
        "closed/budget",
    )

    task_runs = read_json(closed_dir / "task_macro_runs.json")
    check_unique(
        task_runs,
        ["signal", "method", "task_id", "seed", "ratio"],
        errors,
        "closed/task_macro",
    )
    check_rates_and_delays(task_runs, errors, "closed/task_macro")
    expected_task_methods = {"SADA"}
    expected_task_methods.update(
        method
        for method in selected_by_setting["task_balanced/binary_error"]
        if method not in CONTINUOUS_METHODS
    )
    expected_task_methods.update(
        selected_by_setting["task_balanced/correct_option_nll"]
    )
    expected_task = {
        (
            "method_native"
            if method == "SADA"
            else (
                "binary_error"
                if method not in CONTINUOUS_METHODS
                else "correct_option_nll"
            ),
            method,
            task_id,
            seed,
            ratio,
        )
        for method in expected_task_methods
        for task_id in DRIFT_TASKS
        for seed in seeds
        for ratio in TASK_MACRO_RATIOS
    }
    require_run_keys(
        task_runs,
        expected_task,
        ["signal", "method", "task_id", "seed", "ratio"],
        errors,
        "closed/task_macro",
    )

    ordering_runs = read_json(closed_dir / "ordering_runs.json")
    check_unique(
        ordering_runs,
        ["signal", "method", "seed", "ordering"],
        errors,
        "closed/ordering",
    )
    check_rates_and_delays(ordering_runs, errors, "closed/ordering")
    expected_ordering = {
        (
            "method_native"
            if method == "SADA"
            else (
                "binary_error"
                if method not in CONTINUOUS_METHODS
                else "correct_option_nll"
            ),
            method,
            seed,
            ordering,
        )
        for method in expected_budget_methods
        for seed in seeds
        for ordering in ORDERINGS
    }
    require_run_keys(
        ordering_runs,
        expected_ordering,
        ["signal", "method", "seed", "ordering"],
        errors,
        "closed/ordering",
    )

    permutation = read_json(closed_dir / "sada_permutation_invariance.json")
    if int(permutation["detection_disagreement_windows"]) != 0:
        errors.append("closed/permutation: SADA has detection disagreements")
    if float(permutation["max_score_span"]) > 1e-10:
        errors.append(
            "closed/permutation: max score span exceeds 1e-10: "
            f"{permutation['max_score_span']}"
        )

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "counts": {
            "main_runs": len(main_runs),
            "budget_runs": len(budget_runs),
            "task_macro_runs": len(task_runs),
            "ordering_runs": len(ordering_runs),
        },
        "permutation": permutation,
    }


def audit_heterogeneous(heterogeneous_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    config = read_json(heterogeneous_dir / "run_config.json")
    runs = read_json(heterogeneous_dir / "runs.json")
    selection_audit = read_json(heterogeneous_dir / "selection_audit.json")
    signal_audit = read_json(heterogeneous_dir / "signal_audit.json")
    variants = sorted(config["variants"])
    weightings = sorted(
        config.get(
            "weightings",
            {str(row["weighting"]) for row in runs},
        )
    )
    seeds = [int(value) for value in config["seeds"]]
    ratios = [float(value) for value in config["ratios"]]
    target_fpr = float(config["target_fpr"])

    check_unique(
        runs,
        ["variant", "weighting", "signal", "method", "seed", "ratio"],
        errors,
        "heterogeneous/runs",
    )
    check_rates_and_delays(runs, errors, "heterogeneous/runs")
    expected: set[tuple[Any, ...]] = set()
    expected_settings = {
        f"{variant}/{weighting}/{signal}"
        for variant in variants
        for weighting in weightings
        for signal in ["global_nll", "format_aware_nll"]
    }
    if set(selection_audit) != expected_settings:
        missing = sorted(expected_settings - set(selection_audit))
        extra = sorted(set(selection_audit) - expected_settings)
        errors.append(
            "heterogeneous/selection settings mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    if set(signal_audit) != expected_settings:
        missing = sorted(expected_settings - set(signal_audit))
        extra = sorted(set(signal_audit) - expected_settings)
        errors.append(
            "heterogeneous/signal settings mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    for setting in sorted(expected_settings):
        variant, weighting, signal = setting.split("/", maxsplit=2)
        audit = selection_audit.get(setting, {})
        methods = selected_methods(
            audit,
            target_fpr,
            errors,
            f"heterogeneous/{setting}",
        )
        methods.add("SADA")
        for method in methods:
            for seed in seeds:
                for ratio in ratios:
                    expected.add(
                        (variant, weighting, signal, method, seed, ratio)
                    )
    require_run_keys(
        runs,
        expected,
        ["variant", "weighting", "signal", "method", "seed", "ratio"],
        errors,
        "heterogeneous/runs",
    )

    for variant, sizes in config["variant_sizes"].items():
        for split in ["calibration", "no_drift", "drift_validation", "drift_test"]:
            if int(sizes[split]) <= 0:
                errors.append(
                    f"heterogeneous/{variant}: non-positive {split} size"
                )

    for setting, stats in signal_audit.items():
        for split, count_by_group in stats.get("fallback_counts", {}).items():
            if any(int(value) < 0 for value in count_by_group.values()):
                errors.append(
                    f"heterogeneous/{setting}: negative fallback count in {split}"
                )

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "counts": {
            "runs": len(runs),
            "selection_settings": len(selection_audit),
            "signal_settings": len(signal_audit),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "closed_set": audit_closed(args.closed_dir),
    }
    if not args.skip_heterogeneous:
        result["heterogeneous"] = audit_heterogeneous(args.heterogeneous_dir)
    overall_errors = [
        error
        for section in result.values()
        for error in section["errors"]
    ]
    result["overall_status"] = "pass" if not overall_errors else "fail"
    result["overall_error_count"] = len(overall_errors)
    write_json(args.output_dir / "formal_output_audit.json", result)

    lines = [
        "# Formal Label-Aware Output Audit",
        "",
        f"- Overall status: **{result['overall_status']}**",
        f"- Error count: {len(overall_errors)}",
    ]
    for name, section in result.items():
        if not isinstance(section, dict) or "status" not in section:
            continue
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"- Status: {section['status']}",
                f"- Counts: `{json.dumps(section.get('counts', {}), sort_keys=True)}`",
            ]
        )
        for error in section["errors"]:
            lines.append(f"- ERROR: {error}")
    (args.output_dir / "formal_output_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    if overall_errors:
        raise SystemExit(1)
    print(f"Formal output audit passed: {args.output_dir}")


if __name__ == "__main__":
    main()
