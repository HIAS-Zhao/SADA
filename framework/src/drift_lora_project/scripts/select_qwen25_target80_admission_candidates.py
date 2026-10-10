#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("@@WORKSPACE@@/drift_lora_project")
SOURCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "all_model_three_seed_repeats_20260715"
)
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "qwen25_target80_admission_gate_3s_20260914"
)
LAMBDA_PROFILES = (
    PROJECT_ROOT
    / "results"
    / "v2_teacher_closed_loop_20260630_181348"
    / "step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924"
    / "lambda_analysis_rval"
    / "lambda_profiles_group_v2.json"
)
LOAD_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
REFERENCE_THROUGHPUT = 1.5385600458967544


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
        writer.writerows(
            {key: row.get(key, "") for key in fieldnames}
            for row in rows
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def link_authoritative_base_reports(
    *,
    seed: int,
    experiment_root: Path,
    evaluation_rows: list[dict[str, Any]],
) -> None:
    manifest = (
        SOURCE_ROOT
        / f"seed{seed}"
        / "qwen25"
        / "step6"
        / "seed_pareto_grid"
        / "true_accuracy_pareto_grid"
        / "scheduler_true_accuracy_error_rows.csv"
    )
    source_by_key: dict[tuple[str, str], Path] = {}
    for row in read_csv(manifest):
        report = Path(row["base_report"])
        if report.exists():
            source_by_key[(row["group_id"], row["lambda_id"])] = report
    for row in evaluation_rows:
        key = (str(row["group_id"]), str(row["lambda_id"]))
        source = source_by_key.get(key)
        if source is None:
            raise FileNotFoundError(f"Missing authoritative base report: {key}")
        target = (
            experiment_root
            / f"seed{seed}"
            / "true_accuracy_target80_selected_rtest"
            / "base"
            / key[0]
            / key[1]
            / "strict_eval_report.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve())


def decision_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["estimated_utility"],
        row["future_accuracy"],
        row["current_accuracy"],
        row["lambda_accuracy"],
        -row["lambda_cost"],
        row["R"],
    )


def global_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["estimated_utility"],
        row["future_accuracy"],
        row["current_accuracy"],
        row["R"],
    )


def select_group(
    *,
    lambdas: list[dict[str, Any]],
    gammas: list[dict[str, Any]],
    arrival_rate: float,
    horizon_s: float,
    fallback: bool,
    upstream_delay_s: float,
    min_active_fraction: float,
    min_gain: float,
) -> dict[str, Any]:
    grid = {index / 100.0 for index in range(1, 100)}
    for lambda_row in lambdas:
        grid.add(
            min(
                1.0,
                arrival_rate
                / float(lambda_row["throughput_samples_per_sec"]),
            )
        )
    for gamma_row in gammas:
        grid.add(max(0.0, 1.0 - float(gamma_row["resource_cost"])))

    best_by_resource: list[dict[str, Any]] = []
    for inference_resource in sorted(
        value for value in grid if 0.0 < value <= 1.0
    ):
        training_resource = 1.0 - inference_resource
        candidates: list[dict[str, Any]] = []
        no_train: list[dict[str, Any]] = []
        for lambda_row in lambdas:
            lambda_cost = (
                arrival_rate
                / float(lambda_row["throughput_samples_per_sec"])
            )
            if not fallback and lambda_cost > inference_resource + 1e-9:
                continue
            coverage = min(
                1.0,
                inference_resource / max(lambda_cost, 1e-9),
            )
            lambda_accuracy = float(
                lambda_row.get("accuracy")
                or lambda_row.get("teacher_accuracy")
            )
            base_accuracy = lambda_accuracy * coverage
            base = {
                "lambda_id": lambda_row["lambda_id"],
                "gamma_id": "",
                "I": inference_resource,
                "R": training_resource,
                "finish_time_s": "",
                "active_fraction": 0.0,
                "current_accuracy": base_accuracy,
                "future_accuracy": base_accuracy,
                "estimated_utility": base_accuracy,
                "lambda_accuracy": lambda_accuracy,
                "lambda_cost": lambda_cost,
                "coverage": coverage,
                "is_target80": False,
            }
            candidates.append(base)
            no_train.append(base)
            for gamma_row in gammas:
                if (
                    training_resource + 1e-9
                    < float(gamma_row["resource_cost"])
                ):
                    continue
                train_time = float(
                    gamma_row.get("train_time")
                    or gamma_row.get("train_time_s")
                )
                finish_time = train_time / max(training_resource, 1e-9)
                gain = float(
                    gamma_row.get("predicted_final_gain")
                    or gamma_row.get("profile_gain")
                    or 0.0
                )
                updated_accuracy = (
                    min(1.0, lambda_accuracy + gain) * coverage
                )
                if finish_time <= horizon_s:
                    old_ratio = finish_time / horizon_s
                    current_accuracy = (
                        old_ratio * base_accuracy
                        + (1.0 - old_ratio) * updated_accuracy
                    )
                else:
                    current_accuracy = base_accuracy
                active_fraction = max(
                    0.0,
                    1.0
                    - (finish_time + upstream_delay_s) / horizon_s,
                )
                candidates.append(
                    {
                        "lambda_id": lambda_row["lambda_id"],
                        "gamma_id": gamma_row["gamma_id"],
                        "I": inference_resource,
                        "R": training_resource,
                        "finish_time_s": finish_time,
                        "active_fraction": active_fraction,
                        "current_accuracy": current_accuracy,
                        "future_accuracy": updated_accuracy,
                        "estimated_utility": current_accuracy,
                        "lambda_accuracy": lambda_accuracy,
                        "lambda_cost": lambda_cost,
                        "coverage": coverage,
                        "is_target80": str(
                            gamma_row["gamma_id"]
                        ).endswith("target80"),
                    }
                )
        if not no_train:
            continue
        best_no_train = max(no_train, key=decision_key)
        eligible = [
            row
            for row in candidates
            if not row["gamma_id"]
            or (
                row["active_fraction"] >= min_active_fraction
                and row["estimated_utility"]
                - best_no_train["estimated_utility"]
                >= min_gain
            )
        ]
        best_by_resource.append(max(eligible, key=decision_key))
    if not best_by_resource:
        raise RuntimeError("No feasible scheduler decision")
    return max(best_by_resource, key=global_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--upstream-delay-s", type=float, default=3.0)
    parser.add_argument("--min-active-fraction", type=float, default=0.20)
    parser.add_argument("--min-gain", type=float, default=0.005)
    args = parser.parse_args()

    lambda_rows = read_json(LAMBDA_PROFILES)
    lambdas_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in lambda_rows:
        lambdas_by_group.setdefault(str(row["group_id"]), []).append(row)

    for seed in [int(value) for value in args.seeds.split(",") if value]:
        original = read_json(
            SOURCE_ROOT
            / f"seed{seed}"
            / "qwen25"
            / "step5"
            / "gamma_pareto"
            / "gamma_group_profiles_pareto.json"
        )
        target80 = read_json(
            args.experiment_root
            / f"seed{seed}"
            / "target80_gamma_profiles.json"
        )
        gammas_by_group: dict[str, list[dict[str, Any]]] = {}
        for row in original + target80:
            gammas_by_group.setdefault(str(row["group_id"]), []).append(row)

        selections: list[dict[str, Any]] = []
        for load_factor in LOAD_FACTORS:
            arrival_rate = REFERENCE_THROUGHPUT * load_factor
            horizon_s = 110.0 / arrival_rate
            fallback = load_factor >= 1.5
            for group_id, group_lambdas in lambdas_by_group.items():
                selected = select_group(
                    lambdas=group_lambdas,
                    gammas=gammas_by_group[group_id],
                    arrival_rate=arrival_rate,
                    horizon_s=horizon_s,
                    fallback=fallback,
                    upstream_delay_s=args.upstream_delay_s,
                    min_active_fraction=args.min_active_fraction,
                    min_gain=args.min_gain,
                )
                selections.append(
                    {
                        "seed": seed,
                        "load_factor": load_factor,
                        "arrival_rate": arrival_rate,
                        "horizon_s": horizon_s,
                        "fallback_scheduler": fallback,
                        "group_id": group_id,
                        **selected,
                    }
                )
        seed_root = args.experiment_root / f"seed{seed}"
        write_csv(seed_root / "admission_policy_selections.csv", selections)

        unique_target80: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in selections:
            if not row["is_target80"]:
                continue
            key = (row["group_id"], row["gamma_id"], row["lambda_id"])
            unique_target80.setdefault(
                key,
                {
                    "method": "Target80AdmissionSelected",
                    "group_id": row["group_id"],
                    "lambda_id": row["lambda_id"],
                    "gamma_id": row["gamma_id"],
                    "I": row["I"],
                    "R": row["R"],
                    "adapter_accepted": "True",
                    "T_retrain_s": row["horizon_s"],
                    "adapter_ready_time_s": row["finish_time_s"],
                    "train_time_s_effective": row["finish_time_s"],
                    "current_window_accuracy": row["current_accuracy"],
                    "long_avg_accuracy": row["estimated_utility"],
                },
            )
        eval_rows = list(unique_target80.values())
        write_csv(seed_root / "target80_selected_eval_scheduler.csv", eval_rows)
        link_authoritative_base_reports(
            seed=seed,
            experiment_root=args.experiment_root,
            evaluation_rows=eval_rows,
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "policy_rows": len(selections),
                    "selected_target80_eval_rows": len(eval_rows),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
