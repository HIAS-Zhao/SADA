#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("@@WORKSPACE@@/drift_lora_project")
EXPERIMENT_ROOT = (
    PROJECT_ROOT / "results" / "qwen25_target80_admission_gate_3s_20260914"
)
SOURCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "dtop80_current_pool_recompute_20260726"
    / "all_model_rtest_only_formal_profiles_new_manifest"
)
DELAY_ROOT = (
    PROJECT_ROOT
    / "results"
    / "upstream_delay_sensitivity_formal_new_manifest_0to5s_20260907"
)
TRUE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "v2_teacher_closed_loop_20260630_181348"
    / "step5_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924"
    / "inputs"
    / "true_5group_rwindow_root"
)
STATS_SCRIPT = (
    PROJECT_ROOT
    / "results"
    / "v2_teacher_closed_loop_20260630_181348"
    / "step6_v2_5group_framework_fisher5_rwindow_formal_methods_20260703_160924"
    / "oracle_grid_completion_20260706"
    / "recompute_final_statistics.py"
)
SELECTOR_SCRIPT = (
    PROJECT_ROOT / "scripts" / "select_qwen25_target80_admission_candidates.py"
)
SEEDS = (42, 43, 44)
DELAY_S = 3.0
TIE_TOLERANCE = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_stats() -> Any:
    spec = importlib.util.spec_from_file_location("target80_stats", STATS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {STATS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_selector() -> Any:
    spec = importlib.util.spec_from_file_location(
        "target80_selector",
        SELECTOR_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SELECTOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def index_reports(root: Path) -> tuple[
    dict[tuple[str, str], Path],
    dict[tuple[str, str, str], Path],
]:
    base: dict[tuple[str, str], Path] = {}
    adapter: dict[tuple[str, str, str], Path] = {}
    for report in root.rglob("strict_eval_report.json"):
        parts = report.parts
        if "base" in parts:
            index = len(parts) - 1 - tuple(reversed(parts)).index("base")
            if index + 2 < len(parts):
                base[(parts[index + 1], parts[index + 2])] = report
        if "adapter" in parts:
            index = len(parts) - 1 - tuple(reversed(parts)).index("adapter")
            if index + 3 < len(parts):
                adapter[
                    (parts[index + 1], parts[index + 2], parts[index + 3])
                ] = report
    return base, adapter


def correct_count(
    predictions: dict[str, dict[str, Any]],
    uids: list[str],
) -> int:
    return sum(
        int(
            uid in predictions
            and bool(predictions[uid].get("accepted"))
            and bool(predictions[uid].get("correct"))
        )
        for uid in uids
    )


def source_detail_by_rate(seed: int) -> dict[float, list[dict[str, str]]]:
    root = (
        SOURCE_ROOT
        / f"seed{seed}"
        / "pressure"
        / "normalized_load_runs"
        / "qwen25"
    )
    result: dict[float, list[dict[str, str]]] = {}
    for manifest_path in root.glob("*/pressure_recompute_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rate = float(manifest["arrival_rate"])
        result[rate] = read_csv(manifest_path.parent / "group_detail_native.csv")
    return result


def closest_rate_rows(
    rows_by_rate: dict[float, list[dict[str, str]]],
    rate: float,
) -> list[dict[str, str]]:
    matched = min(rows_by_rate, key=lambda value: abs(value - rate))
    if not math.isclose(matched, rate, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError(f"No source scenario matching arrival rate {rate}")
    return rows_by_rate[matched]


def load_existing_summary() -> tuple[dict[str, float], dict[str, float]]:
    rows = read_csv(DELAY_ROOT / "ours_vs_baselines_by_scenario.csv")
    baselines: dict[str, float] = {}
    original_ours: dict[str, float] = {}
    for row in rows:
        if (
            row["family"] == "qwen25"
            and row["scenario_type"] == "normalized"
            and math.isclose(float(row["delay_s"]), DELAY_S)
        ):
            scenario = row["scenario"]
            original_ours[scenario] = float(row["ours_utility"])
            baselines[scenario] = max(
                baselines.get(scenario, -math.inf),
                float(row["baseline_utility"]),
            )
    return baselines, original_ours


def main() -> None:
    stats = load_stats()
    selector = load_selector()
    detail_out: list[dict[str, Any]] = []
    gate_only_out: list[dict[str, Any]] = []
    prediction_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    def predictions(report: Path) -> dict[str, dict[str, Any]]:
        if report not in prediction_cache:
            prediction_cache[report] = stats.prediction_map(report)
        return prediction_cache[report]

    for seed in SEEDS:
        selection_rows = read_csv(
            EXPERIMENT_ROOT / f"seed{seed}" / "admission_policy_selections.csv"
        )
        source_rates = source_detail_by_rate(seed)
        original_root = (
            PROJECT_ROOT
            / "results"
            / "all_model_three_seed_repeats_20260715"
            / f"seed{seed}"
            / "qwen25"
            / "step6"
        )
        base_reports, _ = index_reports(original_root)
        formal_rtest_base, _ = index_reports(
            original_root / "true_accuracy_full_r" / "ar0p05"
        )
        base_reports.update(formal_rtest_base)
        _, target_reports = index_reports(
            EXPERIMENT_ROOT
            / f"seed{seed}"
            / "true_accuracy_target80_selected_rtest"
        )
        original_gammas = selector.read_json(
            selector.SOURCE_ROOT
            / f"seed{seed}"
            / "qwen25"
            / "step5"
            / "gamma_pareto"
            / "gamma_group_profiles_pareto.json"
        )
        gammas_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in original_gammas:
            gammas_by_group[str(row["group_id"])].append(row)
        lambda_rows = selector.read_json(selector.LAMBDA_PROFILES)
        lambdas_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in lambda_rows:
            lambdas_by_group[str(row["group_id"])].append(row)

        for selection in selection_rows:
            rate = float(selection["arrival_rate"])
            load = float(selection["load_factor"])
            group_id = selection["group_id"]
            lambda_id = selection["lambda_id"]
            gamma_id = selection["gamma_id"]
            coverage = float(selection["coverage"])
            uids = [
                str(row["uid"])
                for row in json.loads(
                    (
                        TRUE_ROOT
                        / "groups"
                        / group_id
                        / "R_test"
                        / "data.json"
                    ).read_text(encoding="utf-8")
                )
            ]
            source_rows = closest_rate_rows(source_rates, rate)
            no_retrain = next(
                row
                for row in source_rows
                if row["方法代码"] == "NoRetrain" and row["组ID"] == group_id
            )
            pre_report = Path(no_retrain["base报告"])
            base_report = base_reports.get((group_id, lambda_id))
            if base_report is None:
                raise FileNotFoundError(
                    f"Missing base report: seed={seed} {group_id}/{lambda_id}"
                )
            pre_preds = predictions(pre_report)
            base_preds = predictions(base_report)

            if gamma_id:
                adapter_report = target_reports.get(
                    (group_id, gamma_id, lambda_id)
                )
                if adapter_report is None:
                    raise FileNotFoundError(
                        "Missing target80 report: "
                        f"seed={seed} {group_id}/{gamma_id}/{lambda_id}"
                    )
                adapter_preds = predictions(adapter_report)
                delay_count = min(len(uids), math.ceil(rate * DELAY_S))
                ready_s = DELAY_S + float(selection["finish_time_s"])
                ready_count = min(len(uids), math.ceil(rate * ready_s))
                pre_uids = uids[:delay_count]
                base_uids = uids[delay_count:ready_count]
                adapter_uids = uids[ready_count:]
                strict_correct = (
                    correct_count(pre_preds, pre_uids)
                    + coverage
                    * (
                        correct_count(base_preds, base_uids)
                        + correct_count(adapter_preds, adapter_uids)
                    )
                )
            else:
                adapter_report = None
                delay_count = 0
                ready_count = len(uids)
                pre_uids = []
                base_uids = uids
                adapter_uids = []
                strict_correct = coverage * correct_count(base_preds, base_uids)

            detail_out.append(
                {
                    "seed": seed,
                    "load_factor": load,
                    "scenario": (
                        f"load{load:g}x".replace(".", "p")
                    ),
                    "arrival_rate": rate,
                    "group_id": group_id,
                    "lambda_id": lambda_id,
                    "gamma_id": gamma_id,
                    "I": selection["I"],
                    "R": selection["R"],
                    "coverage": coverage,
                    "delay_samples": delay_count,
                    "ready_samples": ready_count,
                    "adapter_samples": len(adapter_uids),
                    "pre_correct": correct_count(pre_preds, pre_uids),
                    "base_correct": correct_count(base_preds, base_uids),
                    "adapter_correct": (
                        correct_count(predictions(adapter_report), adapter_uids)
                        if adapter_report is not None
                        else 0
                    ),
                    "actual_utility": strict_correct / len(uids),
                    "base_report": str(base_report),
                    "adapter_report": (
                        str(adapter_report) if adapter_report is not None else ""
                    ),
                }
            )

        for load in selector.LOAD_FACTORS:
            rate = selector.REFERENCE_THROUGHPUT * load
            horizon_s = 110.0 / rate
            for group_id, lambdas in lambdas_by_group.items():
                selected = selector.select_group(
                    lambdas=lambdas,
                    gammas=gammas_by_group[group_id],
                    arrival_rate=rate,
                    horizon_s=horizon_s,
                    fallback=load >= 1.5,
                    upstream_delay_s=DELAY_S,
                    min_active_fraction=0.20,
                    min_gain=0.005,
                )
                if selected["gamma_id"]:
                    raise RuntimeError(
                        "Expected the original-profile admission gate to "
                        f"reject training: seed={seed} load={load} "
                        f"group={group_id}"
                    )
                report = base_reports.get((group_id, selected["lambda_id"]))
                if report is None:
                    raise FileNotFoundError(
                        "Missing gate-only base report: "
                        f"seed={seed} {group_id}/{selected['lambda_id']}"
                    )
                uids = [
                    str(row["uid"])
                    for row in json.loads(
                        (
                            TRUE_ROOT
                            / "groups"
                            / group_id
                            / "R_test"
                            / "data.json"
                        ).read_text(encoding="utf-8")
                    )
                ]
                gate_only_out.append(
                    {
                        "seed": seed,
                        "load_factor": load,
                        "scenario": f"load{load:g}x".replace(".", "p"),
                        "group_id": group_id,
                        "lambda_id": selected["lambda_id"],
                        "coverage": selected["coverage"],
                        "actual_utility": (
                            selected["coverage"]
                            * correct_count(predictions(report), uids)
                            / len(uids)
                        ),
                        "base_report": str(report),
                    }
                )

    seed_load: list[dict[str, Any]] = []
    grouped: dict[tuple[int, float, str], list[float]] = defaultdict(list)
    for row in detail_out:
        grouped[
            (row["seed"], row["load_factor"], row["scenario"])
        ].append(float(row["actual_utility"]))
    for (seed, load, scenario), values in sorted(grouped.items()):
        seed_load.append(
            {
                "seed": seed,
                "load_factor": load,
                "scenario": scenario,
                "ours_utility": sum(values) / len(values),
            }
        )

    baselines, original_ours = load_existing_summary()
    gate_only_by_scenario: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in gate_only_out:
        gate_only_by_scenario[
            (row["load_factor"], row["scenario"])
        ].append(float(row["actual_utility"]))
    summary: list[dict[str, Any]] = []
    by_scenario: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in seed_load:
        by_scenario[(row["load_factor"], row["scenario"])].append(
            row["ours_utility"]
        )
    wins = ties = losses = 0
    for (load, scenario), values in sorted(by_scenario.items()):
        ours = sum(values) / len(values)
        gate_only = sum(gate_only_by_scenario[(load, scenario)]) / len(
            gate_only_by_scenario[(load, scenario)]
        )
        original = original_ours[scenario]
        baseline = baselines[scenario]
        margin = ours - baseline
        outcome = (
            "win"
            if margin > TIE_TOLERANCE
            else "loss"
            if margin < -TIE_TOLERANCE
            else "tie"
        )
        wins += outcome == "win"
        ties += outcome == "tie"
        losses += outcome == "loss"
        summary.append(
            {
                "load_factor": load,
                "scenario": scenario,
                "original_fixed3_utility": original,
                "ours_utility": ours,
                "delta_vs_original_fixed3": ours - original,
                "gate_only_utility": gate_only,
                "target80_increment": ours - gate_only,
                "strongest_baseline": baseline,
                "margin": margin,
                "outcome": outcome,
            }
        )

    write_csv(EXPERIMENT_ROOT / "target80_group_actual_results.csv", detail_out)
    write_csv(EXPERIMENT_ROOT / "gate_only_group_actual_results.csv", gate_only_out)
    write_csv(EXPERIMENT_ROOT / "target80_seed_load_results.csv", seed_load)
    write_csv(EXPERIMENT_ROOT / "target80_normalized_summary.csv", summary)
    overall = {
        "delay_s": DELAY_S,
        "mean_ours_utility": sum(row["ours_utility"] for row in summary)
        / len(summary),
        "mean_original_fixed3_utility": sum(
            row["original_fixed3_utility"] for row in summary
        )
        / len(summary),
        "delta_vs_original_fixed3": sum(
            row["delta_vs_original_fixed3"] for row in summary
        )
        / len(summary),
        "mean_gate_only_utility": sum(
            row["gate_only_utility"] for row in summary
        )
        / len(summary),
        "target80_increment": sum(
            row["target80_increment"] for row in summary
        )
        / len(summary),
        "mean_strongest_baseline": sum(
            row["strongest_baseline"] for row in summary
        )
        / len(summary),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }
    (EXPERIMENT_ROOT / "target80_overall_summary.json").write_text(
        json.dumps(overall, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
