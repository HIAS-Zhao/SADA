#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_label_aware_detector_protocol import (  # noqa: E402
    Candidate,
    RiverAdapter,
    river_candidates,
)
from src.label_aware_detectors import (  # noqa: E402
    OPTWIN,
    detects_after_onset,
    detects_in_window,
    warm_detector,
)
from src.label_aware_formal import (  # noqa: E402
    WindowPlan,
    build_window_plans,
    correct_option_nll,
    materialize_signal_window,
    parse_option_scores,
    reorder_window_plans,
    supervised_margin_loss,
)
from src.mahalanobis_detector import MahalanobisDetector  # noqa: E402


MAIN_RATIOS = [0.0, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
BUDGETS = [10, 20, 40, 50, 100, 200]
TASK_MACRO_RATIOS = [0.05, 0.10, 0.20]
DRIFT_TASKS = [1, 3, 5, 6, 7]
BINARY_METHODS = [
    "DDM",
    "EDDM",
    "ADWIN",
    "HDDM-W",
    "Page-Hinkley",
    "KSWIN",
    "OPTWIN",
]
CONTINUOUS_METHODS = [
    "ADWIN",
    "Page-Hinkley",
    "KSWIN",
    "OPTWIN",
]


@dataclass
class ClosedSetPool:
    split_name: str
    uids: np.ndarray
    task_ids: np.ndarray
    errors: np.ndarray
    option_nll: np.ndarray
    margin_loss: np.ndarray
    features: dict[str, np.ndarray]

    def subset(self, mask: np.ndarray) -> "ClosedSetPool":
        return ClosedSetPool(
            split_name=self.split_name,
            uids=self.uids[mask],
            task_ids=self.task_ids[mask],
            errors=self.errors[mask],
            option_nll=self.option_nll[mask],
            margin_loss=self.margin_loss[mask],
            features={
                key: values[mask]
                for key, values in self.features.items()
            },
        )


@dataclass
class SelectedMethod:
    method: str
    config: dict[str, Any]
    factory: Callable[[], Any]
    calibration_fpr: float
    validation_power: float


class SadaScorer:
    def __init__(
        self,
        reference_features: dict[str, np.ndarray],
        pca_dim: int,
        window_size: int,
        cov_eps: float,
        seed: int,
    ) -> None:
        self.window_size = window_size
        self.detectors: dict[str, MahalanobisDetector] = {}
        for stream in ["A", "E_text"]:
            detector = MahalanobisDetector(
                pca_dim=pca_dim,
                window_size=window_size,
                calibration_windows=1,
                calibration_quantile=0.99,
                cov_eps=cov_eps,
                random_seed=seed,
            )
            detector.fit_reference(reference_features[stream])
            self.detectors[stream] = detector

    def project(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            stream: self.detectors[stream].transform(features[stream])
            for stream in self.detectors
        }

    def stream_scores(
        self,
        clean_projected: dict[str, np.ndarray],
        drift_projected: dict[str, np.ndarray],
        plans: Sequence[WindowPlan],
    ) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for stream, detector in self.detectors.items():
            means = np.empty(
                (len(plans), int(detector.actual_pca_dim)),
                dtype=np.float64,
            )
            for index, plan in enumerate(plans):
                total = np.zeros(int(detector.actual_pca_dim), dtype=np.float64)
                if plan.n_clean:
                    total += clean_projected[stream][plan.clean_indices].sum(axis=0)
                if plan.n_drift:
                    total += drift_projected[stream][plan.drift_indices].sum(axis=0)
                means[index] = total / (plan.n_clean + plan.n_drift)
            diff = means - detector.mu_base
            squared = np.einsum(
                "ni,ij,nj->n",
                diff,
                detector.sigma_inv,
                diff,
                optimize=True,
            )
            output[stream] = np.sqrt(np.maximum(squared, 0.0))
        return output

    def calibrate(
        self,
        calibration_projected: dict[str, np.ndarray],
        calibration_task_ids: np.ndarray,
        weighting: str,
        windows: int,
        quantile: float,
        seed: int,
    ) -> dict[str, float]:
        plans = build_window_plans(
            clean_task_ids=calibration_task_ids,
            drift_task_ids=np.empty(0, dtype=np.int64),
            n_clean=self.window_size,
            n_drift=0,
            count=windows,
            seed=seed,
            weighting=weighting,
        )
        scores = self.stream_scores(
            clean_projected=calibration_projected,
            drift_projected={
                stream: np.empty((0, values.shape[1]), dtype=np.float64)
                for stream, values in calibration_projected.items()
            },
            plans=plans,
        )
        return {
            stream: float(np.quantile(values, quantile))
            for stream, values in scores.items()
        }

    def fused_scores(
        self,
        clean_projected: dict[str, np.ndarray],
        drift_projected: dict[str, np.ndarray],
        plans: Sequence[WindowPlan],
        thresholds: dict[str, float],
    ) -> np.ndarray:
        stream_scores = self.stream_scores(
            clean_projected=clean_projected,
            drift_projected=drift_projected,
            plans=plans,
        )
        normalized = np.stack(
            [
                stream_scores[stream] / max(thresholds[stream], 1e-12)
                for stream in ["A", "E_text"]
            ],
            axis=0,
        )
        return np.max(normalized, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal closed-set label-aware drift comparison."
    )
    parser.add_argument(
        "--strict-input-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "strict_inputs"
        ),
    )
    parser.add_argument(
        "--strict-prediction-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "strict_predictions"
        ),
    )
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "features_pooled_e_fusion")
    parser.add_argument("--river-path", type=Path, default=Path("/tmp/label_aware_audit_river"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results_detection"
            / "label_aware_formal_20260719"
            / "closed_set"
        ),
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--windows-per-ratio", type=int, default=1000)
    parser.add_argument("--task-macro-windows", type=int, default=1000)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--sada-calibration-windows", type=int, default=10000)
    parser.add_argument("--validation-windows", type=int, default=200)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--calibration-seed", type=int, default=20260719)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--sada-quantile", type=float, default=0.99)
    parser.add_argument("--cov-eps", type=float, default=1e-6)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["main", "budget", "task_macro", "ordering", "permutation"],
        default=["main", "budget", "task_macro", "ordering", "permutation"],
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_closed_pool(
    split_name: str,
    strict_input_dir: Path,
    strict_prediction_dir: Path,
    features_dir: Path,
) -> ClosedSetPool:
    input_rows = read_json(strict_input_dir / split_name / "data.json")
    predictions = {
        str(row["uid"]): row
        for row in read_jsonl(
            strict_prediction_dir / split_name / "strict_predictions.jsonl"
        )
    }
    payload = torch.load(
        features_dir / split_name / "features.pt",
        map_location="cpu",
    )
    feature_uids = [str(value) for value in payload["metadata"]["uids"]]
    feature_indices = {uid: index for index, uid in enumerate(feature_uids)}
    selected_rows = []
    errors = []
    option_nll = []
    margin_loss = []
    indices = []

    for row in input_rows:
        uid = str(row["uid"])
        prediction = predictions.get(uid)
        if prediction is None:
            raise RuntimeError(f"{split_name}: strict prediction missing for {uid}")
        if not prediction.get("accepted"):
            raise RuntimeError(f"{split_name}/{uid}: strict prediction was rejected")
        scores = parse_option_scores(str(prediction["raw_output"]))
        nll = correct_option_nll(scores, row["ground_truth"])
        margin = supervised_margin_loss(scores, row["ground_truth"])
        predicted_correct = int(
            str(prediction.get("prediction", "")).strip().casefold()
            == str(row["ground_truth"]).strip().casefold()
        )
        if predicted_correct != int(bool(prediction.get("correct"))):
            raise RuntimeError(f"{split_name}/{uid}: stored correctness mismatch")
        selected_rows.append(row)
        errors.append(1 - predicted_correct)
        option_nll.append(nll)
        margin_loss.append(margin)
        indices.append(feature_indices[uid])

    index_array = np.asarray(indices, dtype=np.int64)
    return ClosedSetPool(
        split_name=split_name,
        uids=np.asarray([str(row["uid"]) for row in selected_rows], dtype=object),
        task_ids=np.asarray([int(row["task_id"]) for row in selected_rows], dtype=np.int64),
        errors=np.asarray(errors, dtype=np.float64),
        option_nll=np.asarray(option_nll, dtype=np.float64),
        margin_loss=np.asarray(margin_loss, dtype=np.float64),
        features={
            stream: payload["features"][stream][index_array].float().numpy().astype(np.float64)
            for stream in ["A", "E_text"]
        },
    )


def load_reference_features(
    features_dir: Path,
    tasks: set[int],
) -> dict[str, np.ndarray]:
    payload = torch.load(
        features_dir / "0_Reference_Fit" / "features.pt",
        map_location="cpu",
    )
    task_ids = np.asarray(payload["metadata"]["task_ids"], dtype=np.int64)
    mask = np.isin(task_ids, sorted(tasks))
    return {
        stream: payload["features"][stream][mask].float().numpy().astype(np.float64)
        for stream in ["A", "E_text"]
    }


def stable_validation_mask(
    uids: np.ndarray,
    fraction: float,
) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    modulus = 10_000
    cutoff = int(round(fraction * modulus))
    return np.asarray(
        [
            int(hashlib.sha256(str(uid).encode("utf-8")).hexdigest()[:8], 16)
            % modulus
            < cutoff
            for uid in uids
        ],
        dtype=bool,
    )


def detector_candidates(river_path: Path) -> dict[str, list[Candidate]]:
    river = river_candidates(river_path)
    output = {
        "DDM": river["River DDM"],
        "EDDM": river["River EDDM"],
        "ADWIN": river["River ADWIN"],
        "HDDM-W": river["River HDDM-W two-sided"],
        "Page-Hinkley": river["River Page-Hinkley"],
        "KSWIN": river["River KSWIN"],
        "OPTWIN": [],
    }
    for delta in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
        config = {
            "delta": delta,
            "rigor": 0.5,
            "max_window": 200,
            "min_subwindow": 20,
            "clock": 5,
            "cut_step": 5,
            "use_variance_test": True,
            "two_sided": True,
            "implementation": "local_optwin_style_reproduction",
        }
        kwargs = {
            key: value
            for key, value in config.items()
            if key != "implementation"
        }
        output["OPTWIN"].append(
            (config, lambda cfg=kwargs: OPTWIN(**cfg))
        )
    return output


def calibration_sequence(
    values: np.ndarray,
    task_ids: np.ndarray,
    weighting: str,
    seed: int,
    length: int = 1000,
) -> np.ndarray:
    plans = build_window_plans(
        clean_task_ids=task_ids,
        drift_task_ids=np.empty(0, dtype=np.int64),
        n_clean=length,
        n_drift=0,
        count=1,
        seed=seed,
        weighting=weighting,
    )
    return values[plans[0].clean_indices]


def stationary_windows(
    values: np.ndarray,
    task_ids: np.ndarray,
    weighting: str,
    count: int,
    window_size: int,
    seed: int,
) -> np.ndarray:
    plans = build_window_plans(
        clean_task_ids=task_ids,
        drift_task_ids=np.empty(0, dtype=np.int64),
        n_clean=window_size,
        n_drift=0,
        count=count,
        seed=seed,
        weighting=weighting,
    )
    return np.stack(
        [values[plan.clean_indices] for plan in plans],
        axis=0,
    )


def detector_rate(
    factory: Callable[[], Any],
    calibration: np.ndarray,
    windows: Sequence[np.ndarray],
) -> float:
    warmed = warm_detector(factory(), calibration)
    detections = 0
    for window in windows:
        detected, _ = detects_in_window(warmed, window)
        detections += int(detected)
    return detections / len(windows)


def validation_power(
    factory: Callable[[], Any],
    calibration: np.ndarray,
    clean_values: np.ndarray,
    drift_values: np.ndarray,
    plans_by_ratio: dict[float, list[WindowPlan]],
) -> float:
    warmed = warm_detector(factory(), calibration)
    rates = []
    for plans in plans_by_ratio.values():
        detections = 0
        for plan in plans:
            clean = clean_values[plan.clean_indices]
            changed = drift_values[plan.drift_indices]
            detected, _, _ = detects_after_onset(warmed, clean, changed)
            detections += int(detected)
        rates.append(detections / len(plans))
    return float(np.mean(rates))


def select_methods(
    methods: Sequence[str],
    candidates: dict[str, list[Candidate]],
    calibration_values: np.ndarray,
    calibration_task_ids: np.ndarray,
    validation_clean_values: np.ndarray,
    validation_clean_task_ids: np.ndarray,
    validation_drift_values: np.ndarray,
    validation_drift_task_ids: np.ndarray,
    weighting: str,
    target_fpr: float,
    calibration_windows: int,
    validation_windows: int,
    window_size: int,
    seed: int,
) -> tuple[dict[str, SelectedMethod], dict[str, Any]]:
    calibration = calibration_sequence(
        calibration_values,
        calibration_task_ids,
        weighting=weighting,
        seed=seed,
    )
    stationary = stationary_windows(
        calibration_values,
        calibration_task_ids,
        weighting=weighting,
        count=calibration_windows,
        window_size=window_size,
        seed=seed + 1,
    )
    validation_plans = {
        ratio: build_window_plans(
            clean_task_ids=validation_clean_task_ids,
            drift_task_ids=validation_drift_task_ids,
            n_clean=window_size - int(round(window_size * ratio)),
            n_drift=int(round(window_size * ratio)),
            count=validation_windows,
            seed=seed + 100 + int(round(100 * ratio)),
            weighting=weighting,
        )
        for ratio in [0.05, 0.10, 0.20]
    }

    selected: dict[str, SelectedMethod] = {}
    audit: dict[str, Any] = {}
    for method in methods:
        candidate_rows = []
        feasible = []
        for index, (config, factory) in enumerate(candidates[method]):
            fpr = detector_rate(factory, calibration, stationary)
            power = None
            if fpr <= target_fpr:
                power = validation_power(
                    factory=factory,
                    calibration=calibration,
                    clean_values=validation_clean_values,
                    drift_values=validation_drift_values,
                    plans_by_ratio=validation_plans,
                )
                feasible.append((power, fpr, -index, index, factory))
            candidate_rows.append(
                {
                    "index": index,
                    "config": config,
                    "calibration_fpr": fpr,
                    "validation_power": power,
                    "feasible": fpr <= target_fpr,
                }
            )
        if not feasible:
            audit[method] = {
                "status": "not_calibratable",
                "target_fpr": target_fpr,
                "candidates": candidate_rows,
            }
            continue
        power, fpr, _, index, factory = max(feasible)
        selected[method] = SelectedMethod(
            method=method,
            config=candidates[method][index][0],
            factory=factory,
            calibration_fpr=fpr,
            validation_power=power,
        )
        audit[method] = {
            "status": "selected",
            "selected_index": index,
            "selected": candidate_rows[index],
            "target_fpr": target_fpr,
            "selection_rule": "maximize_validation_power_subject_to_calibration_fpr",
            "candidates": candidate_rows,
        }
    return selected, audit


def evaluate_sequential_plans(
    method: str,
    signal_name: str,
    selected: SelectedMethod,
    calibration_values: np.ndarray,
    calibration_task_ids: np.ndarray,
    clean_values: np.ndarray,
    drift_values: np.ndarray,
    plans: Sequence[WindowPlan],
    weighting: str,
    seed: int,
    layout: str,
) -> dict[str, Any]:
    calibration = calibration_sequence(
        calibration_values,
        calibration_task_ids,
        weighting=weighting,
        seed=20_000 + seed,
    )
    warmed = warm_detector(selected.factory(), calibration)
    detections = 0
    prechange_alarm_windows = 0
    detected_delays = []
    censored_delays = []
    for plan in plans:
        if layout == "contiguous_tail" and plan.n_drift > 0:
            clean = clean_values[plan.clean_indices]
            changed = drift_values[plan.drift_indices]
            detected, delay, prechange_alarms = detects_after_onset(
                warmed,
                clean,
                changed,
            )
            prechange_alarm_windows += int(prechange_alarms > 0)
            if detected and delay is not None:
                detected_delays.append(int(delay) + 1)
                censored_delays.append(int(delay) + 1)
            else:
                censored_delays.append(plan.n_drift)
        else:
            window = materialize_signal_window(clean_values, drift_values, plan)
            detected, delay = detects_in_window(warmed, window)
            if detected and delay is not None:
                detected_delays.append(int(delay) + 1)
            if plan.n_drift:
                censored_delays.append(
                    int(delay) + 1 if detected and delay is not None else len(window)
                )
        detections += int(detected)

    return {
        "method": method,
        "signal": signal_name,
        "seed": seed,
        "weighting": weighting,
        "layout": layout,
        "windows": len(plans),
        "n_clean": plans[0].n_clean if plans else 0,
        "n_drift": plans[0].n_drift if plans else 0,
        "detection_rate": detections / len(plans),
        "prechange_alarm_window_rate": (
            prechange_alarm_windows / len(plans)
            if layout == "contiguous_tail" and plans and plans[0].n_drift
            else None
        ),
        "conditional_mean_delay": (
            float(np.mean(detected_delays))
            if detected_delays
            else None
        ),
        "censored_mean_delay": (
            float(np.mean(censored_delays))
            if censored_delays
            else None
        ),
        "calibration_fpr": selected.calibration_fpr,
        "validation_power": selected.validation_power,
    }


def evaluate_sada_plans(
    scorer: SadaScorer,
    clean_projected: dict[str, np.ndarray],
    drift_projected: dict[str, np.ndarray],
    plans: Sequence[WindowPlan],
    thresholds: dict[str, float],
    weighting: str,
    seed: int,
    signal_name: str,
    layout: str,
) -> dict[str, Any]:
    scores = scorer.fused_scores(
        clean_projected=clean_projected,
        drift_projected=drift_projected,
        plans=plans,
        thresholds=thresholds,
    )
    detected = scores > 1.0
    n_drift = plans[0].n_drift if plans else 0
    return {
        "method": "SADA",
        "signal": signal_name,
        "seed": seed,
        "weighting": weighting,
        "layout": layout,
        "windows": len(plans),
        "n_clean": plans[0].n_clean if plans else 0,
        "n_drift": n_drift,
        "detection_rate": float(detected.mean()),
        "prechange_alarm_window_rate": 0.0 if n_drift else None,
        "conditional_mean_delay": float(n_drift) if n_drift and detected.any() else None,
        "censored_mean_delay": float(n_drift) if n_drift else None,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std(ddof=0)),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, selected in grouped.items():
        aggregate = {
            name: value
            for name, value in zip(group_keys, key)
        }
        aggregate["n_runs"] = len(selected)
        for metric in [
            "detection_rate",
            "prechange_alarm_window_rate",
            "conditional_mean_delay",
            "censored_mean_delay",
            "mean_score",
        ]:
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) is not None
            ]
            if values:
                aggregate[f"{metric}_mean"] = float(np.mean(values))
                aggregate[f"{metric}_std"] = (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                )
        output.append(aggregate)
    return sorted(
        output,
        key=lambda row: tuple(str(row.get(name)) for name in group_keys),
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def selected_config_payload(selected: dict[str, SelectedMethod]) -> dict[str, Any]:
    return {
        method: {
            "config": row.config,
            "calibration_fpr": row.calibration_fpr,
            "validation_power": row.validation_power,
        }
        for method, row in selected.items()
    }


def run_main(
    args: argparse.Namespace,
    calibration: ClosedSetPool,
    no_drift: ClosedSetPool,
    drift_validation: ClosedSetPool,
    drift_test: ClosedSetPool,
    scorer: SadaScorer,
    projected: dict[str, dict[str, np.ndarray]],
    candidates: dict[str, list[Candidate]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, SelectedMethod]],
    dict[str, Any],
    dict[str, dict[str, float]],
]:
    rows: list[dict[str, Any]] = []
    selected_by_key: dict[str, dict[str, SelectedMethod]] = {}
    selection_audit: dict[str, Any] = {}
    sada_thresholds: dict[str, dict[str, float]] = {}

    signal_specs = {
        "binary_error": (
            BINARY_METHODS,
            calibration.errors,
            no_drift.errors,
            drift_validation.errors,
            drift_test.errors,
        ),
        "correct_option_nll": (
            CONTINUOUS_METHODS,
            calibration.option_nll,
            no_drift.option_nll,
            drift_validation.option_nll,
            drift_test.option_nll,
        ),
    }

    for weighting in ["natural", "task_balanced"]:
        thresholds = scorer.calibrate(
            calibration_projected=projected["calibration"],
            calibration_task_ids=calibration.task_ids,
            weighting=weighting,
            windows=args.sada_calibration_windows,
            quantile=args.sada_quantile,
            seed=args.calibration_seed,
        )
        sada_thresholds[weighting] = thresholds

        for signal_name, (
            method_names,
            calibration_values,
            no_drift_values,
            validation_values,
            test_values,
        ) in signal_specs.items():
            selection_key = f"{weighting}/{signal_name}"
            selected, audit = select_methods(
                methods=method_names,
                candidates=candidates,
                calibration_values=calibration_values,
                calibration_task_ids=calibration.task_ids,
                validation_clean_values=no_drift_values,
                validation_clean_task_ids=no_drift.task_ids,
                validation_drift_values=validation_values,
                validation_drift_task_ids=drift_validation.task_ids,
                weighting=weighting,
                target_fpr=args.target_fpr,
                calibration_windows=args.calibration_windows,
                validation_windows=args.validation_windows,
                window_size=args.window_size,
                seed=args.calibration_seed,
            )
            selected_by_key[selection_key] = selected
            selection_audit[selection_key] = audit

            for seed in args.seeds:
                for ratio in MAIN_RATIOS:
                    n_drift = int(round(args.window_size * ratio))
                    plans = build_window_plans(
                        clean_task_ids=no_drift.task_ids,
                        drift_task_ids=drift_test.task_ids,
                        n_clean=args.window_size - n_drift,
                        n_drift=n_drift,
                        count=args.windows_per_ratio,
                        seed=seed + int(round(ratio * 10_000)),
                        weighting=weighting,
                    )
                    if signal_name == "binary_error":
                        sada_row = evaluate_sada_plans(
                            scorer=scorer,
                            clean_projected=projected["no_drift"],
                            drift_projected=projected["drift_test"],
                            plans=plans,
                            thresholds=thresholds,
                            weighting=weighting,
                            seed=seed,
                            signal_name=signal_name,
                            layout="contiguous_tail",
                        )
                        sada_row["ratio"] = ratio
                        rows.append(sada_row)
                    for method, selected_method in selected.items():
                        row = evaluate_sequential_plans(
                            method=method,
                            signal_name=signal_name,
                            selected=selected_method,
                            calibration_values=calibration_values,
                            calibration_task_ids=calibration.task_ids,
                            clean_values=no_drift_values,
                            drift_values=test_values,
                            plans=plans,
                            weighting=weighting,
                            seed=seed,
                            layout="contiguous_tail",
                        )
                        row["ratio"] = ratio
                        rows.append(row)

    write_json(args.output_dir / "main_runs.json", rows)
    write_json(
        args.output_dir / "main_summary.json",
        summarize_rows(
            rows,
            ["weighting", "signal", "method", "ratio"],
        ),
    )
    write_json(args.output_dir / "selection_audit.json", selection_audit)
    write_json(
        args.output_dir / "selected_configs.json",
        {
            key: selected_config_payload(value)
            for key, value in selected_by_key.items()
        },
    )
    write_json(args.output_dir / "sada_thresholds.json", sada_thresholds)
    return rows, selected_by_key, selection_audit, sada_thresholds


def run_budget(
    args: argparse.Namespace,
    calibration: ClosedSetPool,
    no_drift: ClosedSetPool,
    drift_test: ClosedSetPool,
    scorer: SadaScorer,
    projected: dict[str, dict[str, np.ndarray]],
    selected_by_key: dict[str, dict[str, SelectedMethod]],
    sada_thresholds: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    binary_selected = selected_by_key["natural/binary_error"]
    continuous_selected = selected_by_key["natural/correct_option_nll"]
    for seed in args.seeds:
        for budget in BUDGETS:
            plans = build_window_plans(
                clean_task_ids=no_drift.task_ids,
                drift_task_ids=drift_test.task_ids,
                n_clean=args.window_size - budget,
                n_drift=budget,
                count=args.windows_per_ratio,
                seed=100_000 + seed + budget,
                weighting="natural",
            )
            sada_row = evaluate_sada_plans(
                scorer=scorer,
                clean_projected=projected["no_drift"],
                drift_projected=projected["drift_test"],
                plans=plans,
                thresholds=sada_thresholds["natural"],
                weighting="natural",
                seed=seed,
                signal_name="method_native",
                layout="contiguous_tail",
            )
            sada_row["budget"] = budget
            rows.append(sada_row)
            for method, selected in binary_selected.items():
                if method in CONTINUOUS_METHODS:
                    continue
                row = evaluate_sequential_plans(
                    method=method,
                    signal_name="binary_error",
                    selected=selected,
                    calibration_values=calibration.errors,
                    calibration_task_ids=calibration.task_ids,
                    clean_values=no_drift.errors,
                    drift_values=drift_test.errors,
                    plans=plans,
                    weighting="natural",
                    seed=seed,
                    layout="contiguous_tail",
                )
                row["budget"] = budget
                rows.append(row)
            for method, selected in continuous_selected.items():
                row = evaluate_sequential_plans(
                    method=method,
                    signal_name="correct_option_nll",
                    selected=selected,
                    calibration_values=calibration.option_nll,
                    calibration_task_ids=calibration.task_ids,
                    clean_values=no_drift.option_nll,
                    drift_values=drift_test.option_nll,
                    plans=plans,
                    weighting="natural",
                    seed=seed,
                    layout="contiguous_tail",
                )
                row["budget"] = budget
                rows.append(row)
    write_json(args.output_dir / "budget_runs.json", rows)
    write_json(
        args.output_dir / "budget_summary.json",
        summarize_rows(rows, ["signal", "method", "budget"]),
    )
    return rows


def run_task_macro(
    args: argparse.Namespace,
    calibration: ClosedSetPool,
    no_drift: ClosedSetPool,
    drift_test: ClosedSetPool,
    scorer: SadaScorer,
    projected: dict[str, dict[str, np.ndarray]],
    selected_by_key: dict[str, dict[str, SelectedMethod]],
    sada_thresholds: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    binary_selected = selected_by_key["task_balanced/binary_error"]
    continuous_selected = selected_by_key["task_balanced/correct_option_nll"]
    for task_id in DRIFT_TASKS:
        task_mask = drift_test.task_ids == task_id
        task_pool = drift_test.subset(task_mask)
        task_projected = {
            stream: values[task_mask]
            for stream, values in projected["drift_test"].items()
        }
        for seed in args.seeds:
            for ratio in TASK_MACRO_RATIOS:
                n_drift = int(round(args.window_size * ratio))
                plans = build_window_plans(
                    clean_task_ids=no_drift.task_ids,
                    drift_task_ids=task_pool.task_ids,
                    n_clean=args.window_size - n_drift,
                    n_drift=n_drift,
                    count=args.task_macro_windows,
                    seed=200_000 + seed + 1000 * task_id + int(100 * ratio),
                    weighting="task_balanced",
                )
                sada_row = evaluate_sada_plans(
                    scorer=scorer,
                    clean_projected=projected["no_drift"],
                    drift_projected=task_projected,
                    plans=plans,
                    thresholds=sada_thresholds["task_balanced"],
                    weighting="task_balanced",
                    seed=seed,
                    signal_name="method_native",
                    layout="contiguous_tail",
                )
                sada_row.update({"task_id": task_id, "ratio": ratio})
                rows.append(sada_row)
                for method, selected in binary_selected.items():
                    if method in CONTINUOUS_METHODS:
                        continue
                    row = evaluate_sequential_plans(
                        method=method,
                        signal_name="binary_error",
                        selected=selected,
                        calibration_values=calibration.errors,
                        calibration_task_ids=calibration.task_ids,
                        clean_values=no_drift.errors,
                        drift_values=task_pool.errors,
                        plans=plans,
                        weighting="task_balanced",
                        seed=seed,
                        layout="contiguous_tail",
                    )
                    row.update({"task_id": task_id, "ratio": ratio})
                    rows.append(row)
                for method, selected in continuous_selected.items():
                    row = evaluate_sequential_plans(
                        method=method,
                        signal_name="correct_option_nll",
                        selected=selected,
                        calibration_values=calibration.option_nll,
                        calibration_task_ids=calibration.task_ids,
                        clean_values=no_drift.option_nll,
                        drift_values=task_pool.option_nll,
                        plans=plans,
                        weighting="task_balanced",
                        seed=seed,
                        layout="contiguous_tail",
                    )
                    row.update({"task_id": task_id, "ratio": ratio})
                    rows.append(row)
    per_task = summarize_rows(
        rows,
        ["signal", "method", "task_id", "ratio"],
    )
    macro_rows = []
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in per_task:
        key = (str(row["signal"]), str(row["method"]), float(row["ratio"]))
        grouped.setdefault(key, []).append(row)
    for (signal, method, ratio), selected in grouped.items():
        macro_rows.append(
            {
                "signal": signal,
                "method": method,
                "ratio": ratio,
                "tasks": len(selected),
                "task_macro_detection_rate": float(
                    np.mean([row["detection_rate_mean"] for row in selected])
                ),
                "task_macro_detection_rate_std_across_tasks": float(
                    np.std(
                        [row["detection_rate_mean"] for row in selected],
                        ddof=1,
                    )
                    if len(selected) > 1
                    else 0.0
                ),
            }
        )
    write_json(args.output_dir / "task_macro_runs.json", rows)
    write_json(args.output_dir / "task_macro_per_task_summary.json", per_task)
    write_json(args.output_dir / "task_macro_summary.json", macro_rows)
    return rows


def run_ordering(
    args: argparse.Namespace,
    calibration: ClosedSetPool,
    no_drift: ClosedSetPool,
    drift_test: ClosedSetPool,
    scorer: SadaScorer,
    projected: dict[str, dict[str, np.ndarray]],
    selected_by_key: dict[str, dict[str, SelectedMethod]],
    sada_thresholds: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[WindowPlan]]:
    rows = []
    base_by_seed: dict[int, list[WindowPlan]] = {}
    binary_selected = selected_by_key["natural/binary_error"]
    continuous_selected = selected_by_key["natural/correct_option_nll"]
    layouts = [
        ("shuffled", "shuffled", None),
        ("block_1", "block", 1),
        ("block_2", "block", 2),
        ("block_5", "block", 5),
        ("block_10", "block", 10),
        ("block_20", "block", 20),
        ("contiguous", "contiguous_tail", None),
    ]
    for seed in args.seeds:
        base = build_window_plans(
            clean_task_ids=no_drift.task_ids,
            drift_task_ids=drift_test.task_ids,
            n_clean=160,
            n_drift=40,
            count=args.windows_per_ratio,
            seed=300_000 + seed,
            weighting="natural",
            layout="contiguous_tail",
        )
        base_by_seed[seed] = base
        for index, (label, layout, block_size) in enumerate(layouts):
            plans = reorder_window_plans(
                base,
                layout=layout,
                seed=310_000 + 100 * seed + index,
                block_size=block_size,
            )
            sada_row = evaluate_sada_plans(
                scorer=scorer,
                clean_projected=projected["no_drift"],
                drift_projected=projected["drift_test"],
                plans=plans,
                thresholds=sada_thresholds["natural"],
                weighting="natural",
                seed=seed,
                signal_name="method_native",
                layout=layout,
            )
            sada_row["ordering"] = label
            rows.append(sada_row)
            for method, selected in binary_selected.items():
                if method in CONTINUOUS_METHODS:
                    continue
                row = evaluate_sequential_plans(
                    method=method,
                    signal_name="binary_error",
                    selected=selected,
                    calibration_values=calibration.errors,
                    calibration_task_ids=calibration.task_ids,
                    clean_values=no_drift.errors,
                    drift_values=drift_test.errors,
                    plans=plans,
                    weighting="natural",
                    seed=seed,
                    layout=layout,
                )
                row["ordering"] = label
                rows.append(row)
            for method, selected in continuous_selected.items():
                row = evaluate_sequential_plans(
                    method=method,
                    signal_name="correct_option_nll",
                    selected=selected,
                    calibration_values=calibration.option_nll,
                    calibration_task_ids=calibration.task_ids,
                    clean_values=no_drift.option_nll,
                    drift_values=drift_test.option_nll,
                    plans=plans,
                    weighting="natural",
                    seed=seed,
                    layout=layout,
                )
                row["ordering"] = label
                rows.append(row)
    write_json(args.output_dir / "ordering_runs.json", rows)
    write_json(
        args.output_dir / "ordering_summary.json",
        summarize_rows(rows, ["signal", "method", "ordering"]),
    )
    return rows, base_by_seed[args.seeds[0]]


def run_permutation(
    args: argparse.Namespace,
    scorer: SadaScorer,
    no_drift_projected: dict[str, np.ndarray],
    drift_projected: dict[str, np.ndarray],
    base_plans: list[WindowPlan],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    checked = base_plans[: min(100, len(base_plans))]
    permutations = 20
    rng = np.random.default_rng(20260719)
    max_abs_difference = 0.0
    disagreements = 0
    comparisons = 0
    per_window_spans = []
    for plan in checked:
        fused = []
        for _ in range(permutations):
            order = rng.permutation(plan.n_clean + plan.n_drift)
            scores = []
            for stream, detector in scorer.detectors.items():
                parts = []
                if plan.n_clean:
                    parts.append(no_drift_projected[stream][plan.clean_indices])
                if plan.n_drift:
                    parts.append(drift_projected[stream][plan.drift_indices])
                window = np.concatenate(parts, axis=0)[order]
                score = detector.score_projected_window(window)
                scores.append(score / max(thresholds[stream], 1e-12))
            fused.append(max(scores))
        span = float(max(fused) - min(fused))
        per_window_spans.append(span)
        max_abs_difference = max(max_abs_difference, span)
        decisions = [value > 1.0 for value in fused]
        disagreements += int(any(value != decisions[0] for value in decisions[1:]))
        comparisons += 1
    result = {
        "windows": comparisons,
        "permutations_per_window": permutations,
        "max_score_span": max_abs_difference,
        "mean_score_span": float(np.mean(per_window_spans)),
        "detection_disagreement_windows": disagreements,
        "detection_disagreement_rate": disagreements / comparisons,
    }
    write_json(args.output_dir / "sada_permutation_invariance.json", result)
    return result


def write_report(
    path: Path,
    config: dict[str, Any],
    selection_audit: dict[str, Any],
) -> None:
    lines = [
        "# Formal Closed-Set Label-Aware Drift Experiments",
        "",
        "This directory contains feature-aligned SADA and label-aware detector results "
        "using strict task-aware option scoring.",
        "",
        "## Protocol",
        "",
        f"- Window size: {config['window_size']}",
        f"- Windows per ratio: {config['windows_per_ratio']}",
        f"- Seeds: {config['seeds']}",
        f"- Target calibration FPR: {100 * config['target_fpr']:.1f}%",
        "- Detector selection: maximize held-out drift validation power subject to the FPR constraint.",
        "- OPTWIN is explicitly a local OPTWIN-style reproduction; River official implementations are used for the other methods.",
        "",
        "## Calibration Status",
        "",
        "| Setting | Method | Status |",
        "|---|---|---|",
    ]
    for setting, methods in selection_audit.items():
        for method, row in methods.items():
            lines.append(f"| {setting} | {method} | {row['status']} |")
    lines.extend(
        [
            "",
            "Machine-readable runs and summaries are stored in the adjacent JSON files.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_closed_pool(
        "1_Threshold_Cal",
        args.strict_input_dir,
        args.strict_prediction_dir,
        args.features_dir,
    )
    no_drift = load_closed_pool(
        "2_Stream_Sim_NoDrift",
        args.strict_input_dir,
        args.strict_prediction_dir,
        args.features_dir,
    )
    drift = load_closed_pool(
        "2_Stream_Sim_Drift",
        args.strict_input_dir,
        args.strict_prediction_dir,
        args.features_dir,
    )
    validation_mask = stable_validation_mask(
        drift.uids,
        fraction=args.validation_fraction,
    )
    drift_validation = drift.subset(validation_mask)
    drift_test = drift.subset(~validation_mask)
    for task_id in DRIFT_TASKS:
        if not np.any(drift_validation.task_ids == task_id):
            raise RuntimeError(f"validation partition has no Task {task_id}")
        if not np.any(drift_test.task_ids == task_id):
            raise RuntimeError(f"test partition has no Task {task_id}")

    scorer = SadaScorer(
        reference_features=load_reference_features(args.features_dir, {8, 9}),
        pca_dim=args.pca_dim,
        window_size=args.window_size,
        cov_eps=args.cov_eps,
        seed=args.calibration_seed,
    )
    projected = {
        "calibration": scorer.project(calibration.features),
        "no_drift": scorer.project(no_drift.features),
        "drift_validation": scorer.project(drift_validation.features),
        "drift_test": scorer.project(drift_test.features),
    }
    candidates = detector_candidates(args.river_path)

    config = {
        "protocol": "closed_set_strict_option_scoring",
        "strict_input_dir": str(args.strict_input_dir),
        "strict_prediction_dir": str(args.strict_prediction_dir),
        "features_dir": str(args.features_dir),
        "river_path": str(args.river_path),
        "window_size": args.window_size,
        "windows_per_ratio": args.windows_per_ratio,
        "task_macro_windows": args.task_macro_windows,
        "calibration_windows": args.calibration_windows,
        "sada_calibration_windows": args.sada_calibration_windows,
        "validation_windows": args.validation_windows,
        "target_fpr": args.target_fpr,
        "validation_fraction": args.validation_fraction,
        "seeds": args.seeds,
        "main_ratios": MAIN_RATIOS,
        "budgets": BUDGETS,
        "drift_tasks": DRIFT_TASKS,
        "pool_sizes": {
            "calibration": len(calibration.uids),
            "no_drift": len(no_drift.uids),
            "drift_validation": len(drift_validation.uids),
            "drift_test": len(drift_test.uids),
        },
        "error_rates": {
            "calibration": float(calibration.errors.mean()),
            "no_drift": float(no_drift.errors.mean()),
            "drift_validation": float(drift_validation.errors.mean()),
            "drift_test": float(drift_test.errors.mean()),
        },
        "option_nll_means": {
            "calibration": float(calibration.option_nll.mean()),
            "no_drift": float(no_drift.option_nll.mean()),
            "drift_validation": float(drift_validation.option_nll.mean()),
            "drift_test": float(drift_test.option_nll.mean()),
        },
        "sada": {
            "feature_group": "A+E_text",
            "fusion_rule": "max_norm",
            "pca_dim": args.pca_dim,
            "calibration_quantile": args.sada_quantile,
        },
        "optwin_provenance": "local OPTWIN-style reproduction with two-sided mean and variance tests",
    }
    write_json(args.output_dir / "run_config.json", config)

    required_followups = set(args.experiments) - {"main"}
    if required_followups and "main" not in args.experiments:
        raise ValueError("follow-up experiments currently require main in the same invocation")

    selected_by_key: dict[str, dict[str, SelectedMethod]] = {}
    selection_audit: dict[str, Any] = {}
    sada_thresholds: dict[str, dict[str, float]] = {}
    if "main" in args.experiments:
        _, selected_by_key, selection_audit, sada_thresholds = run_main(
            args=args,
            calibration=calibration,
            no_drift=no_drift,
            drift_validation=drift_validation,
            drift_test=drift_test,
            scorer=scorer,
            projected=projected,
            candidates=candidates,
        )

    if "budget" in args.experiments:
        run_budget(
            args,
            calibration,
            no_drift,
            drift_test,
            scorer,
            projected,
            selected_by_key,
            sada_thresholds,
        )
    if "task_macro" in args.experiments:
        run_task_macro(
            args,
            calibration,
            no_drift,
            drift_test,
            scorer,
            projected,
            selected_by_key,
            sada_thresholds,
        )

    permutation_plans = None
    if "ordering" in args.experiments:
        _, permutation_plans = run_ordering(
            args,
            calibration,
            no_drift,
            drift_test,
            scorer,
            projected,
            selected_by_key,
            sada_thresholds,
        )
    if "permutation" in args.experiments:
        if permutation_plans is None:
            permutation_plans = build_window_plans(
                clean_task_ids=no_drift.task_ids,
                drift_task_ids=drift_test.task_ids,
                n_clean=160,
                n_drift=40,
                count=min(100, args.windows_per_ratio),
                seed=300_000 + args.seeds[0],
                weighting="natural",
            )
        run_permutation(
            args,
            scorer,
            projected["no_drift"],
            projected["drift_test"],
            permutation_plans,
            sada_thresholds["natural"],
        )

    write_report(
        args.output_dir / "closed_set_report.md",
        config=config,
        selection_audit=selection_audit,
    )
    print(f"Closed-set formal results written to {args.output_dir}")


if __name__ == "__main__":
    main()
