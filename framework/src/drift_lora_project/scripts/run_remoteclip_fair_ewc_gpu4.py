#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "results" / "all_model_three_seed_repeats_20260715"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "remoteclip_fair_ewc_gpu4_20260724"
DEFAULT_REMOTECLIP_DIR = Path("@@WORKSPACE@@/qwen_eval/models/RemoteCLIP")
RUNNER_PATH = SCRIPT_DIR / "run_omniearth_remoteclip_formal_framework.py"
PRESSURE_PATH = SCRIPT_DIR / "recompute_pressure_scenarios_all_models.py"
HEAD_EWC_PATH = SCRIPT_DIR / "nonqwen_head_ewc.py"

ABSOLUTE_RATES = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1, 2, 4, 8, 10, 12, 16, 24, 32]
LOAD_FACTORS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4]
PLAIN_EPOCHS = [1, 3, 5]
MAIN_METHODS = [
    "NoRetrain",
    "PeriodicFixedRetrain",
    "StaticSplitContinuous",
    "EWC",
    "Ours",
    "OfflineJointOracle",
]
ALL_METHODS = [
    "NoRetrain",
    "PeriodicFixedRetrain",
    "StaticSplitContinuous",
    "EWC",
    "EWCMinFeasibleDiagnostic",
    "Ours",
    "OfflineJointOracle",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(row["uid"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def unique_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            uid = str(row["uid"])
            if uid in seen:
                continue
            out.append(row)
            seen.add(uid)
    return out


def load_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions[str(row["uid"])] = str(row["prediction"])
    return predictions


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    predictions: dict[str, str],
    pseudo: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            uid = str(row["uid"])
            payload = {
                "uid": uid,
                "task_key": row["task_key"],
                "prediction": predictions.get(uid, ""),
                "answer": str(row["answer"]),
                "pseudo_label": pseudo.get(uid, ""),
                "true_correct": int(predictions.get(uid) == str(row["answer"])),
                "pseudo_correct": (
                    int(predictions.get(uid) == pseudo[uid]) if uid in pseudo else ""
                ),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_feature_cache(
    path: Path,
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uids = [str(row["uid"]) for row in rows]
    matrix = np.stack([np.asarray(features[uid], dtype=np.float32) for uid in uids])
    np.savez_compressed(path, uids=np.asarray(uids, dtype=str), features=matrix)
    write_json(path.with_suffix(".manifest.json"), manifest)


def load_feature_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        uids = [str(value) for value in payload["uids"].tolist()]
        matrix = np.asarray(payload["features"], dtype=np.float64)
    return {uid: matrix[index] for index, uid in enumerate(uids)}


def feature_matrix(
    features: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> np.ndarray:
    return np.stack([np.asarray(features[str(row["uid"])], dtype=np.float64) for row in rows])


def extract_features_for_lambda(
    *,
    runner: Any,
    model: Any,
    preprocess: Any,
    config: Any,
    rows: list[dict[str, Any]],
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    amp_enabled = config.model_precision == "fp16" and device.startswith("cuda")
    dtype = torch.float16 if amp_enabled else torch.float32
    features: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            image = Image.open(row["image_path"]).convert("RGB")
            image = runner.maybe_downsample_image(image, config.resolution_keep_ratio)
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
                vector = functional.normalize(model.encode_image(tensor), dim=-1)
            features[str(row["uid"])] = vector.float().cpu().numpy()[0]
            if index % 50 == 0 or index == len(rows):
                print(
                    f"[features] {config.lambda_id}: {index}/{len(rows)}",
                    flush=True,
                )
    return features


def ensure_feature_caches(
    *,
    runner: Any,
    source_dir: Path,
    output_root: Path,
    remoteclip_dir: Path,
    rows: list[dict[str, Any]],
    device: str,
    force: bool,
) -> dict[str, dict[str, np.ndarray]]:
    lambda_rows = read_csv(source_dir / "lambda_profiles" / "lambda_profiles_group_v2.csv")
    model_hash = sha256_file(remoteclip_dir / "RemoteCLIP-RN50.pt")
    row_uids_hash = uid_hash(rows)
    caches: dict[str, dict[str, np.ndarray]] = {}
    model = preprocess = None
    for row in lambda_rows:
        lambda_id = str(row["lambda_id"])
        cache_path = output_root / "feature_cache" / f"{lambda_id}.npz"
        manifest_path = cache_path.with_suffix(".manifest.json")
        expected = {
            "lambda_id": lambda_id,
            "resolution_keep_ratio": float(row["resolution_keep_ratio"]),
            "model_precision": str(row["model_precision"]),
            "row_count": len(rows),
            "row_uid_sha256": row_uids_hash,
            "model_sha256": model_hash,
            "remoteclip_weight": str(remoteclip_dir / "RemoteCLIP-RN50.pt"),
            "feature_extraction": "same preprocessing/downsample/autocast as formal runner",
        }
        if cache_path.exists() and manifest_path.exists() and not force:
            observed = read_json(manifest_path)
            if all(observed.get(key) == value for key, value in expected.items()):
                cached = load_feature_cache(cache_path)
                if set(cached) == {str(item["uid"]) for item in rows}:
                    caches[lambda_id] = cached
                    print(f"[features] reuse {cache_path}", flush=True)
                    continue
        if model is None:
            model, preprocess, _tokenizer, _params = runner.load_remoteclip(
                remoteclip_dir,
                device,
            )
        config = runner.LambdaConfig(
            lambda_id=lambda_id,
            resolution_keep_ratio=float(row["resolution_keep_ratio"]),
            model_precision=str(row["model_precision"]),
            batch_size=int(row.get("batch_size") or 32),
        )
        extracted = extract_features_for_lambda(
            runner=runner,
            model=model,
            preprocess=preprocess,
            config=config,
            rows=rows,
            device=device,
        )
        save_feature_cache(cache_path, rows, extracted, expected)
        caches[lambda_id] = {
            uid: np.asarray(value, dtype=np.float64) for uid, value in extracted.items()
        }
    return caches


def semantic_labels(
    runner: Any,
    rows: list[dict[str, Any]],
    *,
    letters: dict[str, str] | None,
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        uid = str(row["uid"])
        letter = str(row["answer"]) if letters is None else str(letters[uid])
        label = runner.option_label(row, letter)
        if not label:
            raise ValueError(f"Cannot resolve semantic label for uid={uid}")
        labels[uid] = str(label)
    return labels


def indices_for_rows(
    rows: list[dict[str, Any]],
    labels_by_uid: dict[str, str],
    class_to_index: dict[str, int],
) -> np.ndarray:
    return np.asarray(
        [class_to_index[labels_by_uid[str(row["uid"])]] for row in rows],
        dtype=np.int64,
    )


def option_predictions(
    *,
    runner: Any,
    head_ewc: Any,
    state: Any,
    features: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    matrix = feature_matrix(features, rows)
    scores = head_ewc.predict_scores(state, matrix)
    class_to_index = {label: index for index, label in enumerate(state.labels)}
    predictions: dict[str, str] = {}
    for row_index, row in enumerate(rows):
        candidates: list[tuple[float, str]] = []
        for letter, option in row["options"].items():
            label = runner.screening.clean_label(str(option))
            if label in class_to_index:
                candidates.append((float(scores[row_index, class_to_index[label]]), str(letter)))
        predictions[str(row["uid"])] = max(candidates)[1] if candidates else sorted(row["options"])[0]
    return predictions


def timed_deterministic(
    fn: Callable[[], Any],
    *,
    repeats: int,
    state_arrays: Callable[[Any], tuple[np.ndarray, ...]],
) -> tuple[Any, float, list[float]]:
    warm = fn()
    reference = tuple(np.asarray(value) for value in state_arrays(warm))
    times: list[float] = []
    result = warm
    for _ in range(repeats):
        started = time.perf_counter()
        candidate = fn()
        times.append(time.perf_counter() - started)
        observed = state_arrays(candidate)
        if len(observed) != len(reference):
            raise AssertionError("Deterministic state array count changed")
        for expected, actual in zip(reference, observed):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        result = candidate
    return result, float(statistics.median(times)), times


def timed_train(
    *,
    head_ewc: Any,
    x: np.ndarray,
    y: np.ndarray,
    initial: Any,
    epochs: int,
    seed: str,
    alpha: float,
    repeats: int,
    anchor: Any | None = None,
    fisher: Any | None = None,
    ewc_lambda: float = 0.0,
) -> tuple[Any, dict[str, Any], float, list[float]]:
    def run() -> tuple[Any, dict[str, Any]]:
        return head_ewc.train_head(
            x,
            y,
            initial,
            epochs=epochs,
            seed=seed,
            alpha=alpha,
            anchor=anchor,
            fisher=fisher,
            ewc_lambda=ewc_lambda,
        )

    result, runtime, samples = timed_deterministic(
        run,
        repeats=repeats,
        state_arrays=lambda value: (value[0].weight, value[0].bias),
    )
    state, metrics = result
    return state, metrics, runtime, samples


def timed_fisher(
    *,
    head_ewc: Any,
    x: np.ndarray,
    y: np.ndarray,
    state: Any,
    repeats: int,
) -> tuple[Any, float, list[float]]:
    result, runtime, samples = timed_deterministic(
        lambda: head_ewc.compute_empirical_fisher(x, y, state),
        repeats=repeats,
        state_arrays=lambda value: (value.weight, value.bias),
    )
    return result, runtime, samples


def true_accuracy(rows: list[dict[str, Any]], predictions: dict[str, str]) -> float:
    if not rows:
        return 0.0
    return sum(
        int(predictions.get(str(row["uid"])) == str(row["answer"])) for row in rows
    ) / len(rows)


def pseudo_accuracy(
    rows: list[dict[str, Any]],
    predictions: dict[str, str],
    pseudo: dict[str, str],
) -> float:
    labeled = [row for row in rows if str(row["uid"]) in pseudo]
    if not labeled:
        return 0.0
    return sum(
        int(predictions.get(str(row["uid"])) == pseudo[str(row["uid"])])
        for row in labeled
    ) / len(labeled)


def assign_resource_costs(
    rows: list[dict[str, Any]],
    *,
    max_resource_cost: float,
) -> float:
    maximum = max(float(row["counted_train_runtime_s"]) for row in rows)
    for row in rows:
        ratio = float(row["counted_train_runtime_s"]) / max(maximum, 1e-12)
        row["resource_cost"] = round(max(0.01, min(0.95, ratio * max_resource_cost)), 6)
        row["resource_cost_formula"] = (
            "counted_train_runtime_s / max_common_runtime_s * gamma_max_resource_cost"
        )
        row["max_common_runtime_s"] = maximum
    return maximum


def validate_disjoint(
    train_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train = {str(row["uid"]) for row in train_rows}
    selection = {str(row["uid"]) for row in selection_rows}
    evaluation = {str(row["uid"]) for row in eval_rows}
    overlaps = {
        "train_selection": sorted(train & selection),
        "train_evaluation": sorted(train & evaluation),
        "selection_evaluation": sorted(selection & evaluation),
    }
    if any(overlaps.values()):
        raise ValueError(
            "Train/selection/evaluation overlap detected: "
            + json.dumps({key: len(value) for key, value in overlaps.items()})
        )
    return {
        "train_count": len(train),
        "selection_count": len(selection),
        "evaluation_count": len(evaluation),
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
    }


def train_seed(
    *,
    seed: int,
    runner: Any,
    head_ewc: Any,
    source_dir: Path,
    output_dir: Path,
    features_by_lambda: dict[str, dict[str, np.ndarray]],
    repeats: int,
    alpha: float,
    ewc_lambda: float,
    gamma_max_resource_cost: float,
) -> dict[str, Any]:
    splits = runner.load_existing_splits(source_dir / "splits")
    pseudo = runner.load_external_pseudo_labels(source_dir / "teacher" / "pseudo_labels")
    disjoint = validate_disjoint(splits["R_train"], splits["R_val"], splits["R_test"])
    old_labels = semantic_labels(runner, splits["history_no_drift"], letters=None)
    target_labels = semantic_labels(runner, splits["R_train"], letters=pseudo)
    labels = sorted(set(old_labels.values()) | set(target_labels.values()))
    class_to_index = {label: index for index, label in enumerate(labels)}
    target_y = indices_for_rows(splits["R_train"], target_labels, class_to_index)
    old_y = indices_for_rows(splits["history_no_drift"], old_labels, class_to_index)
    eval_rows = unique_rows(
        splits["R_train"],
        splits["R_val"],
        splits["R_test"],
        splits["R_no_drift"],
    )
    lambda_rows = read_csv(source_dir / "lambda_profiles" / "lambda_profiles_group_v2.csv")
    base_predictions = {
        row["lambda_id"]: load_predictions(Path(row["raw_predictions"])) for row in lambda_rows
    }
    plain_rows: list[dict[str, Any]] = []
    plain_predictions: dict[tuple[str, str], dict[str, str]] = {}
    head_states: dict[tuple[str, int], Any] = {}
    scaler_by_lambda: dict[str, Any] = {}
    initial_by_lambda: dict[str, Any] = {}
    transformed_target_by_lambda: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, Any]] = []

    for lambda_row in lambda_rows:
        lambda_id = str(lambda_row["lambda_id"])
        features = features_by_lambda[lambda_id]
        target_raw = feature_matrix(features, splits["R_train"])
        scaler = head_ewc.fit_scaler(target_raw)
        initial = head_ewc.initialize_head(labels, target_raw.shape[1], scaler)
        target_x = head_ewc.transform(target_raw, scaler)
        scaler_by_lambda[lambda_id] = scaler
        initial_by_lambda[lambda_id] = initial
        transformed_target_by_lambda[lambda_id] = target_x
        base_pseudo = pseudo_accuracy(splits["R_val"], base_predictions[lambda_id], pseudo)
        ep1_gain: float | None = None
        for epochs in PLAIN_EPOCHS:
            gamma_id = f"rc_r50_linear_ep{epochs}"
            paired_seed = f"remoteclip_fair_seed{seed}:{lambda_id}:ep{epochs}"
            state, train_metrics, train_runtime, timing_samples = timed_train(
                head_ewc=head_ewc,
                x=target_x,
                y=target_y,
                initial=initial,
                epochs=epochs,
                seed=paired_seed,
                alpha=alpha,
                repeats=repeats,
            )
            predictions = option_predictions(
                runner=runner,
                head_ewc=head_ewc,
                state=state,
                features=features,
                rows=eval_rows,
            )
            adapter_pseudo = pseudo_accuracy(splits["R_val"], predictions, pseudo)
            actual_gain = adapter_pseudo - base_pseudo
            if epochs == 1:
                ep1_gain = actual_gain
            adapter_dir = output_dir / "plain" / lambda_id / gamma_id
            head_ewc.save_head(adapter_dir / "head.npz", state)
            prediction_path = adapter_dir / "predictions.jsonl"
            write_predictions(prediction_path, eval_rows, predictions, pseudo)
            write_json(
                adapter_dir / "train_metrics.json",
                {
                    "seed": seed,
                    "paired_seed": paired_seed,
                    "lambda_id": lambda_id,
                    "gamma_id": gamma_id,
                    "trainer": "common_nonqwen_head_ewc_lambda0",
                    "epochs": epochs,
                    "alpha": alpha,
                    "labels": labels,
                    "runtime_median_s": train_runtime,
                    "runtime_samples_s": timing_samples,
                    "metrics": train_metrics,
                },
            )
            row = {
                "group_id": runner.GROUP_ID,
                "feature_lambda_id": lambda_id,
                "gamma_id": gamma_id,
                "method": "common_head_plain",
                "target_epochs": epochs,
                "microprofile_epochs": 1,
                "train_samples": len(splits["R_train"]),
                "rval_base_accuracy_before_1epoch": base_pseudo,
                "rval_adapter_accuracy_after_target": adapter_pseudo,
                "profile_gain": ep1_gain if ep1_gain is not None else actual_gain,
                "predicted_final_gain": actual_gain,
                "actual_target_gain": actual_gain,
                "microprofile_train_time_s": "",
                "target_train_time_s_est": train_runtime,
                "measured_train_time_s": train_runtime,
                "counted_train_runtime_s": train_runtime,
                "peak_memory_gib": 0.0,
                "adapter_path": str(prediction_path),
                "adapter_predictions_path": str(prediction_path),
                "trainer_seed": paired_seed,
            }
            plain_rows.append(row)
            plain_predictions[(lambda_id, gamma_id)] = predictions
            head_states[(lambda_id, epochs)] = state
            metrics_rows.append(
                {
                    "seed": seed,
                    "method": "plain",
                    "lambda_id": lambda_id,
                    "epochs": epochs,
                    "R_train_true": true_accuracy(splits["R_train"], predictions),
                    "R_train_pseudo": pseudo_accuracy(splits["R_train"], predictions, pseudo),
                    "R_val_true": true_accuracy(splits["R_val"], predictions),
                    "R_val_pseudo": adapter_pseudo,
                    "R_test_true": true_accuracy(splits["R_test"], predictions),
                    "runtime_s": train_runtime,
                }
            )

    source_methods = read_csv(source_dir / "method_comparison.csv")
    static_source = next(
        row for row in source_methods if row["method"] == "StaticSplitContinuous"
    )
    fixed_lambda_id = str(static_source["lambda_id"])
    fixed_features = features_by_lambda[fixed_lambda_id]
    fixed_scaler = scaler_by_lambda[fixed_lambda_id]
    fixed_initial = initial_by_lambda[fixed_lambda_id]
    old_x = head_ewc.transform(
        feature_matrix(fixed_features, splits["history_no_drift"]),
        fixed_scaler,
    )
    fisher, fisher_runtime, fisher_samples = timed_fisher(
        head_ewc=head_ewc,
        x=old_x,
        y=old_y,
        state=fixed_initial,
        repeats=repeats,
    )
    paired_seed = f"remoteclip_fair_seed{seed}:{fixed_lambda_id}:ep3"
    ewc_state, ewc_metrics, ewc_train_runtime, ewc_timing_samples = timed_train(
        head_ewc=head_ewc,
        x=transformed_target_by_lambda[fixed_lambda_id],
        y=target_y,
        initial=fixed_initial,
        epochs=3,
        seed=paired_seed,
        alpha=alpha,
        repeats=repeats,
        anchor=fixed_initial,
        fisher=fisher,
        ewc_lambda=ewc_lambda,
    )
    ewc_predictions = option_predictions(
        runner=runner,
        head_ewc=head_ewc,
        state=ewc_state,
        features=fixed_features,
        rows=eval_rows,
    )
    ewc_dir = output_dir / "ewc"
    head_ewc.save_head(ewc_dir / "anchor_zero.npz", fixed_initial)
    head_ewc.save_fisher(ewc_dir / "fisher.npz", fisher)
    head_ewc.save_head(ewc_dir / "head_ewc.npz", ewc_state)
    ewc_prediction_path = ewc_dir / "predictions.jsonl"
    write_predictions(ewc_prediction_path, eval_rows, ewc_predictions, pseudo)
    ewc_counted_runtime = fisher_runtime + ewc_train_runtime
    ewc_base_pseudo = pseudo_accuracy(
        splits["R_val"],
        base_predictions[fixed_lambda_id],
        pseudo,
    )
    ewc_adapter_pseudo = pseudo_accuracy(splits["R_val"], ewc_predictions, pseudo)
    ewc_row = {
        "group_id": runner.GROUP_ID,
        "feature_lambda_id": fixed_lambda_id,
        "gamma_id": f"rc_r50_linear_ep3_ewc_l{ewc_lambda:g}_fair",
        "method": "common_head_ewc",
        "target_epochs": 3,
        "microprofile_epochs": 1,
        "train_samples": len(splits["R_train"]),
        "rval_base_accuracy_before_1epoch": ewc_base_pseudo,
        "rval_adapter_accuracy_after_target": ewc_adapter_pseudo,
        "profile_gain": ewc_adapter_pseudo - ewc_base_pseudo,
        "predicted_final_gain": ewc_adapter_pseudo - ewc_base_pseudo,
        "actual_target_gain": ewc_adapter_pseudo - ewc_base_pseudo,
        "target_train_time_s_est": ewc_counted_runtime,
        "measured_train_time_s": ewc_counted_runtime,
        "counted_train_runtime_s": ewc_counted_runtime,
        "fisher_runtime_s": fisher_runtime,
        "target_ewc_train_runtime_s": ewc_train_runtime,
        "ewc_lambda": ewc_lambda,
        "peak_memory_gib": 0.0,
        "adapter_path": str(ewc_prediction_path),
        "adapter_predictions_path": str(ewc_prediction_path),
        "trainer_seed": paired_seed,
    }
    maximum_runtime = assign_resource_costs(
        plain_rows + [ewc_row],
        max_resource_cost=gamma_max_resource_cost,
    )
    write_json(
        ewc_dir / "train_metrics.json",
        {
            "seed": seed,
            "paired_seed": paired_seed,
            "trainer": "common_nonqwen_head_ewc",
            "anchor_mode": "initial_zero",
            "anchor_optimizer_updates": 0,
            "labels": labels,
            "ewc_lambda": ewc_lambda,
            "fisher_runtime_median_s": fisher_runtime,
            "fisher_runtime_samples_s": fisher_samples,
            "target_runtime_median_s": ewc_train_runtime,
            "target_runtime_samples_s": ewc_timing_samples,
            "counted_runtime_s": ewc_counted_runtime,
            "resource_cost": ewc_row["resource_cost"],
            "max_common_runtime_s": maximum_runtime,
            "metrics": ewc_metrics,
            "penalty_at_anchor": head_ewc.ewc_penalty(
                fixed_initial,
                fixed_initial,
                head_ewc.normalized_fisher(fisher),
            ),
        },
    )
    metrics_rows.append(
        {
            "seed": seed,
            "method": "EWC",
            "lambda_id": fixed_lambda_id,
            "epochs": 3,
            "R_train_true": true_accuracy(splits["R_train"], ewc_predictions),
            "R_train_pseudo": pseudo_accuracy(splits["R_train"], ewc_predictions, pseudo),
            "R_val_true": true_accuracy(splits["R_val"], ewc_predictions),
            "R_val_pseudo": ewc_adapter_pseudo,
            "R_test_true": true_accuracy(splits["R_test"], ewc_predictions),
            "runtime_s": ewc_counted_runtime,
        }
    )
    write_csv(output_dir / "gamma_profiles_common.csv", plain_rows)
    write_json(output_dir / "gamma_profiles_common.json", plain_rows)
    write_json(output_dir / "ewc_gamma_profile.json", ewc_row)
    write_csv(output_dir / "head_accuracy_metrics.csv", metrics_rows)
    write_json(
        output_dir / "training_protocol.json",
        {
            "seed": seed,
            "source_dir": str(source_dir),
            "train_selection_evaluation": disjoint,
            "labels": labels,
            "old_label_count": len(set(old_labels.values())),
            "target_label_count": len(set(target_labels.values())),
            "union_label_count": len(labels),
            "fixed_lambda_id": fixed_lambda_id,
            "common_trainer": str(HEAD_EWC_PATH),
            "plain_ewc_shared_seed": paired_seed,
            "alpha": alpha,
            "ewc_lambda": ewc_lambda,
            "timing_repeats": repeats,
            "gamma_max_resource_cost": gamma_max_resource_cost,
            "max_common_runtime_s": maximum_runtime,
        },
    )
    return {
        "splits": splits,
        "pseudo": pseudo,
        "lambda_rows": lambda_rows,
        "base_predictions": base_predictions,
        "plain_gamma_rows": plain_rows,
        "plain_predictions": plain_predictions,
        "ewc_gamma_row": ewc_row,
        "ewc_predictions": ewc_predictions,
        "fixed_lambda_id": fixed_lambda_id,
        "head_metrics": metrics_rows,
        "disjoint": disjoint,
    }


def decision_args_for_rate(
    pressure: Any,
    source_dir: Path,
    output_dir: Path,
    rate: float,
) -> Namespace:
    return pressure.nonqwen_args_for_rate(source_dir, output_dir, rate)


def evaluate_rates(
    *,
    runner: Any,
    pressure: Any,
    source_dir: Path,
    output_dir: Path,
    seed_bundle: dict[str, Any],
    rates: list[float],
    load_factors: list[float] | None,
) -> list[dict[str, Any]]:
    splits = dict(seed_bundle["splits"])
    splits["R_all"] = list(seed_bundle["splits"]["R_test"])
    rows: list[dict[str, Any]] = []
    for index, rate in enumerate(rates):
        scenario = (
            f"load{load_factors[index]:g}x".replace(".", "p")
            if load_factors is not None
            else f"ar{rate:g}".replace(".", "p")
        )
        scenario_dir = output_dir / scenario
        args = decision_args_for_rate(pressure, source_dir, scenario_dir, rate)
        lambda_rows = pressure.pressure_adjust_lambda_rows(
            seed_bundle["lambda_rows"],
            rate,
        )
        baseline_rows = runner.build_baseline_decisions(
            lambda_rows=lambda_rows,
            gamma_rows=seed_bundle["plain_gamma_rows"],
            predictions_by_lambda=seed_bundle["base_predictions"],
            adapter_predictions=seed_bundle["plain_predictions"],
            splits=splits,
            pseudo=seed_bundle["pseudo"],
            args=args,
        )
        ours = runner.choose_ours(
            lambda_rows=lambda_rows,
            gamma_rows=seed_bundle["plain_gamma_rows"],
            predictions_by_lambda=seed_bundle["base_predictions"],
            adapter_predictions=seed_bundle["plain_predictions"],
            splits=splits,
            pseudo=seed_bundle["pseudo"],
            triggered=True,
            args=args,
        )
        oracle = pressure.choose_nonqwen_offline_oracle_by_actual_long(
            runner=runner,
            lambda_rows=lambda_rows,
            gamma_rows=seed_bundle["plain_gamma_rows"],
            predictions_by_lambda=seed_bundle["base_predictions"],
            adapter_predictions=seed_bundle["plain_predictions"],
            splits=splits,
            pseudo=seed_bundle["pseudo"],
            args=args,
        )
        fixed_lambda = next(
            row
            for row in lambda_rows
            if row["lambda_id"] == seed_bundle["fixed_lambda_id"]
        )
        ewc_gamma = seed_bundle["ewc_gamma_row"]
        ewc = runner.build_decision(
            method="EWC",
            lambda_row=fixed_lambda,
            gamma_row=ewc_gamma,
            r_all=splits["R_all"],
            no_drift_rows=splits["R_no_drift"],
            r_val=splits["R_val"],
            r_test=splits["R_test"],
            pseudo=seed_bundle["pseudo"],
            base_predictions=seed_bundle["base_predictions"][seed_bundle["fixed_lambda_id"]],
            adapter_predictions=seed_bundle["ewc_predictions"],
            inference_resource=0.70,
            training_resource=0.30,
            triggered=True,
            adapter_accepted=True,
            args=args,
            training_start_delay_s=0.0,
        )
        minimum_r = min(0.99, max(0.30, float(ewc_gamma["resource_cost"]) + 1e-6))
        ewc_feasible = runner.build_decision(
            method="EWCMinFeasibleDiagnostic",
            lambda_row=fixed_lambda,
            gamma_row=ewc_gamma,
            r_all=splits["R_all"],
            no_drift_rows=splits["R_no_drift"],
            r_val=splits["R_val"],
            r_test=splits["R_test"],
            pseudo=seed_bundle["pseudo"],
            base_predictions=seed_bundle["base_predictions"][seed_bundle["fixed_lambda_id"]],
            adapter_predictions=seed_bundle["ewc_predictions"],
            inference_resource=1.0 - minimum_r,
            training_resource=minimum_r,
            triggered=True,
            adapter_accepted=True,
            args=args,
            training_start_delay_s=0.0,
        )
        method_rows = baseline_rows + [ewc, ewc_feasible, ours, oracle]
        for row in method_rows:
            row["pressure_scenario"] = scenario
            row["evaluation_stream"] = "R_test_only"
            if load_factors is not None:
                row["load_factor_vs_noretrain_throughput"] = load_factors[index]
        write_csv(scenario_dir / "method_comparison.csv", method_rows)
        write_json(scenario_dir / "method_comparison.json", method_rows)
        rows.extend(method_rows)
    return rows


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    scenario_key: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row[scenario_key]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (scenario, method), group in groups.items():
        values = [float(row["actual_long_avg_accuracy"]) for row in group]
        current = [float(row["observed_time_weighted_accuracy"]) for row in group]
        final = [float(row["observed_final_accuracy"]) for row in group]
        out.append(
            {
                scenario_key: scenario,
                "arrival_rate": float(group[0]["arrival_rate"]),
                "load_factor_vs_noretrain_throughput": group[0].get(
                    "load_factor_vs_noretrain_throughput",
                    "",
                ),
                "method": method,
                "actual_long_mean": statistics.mean(values),
                "actual_long_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "time_weighted_mean": statistics.mean(current),
                "time_weighted_std": statistics.stdev(current) if len(current) > 1 else 0.0,
                "final_mean": statistics.mean(final),
                "final_std": statistics.stdev(final) if len(final) > 1 else 0.0,
                "adapter_replaced_count": sum(bool(row["adapter_replaced"]) for row in group),
                "seed_count": len(group),
                "lambda_ids": ";".join(sorted({str(row["lambda_id"]) for row in group})),
                "gamma_ids": ";".join(sorted({str(row["gamma_id"]) for row in group})),
                "I_values": ";".join(sorted({str(row["I"]) for row in group})),
                "R_values": ";".join(sorted({str(row["R"]) for row in group})),
            }
        )
    out.sort(
        key=lambda row: (
            float(row.get("load_factor_vs_noretrain_throughput") or row["arrival_rate"]),
            ALL_METHODS.index(str(row["method"])),
        )
    )
    return out


def comparison_counts(
    aggregate: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    scenario_key: str,
) -> dict[str, Any]:
    by_scenario: dict[str, dict[str, float]] = {}
    for row in aggregate:
        by_scenario.setdefault(str(row[scenario_key]), {})[str(row["method"])] = float(
            row["actual_long_mean"]
        )
    deltas = [
        values[left] - values[right]
        for values in by_scenario.values()
        if left in values and right in values
    ]
    return {
        "scenarios": len(deltas),
        "wins": sum(value > 1e-12 for value in deltas),
        "ties": sum(abs(value) <= 1e-12 for value in deltas),
        "losses": sum(value < -1e-12 for value in deltas),
        "mean_delta": statistics.mean(deltas) if deltas else 0.0,
    }


def write_report(
    *,
    output_root: Path,
    absolute_aggregate: list[dict[str, Any]],
    normalized_aggregate: list[dict[str, Any]],
    head_metrics: list[dict[str, Any]],
    seed_protocols: list[dict[str, Any]],
) -> dict[str, Any]:
    absolute_ewc = comparison_counts(
        absolute_aggregate,
        "EWC",
        "Ours",
        scenario_key="pressure_scenario",
    )
    absolute_diag = comparison_counts(
        absolute_aggregate,
        "EWCMinFeasibleDiagnostic",
        "Ours",
        scenario_key="pressure_scenario",
    )
    normalized_ewc = comparison_counts(
        normalized_aggregate,
        "EWC",
        "Ours",
        scenario_key="pressure_scenario",
    )
    normalized_diag = comparison_counts(
        normalized_aggregate,
        "EWCMinFeasibleDiagnostic",
        "Ours",
        scenario_key="pressure_scenario",
    )
    paired_rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in head_metrics}):
        plain = next(
            row
            for row in head_metrics
            if int(row["seed"]) == seed
            and row["method"] == "plain"
            and row["lambda_id"] == "rc_r50_keep100_fp32_b32"
            and int(row["epochs"]) == 3
        )
        ewc = next(
            row
            for row in head_metrics
            if int(row["seed"]) == seed and row["method"] == "EWC"
        )
        paired_rows.append(
            {
                "seed": seed,
                "plain_R_test_true": plain["R_test_true"],
                "ewc_R_test_true": ewc["R_test_true"],
                "ewc_minus_plain": float(ewc["R_test_true"]) - float(plain["R_test_true"]),
                "plain_runtime_s": plain["runtime_s"],
                "ewc_counted_runtime_s": ewc["runtime_s"],
            }
        )
    result = {
        "absolute_EWC_vs_Ours": absolute_ewc,
        "absolute_EWC_min_feasible_vs_Ours": absolute_diag,
        "normalized_EWC_vs_Ours": normalized_ewc,
        "normalized_EWC_min_feasible_vs_Ours": normalized_diag,
        "paired_head_results": paired_rows,
        "seed_protocols": seed_protocols,
    }
    write_json(output_root / "aggregate" / "summary.json", result)
    write_csv(output_root / "aggregate" / "paired_head_results.csv", paired_rows)
    lines = [
        "# RemoteCLIP Fair Plain-vs-EWC Rerun",
        "",
        "## Corrected protocol",
        "",
        "- Frozen RemoteCLIP-RN50 backbone and shared feature tensors.",
        "- Plain adaptation and EWC use the same linear-head trainer, zero initialization, 17-label space, and paired shuffle seed.",
        "- Plain differs from EWC only by `ewc_lambda=0` versus `ewc_lambda=10` plus Fisher overhead.",
        "- `R_train` is adaptation-only, `R_val` is selection-only, and `R_test` is the only headline evaluation stream.",
        "- Gamma resource costs are recomputed from median counted runtimes on one common scale.",
        "",
        "## Paired held-out head accuracy",
        "",
        "| seed | plain ep3 R_test | EWC ep3 R_test | EWC - plain | plain runtime(s) | EWC counted runtime(s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['seed']} | {float(row['plain_R_test_true']):.6f} | "
            f"{float(row['ewc_R_test_true']):.6f} | "
            f"{float(row['ewc_minus_plain']):+.6f} | "
            f"{float(row['plain_runtime_s']):.6f} | "
            f"{float(row['ewc_counted_runtime_s']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Pressure comparison",
            "",
            f"- Absolute EWC vs Ours: {absolute_ewc['wins']}/{absolute_ewc['ties']}/{absolute_ewc['losses']} "
            f"(mean delta {absolute_ewc['mean_delta']:+.6f}).",
            f"- Absolute minimum-feasible EWC diagnostic vs Ours: "
            f"{absolute_diag['wins']}/{absolute_diag['ties']}/{absolute_diag['losses']} "
            f"(mean delta {absolute_diag['mean_delta']:+.6f}).",
            f"- Normalized EWC vs Ours: {normalized_ewc['wins']}/{normalized_ewc['ties']}/{normalized_ewc['losses']} "
            f"(mean delta {normalized_ewc['mean_delta']:+.6f}).",
            f"- Normalized minimum-feasible EWC diagnostic vs Ours: "
            f"{normalized_diag['wins']}/{normalized_diag['ties']}/{normalized_diag['losses']} "
            f"(mean delta {normalized_diag['mean_delta']:+.6f}).",
            "",
            "The `EWC` row keeps the matched Static split I/R=0.7/0.3 and may be infeasible after corrected cost accounting. "
            "`EWCMinFeasibleDiagnostic` gives EWC the smallest training share that satisfies its measured resource cost; it is diagnostic, not a main-table method.",
        ]
    )
    (output_root / "aggregate" / "FINAL_FAIR_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return result


def validate_source_identity(source_root: Path, seeds: list[int]) -> dict[str, Any]:
    files = [
        Path("splits/R_train/data.json"),
        Path("splits/R_val/data.json"),
        Path("splits/R_test/data.json"),
        Path("splits/history_no_drift/data.json"),
        Path("teacher/pseudo_labels/R_train_pseudo.jsonl"),
        Path("teacher/pseudo_labels/R_val_pseudo.jsonl"),
    ]
    hashes: dict[str, dict[int, str]] = {}
    for relative in files:
        observed = {
            seed: sha256_file(source_root / f"seed{seed}" / "remoteclip" / relative)
            for seed in seeds
        }
        if len(set(observed.values())) != 1:
            raise ValueError(f"Source inputs differ across seeds for {relative}: {observed}")
        hashes[str(relative)] = observed
    return {"cross_seed_identical_files": hashes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a corrected paired RemoteCLIP plain-vs-EWC experiment."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--remoteclip-dir", type=Path, default=DEFAULT_REMOTECLIP_DIR)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--gamma-max-resource-cost", type=float, default=0.35)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.remoteclip_dir = args.remoteclip_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.timing_repeats < 3:
        raise ValueError("Use at least three timing repeats")

    runner = load_module(RUNNER_PATH, "remoteclip_fair_runner")
    pressure = load_module(PRESSURE_PATH, "remoteclip_fair_pressure")
    head_ewc = load_module(HEAD_EWC_PATH, "remoteclip_fair_head_ewc")
    pressure.configure_nonqwen_runner(runner, "remoteclip")

    identity = validate_source_identity(args.source_root, seeds)
    reference_source = args.source_root / f"seed{seeds[0]}" / "remoteclip"
    reference_splits = runner.load_existing_splits(reference_source / "splits")
    feature_rows = unique_rows(
        reference_splits["history_no_drift"],
        reference_splits["R_train"],
        reference_splits["R_val"],
        reference_splits["R_test"],
        reference_splits["R_no_drift"],
    )
    features_by_lambda = ensure_feature_caches(
        runner=runner,
        source_dir=reference_source,
        output_root=args.output_root,
        remoteclip_dir=args.remoteclip_dir,
        rows=feature_rows,
        device=args.device,
        force=args.force_features,
    )

    all_absolute: list[dict[str, Any]] = []
    all_normalized: list[dict[str, Any]] = []
    all_head_metrics: list[dict[str, Any]] = []
    seed_protocols: list[dict[str, Any]] = []
    for seed in seeds:
        source_dir = args.source_root / f"seed{seed}" / "remoteclip"
        output_dir = args.output_root / f"seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = output_dir / "complete.json"
        if completed.exists() and not args.force:
            raise FileExistsError(
                f"{completed} exists; use --force to rebuild the fair run"
            )
        print(f"[seed {seed}] training paired heads", flush=True)
        bundle = train_seed(
            seed=seed,
            runner=runner,
            head_ewc=head_ewc,
            source_dir=source_dir,
            output_dir=output_dir,
            features_by_lambda=features_by_lambda,
            repeats=args.timing_repeats,
            alpha=args.alpha,
            ewc_lambda=args.ewc_lambda,
            gamma_max_resource_cost=args.gamma_max_resource_cost,
        )
        full_lambda = next(
            row
            for row in bundle["lambda_rows"]
            if row["lambda_id"] == bundle["fixed_lambda_id"]
        )
        reference_throughput = float(full_lambda["throughput_samples_per_sec"])
        normalized_rates = [reference_throughput * factor for factor in LOAD_FACTORS]
        absolute_rows = evaluate_rates(
            runner=runner,
            pressure=pressure,
            source_dir=source_dir,
            output_dir=output_dir / "absolute",
            seed_bundle=bundle,
            rates=ABSOLUTE_RATES,
            load_factors=None,
        )
        normalized_rows = evaluate_rates(
            runner=runner,
            pressure=pressure,
            source_dir=source_dir,
            output_dir=output_dir / "normalized",
            seed_bundle=bundle,
            rates=normalized_rates,
            load_factors=LOAD_FACTORS,
        )
        for row in absolute_rows:
            row["seed"] = seed
        for row in normalized_rows:
            row["seed"] = seed
            row["reference_noretrain_throughput_samples_per_sec"] = reference_throughput
        write_csv(output_dir / "absolute_method_results.csv", absolute_rows)
        write_csv(output_dir / "normalized_method_results.csv", normalized_rows)
        write_json(
            completed,
            {
                "seed": seed,
                "completed_at_unix": time.time(),
                "absolute_rows": len(absolute_rows),
                "normalized_rows": len(normalized_rows),
                "evaluation_stream": "R_test_only",
                "physical_gpu": args.physical_gpu,
            },
        )
        all_absolute.extend(absolute_rows)
        all_normalized.extend(normalized_rows)
        all_head_metrics.extend(bundle["head_metrics"])
        seed_protocols.append(read_json(output_dir / "training_protocol.json"))

    aggregate_dir = args.output_root / "aggregate"
    absolute_aggregate = aggregate_rows(
        all_absolute,
        scenario_key="pressure_scenario",
    )
    normalized_aggregate = aggregate_rows(
        all_normalized,
        scenario_key="pressure_scenario",
    )
    write_csv(aggregate_dir / "absolute_seed_runs.csv", all_absolute)
    write_csv(aggregate_dir / "normalized_seed_runs.csv", all_normalized)
    write_csv(aggregate_dir / "absolute_mean_std.csv", absolute_aggregate)
    write_csv(aggregate_dir / "normalized_mean_std.csv", normalized_aggregate)
    write_csv(aggregate_dir / "head_accuracy_metrics.csv", all_head_metrics)
    summary = write_report(
        output_root=args.output_root,
        absolute_aggregate=absolute_aggregate,
        normalized_aggregate=normalized_aggregate,
        head_metrics=all_head_metrics,
        seed_protocols=seed_protocols,
    )
    write_json(
        args.output_root / "protocol_manifest.json",
        {
            "date": "2026-07-24",
            "physical_gpu": args.physical_gpu,
            "source_root": str(args.source_root),
            "output_root": str(args.output_root),
            "remoteclip_dir": str(args.remoteclip_dir),
            "remoteclip_model_sha256": sha256_file(
                args.remoteclip_dir / "RemoteCLIP-RN50.pt"
            ),
            "seeds": seeds,
            "identity": identity,
            "feature_row_count": len(feature_rows),
            "feature_row_uid_sha256": uid_hash(feature_rows),
            "train_split": "R_train",
            "selection_split": "R_val",
            "evaluation_stream": "R_test_only",
            "common_trainer": str(HEAD_EWC_PATH),
            "plain_ewc_only_delta": "ewc_lambda=0 versus 10 plus Fisher overhead",
            "timing_repeats": args.timing_repeats,
            "gamma_max_resource_cost": args.gamma_max_resource_cost,
            "absolute_rates": ABSOLUTE_RATES,
            "load_factors": LOAD_FACTORS,
            "summary": summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[complete] {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
