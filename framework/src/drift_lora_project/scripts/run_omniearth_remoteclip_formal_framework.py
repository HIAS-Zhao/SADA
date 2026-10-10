#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCREENING_SCRIPT = SCRIPT_DIR / "screen_omniearth_small_experiments.py"
FLOW_SCRIPT = SCRIPT_DIR / "run_omniearth_small_specialist_flow.py"
SEARCH_SCRIPT = SCRIPT_DIR / "search_omniearth_reference_drift_tasks.py"

REFERENCE_TASK = "perception_a1_2_land_cover_classification"
DRIFT_TASK = "reasoning_b3_2_disaster_cause_inference"
MODEL_ID = "RemoteCLIP-RN50"
GROUP_ID = "remoteclip_rn50_landcover_disaster"
METHODS = ["NoRetrain", "PeriodicFixedRetrain", "StaticSplitContinuous", "Ours", "OfflineJointOracle"]
DEFAULT_CANONICAL_TIMING_JSON = (
    PROJECT_ROOT
    / "results"
    / "omniearth_remoteclip_student_vitb32teacher_formal_landcover200_disaster365_q095_r50_20260705_no_pareto"
    / "lambda_profiles"
    / "lambda_profiles_group_v2.json"
)


@dataclass(frozen=True)
class LambdaConfig:
    lambda_id: str
    resolution_keep_ratio: float
    model_precision: str
    batch_size: int = 32


@dataclass(frozen=True)
class GammaConfig:
    gamma_id: str
    epochs: int


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


screening = load_module(SCREENING_SCRIPT, "screen_omniearth_small_experiments")
flow = load_module(FLOW_SCRIPT, "run_omniearth_small_specialist_flow")
search = load_module(SEARCH_SCRIPT, "search_omniearth_reference_drift_tasks")


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


def load_canonical_lambda_timing(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a list")
    out = {str(row["lambda_id"]): dict(row) for row in rows}
    if not out:
        raise ValueError(f"No canonical lambda timing rows in {path}")
    return out


def lambda_timing_fields(args: argparse.Namespace, lambda_id: str, run: dict[str, Any]) -> dict[str, Any]:
    timing_by_id = getattr(args, "canonical_lambda_timing_by_id", {})
    if lambda_id not in timing_by_id:
        raise KeyError(f"Missing canonical timing for {lambda_id}")
    canonical = timing_by_id[lambda_id]
    latency = float(canonical.get("latency_mean_sec", canonical.get("latency_mean_s")))
    throughput = float(canonical["throughput_samples_per_sec"])
    return {
        "latency_mean_sec": latency,
        "latency_p50_sec": float(canonical.get("latency_p50_sec", latency)),
        "latency_p95_sec": float(canonical.get("latency_p95_sec", latency)),
        "throughput_samples_per_sec": throughput,
        "timing_source": str(args.canonical_lambda_timing_json),
        "timing_reused": True,
        "observed_latency_mean_sec": float(run["latency_mean_sec"]),
        "observed_throughput_samples_per_sec": float(run["throughput_samples_per_sec"]),
        "observed_elapsed_sec": float(run["elapsed_sec"]),
    }


def read_split(split_root: Path, name: str) -> list[dict[str, Any]]:
    path = split_root / name / "data.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_splits(split_root: Path) -> dict[str, list[dict[str, Any]]]:
    split_names = [
        "history_no_drift",
        "history_reference_fit",
        "history_threshold_cal",
        "teacher_ft_train",
        "teacher_ft_train_no_drift",
        "teacher_ft_train_drift",
        "teacher_ft_holdout_drift",
        "D_no_drift",
        "D_drift",
        "D_window_drift",
        "D_window",
        "D_top40",
        "D_top80",
        "R_no_drift",
        "R_drift",
        "R_mixed_drift",
        "R_mixed_diagnostic",
        "R_all",
        "R_train",
        "R_val",
        "R_test",
    ]
    splits = {name: read_split(split_root, name) for name in split_names}
    required = [
        "history_reference_fit",
        "history_threshold_cal",
        "teacher_ft_train",
        "D_no_drift",
        "D_drift",
        "D_window",
        "R_no_drift",
        "R_all",
        "R_train",
        "R_val",
        "R_test",
    ]
    missing = [name for name in required if not splits.get(name)]
    if missing:
        raise FileNotFoundError(f"Missing required split(s) under {split_root}: {missing}")
    return splits


def stable_seed(text: str) -> int:
    return flow.stable_seed(text)


def stable_order(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_seed(f"{seed}:{row['uid']}"))


def unique_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            uid = str(row["uid"])
            if uid not in seen:
                out.append(row)
                seen.add(uid)
    return out


def take_stratified(
    rows: list[dict[str, Any]],
    count: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if count <= 0:
        return [], list(rows)
    if count >= len(rows):
        return stable_order(rows, seed), []
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    labels = sorted(by_label)
    for label in labels:
        by_label[label] = stable_order(by_label[label], f"{seed}:{label}")
    allocations = {label: int(math.floor(count * len(by_label[label]) / len(rows))) for label in labels}
    remaining = count - sum(allocations.values())
    remainders = sorted(
        (
            count * len(by_label[label]) / len(rows) - allocations[label],
            stable_seed(f"{seed}:rem:{label}"),
            label,
        )
        for label in labels
    )
    for _frac, _tie, label in reversed(remainders):
        if remaining <= 0:
            break
        if allocations[label] < len(by_label[label]):
            allocations[label] += 1
            remaining -= 1
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label in labels:
        picked = by_label[label][: allocations[label]]
        selected.extend(picked)
        seen.update(str(row["uid"]) for row in picked)
    if len(selected) < count:
        for row in stable_order(rows, f"{seed}:fill"):
            uid = str(row["uid"])
            if uid in seen:
                continue
            selected.append(row)
            seen.add(uid)
            if len(selected) == count:
                break
    rest = [row for row in rows if str(row["uid"]) not in seen]
    return stable_order(selected, f"{seed}:selected"), stable_order(rest, f"{seed}:rest")


def row_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": row["uid"],
        "task_key": row["task_key"],
        "category": row["category"],
        "image_path": row["image_path"],
        "question": row["question"],
        "options": row["options"],
        "answer": row["answer"],
        "label": row["label"],
    }


def write_split(path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(path / "data.json", [row_brief(row) for row in rows])


def build_mixed_stream(no_rows: list[dict[str, Any]], drift_rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    if len(no_rows) != len(drift_rows):
        raise ValueError("Mixed stream requires equal no-drift and drift rows")
    rng = np.random.default_rng(stable_seed(seed))
    indexes = np.arange(len(no_rows))
    rng.shuffle(indexes)
    stream: list[dict[str, Any]] = []
    for idx in indexes:
        pair = [no_rows[int(idx)], drift_rows[int(idx)]]
        if bool(rng.integers(0, 2)):
            pair.reverse()
        stream.extend(pair)
    return stream


def split_r(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(rows)
    train_n = int(round(total * (180 / 390)))
    val_n = int(round(total * (100 / 390)))
    if train_n + val_n >= total:
        train_n = max(1, total - 2)
        val_n = 1
    return rows[:train_n], rows[train_n : train_n + val_n], rows[train_n + val_n :]


def split_dataset(
    reference_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    need_no = (
        args.history_reference_count
        + args.history_threshold_count
        + args.teacher_no_count
        + args.d_no_count
        + args.r_no_count
    )
    need_drift = (
        args.teacher_drift_count
        + args.teacher_drift_holdout_count
        + args.d_drift_count
        + args.r_drift_count
    )
    if need_no > len(reference_rows):
        raise ValueError(f"Need {need_no} reference rows, got {len(reference_rows)}")
    if need_drift > len(drift_rows):
        raise ValueError(f"Need {need_drift} drift rows, got {len(drift_rows)}")

    history_reference, no_rest = take_stratified(
        reference_rows,
        args.history_reference_count,
        f"{args.split_seed}:history:reference",
    )
    history_threshold, no_rest = take_stratified(
        no_rest,
        args.history_threshold_count,
        f"{args.split_seed}:history:threshold",
    )
    teacher_no, no_rest = take_stratified(no_rest, args.teacher_no_count, f"{args.split_seed}:teacher:no")
    d_no, no_rest = take_stratified(no_rest, args.d_no_count, f"{args.split_seed}:D:no")
    r_no, _no_unused = take_stratified(no_rest, args.r_no_count, f"{args.split_seed}:R:no")

    teacher_drift, drift_rest = take_stratified(drift_rows, args.teacher_drift_count, f"{args.split_seed}:teacher:drift")
    teacher_drift_holdout, drift_rest = take_stratified(
        drift_rest,
        args.teacher_drift_holdout_count,
        f"{args.split_seed}:teacher:drift_holdout",
    )
    d_drift, drift_rest = take_stratified(drift_rest, args.d_drift_count, f"{args.split_seed}:D:drift")
    r_drift, _drift_unused = take_stratified(drift_rest, args.r_drift_count, f"{args.split_seed}:R:drift")

    d_window_drift, _ = take_stratified(
        d_drift,
        min(len(d_no), len(d_drift)),
        f"{args.split_seed}:D:window_drift_subset",
    )
    r_mixed_drift, _ = take_stratified(
        r_drift,
        min(len(r_no), len(r_drift)),
        f"{args.split_seed}:R:mixed_drift_subset",
    )
    d_window = build_mixed_stream(d_no, d_window_drift, f"{args.split_seed}:D:window")
    r_mixed_diagnostic = build_mixed_stream(r_no, r_mixed_drift, f"{args.split_seed}:R:diagnostic_mixed")
    r_all = stable_order(r_drift, f"{args.split_seed}:R:drift_only_formal")
    r_train, r_val, r_test = split_r(r_all)
    return {
        "history_no_drift": history_reference + history_threshold,
        "history_reference_fit": history_reference,
        "history_threshold_cal": history_threshold,
        "teacher_ft_train": teacher_no + teacher_drift,
        "teacher_ft_train_no_drift": teacher_no,
        "teacher_ft_train_drift": teacher_drift,
        "teacher_ft_holdout_drift": teacher_drift_holdout,
        "D_no_drift": d_no,
        "D_drift": d_drift,
        "D_window_drift": d_window_drift,
        "D_window": d_window,
        "R_no_drift": r_no,
        "R_drift": r_drift,
        "R_mixed_drift": r_mixed_drift,
        "R_mixed_diagnostic": r_mixed_diagnostic,
        "R_all": r_all,
        "R_train": r_train,
        "R_val": r_val,
        "R_test": r_test,
    }


def option_label(row: dict[str, Any], letter: str) -> str | None:
    value = row.get("options", {}).get(letter)
    if value is None:
        return None
    return screening.clean_label(str(value))


def feature_matrix(features: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([features[row["uid"]] for row in rows], dtype=np.float64)


def fit_mahalanobis_detector(
    features: dict[str, np.ndarray],
    fit_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fit_x = feature_matrix(features, fit_rows)
    cal_x = feature_matrix(features, cal_rows)
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_x)
    cal_scaled = scaler.transform(cal_x)
    n_components = max(1, min(args.pca_components, fit_scaled.shape[0] - 1, fit_scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=stable_seed(f"{args.split_seed}:detector:pca"))
    fit_z = pca.fit_transform(fit_scaled)
    _cal_z = pca.transform(cal_scaled)
    mu = fit_z.mean(axis=0)
    cov = np.cov(fit_z, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]])
    cov = cov + np.eye(cov.shape[0]) * 1e-4
    inv_cov = np.linalg.pinv(cov)

    def sample_distances(rows: list[dict[str, Any]]) -> np.ndarray:
        x = feature_matrix(features, rows)
        z = pca.transform(scaler.transform(x))
        delta = z - mu
        sq = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
        return np.sqrt(np.maximum(0.0, sq))

    def score_window(rows: list[dict[str, Any]]) -> float:
        z = pca.transform(scaler.transform(feature_matrix(features, rows)))
        delta = z.mean(axis=0) - mu
        return float(np.sqrt(max(0.0, delta @ inv_cov @ delta.T)))

    rng = np.random.default_rng(stable_seed(f"{args.split_seed}:detector:threshold"))
    cal_scores = []
    for _ in range(args.calibration_windows):
        indexes = rng.integers(0, len(cal_rows), size=args.window_size)
        window = [cal_rows[int(idx)] for idx in indexes]
        cal_scores.append(score_window(window))
    threshold = float(np.quantile(cal_scores, args.threshold_quantile))
    return {
        "scaler": scaler,
        "pca": pca,
        "sample_distances": sample_distances,
        "score_window": score_window,
        "threshold": threshold,
        "threshold_quantile": args.threshold_quantile,
        "pca_components": n_components,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "calibration_scores": cal_scores,
    }


def sample_with_replacement(rows: list[dict[str, Any]], count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    indexes = rng.integers(0, len(rows), size=count)
    return [rows[int(idx)] for idx in indexes]


def evaluate_detector(
    detector: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"{args.split_seed}:detector:eval"))
    eval_rows = []
    for ratio in (0.0, args.drift_ratio):
        drift_n = int(round(args.window_size * ratio))
        no_n = args.window_size - drift_n
        positives = 0
        scores = []
        for _ in range(args.eval_windows):
            window = sample_with_replacement(splits["D_no_drift"], no_n, rng)
            if drift_n:
                window += sample_with_replacement(splits["D_drift"], drift_n, rng)
            rng.shuffle(window)
            score = detector["score_window"](window)
            scores.append(score)
            positives += int(score > detector["threshold"])
        eval_rows.append(
            {
                "drift_ratio": ratio,
                "positive_rate": positives / args.eval_windows,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "threshold": detector["threshold"],
            }
        )
    d_window_scores = []
    trigger_position = None
    for start in range(0, max(0, len(splits["D_window"]) - args.window_size + 1), args.window_stride):
        window = splits["D_window"][start : start + args.window_size]
        score = detector["score_window"](window)
        positive = bool(score > detector["threshold"])
        if positive and trigger_position is None:
            trigger_position = start
        d_window_scores.append(
            {
                "start": start,
                "end": start + args.window_size,
                "score": float(score),
                "drift_ratio": sum(1 for row in window if row["task_key"] == args.drift_task) / len(window),
                "positive": positive,
            }
        )
    by_ratio = {round(row["drift_ratio"], 6): row for row in eval_rows}
    return {
        "threshold": detector["threshold"],
        "threshold_quantile": args.threshold_quantile,
        "fpr_at_0": by_ratio[0.0]["positive_rate"],
        "recall_at_50": by_ratio[round(args.drift_ratio, 6)]["positive_rate"],
        "mean_score_0": by_ratio[0.0]["mean_score"],
        "mean_score_50": by_ratio[round(args.drift_ratio, 6)]["mean_score"],
        "actual_D_window_triggered": trigger_position is not None,
        "actual_D_window_trigger_position": trigger_position if trigger_position is not None else "",
        "D_window_scores": d_window_scores,
        "eval_rows": eval_rows,
    }


def select_d_top_rows(
    detector: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    top_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    distances = detector["sample_distances"](splits["D_window"])
    scored = []
    for row, score in zip(splits["D_window"], distances):
        scored.append({"uid": row["uid"], "task_key": row["task_key"], "score": float(score), "answer": row["answer"]})
    scored_sorted = sorted(scored, key=lambda item: (-item["score"], item["uid"]))
    top_uids = {row["uid"] for row in scored_sorted[: min(top_count, len(scored_sorted))]}
    top_rows = [row for row in splits["D_window"] if row["uid"] in top_uids]
    top_rows = sorted(top_rows, key=lambda row: (-next(item["score"] for item in scored if item["uid"] == row["uid"]), row["uid"]))
    return top_rows, scored_sorted


def resolve_d_top_count(splits: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> int:
    if args.d_top_count is not None:
        return min(args.d_top_count, len(splits["D_window"]))
    return max(1, int(math.ceil(len(splits["D_window"]) * args.d_top_fraction)))


def train_head_sgd(
    features: dict[str, np.ndarray],
    train_rows: list[dict[str, Any]],
    *,
    label_letters: dict[str, str] | None,
    epochs: int,
    seed: str,
) -> tuple[dict[str, Any], float]:
    x_rows = []
    y_labels = []
    skipped = 0
    for row in train_rows:
        if label_letters is None:
            label = str(row["label"])
        else:
            letter = label_letters.get(row["uid"])
            label = option_label(row, letter) if letter else None
        if not label:
            skipped += 1
            continue
        x_rows.append(features[row["uid"]])
        y_labels.append(label)
    if len(set(y_labels)) < 2:
        raise ValueError("Need at least two classes to train a linear head")
    x = np.asarray(x_rows, dtype=np.float64)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        random_state=stable_seed(seed),
        learning_rate="optimal",
        max_iter=1,
        tol=None,
    )
    rng = np.random.default_rng(stable_seed(f"{seed}:shuffle"))
    classes = np.arange(len(encoder.classes_))
    started = time.perf_counter()
    for epoch in range(epochs):
        order = np.arange(len(y))
        rng.shuffle(order)
        classifier.partial_fit(x_scaled[order], y[order], classes=classes)
    elapsed = time.perf_counter() - started
    return {"scaler": scaler, "encoder": encoder, "classifier": classifier, "skipped": skipped}, elapsed


def predict_head(
    model: dict[str, Any],
    features: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    scaler: StandardScaler = model["scaler"]
    encoder: LabelEncoder = model["encoder"]
    classifier: SGDClassifier = model["classifier"]
    class_index = {label: idx for idx, label in enumerate(encoder.classes_)}
    out: dict[str, str] = {}
    for row in rows:
        x = scaler.transform(np.asarray([features[row["uid"]]], dtype=np.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            probs = classifier.predict_proba(x)[0]
        use_scores = probs
        if not np.all(np.isfinite(use_scores)) or float(np.nansum(use_scores)) <= 0.0:
            decision = classifier.decision_function(x)
            use_scores = np.asarray(decision[0] if np.ndim(decision) > 1 else decision, dtype=np.float64)
            if use_scores.shape[0] != len(encoder.classes_):
                use_scores = np.asarray([-float(use_scores[0]), float(use_scores[0])], dtype=np.float64)
        best_score = -float("inf")
        best_letter = None
        for letter, option in row["options"].items():
            label = screening.clean_label(str(option))
            if label not in class_index:
                continue
            score = float(use_scores[class_index[label]])
            if score > best_score:
                best_score = score
                best_letter = letter
        out[row["uid"]] = best_letter if best_letter is not None else sorted(row["options"])[0]
    return out


def accuracy_true(rows: list[dict[str, Any]], predictions: dict[str, str]) -> float:
    if not rows:
        return 0.0
    return sum(int(predictions.get(row["uid"]) == row["answer"]) for row in rows) / len(rows)


def accuracy_pseudo(rows: list[dict[str, Any]], predictions: dict[str, str], pseudo: dict[str, str]) -> float:
    rows_with_labels = [row for row in rows if row["uid"] in pseudo]
    if not rows_with_labels:
        return 0.0
    return sum(int(predictions.get(row["uid"]) == pseudo[row["uid"]]) for row in rows_with_labels) / len(rows_with_labels)


def load_external_pseudo_labels(pseudo_dir: Path) -> dict[str, str]:
    pseudo: dict[str, str] = {}
    for split in ("D_window", "D_top40", "R_train", "R_val"):
        path = pseudo_dir / f"{split}_pseudo.jsonl"
        if split == "D_window" and not path.exists():
            continue
        if split == "D_top40" and not path.exists():
            path = pseudo_dir / "D_top80_pseudo.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing external pseudo-label file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = row.get("pseudo_label") or row.get("prediction")
                if not label:
                    raise ValueError(f"Missing pseudo label for row in {path}: {row}")
                pseudo[str(row["uid"])] = str(label)
    return pseudo


def pseudo_accuracy_true(rows: list[dict[str, Any]], pseudo: dict[str, str]) -> float:
    if not rows:
        return 0.0
    return sum(int(pseudo.get(row["uid"]) == row["answer"]) for row in rows) / len(rows)


def write_predictions(path: Path, rows: list[dict[str, Any]], predictions: dict[str, str], pseudo: dict[str, str] | None = None) -> None:
    payload = []
    for row in rows:
        payload.append(
            {
                "uid": row["uid"],
                "task_key": row["task_key"],
                "prediction": predictions.get(row["uid"], ""),
                "answer": row["answer"],
                "pseudo_label": pseudo.get(row["uid"], "") if pseudo else "",
                "true_correct": int(predictions.get(row["uid"]) == row["answer"]),
                "pseudo_correct": int(predictions.get(row["uid"]) == pseudo.get(row["uid"])) if pseudo else "",
            }
        )
    write_jsonl(path, payload)


def remoteclip_configs() -> list[LambdaConfig]:
    configs = []
    for keep in (0.50, 0.75, 1.00):
        for precision in ("fp32", "fp16"):
            configs.append(
                LambdaConfig(
                    lambda_id=f"rc_r50_keep{int(keep * 100):03d}_{precision}_b32",
                    resolution_keep_ratio=keep,
                    model_precision=precision,
                )
            )
    return configs


def gamma_configs() -> list[GammaConfig]:
    return [GammaConfig(gamma_id=f"rc_r50_linear_ep{epochs}", epochs=epochs) for epochs in (1, 3, 5)]


def load_remoteclip(model_dir: Path, device: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    import open_clip

    weight_path = model_dir / "RemoteCLIP-RN50.pt"
    model, _, preprocess = open_clip.create_model_and_transforms("RN50", pretrained=str(weight_path), device=device)
    tokenizer = open_clip.get_tokenizer("RN50")
    model.eval()
    params = screening.remoteclip_param_count(weight_path)
    return model, preprocess, tokenizer, params


def maybe_downsample_image(image: Any, keep_ratio: float) -> Any:
    if keep_ratio >= 0.999:
        return image
    from PIL import Image

    width, height = image.size
    new_size = (max(1, int(round(width * keep_ratio))), max(1, int(round(height * keep_ratio))))
    return image.resize(new_size, Image.BICUBIC)


def extract_remoteclip_for_config(
    *,
    config: LambdaConfig,
    rows: list[dict[str, Any]],
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: str,
    out_path: Path,
    pseudo: dict[str, str] | None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    features: dict[str, np.ndarray] = {}
    predictions: dict[str, str] = {}
    started = time.perf_counter()
    dtype = torch.float16 if config.model_precision == "fp16" and device.startswith("cuda") else torch.float32
    amp_enabled = config.model_precision == "fp16" and device.startswith("cuda")
    with torch.no_grad():
        for row in rows:
            image = Image.open(row["image_path"]).convert("RGB")
            image = maybe_downsample_image(image, config.resolution_keep_ratio)
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
                image_feature = F.normalize(model.encode_image(tensor), dim=-1)
                scores: list[tuple[float, str]] = []
                for letter, option in row["options"].items():
                    prompts = screening.remoteclip_prompts(row["question"], str(option))
                    text_tokens = tokenizer(prompts).to(device)
                    text_feature = F.normalize(model.encode_text(text_tokens), dim=-1).mean(dim=0)
                    text_feature = F.normalize(text_feature.unsqueeze(0), dim=-1)
                    scores.append((float((image_feature * text_feature).sum().detach().cpu()), letter))
            features[row["uid"]] = image_feature.float().detach().cpu().numpy()[0]
            predictions[row["uid"]] = max(scores)[1]
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.startswith("cuda") else 0.0
    write_predictions(out_path, rows, predictions, pseudo)
    return {
        "config": config,
        "features": features,
        "predictions": predictions,
        "elapsed_sec": elapsed,
        "latency_mean_sec": elapsed / len(rows) if rows else 0.0,
        "throughput_samples_per_sec": len(rows) / elapsed if elapsed > 0 else 0.0,
        "peak_memory_gib": peak,
        "raw_predictions": str(out_path),
    }


def pareto_selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                float(other["teacher_accuracy"]) >= float(row["teacher_accuracy"])
                and float(other["resource_cost"]) <= float(row["resource_cost"])
                and (
                    float(other["teacher_accuracy"]) > float(row["teacher_accuracy"])
                    or float(other["resource_cost"]) < float(row["resource_cost"])
                )
            ):
                dominated = True
                break
        if not dominated:
            selected.append(row)
    return selected or [max(rows, key=lambda item: (float(item["teacher_accuracy"]), -float(item["resource_cost"])))]


def normalize_lambda_costs(rows: list[dict[str, Any]], max_cost: float) -> None:
    max_latency = max(float(row["latency_mean_sec"]) for row in rows) if rows else 1.0
    for row in rows:
        latency_ratio = float(row["latency_mean_sec"]) / max(max_latency, 1e-9)
        row["resource_cost"] = round(max(0.01, min(0.95, latency_ratio * max_cost)), 6)


def scheduler_row_accuracy(
    rows: list[dict[str, Any]],
    base_predictions: dict[str, str],
    adapter_predictions: dict[str, str] | None,
    *,
    ready_index: int,
    coverage: float,
) -> dict[str, Any]:
    total = len(rows)
    ready_index = min(total, max(0, ready_index))
    before_rows = rows[:ready_index]
    after_rows = rows[ready_index:] if adapter_predictions is not None else []
    before_correct = sum(int(base_predictions.get(row["uid"]) == row["answer"]) for row in before_rows)
    after_correct = sum(int(adapter_predictions.get(row["uid"]) == row["answer"]) for row in after_rows)
    raw_correct = before_correct + after_correct
    processed_only = raw_correct / total if total else 0.0
    observed_current = coverage * processed_only
    processed = coverage * total
    final_predictions = adapter_predictions if adapter_predictions is not None else base_predictions
    final_correct = sum(int(final_predictions.get(row["uid"]) == row["answer"]) for row in rows)
    by_task: dict[str, Any] = {}
    for prefix, task_key in (("no_drift", REFERENCE_TASK), ("drift", DRIFT_TASK)):
        task_rows = [row for row in rows if row["task_key"] == task_key]
        task_correct = 0
        task_final_correct = 0
        for index, row in enumerate(rows):
            if row["task_key"] != task_key:
                continue
            pred_source = adapter_predictions if adapter_predictions is not None and index >= ready_index else base_predictions
            task_correct += int(pred_source.get(row["uid"]) == row["answer"])
            task_final_correct += int(final_predictions.get(row["uid"]) == row["answer"])
        task_base_correct = sum(int(base_predictions.get(row["uid"]) == row["answer"]) for row in task_rows)
        task_processed_only = task_correct / len(task_rows) if task_rows else 0.0
        task_final_processed_only = task_final_correct / len(task_rows) if task_rows else 0.0
        task_base_acc = task_base_correct / len(task_rows) if task_rows else 0.0
        by_task[f"{prefix}_base_correct"] = task_base_correct
        by_task[f"{prefix}_base_acc"] = task_base_acc
        by_task[f"{prefix}_base_accuracy"] = task_base_acc
        by_task[f"{prefix}_observed_acc_processed_only"] = task_processed_only
        by_task[f"{prefix}_observed_acc_strict_with_coverage"] = coverage * task_processed_only
        by_task[f"{prefix}_observed_time_weighted_accuracy"] = task_processed_only
        by_task[f"{prefix}_observed_current_accuracy"] = coverage * task_processed_only
        by_task[f"{prefix}_final_acc_processed_only"] = task_final_processed_only
        by_task[f"{prefix}_final_acc_strict_with_coverage"] = coverage * task_final_processed_only
        by_task[f"{prefix}_observed_final_accuracy"] = coverage * task_final_processed_only
        by_task[f"{prefix}_correct"] = coverage * task_correct
        by_task[f"{prefix}_total"] = len(task_rows)
    return {
        "adapter_ready_sample_count": ready_index,
        "before_adapter_accuracy": before_correct / len(before_rows) if before_rows else 0.0,
        "before_adapter_correct": before_correct,
        "before_adapter_total": len(before_rows),
        "after_adapter_accuracy": after_correct / len(after_rows) if after_rows else 0.0,
        "after_adapter_correct": after_correct,
        "after_adapter_total": len(after_rows),
        "observed_time_weighted_accuracy": processed_only,
        "observed_current_accuracy": observed_current,
        "observed_final_accuracy": coverage * final_correct / total if total else 0.0,
        "num_arrived": total,
        "num_processed": processed,
        "num_timeout": total - processed,
        "correct": coverage * raw_correct,
        "total": total,
        "strict_group_acc": observed_current,
        "processed_only_acc": processed_only,
        "base_correct": sum(int(base_predictions.get(row["uid"]) == row["answer"]) for row in rows),
        "base_total": total,
        "adapter_correct": final_correct,
        "adapter_total": total,
        **by_task,
    }


def pseudo_mixed_accuracy(
    rows: list[dict[str, Any]],
    pseudo: dict[str, str],
    base_predictions: dict[str, str],
    adapter_predictions: dict[str, str] | None,
    *,
    ready_index: int,
    coverage: float,
) -> float:
    use_rows = [row for row in rows if row["uid"] in pseudo]
    if not use_rows:
        return 0.0
    correct = 0
    for index, row in enumerate(use_rows):
        preds = adapter_predictions if adapter_predictions is not None and index >= ready_index else base_predictions
        correct += int(preds.get(row["uid"]) == pseudo[row["uid"]])
    return coverage * correct / len(use_rows)


def discounted_average(current_acc: float, future_acc: float, future_windows: int, discount: float) -> float:
    weights = [1.0] + [discount**idx for idx in range(1, max(0, future_windows) + 1)]
    return (weights[0] * current_acc + sum(weight * future_acc for weight in weights[1:])) / sum(weights)


def arrival_rate_arg(args: argparse.Namespace) -> float | None:
    value = getattr(args, "arrival_rate", None)
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def effective_window_seconds(total_rows: int, args: argparse.Namespace) -> float:
    rate = arrival_rate_arg(args)
    if rate:
        return total_rows / rate if total_rows > 0 else 0.0
    return float(args.retraining_window_sec)


def ready_index_for_finish(
    *,
    finish_time: float,
    total_rows: int,
    args: argparse.Namespace,
    reference_total_rows: int | None = None,
) -> int:
    if total_rows <= 0:
        return 0
    rate = arrival_rate_arg(args)
    if rate:
        reference_total = max(1, int(reference_total_rows or total_rows))
        arrived_in_reference = max(0.0, rate * finish_time)
        ready_ratio = min(1.0, arrived_in_reference / reference_total)
        return min(total_rows, math.ceil(ready_ratio * total_rows))
    window = max(1e-9, float(args.retraining_window_sec))
    ready_ratio = min(1.0, max(0.0, finish_time / window))
    return min(total_rows, math.ceil(ready_ratio * total_rows))


def optimus_extrapolate_gain(
    *,
    base_acc: float,
    profile_gain: float,
    profile_steps: int,
    target_steps: int,
    max_accuracy: float = 1.0,
) -> float:
    """Ekya/Optimus-style saturating extrapolation from a short micro-profile.

    We fit the one-point saturating curve A(x)=A0+(C-A0)*x/(x+s), an equivalent
    form of the Optimus curve used by Ekya, through (0, base_acc) and the
    micro-profile point.
    """

    base_acc = max(0.0, min(1.0, float(base_acc)))
    profile_gain = float(profile_gain)
    profile_steps = max(0, int(profile_steps))
    target_steps = max(0, int(target_steps))
    cap = max(base_acc, min(1.0, float(max_accuracy)))
    if profile_steps <= 0 or target_steps <= 0:
        return max(-base_acc, min(cap - base_acc, profile_gain))
    profile_acc = max(0.0, min(cap, base_acc + profile_gain))
    if profile_acc <= base_acc:
        return max(-base_acc, min(0.0, profile_acc - base_acc))
    if profile_acc >= cap - 1e-12:
        return cap - base_acc
    scale = profile_steps * (cap - profile_acc) / max(1e-12, profile_acc - base_acc)
    target_acc = base_acc + (cap - base_acc) * (target_steps / (target_steps + scale))
    return max(-base_acc, min(cap - base_acc, target_acc - base_acc))


def extrapolate_gamma_gain(
    *,
    base_acc: float,
    profile_gain: float,
    target_steps: int,
    profile_steps: int,
    args: argparse.Namespace,
) -> float:
    method = getattr(args, "gamma_extrapolation_method", "direct_measured")
    if method == "optimus":
        return optimus_extrapolate_gain(
            base_acc=base_acc,
            profile_gain=profile_gain,
            profile_steps=profile_steps,
            target_steps=target_steps,
            max_accuracy=getattr(args, "scheduler_optimus_max_accuracy", 1.0),
        )
    return profile_gain


def scheduler_projected_gain(gamma_row: dict[str, Any] | None, args: argparse.Namespace) -> float:
    if not gamma_row:
        return 0.0
    mode = getattr(args, "scheduler_estimate_mode", "target_rval")
    if mode == "profile_gain":
        return max(0.0, float(gamma_row.get("profile_gain", 0.0)))
    if mode == "optimus":
        return max(0.0, float(gamma_row.get("predicted_final_gain", 0.0)))
    return max(0.0, float(gamma_row.get("predicted_final_gain", 0.0)))


def scheduler_uses_target_rval(args: argparse.Namespace) -> bool:
    return getattr(args, "scheduler_estimate_mode", "target_rval") == "target_rval"


def build_decision(
    *,
    method: str,
    lambda_row: dict[str, Any],
    gamma_row: dict[str, Any] | None,
    r_all: list[dict[str, Any]],
    no_drift_rows: list[dict[str, Any]],
    r_val: list[dict[str, Any]],
    r_test: list[dict[str, Any]],
    pseudo: dict[str, str],
    base_predictions: dict[str, str],
    adapter_predictions: dict[str, str] | None,
    inference_resource: float,
    training_resource: float,
    triggered: bool,
    adapter_accepted: bool,
    args: argparse.Namespace,
    training_start_delay_s: float = 0.0,
) -> dict[str, Any]:
    lambda_cost = float(lambda_row["resource_cost"])
    gamma_cost = float(gamma_row["resource_cost"]) if gamma_row else 0.0
    coverage = min(1.0, inference_resource / max(lambda_cost, 1e-9))
    can_train = bool(triggered and adapter_accepted and gamma_row and training_resource >= gamma_cost and training_resource > 0)
    train_time = float(gamma_row["target_train_time_s_est"]) if gamma_row else 0.0
    finish_time = training_start_delay_s + train_time / training_resource if can_train else None
    current_window_sec = effective_window_seconds(len(r_all), args)
    adapter_replaced = bool(finish_time is not None and finish_time <= current_window_sec)
    ready_index = len(r_all)
    if adapter_replaced:
        ready_index = ready_index_for_finish(finish_time=finish_time, total_rows=len(r_all), args=args)
    no_drift_ready_index = len(no_drift_rows)
    if adapter_replaced:
        no_drift_ready_index = ready_index_for_finish(
            finish_time=finish_time,
            total_rows=len(no_drift_rows),
            args=args,
            reference_total_rows=len(r_all),
        )
    adapter_for_eval = adapter_predictions if adapter_replaced else None
    current_actual = scheduler_row_accuracy(
        r_all,
        base_predictions,
        adapter_for_eval,
        ready_index=ready_index,
        coverage=coverage,
    )
    no_drift_actual = scheduler_row_accuracy(
        no_drift_rows,
        base_predictions,
        adapter_for_eval,
        ready_index=no_drift_ready_index,
        coverage=coverage,
    )
    est_ready_val = len(r_val)
    if adapter_replaced:
        est_ready_val = ready_index_for_finish(
            finish_time=finish_time or 0.0,
            total_rows=len(r_val),
            args=args,
            reference_total_rows=len(r_all),
        )
    future_base = accuracy_pseudo(r_val, base_predictions, pseudo)
    if scheduler_uses_target_rval(args):
        estimated_current = pseudo_mixed_accuracy(
            r_val,
            pseudo,
            base_predictions,
            adapter_predictions if adapter_replaced else None,
            ready_index=est_ready_val,
            coverage=coverage,
        )
        future_adapter = (
            accuracy_pseudo(r_val, adapter_predictions, pseudo)
            if adapter_predictions and adapter_replaced
            else future_base
        )
        estimated_future = coverage * future_adapter
    else:
        projected_future = future_base
        if adapter_replaced:
            projected_future = min(1.0, max(0.0, future_base + scheduler_projected_gain(gamma_row, args)))
        ready_ratio_val = min(1.0, est_ready_val / max(1, len(r_val)))
        estimated_processed = (
            ready_ratio_val * future_base + (1.0 - ready_ratio_val) * projected_future
            if adapter_replaced
            else future_base
        )
        estimated_current = coverage * estimated_processed
        estimated_future = coverage * projected_future
    estimated_long = discounted_average(estimated_current, estimated_future, args.future_windows, args.discount_factor)
    true_future_preds = adapter_predictions if adapter_replaced and adapter_predictions else base_predictions
    true_future = coverage * accuracy_true(r_test, true_future_preds)
    actual_long = discounted_average(
        float(current_actual["observed_current_accuracy"]),
        true_future,
        args.future_windows,
        args.discount_factor,
    )
    row = {
        "group_id": GROUP_ID,
        "model_id": MODEL_ID,
        "method": method,
        "triggered": bool(triggered),
        "lambda_id": lambda_row["lambda_id"],
        "gamma_id": gamma_row["gamma_id"] if gamma_row and can_train else "",
        "I": round(inference_resource, 6),
        "R": round(training_resource, 6),
        "adapter_accepted": bool(adapter_accepted and gamma_row),
        "adapter_replaced": adapter_replaced,
        "estimated_current_accuracy": round(estimated_current, 6),
        "estimated_future_or_final_accuracy": round(estimated_future, 6),
        "estimated_long_avg_accuracy": round(estimated_long, 6),
        "actual_long_avg_accuracy": round(actual_long, 6),
        "T_retrain_s": round(current_window_sec, 6),
        "arrival_rate": round(arrival_rate_arg(args), 6) if arrival_rate_arg(args) else "",
        "adapter_ready_time_s": round(finish_time, 6) if finish_time is not None else "",
        "train_time_s_effective": round(finish_time, 6) if finish_time is not None else "",
        "training_start_delay_s": round(training_start_delay_s, 6) if training_start_delay_s else 0.0,
        "scheduler_estimate_mode": getattr(args, "scheduler_estimate_mode", "target_rval"),
        "inference_latency_mean_s": lambda_row.get("latency_mean_sec", ""),
        "lambda_teacher_accuracy": lambda_row.get("teacher_accuracy", ""),
        "lambda_true_accuracy": lambda_row.get("true_accuracy", ""),
        "lambda_teacher_noise_accuracy_gap": lambda_row.get("teacher_noise_accuracy_gap", ""),
        "current_window_accuracy": round(estimated_current, 6),
        "future_window_accuracy": round(estimated_future, 6),
        "long_avg_accuracy": round(estimated_long, 6),
        "lambda_resource_cost_effective": lambda_cost,
        "gamma_resource_cost_effective": gamma_cost,
        "coverage": round(coverage, 6),
    }
    row.update({key: round(value, 6) if isinstance(value, float) else value for key, value in current_actual.items()})
    row["no_drift_adapter_ready_sample_count"] = no_drift_ready_index
    row.update(
        {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in no_drift_actual.items()
            if key.startswith("no_drift_")
        }
    )
    row["current_error_actual_minus_estimated"] = round(row["observed_current_accuracy"] - row["estimated_current_accuracy"], 6)
    row["abs_current_error"] = abs(row["current_error_actual_minus_estimated"])
    row["final_error_actual_minus_estimated_future"] = round(row["observed_final_accuracy"] - row["estimated_future_or_final_accuracy"], 6)
    return row


def choose_ours(
    *,
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    predictions_by_lambda: dict[str, dict[str, str]],
    adapter_predictions: dict[tuple[str, str], dict[str, str]],
    splits: dict[str, list[dict[str, Any]]],
    pseudo: dict[str, str],
    triggered: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = []
    gamma_by_lambda = {(row["feature_lambda_id"], row["gamma_id"]): row for row in gamma_rows}
    for lambda_row in lambda_rows:
        base_preds = predictions_by_lambda[lambda_row["lambda_id"]]
        for gamma_cfg in gamma_configs():
            gamma_row = gamma_by_lambda[(lambda_row["lambda_id"], gamma_cfg.gamma_id)]
            adapter_preds = adapter_predictions[(lambda_row["lambda_id"], gamma_cfg.gamma_id)]
            adapter_accepted = scheduler_projected_gain(gamma_row, args) > args.min_profile_gain
            for idx in range(1, 100):
                I = idx / 100.0
                row = build_decision(
                    method="Ours",
                    lambda_row=lambda_row,
                    gamma_row=gamma_row,
                    r_all=splits["R_all"],
                    no_drift_rows=splits["R_no_drift"],
                    r_val=splits["R_val"],
                    r_test=splits["R_test"],
                    pseudo=pseudo,
                    base_predictions=base_preds,
                    adapter_predictions=adapter_preds,
                    inference_resource=I,
                    training_resource=1.0 - I,
                    triggered=triggered,
                    adapter_accepted=adapter_accepted,
                    args=args,
                )
                candidates.append(row)
    return max(
        candidates,
        key=lambda row: (
            float(row["estimated_long_avg_accuracy"]),
            float(row["estimated_future_or_final_accuracy"]),
            float(row["estimated_current_accuracy"]),
            -float(row["lambda_resource_cost_effective"]),
        ),
    )


def choose_offline_oracle(
    *,
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    predictions_by_lambda: dict[str, dict[str, str]],
    adapter_predictions: dict[tuple[str, str], dict[str, str]],
    splits: dict[str, list[dict[str, Any]]],
    pseudo: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = []
    gamma_by_lambda = {(row["feature_lambda_id"], row["gamma_id"]): row for row in gamma_rows}
    for lambda_row in lambda_rows:
        base_preds = predictions_by_lambda[lambda_row["lambda_id"]]
        for gamma_cfg in gamma_configs():
            gamma_row = gamma_by_lambda[(lambda_row["lambda_id"], gamma_cfg.gamma_id)]
            adapter_preds = adapter_predictions[(lambda_row["lambda_id"], gamma_cfg.gamma_id)]
            for idx in range(1, 100):
                I = idx / 100.0
                row = build_decision(
                    method="OfflineJointOracle",
                    lambda_row=lambda_row,
                    gamma_row=gamma_row,
                    r_all=splits["R_all"],
                    no_drift_rows=splits["R_no_drift"],
                    r_val=splits["R_val"],
                    r_test=splits["R_test"],
                    pseudo=pseudo,
                    base_predictions=base_preds,
                    adapter_predictions=adapter_preds,
                    inference_resource=I,
                    training_resource=1.0 - I,
                    triggered=True,
                    adapter_accepted=True,
                    args=args,
                )
                candidates.append(row)
    return max(
        candidates,
        key=lambda row: (
            float(row["observed_time_weighted_accuracy"]),
            float(row["observed_final_accuracy"]),
            float(row["estimated_long_avg_accuracy"]),
        ),
    )


def build_baseline_decisions(
    *,
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    predictions_by_lambda: dict[str, dict[str, str]],
    adapter_predictions: dict[tuple[str, str], dict[str, str]],
    splits: dict[str, list[dict[str, Any]]],
    pseudo: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    full_lambdas = [row for row in lambda_rows if abs(float(row["resolution_keep_ratio"]) - 1.0) < 1e-9]
    fixed_lambda = max(full_lambdas or lambda_rows, key=lambda row: (row["model_precision"] == "fp32", -float(row["resource_cost"])))
    fixed_gamma = next((row for row in gamma_rows if row["feature_lambda_id"] == fixed_lambda["lambda_id"] and row["gamma_id"].endswith("ep3")), None)
    if fixed_gamma is None:
        fixed_gamma = next(row for row in gamma_rows if row["feature_lambda_id"] == fixed_lambda["lambda_id"])
    base_preds = predictions_by_lambda[fixed_lambda["lambda_id"]]
    adapter_preds = adapter_predictions[(fixed_lambda["lambda_id"], fixed_gamma["gamma_id"])]
    periodic_period_samples = max(0, int(getattr(args, "periodic_period_samples", 0)))
    periodic_start_delay_s = 0.0
    if periodic_period_samples and splits["R_all"]:
        rate = arrival_rate_arg(args)
        if rate:
            periodic_start_delay_s = min(effective_window_seconds(len(splits["R_all"]), args), periodic_period_samples / rate)
        else:
            periodic_start_delay_s = min(
                args.retraining_window_sec,
                periodic_period_samples / len(splits["R_all"]) * args.retraining_window_sec,
            )
    no_retrain = build_decision(
        method="NoRetrain",
        lambda_row=fixed_lambda,
        gamma_row=None,
        r_all=splits["R_all"],
        no_drift_rows=splits["R_no_drift"],
        r_val=splits["R_val"],
        r_test=splits["R_test"],
        pseudo=pseudo,
        base_predictions=base_preds,
        adapter_predictions=None,
        inference_resource=1.0,
        training_resource=0.0,
        triggered=False,
        adapter_accepted=False,
        args=args,
    )
    periodic = build_decision(
        method="PeriodicFixedRetrain",
        lambda_row=fixed_lambda,
        gamma_row=fixed_gamma,
        r_all=splits["R_all"],
        no_drift_rows=splits["R_no_drift"],
        r_val=splits["R_val"],
        r_test=splits["R_test"],
        pseudo=pseudo,
        base_predictions=base_preds,
        adapter_predictions=adapter_preds,
        inference_resource=args.periodic_inference_resource,
        training_resource=1.0 - args.periodic_inference_resource,
        triggered=True,
        adapter_accepted=True,
        args=args,
        training_start_delay_s=periodic_start_delay_s,
    )
    static = build_decision(
        method="StaticSplitContinuous",
        lambda_row=fixed_lambda,
        gamma_row=fixed_gamma,
        r_all=splits["R_all"],
        no_drift_rows=splits["R_no_drift"],
        r_val=splits["R_val"],
        r_test=splits["R_test"],
        pseudo=pseudo,
        base_predictions=base_preds,
        adapter_predictions=adapter_preds,
        inference_resource=args.static_inference_resource,
        training_resource=1.0 - args.static_inference_resource,
        triggered=True,
        adapter_accepted=True,
        args=args,
    )
    return [no_retrain, periodic, static]


def build_detailed_tables(
    out_dir: Path,
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    all_pair_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> None:
    detail = out_dir / "detailed_result_tables"
    selected_pairs = {(row["lambda_id"], row["gamma_id"]) for row in method_rows if row["method"] == "Ours"}
    lambda_detail = []
    for row in lambda_rows:
        lambda_detail.append(
            {
                "group_id": GROUP_ID,
                "lambda_id": row["lambda_id"],
                "selected_by_ours": any(row["lambda_id"] == pair[0] for pair in selected_pairs),
                "dtop40_true_accuracy": row["true_accuracy"],
                "dtop40_teacher_accuracy": row["teacher_accuracy"],
                "dtop40_correct": row["true_correct"],
                "dtop40_total": row["measured_samples"],
                "dtop80_true_accuracy": row["true_accuracy"],
                "dtop80_teacher_accuracy": row["teacher_accuracy"],
                "dtop80_correct": row["true_correct"],
                "dtop80_total": row["measured_samples"],
                "model_precision": row["model_precision"],
                "resolution_keep_ratio": row["resolution_keep_ratio"],
                "batch_size": row["batch_size"],
                "resource_cost": row["resource_cost"],
                "latency_p50_sec": row["latency_p50_sec"],
                "latency_p95_sec": row["latency_p95_sec"],
                "throughput_samples_per_sec": row["throughput_samples_per_sec"],
                "peak_memory_gib": row["peak_memory_gib"],
                "raw_predictions": row["raw_predictions"],
            }
        )
    write_csv(detail / "lambda_dtop40_by_group_config.csv", lambda_detail)
    write_csv(detail / "lambda_dtop80_by_group_config.csv", lambda_detail)
    matrix = {"group_id": GROUP_ID}
    for row in lambda_detail:
        value = row["dtop40_teacher_accuracy"]
        matrix[row["lambda_id"]] = f"{value}*" if row["selected_by_ours"] else value
    write_csv(detail / "lambda_dtop40_accuracy_matrix.csv", [matrix])
    write_csv(detail / "lambda_dtop80_accuracy_matrix.csv", [matrix])
    gamma_detail = []
    for row in gamma_rows:
        gamma_detail.append(
            {
                "group_id": GROUP_ID,
                "gamma_id": row["gamma_id"],
                "feature_lambda_id": row["feature_lambda_id"],
                "selected_by_ours": (row["feature_lambda_id"], row["gamma_id"]) in selected_pairs,
                "rval_base_accuracy_before_1epoch": row["rval_base_accuracy_before_1epoch"],
                "rval_adapter_accuracy_after_target": row["rval_adapter_accuracy_after_target"],
                "small_step_profile_gain": row["profile_gain"],
                "extrapolated_target_gain": row["predicted_final_gain"],
                "target_epochs": row["target_epochs"],
                "microprofile_epochs": row["microprofile_epochs"],
                "target_train_time_s_est": row["target_train_time_s_est"],
                "microprofile_train_time_s": row["microprofile_train_time_s"],
                "resource_cost": row["resource_cost"],
                "peak_vram_gb": row["peak_memory_gib"],
                "adapter_path": row["adapter_path"],
            }
        )
    write_csv(detail / "gamma_smallstep_by_group_config.csv", gamma_detail)
    write_csv(detail / "lambda_gamma_estimates_all_pairs.csv", all_pair_rows)
    selected_rows = []
    by_task_rows = []
    for row in method_rows:
        selected_rows.append(
            {
                "group_id": row["group_id"],
                "method_label": row["method"],
                "method_code": row["method"],
                "selected_lambda_id": row["lambda_id"],
                "selected_gamma_id": row["gamma_id"],
                "I": row["I"],
                "R": row["R"],
                "adapter_accepted": row["adapter_accepted"],
                "selected_lambda_dtop40_true_acc": next((lr["true_accuracy"] for lr in lambda_rows if lr["lambda_id"] == row["lambda_id"]), ""),
                "selected_lambda_dtop40_teacher_acc": next((lr["teacher_accuracy"] for lr in lambda_rows if lr["lambda_id"] == row["lambda_id"]), ""),
                "selected_lambda_dtop80_true_acc": next((lr["true_accuracy"] for lr in lambda_rows if lr["lambda_id"] == row["lambda_id"]), ""),
                "selected_lambda_dtop80_teacher_acc": next((lr["teacher_accuracy"] for lr in lambda_rows if lr["lambda_id"] == row["lambda_id"]), ""),
                "selected_lambda_resource_cost": next((lr["resource_cost"] for lr in lambda_rows if lr["lambda_id"] == row["lambda_id"]), ""),
                "estimated_current_accuracy": row["estimated_current_accuracy"],
                "estimated_future_or_final_accuracy": row["estimated_future_or_final_accuracy"],
                "estimated_long_avg_accuracy": row["estimated_long_avg_accuracy"],
                "estimated_adapter_ready_time_s": row["adapter_ready_time_s"],
                "estimated_train_time_s_effective": row["train_time_s_effective"],
                "actual_rtest_base_accuracy_before_adapter": row["before_adapter_accuracy"],
                "actual_rtest_adapter_accuracy_after_adapter": row["after_adapter_accuracy"],
                "actual_current_accuracy": row["observed_current_accuracy"],
                "actual_final_accuracy": row["observed_final_accuracy"],
                "observed_time_weighted_accuracy": row["observed_time_weighted_accuracy"],
                "current_error_actual_minus_estimated": row["current_error_actual_minus_estimated"],
                "abs_current_error": row["abs_current_error"],
                "final_error_actual_minus_estimated_future": row["final_error_actual_minus_estimated_future"],
                "base_correct": row["base_correct"],
                "base_total": row["base_total"],
                "adapter_correct": row["adapter_correct"],
                "adapter_total": row["adapter_total"],
                "no_drift_base_acc": row.get("no_drift_base_acc", ""),
                "no_drift_observed_acc_processed_only": row.get("no_drift_observed_acc_processed_only", ""),
                "no_drift_observed_acc_strict_with_coverage": row.get("no_drift_observed_acc_strict_with_coverage", ""),
                "no_drift_final_acc_processed_only": row.get("no_drift_final_acc_processed_only", ""),
                "drift_base_acc": row.get("drift_base_acc", ""),
                "drift_observed_acc_processed_only": row.get("drift_observed_acc_processed_only", ""),
                "drift_observed_acc_strict_with_coverage": row.get("drift_observed_acc_strict_with_coverage", ""),
                "drift_final_acc_processed_only": row.get("drift_final_acc_processed_only", ""),
            }
        )
        by_task_rows.append(
            {
                "method": row["method"],
                "lambda_id": row["lambda_id"],
                "gamma_id": row["gamma_id"],
                "I": row["I"],
                "R": row["R"],
                "coverage": row["coverage"],
                "adapter_ready_sample_count": row["adapter_ready_sample_count"],
                "no_drift_adapter_ready_sample_count": row.get("no_drift_adapter_ready_sample_count", ""),
                "formal_R_all_acc_processed_only": row["observed_time_weighted_accuracy"],
                "formal_R_all_acc_with_coverage": row["observed_current_accuracy"],
                "formal_R_all_final_acc_with_coverage": row["observed_final_accuracy"],
                "no_drift_base_correct": row.get("no_drift_base_correct", ""),
                "no_drift_total": row.get("no_drift_total", ""),
                "no_drift_base_acc": row.get("no_drift_base_acc", ""),
                "no_drift_observed_acc_processed_only": row.get("no_drift_observed_acc_processed_only", ""),
                "no_drift_observed_acc_strict_with_coverage": row.get("no_drift_observed_acc_strict_with_coverage", ""),
                "no_drift_final_acc_processed_only": row.get("no_drift_final_acc_processed_only", ""),
                "drift_base_correct": row.get("drift_base_correct", ""),
                "drift_total": row.get("drift_total", ""),
                "drift_base_acc": row.get("drift_base_acc", ""),
                "drift_observed_acc_processed_only": row.get("drift_observed_acc_processed_only", ""),
                "drift_observed_acc_strict_with_coverage": row.get("drift_observed_acc_strict_with_coverage", ""),
                "drift_final_acc_processed_only": row.get("drift_final_acc_processed_only", ""),
            }
        )
    write_csv(detail / "selected_config_estimated_vs_actual.csv", selected_rows)
    write_csv(detail / "method_comparison_by_task.csv", by_task_rows)
    lines = [
        "# Detailed Result Tables",
        "",
        "## Lambda D_top40 Accuracy Matrix",
        "",
        "| group_id | " + " | ".join(row["lambda_id"] for row in lambda_detail) + " |",
        "|---|" + "|".join("---:" for _ in lambda_detail) + "|",
        "| " + GROUP_ID + " | " + " | ".join(str(matrix[row["lambda_id"]]) for row in lambda_detail) + " |",
        "",
        "## Selected Configs",
        "",
        "| method | lambda | gamma | I | R | estimated | actual_R_all_time_weighted |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['method_label']} | {row['selected_lambda_id']} | {row['selected_gamma_id']} | "
            f"{row['I']} | {row['R']} | {row['estimated_long_avg_accuracy']} | {row['observed_time_weighted_accuracy']} |"
        )
    (detail / "detailed_result_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        detail / "summary.json",
        {
            "lambda_dtop40_by_group_config": len(lambda_detail),
            "lambda_dtop80_by_group_config": len(lambda_detail),
            "gamma_smallstep_by_group_config": len(gamma_detail),
            "lambda_gamma_estimates_all_pairs": len(all_pair_rows),
            "selected_config_estimated_vs_actual": len(selected_rows),
            "method_comparison_by_task": len(by_task_rows),
        },
    )


def write_summary(
    out_dir: Path,
    *,
    split_summary: dict[str, int],
    detection: dict[str, Any],
    task_evidence: dict[str, Any],
    teacher_quality: dict[str, Any],
    method_rows: list[dict[str, Any]],
    lambda_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
) -> None:
    teacher_model = teacher_quality.get("teacher_model", MODEL_ID)
    teacher_source = (
        "external pseudo labels"
        if teacher_quality.get("external_teacher_pseudo_dir")
        else "RemoteCLIP-RN50 frozen encoder + teacher linear head"
    )
    lines = [
        "# OmniEarth RemoteCLIP Formal Specialist Framework",
        "",
        "## Audit Fixes",
        "",
        "- D is now a 50/50 mixed detection window.",
        "- Formal R_all/R_train/R_val/R_test are drift-task only, matching the large-model framework.",
        "- R_no_drift is retained only for auxiliary no-drift accuracy diagnostics and is not averaged into headline R_all acc.",
        "- A disjoint part-0 no-drift history split is used to fit the reference distribution and threshold.",
        "- D_top40 is selected by ranking Mahalanobis sample scores over the complete D window and retaining the highest-scoring 40%.",
        f"- Teacher and student are explicit: teacher is {teacher_model} ({teacher_source}); student is RemoteCLIP-RN50 zero-shot/linear-head adaptation.",
        "- Teacher pseudo labels cover the complete D window; each run uses its selected D_top40 plus R_train and R_val.",
        "- Lambda uses 20-sample probes; full D_top40 measurement uses the configured lambda selection policy.",
        "- Gamma uses pseudo labels and measured lightweight-head epochs.",
        "- Ours enumerates I=0.01..0.99.",
        "",
        "## Task Evidence",
        "",
        "| role | task | base_acc | adapted_acc | gain | detector |",
        "|---|---|---:|---:|---:|---|",
        f"| no-drift | {REFERENCE_TASK} | {task_evidence.get('reference_base_acc', '')} | {task_evidence.get('reference_adapted_acc', '')} | {task_evidence.get('reference_gain', '')} | history reference + no-drift stream |",
        f"| drift | {DRIFT_TASK} | {task_evidence.get('drift_base_acc', '')} | {task_evidence.get('drift_adapted_acc', '')} | {task_evidence.get('drift_gain', '')} | FPR@0={detection['fpr_at_0']:.3f}, R@50={detection['recall_at_50']:.3f} |",
        "",
        "## Teacher Quality Gate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| D_top40_teacher_true_acc | {teacher_quality.get('D_top40_teacher_true_acc', teacher_quality.get('D_top80_teacher_true_acc', ''))} |",
        f"| R_train_teacher_true_acc | {teacher_quality.get('R_train_teacher_true_acc', '')} |",
        f"| R_val_teacher_true_acc | {teacher_quality.get('R_val_teacher_true_acc', '')} |",
        f"| teacher_eval_mean_true_acc | {teacher_quality.get('teacher_eval_mean_true_acc', '')} |",
        f"| teacher_min_pseudo_acc | {teacher_quality.get('teacher_min_pseudo_acc', '')} |",
        f"| teacher_quality_gate_passed | {teacher_quality.get('teacher_quality_gate_passed', '')} |",
        "",
        "## Split",
        "",
        "| split | rows |",
        "|---|---:|",
    ]
    for key, value in split_summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Config Counts",
            "",
            f"- lambda full D_top40 configs measured: `{len(lambda_rows)}`",
            f"- gamma profile rows: `{len(gamma_rows)}`",
            f"- scheduler resource grid: `99` points per lambda/gamma pair",
            "",
            "## Method Comparison",
            "",
            "| method | lambda | gamma | I | R | ready_n | estimated_long | R_all_time_weighted | current_with_coverage | final_acc | current_error |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['lambda_id']} | {row['gamma_id']} | {row['I']} | {row['R']} | "
            f"{row['adapter_ready_sample_count']} | {row['estimated_long_avg_accuracy']} | "
            f"{row['observed_time_weighted_accuracy']} | {row['observed_current_accuracy']} | "
            f"{row['observed_final_accuracy']} | {row['current_error_actual_minus_estimated']} |"
        )
    (out_dir / "framework_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal RemoteCLIP-RN50 OmniEarth specialist framework.")
    parser.add_argument("--omniearth-root", type=Path, default=Path("@@SERVER_ROOT@@/datasets/public_benchmarks/OmniEarth"))
    parser.add_argument("--remoteclip-dir", type=Path, default=Path("@@WORKSPACE@@/qwen_eval/models/RemoteCLIP"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--canonical-lambda-timing-json",
        type=Path,
        default=DEFAULT_CANONICAL_TIMING_JSON,
    )
    parser.add_argument("--reuse-splits-dir", type=Path, default=None)
    parser.add_argument("--external-teacher-pseudo-dir", type=Path, default=None)
    parser.add_argument("--external-teacher-quality-json", type=Path, default=None)
    parser.add_argument("--teacher-model-name", default=MODEL_ID)
    parser.add_argument("--reference-task", default=REFERENCE_TASK)
    parser.add_argument("--drift-task", default=DRIFT_TASK)
    parser.add_argument("--split-seed", default="omniearth_landcover200_disaster365_qwen35like_v1")
    parser.add_argument("--max-image-pixels", type=int, default=80_000_000)
    parser.add_argument("--history-reference-count", type=int, default=60)
    parser.add_argument("--history-threshold-count", type=int, default=40)
    parser.add_argument("--teacher-no-count", type=int, default=0)
    parser.add_argument("--teacher-drift-count", type=int, default=120)
    parser.add_argument("--teacher-drift-holdout-count", type=int, default=40)
    parser.add_argument("--d-no-count", type=int, default=50)
    parser.add_argument("--d-drift-count", type=int, default=183)
    parser.add_argument("--r-no-count", type=int, default=50)
    parser.add_argument("--r-drift-count", type=int, default=182)
    parser.add_argument("--d-top-count", type=int, default=None)
    parser.add_argument("--d-top-fraction", type=float, default=0.40)
    parser.add_argument("--teacher-min-pseudo-acc", type=float, default=0.50)
    parser.add_argument("--lambda-probe-count", type=int, default=20)
    parser.add_argument("--drift-ratio", type=float, default=0.50)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--eval-windows", type=int, default=300)
    parser.add_argument("--lambda-max-resource-cost", type=float, default=0.65)
    parser.add_argument("--gamma-max-resource-cost", type=float, default=0.35)
    parser.add_argument("--periodic-inference-resource", type=float, default=0.50)
    parser.add_argument("--static-inference-resource", type=float, default=0.70)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Optional samples/s pressure setting. When set, adapter ready samples use ceil(arrival_rate * finish_time).",
    )
    parser.add_argument(
        "--scheduler-estimate-mode",
        choices=["target_rval", "profile_gain", "optimus"],
        default="target_rval",
        help="target_rval preserves the legacy estimate using full target R_val predictions; profile_gain uses the raw small-step profile; optimus uses the Ekya/Optimus-style extrapolated gain.",
    )
    parser.add_argument(
        "--gamma-extrapolation-method",
        choices=["direct_measured", "optimus"],
        default="direct_measured",
        help="direct_measured preserves the legacy target R_val gain; optimus predicts target gain from the micro-profile curve.",
    )
    parser.add_argument("--scheduler-optimus-max-accuracy", type=float, default=1.0)
    parser.add_argument(
        "--periodic-period-samples",
        type=int,
        default=0,
        help="Delay PeriodicFixedRetrain by this many arriving R samples before training starts. 0 preserves the legacy immediate-start behavior.",
    )
    parser.add_argument("--retraining-window-sec", type=float, default=100.0)
    parser.add_argument("--future-windows", type=int, default=3)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--min-profile-gain", type=float, default=0.0)
    parser.add_argument(
        "--disable-pareto-filter",
        action="store_true",
        help="Measure and schedule all lambda configs after probe20 instead of keeping only the Pareto frontier.",
    )
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="Materialize the corrected complete-window D_top40 split and detector artifacts, then stop.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.canonical_lambda_timing_json = args.canonical_lambda_timing_json.resolve()
    args.canonical_lambda_timing_by_id = load_canonical_lambda_timing(args.canonical_lambda_timing_json)
    if abs(args.drift_ratio - 0.50) > 1e-9:
        raise ValueError("This formal runner expects a 50% drift ratio.")

    if args.reuse_splits_dir:
        splits = load_existing_splits(args.reuse_splits_dir)
    else:
        specs = {spec.task_key: spec for spec in search.discover_specs(args.omniearth_root)}
        reference_rows = search.read_rows(args.omniearth_root, specs[args.reference_task], args.max_image_pixels)
        drift_rows = search.read_rows(args.omniearth_root, specs[args.drift_task], args.max_image_pixels)
        splits = split_dataset(reference_rows, drift_rows, args)

    split_dir = args.output_dir / "splits"
    for name, rows in splits.items():
        write_split(split_dir / name, rows)

    all_seed_rows = unique_rows(
        splits["history_reference_fit"],
        splits["history_threshold_cal"],
        splits["teacher_ft_train"],
        splits["D_no_drift"],
        splits["D_drift"],
        splits["D_window"],
        splits["R_no_drift"],
        splits["R_all"],
    )
    model, preprocess, tokenizer, param_info = load_remoteclip(args.remoteclip_dir, args.device)
    teacher_config = LambdaConfig("teacher_rc_r50_keep100_fp32_b32", 1.0, "fp32", 32)
    teacher_run = extract_remoteclip_for_config(
        config=teacher_config,
        rows=all_seed_rows,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=args.device,
        out_path=args.output_dir / "teacher" / "teacher_seed_predictions.jsonl",
        pseudo=None,
    )
    teacher_features = teacher_run["features"]

    detector = fit_mahalanobis_detector(teacher_features, splits["history_reference_fit"], splits["history_threshold_cal"], args)
    detection = evaluate_detector(detector, splits, args)
    d_top_count = resolve_d_top_count(splits, args)
    selected_d_top80, d_drift_scores = select_d_top_rows(detector, splits, d_top_count)
    d_top80 = selected_d_top80
    splits["D_top80"] = d_top80
    splits["D_top40"] = d_top80
    write_split(split_dir / "D_top40", d_top80)
    write_split(split_dir / "D_top80", d_top80)
    write_jsonl(args.output_dir / "detector" / "D_window_sample_scores.jsonl", d_drift_scores)
    write_json(args.output_dir / "detector" / "detector_summary.json", {k: v for k, v in detection.items() if k != "D_window_scores"})
    write_jsonl(args.output_dir / "detector" / "D_window_scores.jsonl", detection["D_window_scores"])
    if args.detector_only:
        write_json(
            args.output_dir / "detector_only_summary.json",
            {
                "D_window": len(splits["D_window"]),
                "D_top40": len(d_top80),
                "D_top_fraction": args.d_top_fraction if args.d_top_count is None else "",
                "D_top_count": d_top_count,
                "selection": "rank all samples in the complete D window by drift score",
            },
        )
        return 0

    pseudo_rows = unique_rows(d_top80, splits["R_train"], splits["R_val"])
    if args.external_teacher_pseudo_dir:
        pseudo = load_external_pseudo_labels(args.external_teacher_pseudo_dir)
        missing_pseudo = [row["uid"] for row in pseudo_rows if row["uid"] not in pseudo]
        if missing_pseudo:
            raise ValueError(
                f"External pseudo labels missing {len(missing_pseudo)} required uid(s); first={missing_pseudo[:5]}"
            )
        teacher_predictions = {row["uid"]: pseudo[row["uid"]] for row in pseudo_rows}
        teacher_quality = {
            "teacher_quality_eval_splits": ["D_top40", "R_train", "R_val"],
            "teacher_model": args.teacher_model_name,
            "teacher_min_pseudo_acc": args.teacher_min_pseudo_acc,
            "D_top40_teacher_true_acc": pseudo_accuracy_true(d_top80, pseudo),
            "D_top80_teacher_true_acc": pseudo_accuracy_true(d_top80, pseudo),
            "R_train_teacher_true_acc": pseudo_accuracy_true(splits["R_train"], pseudo),
            "R_val_teacher_true_acc": pseudo_accuracy_true(splits["R_val"], pseudo),
            "external_teacher_pseudo_dir": str(args.external_teacher_pseudo_dir),
        }
        if args.external_teacher_quality_json and args.external_teacher_quality_json.exists():
            teacher_quality["external_teacher_quality_json"] = str(args.external_teacher_quality_json)
            teacher_quality["external_teacher_quality"] = json.loads(args.external_teacher_quality_json.read_text(encoding="utf-8"))
        teacher_time = 0.0
    else:
        teacher_head, teacher_time = train_head_sgd(
            teacher_features,
            splits["teacher_ft_train"],
            label_letters=None,
            epochs=5,
            seed=f"{args.split_seed}:teacher_head",
        )
        teacher_predictions = predict_head(teacher_head, teacher_features, pseudo_rows)
        pseudo = {row["uid"]: teacher_predictions[row["uid"]] for row in pseudo_rows}
        teacher_quality = {
            "teacher_quality_eval_splits": ["D_top40", "R_train", "R_val"],
            "teacher_model": args.teacher_model_name,
            "teacher_min_pseudo_acc": args.teacher_min_pseudo_acc,
            "D_top40_teacher_true_acc": accuracy_true(d_top80, teacher_predictions),
            "D_top80_teacher_true_acc": accuracy_true(d_top80, teacher_predictions),
            "R_train_teacher_true_acc": accuracy_true(splits["R_train"], teacher_predictions),
            "R_val_teacher_true_acc": accuracy_true(splits["R_val"], teacher_predictions),
        }
    teacher_quality["teacher_eval_mean_true_acc"] = float(
        np.mean(
            [
                teacher_quality["D_top40_teacher_true_acc"],
                teacher_quality["R_train_teacher_true_acc"],
                teacher_quality["R_val_teacher_true_acc"],
            ]
        )
    )
    teacher_quality["teacher_quality_gate_passed"] = bool(
        teacher_quality["teacher_eval_mean_true_acc"] >= args.teacher_min_pseudo_acc
    )
    teacher_quality["pseudo_labeling_started"] = True
    if not teacher_quality["teacher_quality_gate_passed"]:
        print(
            "[WARN] Teacher quality gate did not pass; pseudo labels are still written for diagnostic continuity.",
            file=sys.stderr,
        )
    write_json(args.output_dir / "teacher" / "teacher_quality_gate.json", teacher_quality)
    pseudo_dir = args.output_dir / "teacher" / "pseudo_labels"
    write_predictions(pseudo_dir / "D_top80_pseudo.jsonl", d_top80, teacher_predictions, pseudo)
    write_predictions(pseudo_dir / "D_top40_pseudo.jsonl", d_top80, teacher_predictions, pseudo)
    write_predictions(pseudo_dir / "R_train_pseudo.jsonl", splits["R_train"], teacher_predictions, pseudo)
    write_predictions(pseudo_dir / "R_val_pseudo.jsonl", splits["R_val"], teacher_predictions, pseudo)
    write_json(
        args.output_dir / "teacher" / "teacher_summary.json",
        {
            "teacher_model": args.teacher_model_name,
            "teacher_head": "external_pseudo_labels" if args.external_teacher_pseudo_dir else "linear_head",
            "teacher_train_rows": len(splits["teacher_ft_train"]),
            "teacher_train_no_drift_rows": len(splits["teacher_ft_train_no_drift"]),
            "teacher_train_drift_rows": len(splits["teacher_ft_train_drift"]),
            "teacher_train_time_sec": teacher_time,
            **teacher_quality,
            "pseudo_splits": ["D_window", "D_top40", "R_train", "R_val"],
            "D_top40_selection": "complete-window top drift-score fraction" if args.d_top_count is None else "complete-window top drift-score count",
            "D_top40_top_fraction": args.d_top_fraction if args.d_top_count is None else "",
            "D_top40_top_count": d_top_count,
            "D_top80_selection": "top drift-score fraction" if args.d_top_count is None else "top drift-score count",
            "D_top80_top_fraction": args.d_top_fraction if args.d_top_count is None else "",
            "D_top80_top_count": d_top_count,
            "D_top80_pseudo_true_acc": accuracy_true(d_top80, teacher_predictions),
            "D_top40_pseudo_true_acc": accuracy_true(d_top80, teacher_predictions),
            "R_train_pseudo_true_acc": accuracy_true(splits["R_train"], teacher_predictions),
            "R_val_pseudo_true_acc": accuracy_true(splits["R_val"], teacher_predictions),
        },
    )

    reference_evidence_rows = unique_rows(splits["D_no_drift"], splits["R_no_drift"])
    drift_evidence_rows = unique_rows(splits["D_drift"], splits["R_drift"])
    reference_base = accuracy_true(reference_evidence_rows, teacher_run["predictions"])
    drift_base = accuracy_true(drift_evidence_rows, teacher_run["predictions"])
    task_evidence = {
        "reference_base_acc": round(reference_base, 6),
        "drift_base_acc": round(drift_base, 6),
        "teacher_model": args.teacher_model_name,
        "teacher_D_top40_acc": round(teacher_quality["D_top40_teacher_true_acc"], 6),
        "teacher_D_top80_acc": round(teacher_quality["D_top40_teacher_true_acc"], 6),
        "teacher_R_train_acc": round(teacher_quality["R_train_teacher_true_acc"], 6),
        "teacher_R_val_acc": round(teacher_quality["R_val_teacher_true_acc"], 6),
    }
    if not args.external_teacher_pseudo_dir:
        evidence_rows = unique_rows(reference_evidence_rows, drift_evidence_rows)
        teacher_adapted_evidence = predict_head(teacher_head, teacher_features, evidence_rows)
        reference_adapted = accuracy_true(reference_evidence_rows, teacher_adapted_evidence)
        drift_adapted = accuracy_true(drift_evidence_rows, teacher_adapted_evidence)
        task_evidence["reference_adapted_acc"] = round(reference_adapted, 6)
        task_evidence["reference_gain"] = round(reference_adapted - reference_base, 6)
        task_evidence["drift_adapted_acc"] = round(drift_adapted, 6)
        task_evidence["drift_gain"] = round(drift_adapted - drift_base, 6)
    write_json(args.output_dir / "task_selection_evidence.json", task_evidence)

    lambda_probe_rows: list[dict[str, Any]] = []
    probe_rows = d_top80[: min(args.lambda_probe_count, len(d_top80))]
    probe_results = []
    for config in remoteclip_configs():
        run = extract_remoteclip_for_config(
            config=config,
            rows=probe_rows,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=args.device,
            out_path=args.output_dir / "lambda_profiles" / "probe20" / f"{config.lambda_id}_predictions.jsonl",
            pseudo=pseudo,
        )
        probe_results.append(run)
        lambda_probe_rows.append(
            {
                "group_id": GROUP_ID,
                "lambda_id": config.lambda_id,
                "resolution_keep_ratio": config.resolution_keep_ratio,
                "model_precision": config.model_precision,
                "batch_size": config.batch_size,
                "teacher_accuracy": accuracy_pseudo(probe_rows, run["predictions"], pseudo),
                "true_accuracy": accuracy_true(probe_rows, run["predictions"]),
                "peak_memory_gib": run["peak_memory_gib"],
                "raw_predictions": run["raw_predictions"],
                **lambda_timing_fields(args, config.lambda_id, run),
            }
        )
    normalize_lambda_costs(lambda_probe_rows, args.lambda_max_resource_cost)
    pareto_probe = pareto_selected(lambda_probe_rows)
    selected_probe = list(lambda_probe_rows) if args.disable_pareto_filter else pareto_probe
    selected_ids = {row["lambda_id"] for row in selected_probe}
    for row in lambda_probe_rows:
        row["pareto_selected"] = row["lambda_id"] in {item["lambda_id"] for item in pareto_probe}
        row["selected_for_full_eval"] = row["lambda_id"] in selected_ids
        row["lambda_selection_policy"] = "all_configs_no_pareto" if args.disable_pareto_filter else "probe20_pareto"
    write_csv(args.output_dir / "lambda_profiles" / "lambda_probe20_all_config.csv", lambda_probe_rows)
    write_json(args.output_dir / "lambda_profiles" / "lambda_probe20_all_config.json", lambda_probe_rows)
    write_csv(args.output_dir / "lambda_profiles" / "lambda_pareto_selected_configs.csv", selected_probe)
    write_json(args.output_dir / "lambda_profiles" / "lambda_pareto_selected_configs.json", selected_probe)

    full_eval_rows = unique_rows(
        d_top80,
        splits["R_no_drift"],
        splits["R_train"],
        splits["R_val"],
        splits["R_test"],
        splits["R_all"],
    )
    lambda_full_rows: list[dict[str, Any]] = []
    lambda_full_runs: dict[str, dict[str, Any]] = {}
    predictions_by_lambda: dict[str, dict[str, str]] = {}
    features_by_lambda: dict[str, dict[str, np.ndarray]] = {}
    selected_configs = [config for config in remoteclip_configs() if config.lambda_id in selected_ids]
    for config in selected_configs:
        run = extract_remoteclip_for_config(
            config=config,
            rows=full_eval_rows,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=args.device,
            out_path=args.output_dir / "lambda_profiles" / "full_dtop40" / f"{config.lambda_id}_predictions.jsonl",
            pseudo=pseudo,
        )
        lambda_full_runs[config.lambda_id] = run
        predictions_by_lambda[config.lambda_id] = run["predictions"]
        features_by_lambda[config.lambda_id] = run["features"]
        true_acc = accuracy_true(d_top80, run["predictions"])
        teacher_acc = accuracy_pseudo(d_top80, run["predictions"], pseudo)
        lambda_full_rows.append(
            {
                "group_id": GROUP_ID,
                "lambda_id": config.lambda_id,
                "resolution_keep_ratio": config.resolution_keep_ratio,
                "model_precision": config.model_precision,
                "batch_size": config.batch_size,
                "accuracy": teacher_acc,
                "teacher_accuracy": teacher_acc,
                "true_accuracy": true_acc,
                "teacher_noise_accuracy_gap": true_acc - teacher_acc,
                "teacher_correct": sum(int(run["predictions"].get(row["uid"]) == pseudo.get(row["uid"])) for row in d_top80),
                "true_correct": sum(int(run["predictions"].get(row["uid"]) == row["answer"]) for row in d_top80),
                "measured_samples": len(d_top80),
                "peak_memory_gib": run["peak_memory_gib"],
                "raw_predictions": run["raw_predictions"],
                **lambda_timing_fields(args, config.lambda_id, run),
            }
        )
    normalize_lambda_costs(lambda_full_rows, args.lambda_max_resource_cost)
    write_csv(args.output_dir / "lambda_profiles" / "lambda_profiles_group_v2.csv", lambda_full_rows)
    write_json(args.output_dir / "lambda_profiles" / "lambda_profiles_group_v2.json", lambda_full_rows)

    gamma_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    adapter_predictions: dict[tuple[str, str], dict[str, str]] = {}
    max_train_time = 1e-9
    for lambda_row in lambda_full_rows:
        lambda_id = lambda_row["lambda_id"]
        features = features_by_lambda[lambda_id]
        base_preds = predictions_by_lambda[lambda_id]
        micro_head, micro_time = train_head_sgd(
            features,
            splits["R_train"],
            label_letters=pseudo,
            epochs=1,
            seed=f"{args.split_seed}:{lambda_id}:gamma:micro",
        )
        micro_preds = predict_head(micro_head, features, full_eval_rows)
        micro_gain = accuracy_pseudo(splits["R_val"], micro_preds, pseudo) - accuracy_pseudo(splits["R_val"], base_preds, pseudo)
        for gamma_cfg in gamma_configs():
            head, train_time = train_head_sgd(
                features,
                splits["R_train"],
                label_letters=pseudo,
                epochs=gamma_cfg.epochs,
                seed=f"{args.split_seed}:{lambda_id}:{gamma_cfg.gamma_id}",
            )
            preds = predict_head(head, features, full_eval_rows)
            adapter_predictions[(lambda_id, gamma_cfg.gamma_id)] = preds
            max_train_time = max(max_train_time, train_time)
            base_pseudo = accuracy_pseudo(splits["R_val"], base_preds, pseudo)
            adapter_pseudo = accuracy_pseudo(splits["R_val"], preds, pseudo)
            actual_target_gain = adapter_pseudo - base_pseudo
            predicted_gain = (
                actual_target_gain
                if args.gamma_extrapolation_method == "direct_measured"
                else extrapolate_gamma_gain(
                    base_acc=base_pseudo,
                    profile_gain=micro_gain,
                    target_steps=gamma_cfg.epochs,
                    profile_steps=1,
                    args=args,
                )
            )
            adapter_pred_path = (
                args.output_dir
                / "gamma_profiles"
                / "adapter_predictions"
                / f"{lambda_id}_{gamma_cfg.gamma_id}.jsonl"
            )
            gamma_rows.append(
                {
                    "group_id": GROUP_ID,
                    "gamma_id": gamma_cfg.gamma_id,
                    "feature_lambda_id": lambda_id,
                    "method": "linear_head_sgd",
                    "target_epochs": gamma_cfg.epochs,
                    "microprofile_epochs": 1,
                    "train_samples": len(splits["R_train"]),
                    "rval_base_accuracy_before_1epoch": base_pseudo,
                    "rval_adapter_accuracy_after_target": adapter_pseudo,
                    "profile_gain": micro_gain,
                    "predicted_final_gain": predicted_gain,
                    "actual_target_gain": actual_target_gain,
                    "microprofile_train_time_s": micro_time,
                    "target_train_time_s_est": train_time,
                    "measured_train_time_s": train_time,
                    "resource_cost": 0.01,
                    "peak_memory_gib": 0.0,
                    "adapter_path": str(adapter_pred_path),
                    "adapter_predictions_path": str(adapter_pred_path),
                    "extrapolation_method": args.gamma_extrapolation_method,
                }
            )
            all_pair_rows.append(
                {
                    "group_id": GROUP_ID,
                    "lambda_id": lambda_id,
                    "gamma_id": gamma_cfg.gamma_id,
                    "is_selected_pair_by_ours": False,
                    "dtop40_true_accuracy": lambda_row["true_accuracy"],
                    "dtop40_teacher_accuracy": lambda_row["teacher_accuracy"],
                    "dtop80_true_accuracy": lambda_row["true_accuracy"],
                    "dtop80_teacher_accuracy": lambda_row["teacher_accuracy"],
                    "lambda_resource_cost": lambda_row["resource_cost"],
                    "small_step_profile_gain": micro_gain,
                    "extrapolated_target_gain": predicted_gain,
                    "rval_base_accuracy_before_1epoch": base_pseudo,
                    "rval_adapter_accuracy_after_target": adapter_pseudo,
                    "target_epochs": gamma_cfg.epochs,
                    "gamma_resource_cost": 0.01,
                    "estimated_current_accuracy_if_selected": "",
                    "estimated_future_or_final_accuracy_if_selected": "",
                    "actual_current_accuracy_if_selected": "",
                    "actual_final_accuracy_if_selected": "",
                    "current_error_if_selected": "",
                    "final_error_if_selected": "",
                }
            )
            write_predictions(adapter_pred_path, full_eval_rows, preds, pseudo)
    for row in gamma_rows:
        row["resource_cost"] = round(max(0.01, min(0.95, float(row["measured_train_time_s"]) / max_train_time * args.gamma_max_resource_cost)), 6)
    for row in all_pair_rows:
        match = next(gr for gr in gamma_rows if gr["feature_lambda_id"] == row["lambda_id"] and gr["gamma_id"] == row["gamma_id"])
        row["gamma_resource_cost"] = match["resource_cost"]
    write_csv(args.output_dir / "gamma_profiles" / "gamma_group_profiles.csv", gamma_rows)
    write_json(args.output_dir / "gamma_profiles" / "gamma_group_profiles.json", gamma_rows)

    baseline_rows = build_baseline_decisions(
        lambda_rows=lambda_full_rows,
        gamma_rows=gamma_rows,
        predictions_by_lambda=predictions_by_lambda,
        adapter_predictions=adapter_predictions,
        splits=splits,
        pseudo=pseudo,
        args=args,
    )
    ours = choose_ours(
        lambda_rows=lambda_full_rows,
        gamma_rows=gamma_rows,
        predictions_by_lambda=predictions_by_lambda,
        adapter_predictions=adapter_predictions,
        splits=splits,
        pseudo=pseudo,
        triggered=detection["actual_D_window_triggered"],
        args=args,
    )
    oracle = choose_offline_oracle(
        lambda_rows=lambda_full_rows,
        gamma_rows=gamma_rows,
        predictions_by_lambda=predictions_by_lambda,
        adapter_predictions=adapter_predictions,
        splits=splits,
        pseudo=pseudo,
        args=args,
    )
    method_rows = baseline_rows + [ours, oracle]
    selected_pair = (ours["lambda_id"], ours["gamma_id"])
    for row in all_pair_rows:
        if (row["lambda_id"], row["gamma_id"]) == selected_pair:
            row["is_selected_pair_by_ours"] = True
            row["estimated_current_accuracy_if_selected"] = ours["estimated_current_accuracy"]
            row["estimated_future_or_final_accuracy_if_selected"] = ours["estimated_future_or_final_accuracy"]
            row["actual_current_accuracy_if_selected"] = ours["observed_current_accuracy"]
            row["actual_final_accuracy_if_selected"] = ours["observed_final_accuracy"]
            row["current_error_if_selected"] = ours["current_error_actual_minus_estimated"]
            row["final_error_if_selected"] = ours["final_error_actual_minus_estimated_future"]

    write_csv(args.output_dir / "scheduler" / "scheduler_end_to_end_results.csv", method_rows)
    write_csv(args.output_dir / "method_comparison.csv", method_rows)
    write_json(args.output_dir / "method_comparison.json", method_rows)
    write_csv(args.output_dir / "true_accuracy" / "scheduler_true_accuracy_error_rows.csv", method_rows)
    write_csv(args.output_dir / "detailed_result_tables" / "lambda_gamma_estimates_all_pairs.csv", all_pair_rows)

    split_summary = {key: len(rows) for key, rows in splits.items()}
    payload = {
        "protocol": {
            "model_id": MODEL_ID,
            "teacher_model": args.teacher_model_name,
            "student_model": MODEL_ID,
            "reference_task": args.reference_task,
            "drift_task": args.drift_task,
            "drift_ratio": args.drift_ratio,
            "detector": "StandardScaler + PCA + Mahalanobis",
            "threshold_quantile": args.threshold_quantile,
            "split_schema": "part0_history_no_drift, part1_teacher_ft_train, part2_D_detection_window_50_50, part3_R_retraining_window_drift_only",
            "headline_accuracy_scope": "formal R_all is drift-task only; R_no_drift is auxiliary and not averaged into headline acc",
            "history_reference_count": len(splits["history_reference_fit"]),
            "history_threshold_count": len(splits["history_threshold_cal"]),
            "D_top80_selection": "top drift-score fraction" if args.d_top_count is None else "top drift-score count",
            "D_top80_top_fraction": args.d_top_fraction if args.d_top_count is None else "",
            "D_top80_top_count": d_top_count,
            "D_top40_selection": "complete-window top drift-score fraction" if args.d_top_count is None else "complete-window top drift-score count",
            "D_top40_top_fraction": args.d_top_fraction if args.d_top_count is None else "",
            "D_top40_top_count": d_top_count,
            "lambda_selection_policy": "all_configs_no_pareto" if args.disable_pareto_filter else "probe20_pareto",
            "disable_pareto_filter": bool(args.disable_pareto_filter),
            "scheduler_estimate_mode": args.scheduler_estimate_mode,
            "gamma_extrapolation_method": args.gamma_extrapolation_method,
            "scheduler_optimus_max_accuracy": args.scheduler_optimus_max_accuracy,
            "periodic_period_samples": args.periodic_period_samples,
            "resource_grid": "I=0.01..0.99, R=1-I",
            "experiment_seed": args.split_seed,
            "arrival_rate": args.arrival_rate if args.arrival_rate is not None else "",
            "elapsed_sec": time.perf_counter() - started,
            "remoteclip_params": param_info,
            "reuse_splits_dir": str(args.reuse_splits_dir) if args.reuse_splits_dir else "",
            "external_teacher_pseudo_dir": str(args.external_teacher_pseudo_dir) if args.external_teacher_pseudo_dir else "",
            "canonical_lambda_timing_json": str(args.canonical_lambda_timing_json),
            "timing_policy": "canonical_reuse_observed_runtime_non_authoritative",
        },
        "split_summary": split_summary,
        "detection": {k: v for k, v in detection.items() if k not in {"D_window_scores", "eval_rows"}},
        "teacher_quality": teacher_quality,
        "lambda_profiles": lambda_full_rows,
        "gamma_profiles": gamma_rows,
        "methods": method_rows,
        "old_run_audit": {
            "previous_v4": "results/omniearth_nonqwen_full_framework_landcover_disaster_q095_r50_20260705_v4",
            "previous_v2_formal": "results/omniearth_remoteclip_formal_landcover_disaster_q095_r50_20260705_v2",
            "not_met": [
                "D_top80 was no-drift land-cover subset, not detector-selected drift-score top samples",
                "Used true labels for adaptation/profile instead of teacher pseudo labels",
                "Used one lambda/gamma instead of config tables",
                "No 20-sample lambda probe or Pareto filtering",
                "Coarse resource grid instead of 0.01 enumeration",
                "Raw predictions were not stored for every measured configuration",
                "v2 formal used D_no_drift to fit reference distribution and threshold instead of a disjoint part-0 history split",
                "v5 mixed land-cover no-drift rows into headline R_all accuracy instead of using drift-only R_all",
            ],
        },
    }
    write_json(args.output_dir / "framework_results.json", payload)
    build_detailed_tables(args.output_dir, lambda_full_rows, gamma_rows, all_pair_rows, method_rows)
    write_summary(
        args.output_dir,
        split_summary=split_summary,
        detection=detection,
        task_evidence=task_evidence,
        teacher_quality=teacher_quality,
        method_rows=method_rows,
        lambda_rows=lambda_full_rows,
        gamma_rows=gamma_rows,
    )
    print((args.output_dir / "framework_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
