from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING_SCRIPT = SCRIPT_DIR / "screen_omniearth_small_experiments.py"


def load_screening_module() -> Any:
    spec = importlib.util.spec_from_file_location("screen_omniearth_small_experiments", SCREENING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCREENING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


screening = load_screening_module()


def stable_seed(text: str) -> int:
    import hashlib

    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def split_reference_rows(rows: list[dict[str, Any]], split_seed: str) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: stable_seed(f"{split_seed}:reference:{row['uid']}"))
    n = len(ordered)
    ref_end = max(1, int(0.40 * n))
    cal_end = max(ref_end + 1, int(0.70 * n))
    return {
        "reference_fit": ordered[:ref_end],
        "threshold_cal": ordered[ref_end:cal_end],
        "no_drift_test": ordered[cal_end:],
    }


def feature_matrix(features: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([features[row["uid"]] for row in rows], dtype=np.float64)


def sample_window(
    rng: np.random.Generator,
    pool: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, pool.shape[1]), dtype=pool.dtype)
    indexes = rng.integers(0, len(pool), size=count)
    return pool[indexes]


def fit_detector(
    ref_features: np.ndarray,
    cal_features: np.ndarray,
    *,
    pca_components: int,
    window_size: int,
    calibration_windows: int,
    threshold_quantile: float,
    score_mode: str,
    seed: int,
) -> dict[str, Any]:
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(ref_features)
    cal_scaled = scaler.transform(cal_features)
    n_components = max(1, min(pca_components, ref_scaled.shape[0] - 1, ref_scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    ref_z = pca.fit_transform(ref_scaled)
    cal_z = pca.transform(cal_scaled)

    mu = ref_z.mean(axis=0)
    cov = np.cov(ref_z, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + np.eye(cov.shape[0]) * 1e-4
    inv_cov = np.linalg.pinv(cov)

    def sample_distances(window_features: np.ndarray) -> np.ndarray:
        z = pca.transform(scaler.transform(window_features))
        delta = z - mu
        sq = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
        return np.sqrt(np.maximum(0.0, sq))

    def score_window(window_features: np.ndarray) -> float:
        distances = sample_distances(window_features)
        if score_mode == "mean_sample":
            return float(np.mean(distances))
        if score_mode == "p90_sample":
            return float(np.quantile(distances, 0.90))
        if score_mode == "window_mean":
            z = pca.transform(scaler.transform(window_features))
            delta = z.mean(axis=0) - mu
            return float(np.sqrt(max(0.0, delta @ inv_cov @ delta.T)))
        raise ValueError(f"Unknown score_mode: {score_mode}")

    rng = np.random.default_rng(seed)
    cal_scores = []
    for _ in range(calibration_windows):
        window = sample_window(rng, cal_features, window_size)
        cal_scores.append(score_window(window))
    threshold = float(np.quantile(cal_scores, threshold_quantile))
    return {
        "scaler": scaler,
        "pca": pca,
        "score_window": score_window,
        "threshold": threshold,
        "calibration_scores": cal_scores,
        "pca_components": n_components,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "score_mode": score_mode,
    }


def evaluate_detector(
    detector: dict[str, Any],
    no_drift_features: np.ndarray,
    drift_features: np.ndarray,
    *,
    window_size: int,
    drift_ratios: list[float],
    eval_windows: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    score_window = detector["score_window"]
    threshold = detector["threshold"]
    for ratio in drift_ratios:
        drift_count = int(round(window_size * ratio))
        drift_count = min(window_size, max(0, drift_count))
        no_drift_count = window_size - drift_count
        positives = 0
        scores = []
        for _ in range(eval_windows):
            no_part = sample_window(rng, no_drift_features, no_drift_count)
            drift_part = sample_window(rng, drift_features, drift_count)
            if len(drift_part):
                window = np.concatenate([no_part, drift_part], axis=0)
            else:
                window = no_part
            rng.shuffle(window)
            score = score_window(window)
            scores.append(score)
            positives += int(score > threshold)
        rows.append(
            {
                "drift_ratio": ratio,
                "positive_rate": positives / eval_windows if eval_windows else 0.0,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "threshold": threshold,
                "eval_windows": eval_windows,
            }
        )
    return rows


def load_screening_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", [])


def screening_row(rows: list[dict[str, Any]], family: str, model: str, task_key: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("family") == family and row.get("model") == model and row.get("task_key") == task_key:
            return row
    return None


def extract_resnet_features(rows_by_task: dict[str, list[dict[str, Any]]], device: str) -> dict[str, np.ndarray]:
    all_rows = [row for rows in rows_by_task.values() for row in rows]
    features, _params, _elapsed = screening.extract_cnn_features(all_rows, "resnet18", device)
    return features


def extract_remoteclip_features(rows_by_task: dict[str, list[dict[str, Any]]], model_dir: Path, device: str) -> dict[str, np.ndarray]:
    import open_clip
    import torch
    import torch.nn.functional as F
    from PIL import Image

    weight_path = model_dir / "RemoteCLIP-RN50.pt"
    model, _, preprocess = open_clip.create_model_and_transforms("RN50", pretrained=str(weight_path), device=device)
    model.eval()
    features: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for row in [item for rows in rows_by_task.values() for item in rows]:
            image = preprocess(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
            feature = F.normalize(model.encode_image(image), dim=-1)
            features[row["uid"]] = feature.cpu().numpy()[0]
    return features


def run_model_flow(
    *,
    model_id: str,
    family: str,
    model_name: str,
    features: dict[str, np.ndarray],
    rows_by_task: dict[str, list[dict[str, Any]]],
    screening_rows: list[dict[str, Any]],
    reference_task: str,
    drift_tasks: list[str],
    split_seed: str,
    pca_components: int,
    window_size: int,
    calibration_windows: int,
    eval_windows: int,
    threshold_quantile: float,
    score_mode: str,
    drift_ratios: list[float],
) -> dict[str, Any]:
    ref_splits = split_reference_rows(rows_by_task[reference_task], split_seed=f"{split_seed}:{model_id}")
    ref_features = feature_matrix(features, ref_splits["reference_fit"])
    cal_features = feature_matrix(features, ref_splits["threshold_cal"])
    no_drift_features = feature_matrix(features, ref_splits["no_drift_test"])
    detector = fit_detector(
        ref_features,
        cal_features,
        pca_components=pca_components,
        window_size=window_size,
        calibration_windows=calibration_windows,
        threshold_quantile=threshold_quantile,
        score_mode=score_mode,
        seed=stable_seed(f"{split_seed}:{model_id}:detector"),
    )
    drift_results = []
    for task_key in drift_tasks:
        drift_features = feature_matrix(features, rows_by_task[task_key])
        detection = evaluate_detector(
            detector,
            no_drift_features,
            drift_features,
            window_size=window_size,
            drift_ratios=drift_ratios,
            eval_windows=eval_windows,
            seed=stable_seed(f"{split_seed}:{model_id}:{task_key}:eval"),
        )
        gain_row = screening_row(screening_rows, family, model_name, task_key)
        drift_results.append(
            {
                "task_key": task_key,
                "category": rows_by_task[task_key][0]["category"],
                "rows": len(rows_by_task[task_key]),
                "screening": gain_row,
                "detection": detection,
            }
        )
    reference_screening = screening_row(screening_rows, family, model_name, reference_task)
    return {
        "model_id": model_id,
        "family": family,
        "model": model_name,
        "reference_task": reference_task,
        "reference_category": rows_by_task[reference_task][0]["category"],
        "reference_rows": {key: len(value) for key, value in ref_splits.items()},
        "reference_screening": reference_screening,
        "detector": {
            "pca_components": detector["pca_components"],
            "explained_variance_ratio_sum": detector["explained_variance_ratio_sum"],
            "threshold": detector["threshold"],
            "threshold_quantile": threshold_quantile,
            "window_size": window_size,
            "score_mode": detector["score_mode"],
        },
        "drift_tasks": drift_results,
    }


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "small_specialist_flow_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# OmniEarth Small/Specialist Drift Flow",
        "",
        f"- reference/no-drift task: `{payload['reference_task']}`",
        f"- drift tasks: `{', '.join(payload['drift_tasks'])}`",
        f"- drift ratios evaluated: `{', '.join(str(ratio) for ratio in payload['drift_ratios'])}`",
        f"- threshold quantile: `{payload['protocol']['threshold_quantile']}`",
        "",
        "## Reference Split",
        "",
        "| model | reference_fit | threshold_cal | no_drift_test |",
        "|---|---:|---:|---:|",
    ]
    for model in payload["models"]:
        rows = model["reference_rows"]
        lines.append(
            "| {model} | {reference_fit} | {threshold_cal} | {no_drift_test} |".format(
                model=model["model_id"],
                reference_fit=rows["reference_fit"],
                threshold_cal=rows["threshold_cal"],
                no_drift_test=rows["no_drift_test"],
            )
        )
    lines.extend(
        [
            "",
            "## Detection Results",
            "",
            "| model | reference base/adapt/gain | drift task | drift base/adapt/gain | FPR@0 | R@50 | threshold |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for model in payload["models"]:
        ref_screen = model.get("reference_screening") or {}
        ref_text = "{:.4f}/{:.4f}/{:+.4f}".format(
            float(ref_screen.get("base_acc", 0.0)),
            float(ref_screen.get("adapted_acc", 0.0)),
            float(ref_screen.get("gain", 0.0)),
        )
        for task in model["drift_tasks"]:
            screen = task.get("screening") or {}
            drift_text = "{:.4f}/{:.4f}/{:+.4f}".format(
                float(screen.get("base_acc", 0.0)),
                float(screen.get("adapted_acc", 0.0)),
                float(screen.get("gain", 0.0)),
            )
            by_ratio = {row["drift_ratio"]: row["positive_rate"] for row in task["detection"]}
            lines.append(
                "| {model} | {ref_text} | {task_key} | {drift_text} | {r0:.3f} | {r50:.3f} | {threshold:.4f} |".format(
                    model=model["model_id"],
                    ref_text=ref_text,
                    task_key=task["task_key"],
                    drift_text=drift_text,
                    r0=by_ratio.get(0.0, 0.0),
                    r50=by_ratio.get(0.50, 0.0),
                    threshold=model["detector"]["threshold"],
                )
            )
    (output_dir / "small_specialist_flow_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniEarth small/specialist drift flow.")
    parser.add_argument("--omniearth-root", type=Path, default=Path("@@SERVER_ROOT@@/datasets/public_benchmarks/OmniEarth"))
    parser.add_argument("--screening-results", type=Path, required=True)
    parser.add_argument("--remoteclip-dir", type=Path, default=Path("@@WORKSPACE@@/qwen_eval/models/RemoteCLIP"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", default="flow_seed1")
    parser.add_argument("--reference-task", default="landcover")
    parser.add_argument("--drift-tasks", default="modality")
    parser.add_argument("--drift-ratios", default="0.50")
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--eval-windows", type=int, default=300)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--score-mode", choices=["mean_sample", "p90_sample", "window_mean"], default="mean_sample")
    parser.add_argument("--max-image-pixels", type=int, default=80_000_000)
    return parser.parse_args()


def parse_drift_ratios(text: str) -> list[float]:
    ratios = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ratio = float(part)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"Drift ratio must be in [0, 1], got {ratio}")
        ratios.append(ratio)
    if 0.0 not in ratios:
        ratios.insert(0, 0.0)
    return sorted(set(ratios))


def main() -> int:
    args = parse_args()
    task_keys = [args.reference_task] + [part.strip() for part in args.drift_tasks.split(",") if part.strip()]
    specs = {spec.task_key: spec for spec in screening.CATEGORIES}
    rows_by_task = {
        task_key: screening.read_category_rows(args.omniearth_root, specs[task_key], args.max_image_pixels)
        for task_key in task_keys
    }
    screening_rows = load_screening_rows(args.screening_results)
    drift_ratios = parse_drift_ratios(args.drift_ratios)

    resnet_features = extract_resnet_features(rows_by_task, args.device)
    remoteclip_features = extract_remoteclip_features(rows_by_task, args.remoteclip_dir, args.device)

    drift_tasks = [part.strip() for part in args.drift_tasks.split(",") if part.strip()]
    models = [
        run_model_flow(
            model_id="ResNet18",
            family="cnn",
            model_name="resnet18",
            features=resnet_features,
            rows_by_task=rows_by_task,
            screening_rows=screening_rows,
            reference_task=args.reference_task,
            drift_tasks=drift_tasks,
            split_seed=args.split_seed,
            pca_components=args.pca_components,
            window_size=args.window_size,
            calibration_windows=args.calibration_windows,
            eval_windows=args.eval_windows,
            threshold_quantile=args.threshold_quantile,
            score_mode=args.score_mode,
            drift_ratios=drift_ratios,
        ),
        run_model_flow(
            model_id="RemoteCLIP-RN50",
            family="remoteclip",
            model_name="RemoteCLIP-RN50",
            features=remoteclip_features,
            rows_by_task=rows_by_task,
            screening_rows=screening_rows,
            reference_task=args.reference_task,
            drift_tasks=drift_tasks,
            split_seed=args.split_seed,
            pca_components=args.pca_components,
            window_size=args.window_size,
            calibration_windows=args.calibration_windows,
            eval_windows=args.eval_windows,
            threshold_quantile=args.threshold_quantile,
            score_mode=args.score_mode,
            drift_ratios=drift_ratios,
        ),
    ]
    payload = {
        "reference_task": args.reference_task,
        "drift_tasks": drift_tasks,
        "drift_ratios": drift_ratios,
        "protocol": {
            "reference_fit": "OmniEarth reference task split, model-specific features",
            "threshold_cal": "held-out reference windows",
            "stream_eval": "mixed windows from held-out reference + drift task rows",
            "detector": "StandardScaler + PCA + Mahalanobis distance",
            "score_mode": args.score_mode,
            "threshold_quantile": args.threshold_quantile,
            "drift_definition": "cross-task OmniEarth categories; no within-task modality/domain split",
            "adaptation_screening": "true labels for small screening; formal framework should replace with teacher pseudo labels",
        },
        "models": models,
    }
    write_summary(args.output_dir, payload)
    print((args.output_dir / "small_specialist_flow_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
