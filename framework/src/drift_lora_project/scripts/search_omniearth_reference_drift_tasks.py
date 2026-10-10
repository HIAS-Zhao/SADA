from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING_SCRIPT = SCRIPT_DIR / "screen_omniearth_small_experiments.py"
FLOW_SCRIPT = SCRIPT_DIR / "run_omniearth_small_specialist_flow.py"


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


@dataclass(frozen=True)
class DynamicTaskSpec:
    task_key: str
    category: str
    json_relpath: str


def slugify(text: str) -> str:
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def discover_specs(root: Path) -> list[DynamicTaskSpec]:
    specs: list[DynamicTaskSpec] = []
    for json_path in sorted(root.glob("*/*/*.json")):
        rel = json_path.relative_to(root)
        category = rel.parent.as_posix()
        task_key = slugify(category)
        specs.append(DynamicTaskSpec(task_key=task_key, category=category, json_relpath=rel.as_posix()))
    return specs


def read_rows(root: Path, spec: DynamicTaskSpec, max_image_pixels: int) -> list[dict[str, Any]]:
    category_spec = screening.CategorySpec(
        task_key=spec.task_key,
        category=spec.category,
        json_relpath=spec.json_relpath,
    )
    return screening.read_category_rows(root, category_spec, max_image_pixels)


def stable_seed(text: str) -> int:
    return flow.stable_seed(text)


def cap_rows(rows: list[dict[str, Any]], max_rows: int, seed: str) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return sorted(rows, key=lambda row: stable_seed(f"{seed}:{row['uid']}"))[:max_rows]


def label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["label"] for row in rows)
    return {
        "rows": len(rows),
        "num_classes": len(counts),
        "min_class_count": min(counts.values()) if counts else 0,
        "median_class_count": float(np.median(list(counts.values()))) if counts else 0.0,
        "max_class_count": max(counts.values()) if counts else 0,
        "top_labels": counts.most_common(8),
    }


def safe_train_eval_linear_head(
    features: dict[str, np.ndarray],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    train_labels = {row["label"] for row in train_rows}
    if len(train_labels) < 2 or not test_rows:
        return None
    try:
        return screening.train_eval_linear_head(features, train_rows, test_rows)
    except Exception as exc:
        return {"error": str(exc), "accuracy": 0.0, "correct": 0, "total": len(test_rows)}


def screen_cnn_tasks(
    rows_by_task: dict[str, list[dict[str, Any]]],
    features: dict[str, np.ndarray],
    *,
    model_name: str,
    params: int,
    split_seed: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task_key, rows in rows_by_task.items():
        train_rows, test_rows = screening.split_by_class(rows, split_seed)
        base = safe_train_eval_linear_head(features, screening.lowshot_rows(train_rows), test_rows)
        adapted = safe_train_eval_linear_head(features, train_rows, test_rows)
        if base is None or adapted is None:
            continue
        results.append(
            {
                "family": "cnn",
                "model": model_name,
                "task_key": task_key,
                "category": rows[0]["category"],
                "params": params,
                "rows": len(rows),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "num_classes": label_summary(rows)["num_classes"],
                "base_protocol": "frozen ImageNet ResNet18 + 1-shot-per-class linear head",
                "adapt_protocol": "frozen ImageNet ResNet18 + full train split linear head",
                "base_acc": float(base.get("accuracy", 0.0)),
                "adapted_acc": float(adapted.get("accuracy", 0.0)),
                "gain": float(adapted.get("accuracy", 0.0)) - float(base.get("accuracy", 0.0)),
                "base_details": base,
                "adapted_details": adapted,
            }
        )
    return results


def extract_remoteclip_and_screen(
    rows_by_task: dict[str, list[dict[str, Any]]],
    model_dir: Path,
    device: str,
    split_seed: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    import open_clip
    import torch
    import torch.nn.functional as F
    from PIL import Image

    weight_path = model_dir / "RemoteCLIP-RN50.pt"
    param_info = screening.remoteclip_param_count(weight_path)
    model, _, preprocess = open_clip.create_model_and_transforms("RN50", pretrained=str(weight_path), device=device)
    tokenizer = open_clip.get_tokenizer("RN50")
    model.eval()

    features: dict[str, np.ndarray] = {}
    zero_predictions: dict[str, str] = {}
    all_rows = [row for rows in rows_by_task.values() for row in rows]
    with torch.no_grad():
        for row in all_rows:
            image = preprocess(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
            image_feature = F.normalize(model.encode_image(image), dim=-1)
            features[row["uid"]] = image_feature.cpu().numpy()[0]
            scores: list[tuple[float, str]] = []
            for letter, option in row["options"].items():
                prompts = screening.remoteclip_prompts(row["question"], option)
                text_feature = F.normalize(model.encode_text(tokenizer(prompts).to(device)), dim=-1).mean(dim=0)
                text_feature = F.normalize(text_feature.unsqueeze(0), dim=-1)
                scores.append((float((image_feature * text_feature).sum().cpu()), letter))
            zero_predictions[row["uid"]] = max(scores)[1]

    results: list[dict[str, Any]] = []
    for task_key, rows in rows_by_task.items():
        train_rows, test_rows = screening.split_by_class(rows, split_seed)
        adapted = safe_train_eval_linear_head(features, train_rows, test_rows)
        if adapted is None or not test_rows:
            continue
        zero_correct = sum(int(zero_predictions[row["uid"]] == row["answer"]) for row in test_rows)
        zero_acc = zero_correct / len(test_rows)
        results.append(
            {
                "family": "remoteclip",
                "model": "RemoteCLIP-RN50",
                "task_key": task_key,
                "category": rows[0]["category"],
                **param_info,
                "rows": len(rows),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "num_classes": label_summary(rows)["num_classes"],
                "base_protocol": "RemoteCLIP image-text zero-shot option scoring",
                "adapt_protocol": "frozen RemoteCLIP image encoder + full train split linear probe",
                "base_acc": float(zero_acc),
                "adapted_acc": float(adapted.get("accuracy", 0.0)),
                "gain": float(adapted.get("accuracy", 0.0)) - float(zero_acc),
                "base_details": {"correct": zero_correct, "total": len(test_rows), "accuracy": zero_acc},
                "adapted_details": adapted,
            }
        )
    return features, results


def feature_matrix(features: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([features[row["uid"]] for row in rows], dtype=np.float64)


def screening_lookup(screening_rows: list[dict[str, Any]], family: str, model: str) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in screening_rows:
        if row["family"] == family and row["model"] == model:
            lookup[row["task_key"]] = row
    return lookup


def select_reference_candidates(
    task_inventory: dict[str, dict[str, Any]],
    screening_rows: list[dict[str, Any]],
    *,
    min_rows: int,
    top_k: int,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in screening_rows:
        by_task.setdefault(row["task_key"], []).append(row)
    candidates = []
    for task_key, rows in by_task.items():
        inventory = task_inventory[task_key]
        if inventory["rows"] < min_rows:
            continue
        models = {row["model"]: row for row in rows}
        if "resnet18" not in models or "RemoteCLIP-RN50" not in models:
            continue
        mean_base = float(np.mean([models["resnet18"]["base_acc"], models["RemoteCLIP-RN50"]["base_acc"]]))
        mean_adapted = float(np.mean([models["resnet18"]["adapted_acc"], models["RemoteCLIP-RN50"]["adapted_acc"]]))
        candidates.append(
            {
                "task_key": task_key,
                "category": inventory["category"],
                "rows": inventory["rows"],
                "num_classes": inventory["num_classes"],
                "resnet18_base": models["resnet18"]["base_acc"],
                "remoteclip_base": models["RemoteCLIP-RN50"]["base_acc"],
                "mean_base": mean_base,
                "mean_adapted": mean_adapted,
            }
        )
    return sorted(candidates, key=lambda row: (row["mean_base"], row["rows"]), reverse=True)[:top_k]


def select_drift_candidates(
    task_inventory: dict[str, dict[str, Any]],
    screening_rows: list[dict[str, Any]],
    *,
    min_rows: int,
    top_k: int,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in screening_rows:
        by_task.setdefault(row["task_key"], []).append(row)
    candidates = []
    for task_key, rows in by_task.items():
        inventory = task_inventory[task_key]
        if inventory["rows"] < min_rows:
            continue
        models = {row["model"]: row for row in rows}
        if "resnet18" not in models or "RemoteCLIP-RN50" not in models:
            continue
        mean_gain = float(np.mean([models["resnet18"]["gain"], models["RemoteCLIP-RN50"]["gain"]]))
        mean_base = float(np.mean([models["resnet18"]["base_acc"], models["RemoteCLIP-RN50"]["base_acc"]]))
        candidates.append(
            {
                "task_key": task_key,
                "category": inventory["category"],
                "rows": inventory["rows"],
                "num_classes": inventory["num_classes"],
                "resnet18_gain": models["resnet18"]["gain"],
                "remoteclip_gain": models["RemoteCLIP-RN50"]["gain"],
                "resnet18_base": models["resnet18"]["base_acc"],
                "remoteclip_base": models["RemoteCLIP-RN50"]["base_acc"],
                "mean_gain": mean_gain,
                "mean_base": mean_base,
                "drift_score": mean_gain + max(0.0, 0.6 - mean_base),
            }
        )
    return sorted(candidates, key=lambda row: (row["drift_score"], row["mean_gain"]), reverse=True)[:top_k]


def run_detection_grid(
    rows_by_task: dict[str, list[dict[str, Any]]],
    features_by_model: dict[str, dict[str, np.ndarray]],
    screening_rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    drifts: list[dict[str, Any]],
    *,
    split_seed: str,
    pca_components: int,
    window_size: int,
    calibration_windows: int,
    eval_windows: int,
    threshold_quantile: float,
    score_modes: list[str],
) -> list[dict[str, Any]]:
    model_meta = {
        "ResNet18": ("cnn", "resnet18"),
        "RemoteCLIP-RN50": ("remoteclip", "RemoteCLIP-RN50"),
    }
    results: list[dict[str, Any]] = []
    for model_id, features in features_by_model.items():
        family, model_name = model_meta[model_id]
        screen = screening_lookup(screening_rows, family, model_name)
        for reference in references:
            ref_task = reference["task_key"]
            splits = flow.split_reference_rows(rows_by_task[ref_task], split_seed=f"{split_seed}:{model_id}:{ref_task}")
            ref_features = feature_matrix(features, splits["reference_fit"])
            cal_features = feature_matrix(features, splits["threshold_cal"])
            no_drift_features = feature_matrix(features, splits["no_drift_test"])
            for score_mode in score_modes:
                detector = flow.fit_detector(
                    ref_features,
                    cal_features,
                    pca_components=pca_components,
                    window_size=window_size,
                    calibration_windows=calibration_windows,
                    threshold_quantile=threshold_quantile,
                    score_mode=score_mode,
                    seed=stable_seed(f"{split_seed}:{model_id}:{ref_task}:{score_mode}:detector"),
                )
                for drift in drifts:
                    drift_task = drift["task_key"]
                    if drift_task == ref_task:
                        continue
                    detection = flow.evaluate_detector(
                        detector,
                        no_drift_features,
                        feature_matrix(features, rows_by_task[drift_task]),
                        window_size=window_size,
                        drift_ratios=[0.0, 0.50],
                        eval_windows=eval_windows,
                        seed=stable_seed(f"{split_seed}:{model_id}:{ref_task}:{drift_task}:{score_mode}:eval"),
                    )
                    by_ratio = {row["drift_ratio"]: row for row in detection}
                    drift_screen = screen.get(drift_task, {})
                    ref_screen = screen.get(ref_task, {})
                    results.append(
                        {
                            "model_id": model_id,
                            "score_mode": score_mode,
                            "reference_task": ref_task,
                            "reference_category": reference["category"],
                            "reference_rows": {key: len(value) for key, value in splits.items()},
                            "reference_base_acc": ref_screen.get("base_acc"),
                            "drift_task": drift_task,
                            "drift_category": drift["category"],
                            "drift_base_acc": drift_screen.get("base_acc"),
                            "drift_adapted_acc": drift_screen.get("adapted_acc"),
                            "drift_gain": drift_screen.get("gain"),
                            "threshold": detector["threshold"],
                            "threshold_quantile": threshold_quantile,
                            "fpr_at_0": by_ratio[0.0]["positive_rate"],
                            "recall_at_50": by_ratio[0.5]["positive_rate"],
                            "mean_score_0": by_ratio[0.0]["mean_score"],
                            "mean_score_50": by_ratio[0.5]["mean_score"],
                        }
                    )
    return results


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference_drift_search_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# OmniEarth Reference/Drift Search",
        "",
        "## Task Inventory",
        "",
        "| task | rows | classes | min/median/max class count | category |",
        "|---|---:|---:|---:|---|",
    ]
    for task_key, row in sorted(payload["task_inventory"].items(), key=lambda item: item[1]["rows"], reverse=True):
        lines.append(
            "| {task} | {rows} | {num_classes} | {min_class_count}/{median_class_count:.1f}/{max_class_count} | {category} |".format(
                task=task_key,
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Reference Candidates",
            "",
            "| task | mean base | ResNet18 base | RemoteCLIP base | rows | classes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["reference_candidates"]:
        lines.append(
            "| {task_key} | {mean_base:.4f} | {resnet18_base:.4f} | {remoteclip_base:.4f} | {rows} | {num_classes} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Drift Candidates",
            "",
            "| task | drift score | mean gain | mean base | ResNet18 gain | RemoteCLIP gain | rows | classes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["drift_candidates"]:
        lines.append(
            "| {task_key} | {drift_score:.4f} | {mean_gain:+.4f} | {mean_base:.4f} | {resnet18_gain:+.4f} | {remoteclip_gain:+.4f} | {rows} | {num_classes} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Best Detection Pairs",
            "",
            "| model | score | reference | drift | drift base/adapt/gain | FPR@0 | R@50 | mean score 0->50 |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    ranked = sorted(
        payload["detection_results"],
        key=lambda row: (row["score_mode"], row["model_id"], row["recall_at_50"], -row["fpr_at_0"]),
        reverse=True,
    )
    for row in ranked[:60]:
        lines.append(
            "| {model_id} | {score_mode} | {reference_task} | {drift_task} | {base:.4f}/{adapt:.4f}/{gain:+.4f} | {fpr:.3f} | {recall:.3f} | {m0:.3f}->{m50:.3f} |".format(
                model_id=row["model_id"],
                score_mode=row["score_mode"],
                reference_task=row["reference_task"],
                drift_task=row["drift_task"],
                base=float(row.get("drift_base_acc") or 0.0),
                adapt=float(row.get("drift_adapted_acc") or 0.0),
                gain=float(row.get("drift_gain") or 0.0),
                fpr=row["fpr_at_0"],
                recall=row["recall_at_50"],
                m0=row["mean_score_0"],
                m50=row["mean_score_50"],
            )
        )

    (output_dir / "reference_drift_search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search OmniEarth reference/drift tasks with small visual encoders.")
    parser.add_argument("--omniearth-root", type=Path, default=Path("@@SERVER_ROOT@@/datasets/public_benchmarks/OmniEarth"))
    parser.add_argument("--remoteclip-dir", type=Path, default=Path("@@WORKSPACE@@/qwen_eval/models/RemoteCLIP"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", default="omniearth_reference_search_seed1")
    parser.add_argument("--max-image-pixels", type=int, default=80_000_000)
    parser.add_argument("--max-rows-per-task", type=int, default=320)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--top-reference", type=int, default=6)
    parser.add_argument("--top-drift", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--calibration-windows", type=int, default=300)
    parser.add_argument("--eval-windows", type=int, default=300)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--score-modes", default="window_mean,mean_sample")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    specs = discover_specs(args.omniearth_root)
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    task_inventory: dict[str, dict[str, Any]] = {}
    for spec in specs:
        rows = read_rows(args.omniearth_root, spec, args.max_image_pixels)
        if not rows:
            continue
        rows = cap_rows(rows, args.max_rows_per_task, args.split_seed)
        if len(rows) < args.min_rows:
            continue
        rows_by_task[spec.task_key] = rows
        task_inventory[spec.task_key] = {
            "category": spec.category,
            **label_summary(rows),
        }

    all_rows = [row for rows in rows_by_task.values() for row in rows]
    resnet_features, resnet_params, resnet_elapsed = screening.extract_cnn_features(all_rows, "resnet18", args.device)
    cnn_rows = screen_cnn_tasks(
        rows_by_task,
        resnet_features,
        model_name="resnet18",
        params=resnet_params,
        split_seed=args.split_seed,
    )
    remote_features, remote_rows = extract_remoteclip_and_screen(
        rows_by_task,
        model_dir=args.remoteclip_dir,
        device=args.device,
        split_seed=args.split_seed,
    )
    screening_rows = cnn_rows + remote_rows

    references = select_reference_candidates(
        task_inventory,
        screening_rows,
        min_rows=args.min_rows,
        top_k=args.top_reference,
    )
    drifts = select_drift_candidates(
        task_inventory,
        screening_rows,
        min_rows=args.min_rows,
        top_k=args.top_drift,
    )
    detection_results = run_detection_grid(
        rows_by_task,
        {
            "ResNet18": resnet_features,
            "RemoteCLIP-RN50": remote_features,
        },
        screening_rows,
        references,
        drifts,
        split_seed=args.split_seed,
        pca_components=args.pca_components,
        window_size=args.window_size,
        calibration_windows=args.calibration_windows,
        eval_windows=args.eval_windows,
        threshold_quantile=args.threshold_quantile,
        score_modes=[part.strip() for part in args.score_modes.split(",") if part.strip()],
    )
    payload = {
        "protocol": {
            "root": args.omniearth_root.as_posix(),
            "max_rows_per_task": args.max_rows_per_task,
            "min_rows": args.min_rows,
            "reference_split": "40% reference_fit, 30% threshold_cal, 30% no_drift_test",
            "threshold_quantile": args.threshold_quantile,
            "drift_ratio": 0.5,
            "window_size": args.window_size,
            "calibration_windows": args.calibration_windows,
            "eval_windows": args.eval_windows,
            "distance": "Mahalanobis in StandardScaler+PCA space",
            "features": {
                "ResNet18": "ImageNet pretrained frozen avgpool feature",
                "RemoteCLIP-RN50": "RemoteCLIP frozen visual embedding",
            },
            "resnet_feature_elapsed_sec": resnet_elapsed,
            "total_elapsed_sec": time.perf_counter() - started,
        },
        "task_inventory": task_inventory,
        "screening_results": screening_rows,
        "reference_candidates": references,
        "drift_candidates": drifts,
        "detection_results": detection_results,
    }
    write_summary(args.output_dir, payload)
    print((args.output_dir / "reference_drift_search_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
