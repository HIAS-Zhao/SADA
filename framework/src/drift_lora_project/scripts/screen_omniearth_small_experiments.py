from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class CategorySpec:
    task_key: str
    category: str
    json_relpath: str


CATEGORIES = [
    CategorySpec(
        task_key="scene",
        category="Perception/A1.1-Scene Classification",
        json_relpath="Perception/A1.1-Scene Classification/scene classification.json",
    ),
    CategorySpec(
        task_key="landcover",
        category="Perception/A1.2-Land-cover Classification",
        json_relpath="Perception/A1.2-Land-cover Classification/land-cover classification.json",
    ),
    CategorySpec(
        task_key="modality",
        category="Perception/A1.3-Image Modality Recognition",
        json_relpath="Perception/A1.3-Image Modality Recognition/image modality recognition.json",
    ),
    CategorySpec(
        task_key="counting",
        category="Perception/A2.3-Object Counting",
        json_relpath="Perception/A2.3-Object Counting/object counting.json",
    ),
    CategorySpec(
        task_key="finegrained",
        category="Perception/A2.4-Fine-grained Category Classification_new",
        json_relpath="Perception/A2.4-Fine-grained Category Classification_new/fine-grained category classification.json",
    ),
    CategorySpec(
        task_key="attribute",
        category="Perception/A2.5-Attribute Recognition",
        json_relpath="Perception/A2.5-Attribute Recognition/attribute recognition.json",
    ),
    CategorySpec(
        task_key="image_condition",
        category="Robustness/C1.1-Image Condition Assessment",
        json_relpath="Robustness/C1.1-Image Condition Assessment/image condition assessment.json",
    ),
]


def clean_label(text: str) -> str:
    label = str(text).strip().lower().replace("_", " ").replace("-", " ")
    label = label.replace("urbun", "urban")
    return re.sub(r"\s+", " ", label)


def stable_digest(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def normalize_options(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key).strip().upper(): str(option).strip() for key, option in value.items()}
    if isinstance(value, list):
        return {LETTERS[index]: str(option).strip() for index, option in enumerate(value)}
    text = str(value).strip()
    markers = list(re.finditer(r"(?<!\w)([A-Z])[\.\):]\s*", text))
    options: dict[str, str] = {}
    for index, match in enumerate(markers):
        key = match.group(1).upper()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        option = text[match.end() : end].strip()
        if key in LETTERS and option:
            options[key] = option
    if not options:
        raise ValueError(f"Cannot parse options: {text[:120]}")
    return options


def normalize_answer(record: dict[str, Any], options: dict[str, str]) -> str:
    candidates = [
        record.get("ground_truth_option"),
        record.get("answer"),
        record.get("label"),
        record.get("ground_truth"),
    ]
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        upper = text.upper()
        if upper in options:
            return upper
        for key, option in options.items():
            if clean_label(text) == clean_label(option):
                return key
    raise ValueError(f"Cannot normalize answer for options={options}")


def resolve_image_path(record_file: Path, image_value: Any) -> Path:
    if isinstance(image_value, list):
        image_value = image_value[0]
    raw = Path(str(image_value))
    if raw.is_absolute():
        return raw
    candidates = [
        record_file.parent / raw,
        record_file.parent / "images" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def image_within_limit(path: Path, max_image_pixels: int) -> bool:
    if max_image_pixels <= 0:
        return True
    from PIL import Image

    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            return int(image.width) * int(image.height) <= max_image_pixels
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit


def read_category_rows(root: Path, spec: CategorySpec, max_image_pixels: int) -> list[dict[str, Any]]:
    record_file = root / spec.json_relpath
    payload = json.loads(record_file.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("data", [])
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "options" not in record:
            continue
        try:
            options = normalize_options(record["options"])
            answer = normalize_answer(record, options)
        except ValueError:
            continue
        image_path = resolve_image_path(record_file, record.get("image_path"))
        if not image_path.exists():
            continue
        if not image_within_limit(image_path, max_image_pixels):
            continue
        prompts = record.get("prompts") or [record.get("question") or ""]
        question = prompts[0] if isinstance(prompts, list) and prompts else str(prompts)
        uid = f"{spec.task_key}_{record.get('question_id', index)}_{index}"
        rows.append(
            {
                "uid": uid,
                "task_key": spec.task_key,
                "category": spec.category,
                "image_path": image_path.as_posix(),
                "question": str(question),
                "options": options,
                "answer": answer,
                "label": clean_label(options[answer]),
            }
        )
    return rows


def split_by_class(rows: list[dict[str, Any]], split_seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label, label_rows in by_label.items():
        ordered = sorted(label_rows, key=lambda row: stable_digest(f"{split_seed}:{row['uid']}"))
        if len(ordered) == 1:
            train.extend(ordered)
            continue
        train_count = max(1, min(len(ordered) - 1, len(ordered) // 2))
        train.extend(ordered[:train_count])
        test.extend(ordered[train_count:])
    return train, test


def lowshot_rows(train_rows: list[dict[str, Any]], samples_per_class: int = 1) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        by_label.setdefault(row["label"], []).append(row)
    picked: list[dict[str, Any]] = []
    for label_rows in by_label.values():
        picked.extend(label_rows[:samples_per_class])
    return picked


def train_eval_linear_head(
    features: dict[str, np.ndarray],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    encoder = LabelEncoder()
    labels = [row["label"] for row in train_rows]
    encoded = encoder.fit_transform(labels)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    started = time.perf_counter()
    classifier.fit(np.array([features[row["uid"]] for row in train_rows]), encoded)
    train_time = time.perf_counter() - started
    class_index = {label: idx for idx, label in enumerate(encoder.classes_)}
    correct = 0
    fallback = 0
    for row in test_rows:
        probabilities = classifier.predict_proba(np.array([features[row["uid"]]]))[0]
        best_score = -1.0
        best_letter = None
        for letter, option in row["options"].items():
            label = clean_label(option)
            if label not in class_index:
                continue
            score = float(probabilities[class_index[label]])
            if score > best_score:
                best_score = score
                best_letter = letter
        if best_letter is None:
            fallback += 1
            best_letter = sorted(row["options"])[0]
        correct += int(best_letter == row["answer"])
    total = len(test_rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "train_time_sec": train_time,
        "num_classes": len(encoder.classes_),
        "fallback_rows": fallback,
    }


def build_cnn_model(model_name: str, device: str) -> tuple[Any, Any, int]:
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        params = sum(p.numel() for p in model.parameters())
        model.fc = nn.Identity()
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        params = sum(p.numel() for p in model.parameters())
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported CNN model: {model_name}")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model.to(device).eval(), transform, params


def extract_cnn_features(rows: list[dict[str, Any]], model_name: str, device: str) -> tuple[dict[str, np.ndarray], int, float]:
    import torch
    from PIL import Image

    model, transform, params = build_cnn_model(model_name, device)
    features: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            image = transform(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
            features[row["uid"]] = model(image).flatten(1).cpu().numpy()[0]
    return features, params, time.perf_counter() - started


def run_cnn_screening(
    rows_by_category: dict[str, list[dict[str, Any]]],
    device: str,
    split_seed: str,
    cnn_models: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_name in cnn_models:
        for spec in CATEGORIES:
            rows = rows_by_category[spec.task_key]
            train_rows, test_rows = split_by_class(rows, split_seed)
            if not rows or not train_rows or not test_rows:
                continue
            features, params, elapsed = extract_cnn_features(rows, model_name, device)
            base = train_eval_linear_head(features, lowshot_rows(train_rows), test_rows)
            adapted = train_eval_linear_head(features, train_rows, test_rows)
            results.append(
                {
                    "family": "cnn",
                    "model": model_name,
                    "task_key": spec.task_key,
                    "category": spec.category,
                    "params": params,
                    "rows": len(rows),
                    "train_rows": len(train_rows),
                    "test_rows": len(test_rows),
                    "base_protocol": "frozen pretrained backbone + 1-shot-per-class head",
                    "adapt_protocol": "frozen pretrained backbone + full train split head",
                    "base_acc": base["accuracy"],
                    "adapted_acc": adapted["accuracy"],
                    "gain": adapted["accuracy"] - base["accuracy"],
                    "feature_elapsed_sec": elapsed,
                    "base_details": base,
                    "adapted_details": adapted,
                }
            )
    return results


def remoteclip_param_count(weight_path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(weight_path, map_location="cpu")
    sd = state.get("state_dict") if isinstance(state, dict) and "state_dict" in state else state
    total = visual = text = other = 0
    for key, value in sd.items():
        if not hasattr(value, "numel"):
            continue
        count = int(value.numel())
        total += count
        lower = key.lower()
        if lower.startswith("visual") or ".visual" in lower or lower.startswith("module.visual"):
            visual += count
        elif (
            lower.startswith("transformer")
            or lower.startswith("token_embedding")
            or lower.startswith("ln_final")
            or lower.startswith("text_projection")
            or lower == "positional_embedding"
            or lower == "attn_mask"
        ):
            text += count
        else:
            other += count
    return {"total_params": total, "visual_params": visual, "text_params": text, "other_params": other}


def remoteclip_prompts(question: str, option: str) -> list[str]:
    return [
        f"a satellite image of {option}.",
        f"a remote sensing image of {option}.",
        f"{question} {option}.",
    ]


def run_remoteclip_one_arch(
    rows_by_category: dict[str, list[dict[str, Any]]],
    model_dir: Path,
    arch: str,
    weight_file: str,
    device: str,
    split_seed: str,
) -> list[dict[str, Any]]:
    import open_clip
    import torch
    import torch.nn.functional as F
    from PIL import Image

    weight_path = model_dir / weight_file
    param_info = remoteclip_param_count(weight_path)
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch,
        pretrained=str(weight_path),
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(arch)
    model.eval()

    def normalize(tensor: Any) -> Any:
        return F.normalize(tensor, dim=-1)

    results: list[dict[str, Any]] = []
    for spec in CATEGORIES:
        rows = rows_by_category[spec.task_key]
        train_rows, test_rows = split_by_class(rows, split_seed)
        if not rows or not train_rows or not test_rows:
            continue
        features: dict[str, np.ndarray] = {}
        zero_predictions: dict[str, str] = {}
        started = time.perf_counter()
        for row in rows:
            image = preprocess(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = normalize(model.encode_image(image))
            features[row["uid"]] = image_feature.cpu().numpy()[0]
            scores: list[tuple[float, str]] = []
            for letter, option in row["options"].items():
                prompts = remoteclip_prompts(row["question"], option)
                with torch.no_grad():
                    text_feature = normalize(model.encode_text(tokenizer(prompts).to(device))).mean(dim=0)
                    text_feature = normalize(text_feature.unsqueeze(0))
                    score = float((image_feature * text_feature).sum().cpu())
                scores.append((score, letter))
            zero_predictions[row["uid"]] = max(scores)[1]
        zero_correct = sum(int(zero_predictions[row["uid"]] == row["answer"]) for row in test_rows)
        zero_total = len(test_rows)
        adapted = train_eval_linear_head(features, train_rows, test_rows)
        zero_acc = zero_correct / zero_total if zero_total else 0.0
        results.append(
            {
                "family": "remoteclip",
                "model": f"RemoteCLIP-{arch}",
                "weight_file": weight_file,
                "task_key": spec.task_key,
                "category": spec.category,
                **param_info,
                "rows": len(rows),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "base_protocol": "RemoteCLIP image-text zero-shot option scoring",
                "adapt_protocol": "frozen RemoteCLIP image encoder + full train split linear probe",
                "base_acc": zero_acc,
                "adapted_acc": adapted["accuracy"],
                "gain": adapted["accuracy"] - zero_acc,
                "feature_elapsed_sec": time.perf_counter() - started,
                "base_details": {"correct": zero_correct, "total": zero_total, "accuracy": zero_acc},
                "adapted_details": adapted,
            }
        )
    return results


def run_remoteclip_screening(
    rows_by_category: dict[str, list[dict[str, Any]]],
    model_dir: Path,
    device: str,
    split_seed: str,
    archs: list[str],
) -> list[dict[str, Any]]:
    all_configs = {
        "RN50": ("RN50", "RemoteCLIP-RN50.pt"),
        "ViT-B-32": ("ViT-B-32", "RemoteCLIP-ViT-B-32.pt"),
    }
    results: list[dict[str, Any]] = []
    for arch_name in archs:
        if arch_name not in all_configs:
            raise ValueError(f"Unknown RemoteCLIP arch {arch_name}; available: {sorted(all_configs)}")
        arch, weight_file = all_configs[arch_name]
        results.extend(
            run_remoteclip_one_arch(
                rows_by_category,
                model_dir=model_dir,
                arch=arch,
                weight_file=weight_file,
                device=device,
                split_seed=split_seed,
            )
        )
    return results


def qwen35_status(model_root: Path) -> dict[str, Any]:
    matches = sorted(
        path.as_posix()
        for path in model_root.glob("*")
        if path.is_dir() and re.search(r"qwen.?3.?5|0.?8b", path.name, flags=re.IGNORECASE)
    )
    return {
        "family": "qwen35",
        "teacher": "Qwen3.5-27B",
        "student": "Qwen3.5-0.8B",
        "status": "not_run_missing_local_checkpoints" if not matches else "checkpoint_candidates_found",
        "local_candidates": matches,
        "note": "Qwen drift screening requires the actual Qwen3.5-27B and Qwen3.5-0.8B checkpoints.",
    }


def write_outputs(output_dir: Path, rows_by_category: dict[str, list[dict[str, Any]]], results: list[dict[str, Any]], qwen_status: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rows in rows_by_category.values():
        manifest.extend(rows)
    (output_dir / "omniearth_screening_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "screening_results.json").write_text(
        json.dumps({"results": results, "qwen35": qwen_status}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# OmniEarth Small Screening",
        "",
        "## Model Results",
        "",
        "| family | model | task | params | base | adapted | gain | train/test | protocol |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(results, key=lambda item: (item["family"], item["model"], -item["gain"])):
        params = row.get("params", row.get("total_params", 0))
        lines.append(
            "| {family} | {model} | {task_key} | {params:.2f}M | {base_acc:.4f} | "
            "{adapted_acc:.4f} | {gain:+.4f} | {train_rows}/{test_rows} | {base_protocol} -> {adapt_protocol} |".format(
                **{**row, "params": params / 1_000_000}
            )
        )
    lines.extend(
        [
            "",
            "## Qwen3.5 Status",
            "",
            f"- status: `{qwen_status['status']}`",
            f"- local candidates: `{len(qwen_status['local_candidates'])}`",
            f"- note: {qwen_status['note']}",
            "",
            "## Category Rows",
            "",
            "| task | category | rows |",
            "|---|---|---:|",
        ]
    )
    for spec in CATEGORIES:
        lines.append(f"| {spec.task_key} | {spec.category} | {len(rows_by_category[spec.task_key])} |")
    (output_dir / "screening_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small true-label screening for OmniEarth RemoteCLIP/CNN tasks.")
    parser.add_argument("--omniearth-root", type=Path, default=Path("@@SERVER_ROOT@@/datasets/public_benchmarks/OmniEarth"))
    parser.add_argument("--remoteclip-dir", type=Path, default=Path("@@WORKSPACE@@/qwen_eval/models/RemoteCLIP"))
    parser.add_argument("--qwen-model-root", type=Path, default=Path("@@WORKSPACE@@/qwen_eval/models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", default="screen_seed1")
    parser.add_argument("--skip-remoteclip", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--max-image-pixels", type=int, default=80_000_000)
    parser.add_argument("--cnn-models", default="resnet18,mobilenet_v3_small")
    parser.add_argument("--remoteclip-archs", default="RN50,ViT-B-32")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows_by_category = {
        spec.task_key: read_category_rows(args.omniearth_root, spec, args.max_image_pixels)
        for spec in CATEGORIES
    }
    results: list[dict[str, Any]] = []
    if not args.skip_cnn:
        results.extend(
            run_cnn_screening(
                rows_by_category,
                device=args.device,
                split_seed=args.split_seed,
                cnn_models=[part.strip() for part in args.cnn_models.split(",") if part.strip()],
            )
        )
    if not args.skip_remoteclip:
        results.extend(
            run_remoteclip_screening(
                rows_by_category,
                model_dir=args.remoteclip_dir,
                device=args.device,
                split_seed=args.split_seed,
                archs=[part.strip() for part in args.remoteclip_archs.split(",") if part.strip()],
            )
        )
    qwen_status = qwen35_status(args.qwen_model_root)
    write_outputs(args.output_dir, rows_by_category, results, qwen_status)
    print((args.output_dir / "screening_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
