#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pseudo-label 2_Stream_Sim_NoDrift with Qwen2.5-VL-72B-Instruct.

Outputs:
  - raw_predictions.jsonl: every model output with confidence and status
  - rejected.jsonl: samples filtered out
  - pseudo_dataset/data.json: accepted rows, original schema with ground_truth replaced
  - pseudo_dataset/images/: hardlinked/copied image folder
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_ROOT = Path("@@WORKSPACE@@")
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "2_Stream_Sim_NoDrift"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "qwen_eval" / "models" / "Qwen2.5-VL-72B-Instruct"
DEFAULT_OUT_DIR = PROJECT_ROOT / "qwen_eval" / "runs" / "qwen72b_nodrift_pseudolabel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--gpu-ids", default="0,1,2,3", help="Visible GPU ids. Default uses four H100s and leaves GPU 4 free.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", choices=["auto", "flash_attention_2", "sdpa", "eager"], default="auto")
    parser.add_argument("--max-memory-gib", type=int, default=76)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size for generate(). Raise for throughput, lower on OOM.")
    parser.add_argument("--max-batch-images", type=int, default=32, help="Upper bound on total images in one generate() batch.")
    parser.add_argument("--freeform-batch-size", type=int, default=1, help="Batch size cap for long/freeform tasks such as Task02.")
    parser.add_argument("--caption-batch-size", type=int, default=4, help="Batch size cap for caption tasks.")
    parser.add_argument("--mcq-scoring", choices=["generate", "logprob"], default="logprob", help="Use constrained option scoring for single-answer MCQ tasks.")
    parser.add_argument("--task-ids", default="", help="Comma separated task ids. Empty means all tasks in NoDrift.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--image-mode", choices=["hardlink", "copy", "none"], default="hardlink")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def import_project_helpers() -> Tuple[Any, Any, Any]:
    code_dir = PROJECT_ROOT / "code"
    sys.path.insert(0, str(code_dir))
    from generic_inference import build_prompt, get_image_paths, load_images
    return build_prompt, get_image_paths, load_images


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_map(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["uid"]] = row
    return rows


def norm_path(p: str, base: Path) -> str:
    if not p:
        return ""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((base / p.lstrip("./")).resolve())


def adapt_item(item: Dict[str, Any], dataset_dir: Path) -> Dict[str, Any]:
    task_id = int(item.get("task_id", 0))
    uid = str(item.get("uid", ""))
    options = item.get("options", {})
    if task_id == 2:
        prompts = item.get("prompts", [])
        question = prompts[0] if prompts else item.get("question", "")
        pre_img = norm_path(item.get("pre_image_path", ""), dataset_dir)
        post_img = norm_path(item.get("post_image_path", ""), dataset_dir)
        return {
            "uid": uid,
            "task_id": task_id,
            "image_path": post_img,
            "gt": item.get("ground_truth", ""),
            "meta": {
                "question": question,
                "pre_image_path": pre_img,
                "post_image_path": post_img,
                "options": {},
                "type": item.get("type", ""),
                "type_id": item.get("type_id"),
            },
        }
    question = item.get("question", "")
    sample: Dict[str, Any] = {
        "uid": uid,
        "task_id": task_id,
        "image_path": norm_path(item.get("image_path", ""), dataset_dir),
        "gt": item.get("ground_truth", ""),
        "meta": {
            "question": question,
            "options": options if isinstance(options, dict) else {},
            "type": item.get("type", ""),
            "type_id": item.get("type_id"),
        },
    }
    if task_id == 1 and isinstance(options, dict):
        first = next(iter(options.values()), "")
        if isinstance(first, str) and Path(first).suffix.lower() in IMG_EXTS:
            full_options = {k: norm_path(v, dataset_dir) for k, v in options.items()}
            sample["image_path"] = next(iter(full_options.values()), "")
            sample["meta"]["options"] = full_options
            sample["meta"]["type"] = "image_options"
    return sample


def task_kind(sample: Dict[str, Any]) -> str:
    task_id = int(sample.get("task_id", 0))
    type_id = sample.get("meta", {}).get("type_id")
    sample_type = sample.get("meta", {}).get("type", "")
    if task_id == 1 and (type_id == 3 or sample_type == "multi_mcq_opt_text"):
        return "multi_mcq"
    if task_id == 1 and (type_id == 4 or sample_type == "yesno"):
        return "yes_no"
    if task_id == 4:
        return "yes_no"
    if task_id == 10:
        return "caption"
    if task_id in {2, 11}:
        return "freeform"
    return "mcq"


def decode_config(kind: str) -> Dict[str, Any]:
    base = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    if kind == "caption":
        return {**base, "num_beams": 3, "max_new_tokens": 80}
    if kind == "freeform":
        return {**base, "num_beams": 1, "max_new_tokens": 300}
    if kind == "multi_mcq":
        return {**base, "num_beams": 1, "max_new_tokens": 20}
    return {**base, "num_beams": 1, "max_new_tokens": 10}


def batch_limit_for_kind(kind: str, args: argparse.Namespace) -> int:
    if kind == "freeform":
        return max(1, min(args.batch_size, args.freeform_batch_size))
    if kind == "caption":
        return max(1, min(args.batch_size, args.caption_batch_size))
    return max(1, args.batch_size)


def clean_answer(raw: str) -> str:
    if "<|im_start|>assistant" in raw:
        raw = raw.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in raw:
        raw = raw.split("<|im_end|>")[0]
    if "assistant\n" in raw:
        raw = raw.split("assistant\n")[-1]
    return raw.strip()


def normalize_prediction(answer: str, sample: Dict[str, Any]) -> str:
    kind = task_kind(sample)
    first = answer.splitlines()[0].strip() if answer else ""
    if kind == "multi_mcq":
        allowed = set(allowed_labels(sample))
        letters = []
        for ch in answer.upper():
            if ch in allowed and ch not in letters:
                letters.append(ch)
        return ",".join(letters)
    if kind == "mcq":
        allowed = allowed_labels(sample)
        for ch in first:
            ch = ch.upper()
            if ch in allowed:
                return ch
        m = re.search(r"\b([A-G])\b", answer.upper())
        return m.group(1) if m and m.group(1) in allowed else first[:1].upper()
    if kind == "yes_no":
        low = answer.lower()
        if re.search(r"\byes\b", low):
            return "Yes"
        if re.search(r"\bno\b", low):
            return "No"
        return first.rstrip(".,!?;:")
    return answer.strip()


def allowed_labels(sample: Dict[str, Any]) -> List[str]:
    kind = task_kind(sample)
    if kind == "yes_no":
        return ["Yes", "No"]
    if kind in {"mcq", "multi_mcq"}:
        opts = sample.get("meta", {}).get("options", {})
        if isinstance(opts, dict) and opts:
            return sorted(str(k).upper() for k in opts.keys())
        return list("ABCDE")
    return []


def validate_prediction(pred: str, sample: Dict[str, Any], confidence: Optional[float], min_confidence: float) -> Tuple[bool, str]:
    if not pred:
        return False, "empty_prediction"
    labels = allowed_labels(sample)
    if task_kind(sample) == "multi_mcq":
        pred_labels = [x for x in re.split(r"[\s,;]+", str(pred).upper()) if x]
        if not pred_labels:
            return False, "empty_prediction"
        invalid = [x for x in pred_labels if x not in labels]
        if invalid:
            return False, f"invalid_label_not_in_{','.join(labels)}"
    elif labels and pred not in labels:
        return False, f"invalid_label_not_in_{','.join(labels)}"
    if task_kind(sample) in {"caption", "freeform"} and len(pred.strip()) < 3:
        return False, "freeform_too_short"
    if confidence is not None and confidence < min_confidence:
        return False, f"confidence_below_{min_confidence}"
    return True, "accepted"


def pseudo_value_for_dataset(pred: Any, sample: Dict[str, Any]) -> Any:
    if task_kind(sample) != "multi_mcq":
        return pred
    if isinstance(pred, list):
        return pred
    labels = [x for x in re.split(r"[\s,;]+", str(pred).upper()) if x]
    allowed = set(allowed_labels(sample))
    return [x for x in labels if x in allowed]


def referenced_images(sample: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    if sample.get("image_path"):
        paths.append(sample["image_path"])
    opts = sample.get("meta", {}).get("options", {})
    if isinstance(opts, dict):
        for v in opts.values():
            if isinstance(v, str) and Path(v).suffix.lower() in IMG_EXTS:
                paths.append(v)
    return sorted(set(paths))


def choose_dtype(dtype_arg: str, torch: Any) -> Any:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def choose_attention(attn_arg: str) -> str:
    if attn_arg != "auto":
        return attn_arg
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


class Qwen72PseudoLabeler:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.args = args
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        dtype = choose_dtype(args.dtype, torch)
        attn = choose_attention(args.attn_implementation)
        gpu_count = len([x for x in args.gpu_ids.split(",") if x.strip()])
        max_memory = {i: f"{args.max_memory_gib}GiB" for i in range(gpu_count)}
        print(f"[load] model={args.model_path}")
        print(f"[load] dtype={dtype}, attn={attn}, device_map=auto, max_memory={max_memory}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            attn_implementation=attn,
            device_map="auto",
            max_memory=max_memory,
        )
        self.model.eval()
        self.input_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.build_prompt, self.get_image_paths, self.load_images = import_project_helpers()

    def predict(self, sample: Dict[str, Any]) -> Tuple[str, str, Optional[float]]:
        return self.predict_batch([sample])[0]

    def predict_batch(self, samples: List[Dict[str, Any]]) -> List[Tuple[str, str, Optional[float]]]:
        if not samples:
            return []
        kinds = {task_kind(s) for s in samples}
        if len(kinds) != 1:
            raise ValueError(f"predict_batch requires homogeneous task kinds, got {sorted(kinds)}")
        kind = next(iter(kinds))
        if kind == "mcq" and self.args.mcq_scoring == "logprob":
            return self.score_mcq_batch(samples)

        texts = []
        all_images = []
        image_counts = []
        for sample in samples:
            prompt = self.build_prompt(
                sample["task_id"],
                sample["meta"].get("question", ""),
                sample["meta"].get("options", {}),
                sample["meta"].get("type", ""),
                type_id=sample["meta"].get("type_id"),
            )
            images = self.load_images(self.get_image_paths(sample))
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}],
            }]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            all_images.extend(images)
            image_counts.append(len(images))

        inputs = self.processor(text=texts, images=all_images, padding=True, return_tensors="pt").to(self.input_device)
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **decode_config(kind),
                return_dict_in_generate=True,
                output_scores=True,
            )
        seq = generated.sequences
        full_outputs = self.processor.batch_decode(seq, skip_special_tokens=False)
        confidences = self._mean_generated_token_probs(generated, seq, input_len)
        results = []
        for sample, full_output, confidence in zip(samples, full_outputs, confidences):
            raw = clean_answer(full_output)
            answer = normalize_prediction(raw, sample)
            results.append((answer, raw, confidence))
        del inputs, generated, seq
        self.torch.cuda.empty_cache()
        return results

    def score_mcq_batch(self, samples: List[Dict[str, Any]]) -> List[Tuple[str, str, Optional[float]]]:
        texts = []
        all_images = []
        labels_by_sample = []
        for sample in samples:
            labels = allowed_labels(sample)
            if not labels:
                raise ValueError(f"No MCQ labels for {sample.get('uid')}")
            labels_by_sample.append(labels)
            prompt = self.build_prompt(
                sample["task_id"],
                sample["meta"].get("question", ""),
                sample["meta"].get("options", {}),
                sample["meta"].get("type", ""),
                type_id=sample["meta"].get("type_id"),
            )
            images = self.load_images(self.get_image_paths(sample))
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}],
            }]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            all_images.extend(images)

        inputs = self.processor(text=texts, images=all_images, padding=True, return_tensors="pt").to(self.input_device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        next_logits = outputs.logits[:, -1, :].detach().float()
        results = []
        for row_idx, labels in enumerate(labels_by_sample):
            label_scores = {}
            for label in labels:
                token_ids = self._label_token_ids(label)
                if not token_ids:
                    continue
                label_scores[label] = max(float(next_logits[row_idx, tid].item()) for tid in token_ids)
            if not label_scores:
                raise ValueError(f"No token ids for labels={labels}")
            pred = max(label_scores, key=label_scores.get)
            ordered = [label_scores[label] for label in labels if label in label_scores]
            probs = self.torch.softmax(self.torch.tensor(ordered), dim=0).tolist()
            conf = float(max(probs)) if probs else None
            raw = json.dumps({"method": "logprob", "scores": label_scores}, ensure_ascii=False)
            results.append((pred, raw, conf))
        del inputs, outputs, next_logits
        self.torch.cuda.empty_cache()
        return results

    def _label_token_ids(self, label: str) -> List[int]:
        tokenizer = self.processor.tokenizer
        ids = []
        for text in (label, f" {label}", f"\n{label}"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.append(encoded[-1])
        return sorted(set(ids))

    def predict_batch_resilient(self, samples: List[Dict[str, Any]]) -> List[Tuple[str, str, Optional[float]]]:
        try:
            return self.predict_batch(samples)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(samples) == 1:
                raise
            mid = len(samples) // 2
            return self.predict_batch_resilient(samples[:mid]) + self.predict_batch_resilient(samples[mid:])

    def predict_legacy_single(self, sample: Dict[str, Any]) -> Tuple[str, str, Optional[float]]:
        prompt = self.build_prompt(
            sample["task_id"],
            sample["meta"].get("question", ""),
            sample["meta"].get("options", {}),
            sample["meta"].get("type", ""),
            type_id=sample["meta"].get("type_id"),
        )
        images = self.load_images(self.get_image_paths(sample))
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, padding=True, return_tensors="pt").to(self.input_device)
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **decode_config(task_kind(sample)),
                return_dict_in_generate=True,
                output_scores=True,
            )
        seq = generated.sequences
        full_output = self.processor.batch_decode(seq, skip_special_tokens=False)[0]
        raw = clean_answer(full_output)
        answer = normalize_prediction(raw, sample)
        confidence = self._mean_generated_token_prob(generated, seq, input_len)
        del inputs, generated, seq
        self.torch.cuda.empty_cache()
        return answer, raw, confidence

    def _mean_generated_token_probs(self, generated: Any, seq: Any, input_len: int) -> List[Optional[float]]:
        scores = getattr(generated, "scores", None)
        batch = int(seq.shape[0])
        if not scores:
            return [None] * batch
        try:
            transition = self.model.compute_transition_scores(seq, scores, normalize_logits=True)
            out: List[Optional[float]] = []
            for row in transition.detach().float().cpu().tolist():
                token_logps = [x for x in row if math.isfinite(x)]
                if token_logps:
                    out.append(float(math.exp(sum(token_logps) / len(token_logps))))
                else:
                    out.append(None)
            return out
        except Exception:
            return [None] * batch

    def _mean_generated_token_prob(self, generated: Any, seq: Any, input_len: int) -> Optional[float]:
        scores = getattr(generated, "scores", None)
        if not scores:
            return None
        try:
            transition = self.model.compute_transition_scores(seq, scores, normalize_logits=True)
            token_logps = transition[0].detach().float().cpu().tolist()
            token_logps = [x for x in token_logps if math.isfinite(x)]
            if not token_logps:
                return None
            return float(math.exp(sum(token_logps) / len(token_logps)))
        except Exception:
            return None


def materialize_images(dataset_dir: Path, pseudo_dir: Path, mode: str) -> None:
    if mode == "none":
        return
    src = dataset_dir / "images"
    dst = pseudo_dir / "images"
    if dst.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        target = dst / p.name
        if p.is_dir():
            continue
        try:
            if mode == "hardlink":
                os.link(p, target)
            else:
                shutil.copy2(p, target)
        except OSError:
            shutil.copy2(p, target)


def flush_outputs(raw_items: List[Dict[str, Any]], rows_by_uid: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    accepted = []
    rejected = []
    for item in raw_items:
        uid = str(item.get("uid", ""))
        row = rows_by_uid.get(uid)
        if not row:
            continue
        if row["accepted"]:
            new_item = copy.deepcopy(item)
            new_item["ground_truth"] = row["pseudo_label"]
            accepted.append(new_item)
        else:
            rejected.append(row)
    pseudo_dir = out_dir / "pseudo_dataset"
    write_json(pseudo_dir / "data.json", accepted)
    write_json(out_dir / "rejected.json", rejected)
    summary = {
        "total_seen": len(rows_by_uid),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "pseudo_dataset": str(pseudo_dir),
    }
    write_json(out_dir / "summary.json", summary)


def main() -> int:
    args = parse_args()
    configure_environment(args)

    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    data_path = dataset_dir / "data.json"
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_items = load_json(data_path)
    task_filter = {int(x) for x in args.task_ids.split(",") if x.strip()} if args.task_ids else set()
    if task_filter:
        raw_items = [x for x in raw_items if int(x.get("task_id", 0)) in task_filter]
    if args.max_samples:
        raw_items = raw_items[: args.max_samples]

    raw_pred_path = out_dir / "raw_predictions.jsonl"
    rows_by_uid = read_jsonl_map(raw_pred_path) if args.resume else {}
    labeler = Qwen72PseudoLabeler(args)

    idx = 0
    while idx < len(raw_items):
        item = raw_items[idx]
        idx += 1
        uid = str(item.get("uid", ""))
        if uid in rows_by_uid:
            continue
        sample = adapt_item(item, dataset_dir)
        missing = [p for p in referenced_images(sample) if not Path(p).exists()]
        if missing:
            row = {
                "uid": uid,
                "task_id": item.get("task_id"),
                "accepted": False,
                "pseudo_label": "",
                "raw_output": "",
                "confidence": None,
                "reason": "missing_images",
                "missing_images": missing,
            }
            rows_by_uid[uid] = row
            append_jsonl(raw_pred_path, row)
        else:
            batch_items = [item]
            batch_samples = [sample]
            kind = task_kind(sample)
            batch_limit = batch_limit_for_kind(kind, args)
            batch_images = len(referenced_images(sample)) or 1
            lookahead = idx
            while len(batch_samples) < batch_limit and lookahead < len(raw_items):
                cand_item = raw_items[lookahead]
                cand_uid = str(cand_item.get("uid", ""))
                if cand_uid in rows_by_uid:
                    lookahead += 1
                    continue
                cand_sample = adapt_item(cand_item, dataset_dir)
                cand_missing = [p for p in referenced_images(cand_sample) if not Path(p).exists()]
                if cand_missing or task_kind(cand_sample) != kind:
                    break
                cand_images = len(referenced_images(cand_sample)) or 1
                if batch_images + cand_images > args.max_batch_images:
                    break
                batch_items.append(cand_item)
                batch_samples.append(cand_sample)
                batch_images += cand_images
                lookahead += 1
            idx = lookahead
            try:
                preds = labeler.predict_batch_resilient(batch_samples)
                for cand_item, cand_sample, (pred, raw, conf) in zip(batch_items, batch_samples, preds):
                    cand_uid = str(cand_item.get("uid", ""))
                    ok, reason = validate_prediction(pred, cand_sample, conf, args.min_confidence)
                    row = {
                        "uid": cand_uid,
                        "task_id": cand_item.get("task_id"),
                        "accepted": ok,
                        "pseudo_label": pseudo_value_for_dataset(pred, cand_sample),
                        "raw_output": raw,
                        "confidence": conf,
                        "reason": reason,
                        "allowed_labels": allowed_labels(cand_sample),
                    }
                    rows_by_uid[cand_uid] = row
                    append_jsonl(raw_pred_path, row)
            except Exception as exc:
                for cand_item in batch_items:
                    cand_uid = str(cand_item.get("uid", ""))
                    row = {
                        "uid": cand_uid,
                        "task_id": cand_item.get("task_id"),
                        "accepted": False,
                        "pseudo_label": "",
                        "raw_output": "",
                        "confidence": None,
                        "reason": f"inference_error:{type(exc).__name__}:{exc}",
                    }
                    rows_by_uid[cand_uid] = row
                    append_jsonl(raw_pred_path, row)
        if idx % args.save_every == 0:
            flush_outputs(raw_items, rows_by_uid, out_dir)
            print(f"[progress] {idx}/{len(raw_items)} seen, accepted={sum(r['accepted'] for r in rows_by_uid.values())}")

    flush_outputs(raw_items, rows_by_uid, out_dir)
    materialize_images(dataset_dir, out_dir / "pseudo_dataset", args.image_mode)
    print(f"[done] out_dir={out_dir}")
    print(f"[done] pseudo_dataset={out_dir / 'pseudo_dataset'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
