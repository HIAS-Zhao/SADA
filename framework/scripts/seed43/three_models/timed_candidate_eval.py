#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import copy
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path("@@WORKSPACE@@")
QWEN_EVAL = PROJECT_ROOT / "qwen_eval"
DATASET_DIR = PROJECT_ROOT / "dataset" / "3_Evaluation"
TASK_IDS = "1,3,4,5,6,7,8,9"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GEOCHAT_REPO = QWEN_EVAL / "third_party" / "GeoChat-main"

sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(QWEN_EVAL / "code"))

from generic_inference import build_prompt, get_image_paths, load_images  # noqa: E402
from qwen72b_pseudolabel_nodrift import (  # noqa: E402
    adapt_item,
    allowed_labels,
    materialize_images,
    normalize_prediction,
    pseudo_value_for_dataset,
    read_jsonl_map,
    referenced_images,
    task_kind,
    validate_prediction,
)


def read_json(path: Path) -> Any:
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
    write_json(
        out_dir / "summary.json",
        {
            "total_seen": len(rows_by_uid),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "pseudo_dataset": str(pseudo_dir),
        },
    )


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


def max_tokens_for(sample: Dict[str, Any], args: Optional[argparse.Namespace] = None) -> int:
    kind = task_kind(sample)
    if args is not None:
        if kind == "caption":
            return args.caption_max_new_tokens
        if kind == "freeform":
            return args.freeform_max_new_tokens
        if kind == "multi_mcq":
            return args.multi_mcq_max_new_tokens
        if kind == "yes_no":
            return args.yes_no_max_new_tokens
        return args.mcq_max_new_tokens
    if kind == "caption":
        return 96
    if kind == "freeform":
        return 300
    if kind == "multi_mcq":
        return 24
    return 12


def clean_raw_answer(text: Any) -> str:
    raw = "" if text is None else str(text)
    for marker in ("<|im_start|>assistant", "assistant\n"):
        if marker in raw:
            raw = raw.split(marker)[-1]
    for marker in ("<|im_end|>", "</s>"):
        if marker in raw:
            raw = raw.split(marker)[0]
    return raw.strip()


def final_after_think(text: str) -> str:
    if "</think>" not in text:
        return ""
    return text.split("</think>", 1)[1].strip()


def infer_family(model_name: str, model_path: str, family: str) -> str:
    if family != "auto":
        return family
    text = f"{model_name} {model_path}".lower()
    if "minicpm" in text:
        return "minicpm"
    if "internvl" in text:
        return "internvl"
    if "geochat" in text:
        return "geochat"
    return "hf_image_text"


def comparable_mcq_scoring(family: str) -> str:
    return "generate_then_extract"


class CandidateVLM:
    def __init__(self, args: argparse.Namespace):
        import torch

        self.args = args
        self.torch = torch
        self.dtype = choose_dtype(args.dtype, torch)
        self.family = infer_family(args.model_name, args.model_path, args.family)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.load()

    def load(self) -> None:
        if self.family == "minicpm":
            self._load_minicpm()
        elif self.family == "internvl":
            self._load_internvl()
        elif self.family == "geochat":
            self._load_geochat()
        else:
            self._load_hf_image_text()

    def _load_hf_image_text(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[load] family=hf_image_text model={self.args.model_path} dtype={self.dtype}")
        if self.args.adapter_path:
            print(f"[load] adapter={self.args.adapter_path}")
        self.processor = AutoProcessor.from_pretrained(self.args.model_path, trust_remote_code=True)
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.args.model_path,
                dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.args.model_path,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        if self.args.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.args.adapter_path, is_trainable=False)
        self.model.eval()

    def _load_minicpm(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        print(f"[load] family=minicpm model={self.args.model_path} dtype={self.dtype}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        model_dir = str(Path(self.args.model_path).resolve())
        try:
            import importlib
            import types

            package_name = "minicpmv45_local_model"
            if package_name not in sys.modules:
                package = types.ModuleType(package_name)
                package.__path__ = [model_dir]  # type: ignore[attr-defined]
                sys.modules[package_name] = package
            MiniCPMV = importlib.import_module(f"{package_name}.modeling_minicpmv").MiniCPMV
            if not hasattr(MiniCPMV, "all_tied_weights_keys"):
                MiniCPMV.all_tied_weights_keys = {}
            self.model = MiniCPMV.from_pretrained(
                model_dir,
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=self.dtype,
            )
        except (ImportError, TypeError):
            kwargs = {
                "trust_remote_code": True,
                "attn_implementation": "sdpa",
                "torch_dtype": self.dtype,
            }
            self.model = AutoModel.from_pretrained(model_dir, **kwargs)
        self.model = self.model.eval().cuda()

    def _load_internvl(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        print(f"[load] family=internvl model={self.args.model_path} dtype={self.dtype}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True, use_fast=False)
        model_dir = str(Path(self.args.model_path).resolve())
        try:
            import importlib
            import types

            package_name = "internvl3_5_local_model"
            if package_name not in sys.modules:
                package = types.ModuleType(package_name)
                package.__path__ = [model_dir]  # type: ignore[attr-defined]
                sys.modules[package_name] = package
            InternVLChatModel = importlib.import_module(
                f"{package_name}.modeling_internvl_chat"
            ).InternVLChatModel
            if not hasattr(InternVLChatModel, "all_tied_weights_keys"):
                InternVLChatModel.all_tied_weights_keys = {}

            self.model = InternVLChatModel.from_pretrained(
                model_dir,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map="auto",
            )
        except (ImportError, TypeError):
            self.model = AutoModel.from_pretrained(
                model_dir,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map="auto",
            )
        self.model.eval()

    def _load_geochat(self) -> None:
        print(f"[load] family=geochat model={self.args.model_path} dtype=float16")
        sys.path.insert(0, str(GEOCHAT_REPO))
        from geochat.mm_utils import get_model_name_from_path
        from geochat.model.builder import load_pretrained_model
        from geochat.utils import disable_torch_init

        disable_torch_init()
        model_path = str(Path(self.args.model_path).resolve())
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="auto",
            device="cuda",
        )
        self.model.eval()

    def prompt_and_images(self, sample: Dict[str, Any]) -> Tuple[str, List[Any]]:
        prompt = build_prompt(
            int(sample["task_id"]),
            sample["meta"].get("question", ""),
            sample["meta"].get("options", {}),
            sample["meta"].get("type", ""),
            type_id=sample["meta"].get("type_id"),
        )
        images = load_images(get_image_paths(sample))
        return prompt, images

    def chat_template_kwargs(self) -> Dict[str, Any]:
        if self.args.enable_thinking == "true":
            return {"enable_thinking": True}
        if self.args.enable_thinking == "false":
            return {"enable_thinking": False}
        return {}

    def extraction_text(self, raw: str) -> str:
        if self.args.answer_source == "final_after_think":
            return final_after_think(raw)
        return raw

    def predict(self, sample: Dict[str, Any]) -> Tuple[str, str, Optional[float]]:
        if self.family == "minicpm":
            raw = self._predict_minicpm(sample)
        elif self.family == "internvl":
            raw = self._predict_internvl(sample)
        elif self.family == "geochat":
            raw = self._predict_geochat(sample)
        else:
            raw = self._predict_hf_image_text(sample)
        pred = normalize_prediction(self.extraction_text(raw), sample)
        return pred, raw, None

    def predict_batch(self, samples: List[Dict[str, Any]]) -> List[Tuple[str, str, Optional[float]]]:
        if self.family != "hf_image_text" or len(samples) <= 1:
            return [self.predict(sample) for sample in samples]
        raws = self._predict_hf_image_text_batch(samples)
        return [(normalize_prediction(self.extraction_text(raw), sample), raw, None) for sample, raw in zip(samples, raws)]

    def _score_hf_labels(self, sample: Dict[str, Any]) -> Tuple[str, str, Optional[float]]:
        labels = allowed_labels(sample)
        if not labels:
            raw = self._predict_hf_image_text(sample)
            return normalize_prediction(self.extraction_text(raw), sample), raw, None

        prompt, images = self.prompt_and_images(sample)
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs(),
        )
        inputs = inputs.to("cuda")
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        next_logits = outputs.logits[:, -1, :].detach().float()[0]
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raw = self._predict_hf_image_text(sample)
            return normalize_prediction(self.extraction_text(raw), sample), raw, None

        label_scores: Dict[str, float] = {}
        for label in labels:
            token_ids = self._label_token_ids(tokenizer, label)
            if token_ids:
                label_scores[label] = max(float(next_logits[tid].item()) for tid in token_ids)
        del inputs, outputs, next_logits
        self.torch.cuda.empty_cache()
        if not label_scores:
            raw = self._predict_hf_image_text(sample)
            return normalize_prediction(self.extraction_text(raw), sample), raw, None
        pred = max(label_scores, key=label_scores.get)
        ordered = [label_scores[label] for label in labels if label in label_scores]
        probs = self.torch.softmax(self.torch.tensor(ordered), dim=0).tolist()
        conf = float(max(probs)) if probs else None
        raw = json.dumps({"method": "logprob", "scores": label_scores}, ensure_ascii=False)
        return pred, raw, conf

    @staticmethod
    def _label_token_ids(tokenizer: Any, label: str) -> List[int]:
        ids = []
        for text in (label, label.lower(), f" {label}", f"\n{label}"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.append(encoded[-1])
        return sorted(set(ids))

    def _predict_hf_image_text(self, sample: Dict[str, Any]) -> str:
        prompt, images = self.prompt_and_images(sample)
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs(),
        )
        inputs = inputs.to("cuda")
        input_len = int(inputs["input_ids"].shape[-1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens_for(sample, self.args),
            )
        if generated.ndim == 2:
            generated = generated[:, input_len:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        del inputs, generated
        self.torch.cuda.empty_cache()
        return clean_raw_answer(raw)

    def _predict_hf_image_text_batch(self, samples: List[Dict[str, Any]]) -> List[str]:
        messages_batch = []
        for sample in samples:
            prompt, images = self.prompt_and_images(sample)
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            messages_batch.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            **self.chat_template_kwargs(),
        )
        inputs = inputs.to("cuda")
        input_len = int(inputs["input_ids"].shape[-1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max(max_tokens_for(sample, self.args) for sample in samples),
            )
        if generated.ndim == 2:
            generated = generated[:, input_len:]
        raws = self.processor.batch_decode(generated, skip_special_tokens=True)
        del inputs, generated
        self.torch.cuda.empty_cache()
        return [clean_raw_answer(raw) for raw in raws]

    def _predict_minicpm(self, sample: Dict[str, Any]) -> str:
        prompt, images = self.prompt_and_images(sample)
        msgs = [{"role": "user", "content": [*images, prompt]}]
        kwargs = {
            "msgs": msgs,
            "tokenizer": self.tokenizer,
            "stream": False,
        }
        for extra in (
            {"sampling": False, "max_new_tokens": max_tokens_for(sample, self.args)},
            {"max_new_tokens": max_tokens_for(sample, self.args)},
            {},
        ):
            try:
                result = self.model.chat(**kwargs, **extra)
                break
            except TypeError:
                result = None
        if result is None:
            result = self.model.chat(**kwargs)
        if not isinstance(result, str):
            try:
                result = "".join(list(result))
            except TypeError:
                result = str(result)
        self.torch.cuda.empty_cache()
        return clean_raw_answer(result)

    def _predict_internvl(self, sample: Dict[str, Any]) -> str:
        prompt, images = self.prompt_and_images(sample)
        pixel_values, num_patches_list = self._internvl_pixels(images)
        if len(images) > 1:
            prefix = "\n".join([f"Image-{i + 1}: <image>" for i in range(len(images))])
            question = f"{prefix}\n{prompt}"
        else:
            question = f"<image>\n{prompt}"
        generation_config = {
            "max_new_tokens": max_tokens_for(sample, self.args),
            "do_sample": False,
        }
        with self.torch.inference_mode():
            try:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                )
            except TypeError:
                response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        self.torch.cuda.empty_cache()
        return clean_raw_answer(response)

    def _predict_geochat(self, sample: Dict[str, Any]) -> str:
        prompt, images = self.prompt_and_images(sample)
        from geochat.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from geochat.conversation import SeparatorStyle, conv_templates
        from geochat.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

        if len(images) > 1:
            image_prefix = "\n".join(f"Image-{i + 1}: {DEFAULT_IMAGE_TOKEN}" for i in range(len(images)))
        else:
            image_prefix = DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_prefix = image_prefix.replace(
                DEFAULT_IMAGE_TOKEN,
                f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}",
            )
        question = f"{image_prefix}\n{prompt}"

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).cuda()
        image_tensor = self.processor.preprocess(
            images,
            crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504},
            return_tensors="pt",
        )["pixel_values"].half().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        input_token_len = int(input_ids.shape[1])
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_tokens_for(sample, self.args),
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        raw = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        raw = raw.strip()
        if raw.endswith(stop_str):
            raw = raw[: -len(stop_str)]
        del input_ids, image_tensor, output_ids
        self.torch.cuda.empty_cache()
        return clean_raw_answer(raw)

    def _internvl_pixels(self, images: List[Any]) -> Tuple[Any, List[int]]:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        values = []
        patch_counts = []
        for image in images:
            tiles = self._dynamic_preprocess(image, max_num=self.args.internvl_max_tiles)
            patch_counts.append(len(tiles))
            values.extend([transform(tile) for tile in tiles])
        if not values:
            values = [transform(images[0])]
            patch_counts = [1]
        pixel_values = self.torch.stack(values).to(dtype=self.dtype, device="cuda")
        return pixel_values, patch_counts

    def _dynamic_preprocess(
        self,
        image: Any,
        min_num: int = 1,
        max_num: int = 6,
        image_size: int = 448,
        use_thumbnail: bool = True,
    ) -> List[Any]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    @staticmethod
    def _find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: List[Tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> Tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif math.isclose(ratio_diff, best_ratio_diff):
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--family", default="auto", choices=["auto", "hf_image_text", "minicpm", "internvl", "geochat"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-ids", default=TASK_IDS)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--image-mode", choices=["hardlink", "copy", "none"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--internvl-max-tiles", type=int, default=6)
    parser.add_argument("--enable-thinking", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--answer-source", choices=["raw", "final_after_think"], default="raw")
    parser.add_argument("--mcq-max-new-tokens", type=int, default=12)
    parser.add_argument("--multi-mcq-max-new-tokens", type=int, default=24)
    parser.add_argument("--yes-no-max-new-tokens", type=int, default=12)
    parser.add_argument("--caption-max-new-tokens", type=int, default=96)
    parser.add_argument("--freeform-max-new-tokens", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(args.seed)

    try:
        import numpy as np
        import torch

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    except Exception:
        pass

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_items = read_json(dataset_dir / "data.json")
    task_filter = {int(x) for x in args.task_ids.split(",") if x.strip()} if args.task_ids else set()
    if task_filter:
        raw_items = [x for x in raw_items if int(x.get("task_id", 0)) in task_filter]
    if args.max_samples:
        raw_items = raw_items[: args.max_samples]

    write_json(
        out_dir / "run_config.json",
        {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "family": infer_family(args.model_name, args.model_path, args.family),
            "dataset_dir": str(dataset_dir),
            "task_ids": sorted(task_filter),
            "max_samples": args.max_samples,
            "prompt_source": "@@WORKSPACE@@/code/generic_inference.py:build_prompt",
            "sample_adapter_source": "@@WORKSPACE@@/qwen_eval/code/qwen72b_pseudolabel_nodrift.py:adapt_item",
            "image_path_source": "@@WORKSPACE@@/code/generic_inference.py:get_image_paths",
            "mcq_scoring": comparable_mcq_scoring(infer_family(args.model_name, args.model_path, args.family)),
            "enable_thinking": args.enable_thinking,
            "answer_source": args.answer_source,
            "max_new_tokens": {
                "mcq": args.mcq_max_new_tokens,
                "multi_mcq": args.multi_mcq_max_new_tokens,
                "yes_no": args.yes_no_max_new_tokens,
                "caption": args.caption_max_new_tokens,
                "freeform": args.freeform_max_new_tokens,
            },
            "comparison_bucket": "main_fair"
            if infer_family(args.model_name, args.model_path, args.family) == "hf_image_text"
            else "adapter_eval",
            "batch_size": args.batch_size,
        },
    )

    raw_pred_path = out_dir / "raw_predictions.jsonl"
    rows_by_uid = read_jsonl_map(raw_pred_path) if args.resume else {}
    labeler = CandidateVLM(args)

    batch_size = max(1, int(args.batch_size))

    def persist_row(idx: int, uid: str, row: Dict[str, Any]) -> None:
        rows_by_uid[uid] = row
        append_jsonl(raw_pred_path, row)
        if idx % args.save_every == 0:
            flush_outputs(raw_items, rows_by_uid, out_dir)
            accepted = sum(1 for r in rows_by_uid.values() if r.get("accepted"))
            print(f"[progress] {idx}/{len(raw_items)} seen, accepted={accepted}", flush=True)

    def run_batch(entries: List[Tuple[int, Dict[str, Any], Dict[str, Any]]]) -> None:
        if not entries:
            return
        try:
            labeler.torch.cuda.synchronize()
            started = time.perf_counter()
            results = labeler.predict_batch([sample for _, _, sample in entries])
            labeler.torch.cuda.synchronize()
            prediction_elapsed_s = (time.perf_counter() - started) / len(entries)
        except Exception as exc:
            if len(entries) == 1:
                idx, item, _sample = entries[0]
                uid = str(item.get("uid", ""))
                row = {
                    "uid": uid,
                    "task_id": item.get("task_id"),
                    "accepted": False,
                    "pseudo_label": "",
                    "raw_output": "",
                    "confidence": None,
                    "reason": f"inference_error:{type(exc).__name__}:{exc}",
                }
                persist_row(idx, uid, row)
                return
            split = max(1, len(entries) // 2)
            print(
                f"[batch-fallback] size={len(entries)} split={split}+{len(entries) - split} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            try:
                labeler.torch.cuda.empty_cache()
            except Exception:
                pass
            run_batch(entries[:split])
            run_batch(entries[split:])
            return

        for (idx, item, sample), (pred, raw, conf) in zip(entries, results):
            uid = str(item.get("uid", ""))
            ok, reason = validate_prediction(pred, sample, conf, args.min_confidence)
            row = {
                "uid": uid,
                "task_id": item.get("task_id"),
                "accepted": ok,
                "pseudo_label": pseudo_value_for_dataset(pred, sample),
                "raw_output": raw,
                "confidence": conf,
                "reason": reason,
                "allowed_labels": allowed_labels(sample),
                "prediction_elapsed_s": prediction_elapsed_s,
            }
            persist_row(idx, uid, row)

    pending: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for idx, item in enumerate(raw_items, start=1):
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
            persist_row(idx, uid, row)
        else:
            pending.append((idx, item, sample))
            if len(pending) >= batch_size:
                run_batch(pending)
                pending = []

    run_batch(pending)

    flush_outputs(raw_items, rows_by_uid, out_dir)
    materialize_images(dataset_dir, out_dir / "pseudo_dataset", args.image_mode)
    print(f"[done] out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
