# -*- coding: utf-8 -*-
"""
Sequence-level feature extraction entrypoint.

This is a NEW script and does not modify existing extraction flow.
It stores per-sample 2D sequence features without global sequence pooling.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

# Memory settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_DIR, DATA_SPLIT_CONFIG, DEVICE, MODEL_DIR
from src.utils.qwen_sequence_hooks import SequenceFeatures, create_sequence_extractor
from src.utils.split_data_loader import SplitDataset, SplitSample
from src.utils.task_prompts import get_task_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_qwen_model(model_dir: Path, device: str = "cuda"):
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_processor(model_dir: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_dir))


def extract_single_sample(
    sample: SplitSample,
    model,
    processor,
    extractor,
    device: str,
) -> SequenceFeatures:
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    prompt = get_task_prompt(
        task_id=sample.task_id,
        question=sample.question,
        options=sample.options,
        type_id=sample.type_id,
    )

    images = [Image.open(p).convert("RGB") for p in sample.image_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    features = extractor.extract_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
    )

    # In this script we intentionally process one sample per forward.
    if len(features) != 1:
        raise RuntimeError(f"Expected exactly one sample feature package, got {len(features)}")

    return features[0]


def extract_split(
    split_name: str,
    dataset_dir: Path,
    output_dir: Path,
    model,
    processor,
    extractor,
    device: str,
):
    split_dir = dataset_dir / split_name
    dataset = SplitDataset(split_dir)

    split_output = output_dir / split_name
    split_output.mkdir(parents=True, exist_ok=True)

    rows_A: List[torch.Tensor] = []
    rows_B: List[torch.Tensor] = []
    rows_C: List[torch.Tensor] = []
    rows_D: List[torch.Tensor] = []
    rows_pred: List[torch.Tensor] = []

    metadata: Dict[str, List] = {
        "uids": [],
        "task_ids": [],
        "ground_truths": [],
        "A_lengths": [],
        "B_lengths": [],
        "C_lengths": [],
        "D_lengths": [],
    }

    for sample in tqdm(dataset, desc=f"Extracting {split_name}"):
        try:
            feat = extract_single_sample(sample, model, processor, extractor, device)

            rows_A.append(feat.A.cpu())
            rows_B.append(feat.B.cpu())
            rows_C.append(feat.C.cpu())
            rows_D.append(feat.D.cpu())
            rows_pred.append(feat.predicted_labels.cpu())

            metadata["uids"].append(sample.uid)
            metadata["task_ids"].append(sample.task_id)
            metadata["ground_truths"].append(sample.ground_truth)
            metadata["A_lengths"].append(int(feat.A.size(0)))
            metadata["B_lengths"].append(int(feat.B.size(0)))
            metadata["C_lengths"].append(int(feat.C.size(0)))
            metadata["D_lengths"].append(int(feat.D.size(0)))

        except Exception as e:
            logger.error("Error on sample %s: %s", sample.uid, e)

    out_file = split_output / "features_sequence.pt"
    torch.save(
        {
            "features": {
                "A": rows_A,
                "B": rows_B,
                "C": rows_C,
                "D": rows_D,
                "predicted_labels": rows_pred,
            },
            "metadata": {
                **metadata,
                "split_name": split_name,
                "n_samples": len(metadata["uids"]),
            },
        },
        out_file,
    )

    logger.info("Saved sequence features: %s", out_file)
    logger.info("Samples: %d", len(metadata["uids"]))


def main():
    parser = argparse.ArgumentParser(description="Extract sequence-level features from pre-split dataset")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    parser.add_argument("--output", type=str, default="features_sequence")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--splits", type=str, nargs="+", default=None)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.splits:
        splits = args.splits
    else:
        splits = [cfg["name"] for cfg in DATA_SPLIT_CONFIG.values()]

    logger.info("Dataset: %s", dataset_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Window size: %d", args.window_size)
    logger.info("Splits: %s", splits)

    model = load_qwen_model(MODEL_DIR, device=args.device)
    processor = load_processor(MODEL_DIR)
    extractor = create_sequence_extractor(model, window_size=args.window_size, device=args.device)

    for split_name in splits:
        extract_split(
            split_name=split_name,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            model=model,
            processor=processor,
            extractor=extractor,
            device=args.device,
        )

    extractor.remove_hooks()
    logger.info("Sequence feature extraction finished")


if __name__ == "__main__":
    main()
