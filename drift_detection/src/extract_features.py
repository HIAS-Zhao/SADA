# -*- coding: utf-8 -*-
"""
Feature extraction from pre-split dataset.

This script extracts quad-stream features (A, B, C, D) from the pre-split dataset:
- 0_Reference_Fit: For PCA fitting
- 1_Threshold_Cal: For threshold calibration
- 2_Stream_Sim_NoDrift: For FPR evaluation
- 2_Stream_Sim_Drift: For Recall evaluation

Usage:
    python src/extract_features.py --dataset @@WORKSPACE@@/dataset --output features_pooled
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

# Enable expandable memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import (
    MODEL_DIR,
    DATASET_DIR,
    DEVICE,
    DATA_SPLIT_CONFIG,
)

# Import utilities
from src.utils.split_data_loader import SplitDataset, SplitSample
from src.utils.qwen_hooks import create_extractor, ExtractedFeatures
from src.utils.task_prompts import get_task_prompt

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_layer_tag(layer_offset: int) -> str:
    if layer_offset < 0:
        return f"m{abs(layer_offset)}"
    return f"p{layer_offset}"


# ============================================================================
# Model Loading
# ============================================================================

def load_qwen_model(model_dir: Path, device: str = "cuda", multi_gpu: bool = False):
    """Load Qwen2.5-VL model in inference mode with multi-GPU support."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    import torch

    logger.info(f"Loading Qwen2.5-VL model from {model_dir}...")

    # Check available GPUs
    if device == "cuda" and multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Found {torch.cuda.device_count()} GPUs, using model parallelism")
        # Use device_map="auto" for automatic multi-GPU distribution
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatically distribute across GPUs
        )
    else:
        logger.info(f"Using single device: {device}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    logger.info(f"✓ Model loaded")
    return model


def load_processor(model_dir: Path):
    """Load Qwen2.5-VL processor."""
    from transformers import AutoProcessor

    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    logger.info("✓ Processor loaded")
    return processor


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features_from_sample(
    sample: SplitSample,
    model,
    processor,
    extractor,
    device: str = "cuda"
) -> ExtractedFeatures:
    """
    Extract features from a single sample with native multi-image support.

    Args:
        sample: SplitSample object
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        extractor: FeatureExtractor instance
        device: Device to run on

    Returns:
        ExtractedFeatures object
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    # Get task-level prompt
    task_prompt = get_task_prompt(
        task_id=sample.task_id,
        question=sample.question,
        options=sample.options,
        type_id=sample.type_id
    )

    # Load images
    sample_images = []
    for img_path in sample.image_paths:
        img = Image.open(img_path).convert('RGB')
        sample_images.append(img)

    # Construct messages in Qwen2.5-VL format
    content = []
    for img in sample_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": task_prompt})

    messages = [{"role": "user", "content": content}]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Process inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Extract features during the single model forward owned by the extractor.
    extractor.clear_cache()

    features = extractor.extract_features(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        pixel_values=inputs.get('pixel_values'),
        image_grid_thw=inputs.get('image_grid_thw')
    )

    return features


def extract_split_features(
    split_name: str,
    dataset_dir: Path,
    output_dir: Path,
    model,
    processor,
    extractor,
    device: str = "cuda",
    start_idx: int = None,
    end_idx: int = None,
    decoder_layer_offset: int = -1,
):
    """
    Extract features from a data split.

    Args:
        split_name: Split name (e.g., '0_Reference_Fit')
        dataset_dir: Root dataset directory
        output_dir: Output directory for features
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        extractor: FeatureExtractor instance
        device: Device to run on
        start_idx: Start sample index (optional, for splitting work)
        end_idx: End sample index (optional, for splitting work)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting features from: {split_name}")
    if start_idx is not None or end_idx is not None:
        logger.info(f"Sample range: {start_idx or 0} to {end_idx or 'end'}")
    logger.info(f"{'='*60}")

    # Load dataset
    split_dir = dataset_dir / split_name
    dataset = SplitDataset(split_dir)

    # Apply sample range if specified
    if start_idx is not None or end_idx is not None:
        start = start_idx or 0
        end = end_idx or len(dataset)
        dataset.samples = dataset.samples[start:end]
        logger.info(f"Processing samples {start} to {end-1} ({len(dataset.samples)} samples)")
    else:
        logger.info(f"Loaded {len(dataset)} samples")

    # Create output directory
    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract features for each sample
    all_features = {
        'A': [],
        'B': [],
        'C': [],
        'D': [],
        'E_last': [],
        'E_text': [],
        'predicted_labels': [],
        'uids': [],
        'task_ids': [],
        'ground_truths': [],
    }

    for sample in tqdm(dataset, desc=f"Extracting {split_name}"):
        try:
            features = extract_features_from_sample(
                sample=sample,
                model=model,
                processor=processor,
                extractor=extractor,
                device=device
            )

            # Collect features
            all_features['A'].append(features.A)
            all_features['B'].append(features.B)
            all_features['C'].append(features.C)
            all_features['D'].append(features.D)
            all_features['E_last'].append(features.E_last)
            all_features['E_text'].append(features.E_text)
            all_features['predicted_labels'].append(features.predicted_labels)
            all_features['uids'].append(sample.uid)
            all_features['task_ids'].append(sample.task_id)
            all_features['ground_truths'].append(sample.ground_truth)

        except Exception as e:
            logger.error(f"Error extracting features from {sample.uid}: {e}")
            continue

    # Concatenate all features
    features_tensor = {
        'A': torch.cat(all_features['A'], dim=0),
        'B': torch.cat(all_features['B'], dim=0),
        'C': torch.cat(all_features['C'], dim=0),
        'D': torch.cat(all_features['D'], dim=0),
        'E_last': torch.cat(all_features['E_last'], dim=0),
        'E_text': torch.cat(all_features['E_text'], dim=0),
        'predicted_labels': torch.cat(all_features['predicted_labels'], dim=0),
    }

    # Save features
    output_file = split_output_dir / "features.pt"
    torch.save({
        'features': features_tensor,
        'metadata': {
            'uids': all_features['uids'],
            'task_ids': all_features['task_ids'],
            'ground_truths': all_features['ground_truths'],
            'split_name': split_name,
            'n_samples': len(all_features['uids']),
            'decoder_layer_offset': decoder_layer_offset,
            'decoder_layer_index': getattr(extractor, 'decoder_layer_index', None),
        }
    }, output_file)

    logger.info(f"✓ Saved features to: {output_file}")
    logger.info(f"  - Stream A: {features_tensor['A'].shape}")
    logger.info(f"  - Stream B: {features_tensor['B'].shape}")
    logger.info(f"  - Stream C: {features_tensor['C'].shape}")
    logger.info(f"  - Stream D: {features_tensor['D'].shape}")
    logger.info(f"  - Stream E_last: {features_tensor['E_last'].shape}")
    logger.info(f"  - Stream E_text: {features_tensor['E_text'].shape}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract features from pre-split dataset")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR), help="Dataset root directory")
    parser.add_argument("--output", type=str, default="features_pooled", help="Output directory")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cuda/cpu)")
    parser.add_argument("--splits", type=str, nargs='+', default=None, help="Specific splits to process")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs if available")
    parser.add_argument("--start-idx", type=int, default=None, help="Start sample index (for splitting work)")
    parser.add_argument("--end-idx", type=int, default=None, help="End sample index (for splitting work)")
    parser.add_argument(
        "--decoder-layer-offset",
        type=int,
        default=-1,
        help="Relative decoder layer index for Stream B/D extraction, -1 means last layer, -3 means third from last",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_name = args.output
    if args.decoder_layer_offset != -1 and output_name == "features_pooled":
        output_name = f"features_pooled_layer_{format_layer_tag(args.decoder_layer_offset)}"

    output_dir = Path(output_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which splits to process
    if args.splits:
        splits_to_process = args.splits
    else:
        splits_to_process = [config['name'] for config in DATA_SPLIT_CONFIG.values()]

    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Splits to process: {splits_to_process}")
    logger.info(f"Decoder layer offset: {args.decoder_layer_offset}")

    # Load model and processor
    model = load_qwen_model(MODEL_DIR, device=args.device, multi_gpu=args.multi_gpu)
    processor = load_processor(MODEL_DIR)

    # Create feature extractor
    extractor = create_extractor(
        model,
        device=args.device,
        decoder_layer_offset=args.decoder_layer_offset,
    )

    # Extract features from each split
    for split_name in splits_to_process:
        try:
            extract_split_features(
                split_name=split_name,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                model=model,
                processor=processor,
                extractor=extractor,
                device=args.device,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
                decoder_layer_offset=args.decoder_layer_offset,
            )
        except Exception as e:
            logger.error(f"Failed to process split {split_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    extractor.remove_hooks()

    logger.info(f"\n{'='*60}")
    logger.info("Feature extraction completed!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
