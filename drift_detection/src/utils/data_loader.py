# -*- coding: utf-8 -*-
"""
Data Loader for Drift Detection Experiment.

This module provides a unified DriftDataset class that can load data from:
1. Original task JSON files (via baseline data_adapters)
2. Preprocessed concatenated images from the cache directory

The DriftDataset provides a consistent interface regardless of the data source.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import glob
import logging
import json

# Setup path and get baseline modules using dynamic loading
from .path_setup import get_baseline_module, BASELINE_ROOT, BASELINE_CODE_DIR

# Dynamically load baseline modules
_data_adapters = get_baseline_module("data_adapters")
_io_utils = get_baseline_module("io_utils")

# Extract functions we need
load_and_adapt = _data_adapters.load_and_adapt
load_json = _io_utils.load_json

# Import config (use relative import from project root perspective)
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
from config import (
    BASELINE_ROOT as CFG_BASELINE_ROOT,
    BASELINE_DATASET_DIR,
    BASELINE_JSON_TEST_DIR,
    CONCAT_CACHE_DIR,
    ID_TASKS,
    OOD_TASKS,
    ALL_TASKS,
    IMAGE_EXTENSIONS,
)

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataSource(Enum):
    """Enum to track the source of data samples."""
    ORIGINAL_JSON = "original_json"      # From task JSON files
    CONCAT_CACHE = "concat_cache"        # From preprocessed cache


@dataclass
class DriftSample:
    """
    Unified sample structure for drift detection.

    Attributes:
        uid: Unique identifier for the sample
        task_id: Task ID (1-11)
        image_path: Path to the image file (single image or concatenated)
        image_paths: List of image paths (for multi-image tasks)
        meta: Task-specific metadata (question, options, etc.)
        gt: Ground truth answer
        source: Data source (original JSON or concat cache)
    """
    uid: str
    task_id: int
    image_path: str
    image_paths: List[str]  # For multi-image tasks
    meta: Dict[str, Any]
    gt: str
    source: DataSource

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return {
            "uid": self.uid,
            "task_id": self.task_id,
            "image_path": self.image_path,
            "image_paths": self.image_paths,
            "meta": self.meta,
            "gt": self.gt,
            "source": self.source.value,
        }


class DriftDataset:
    """
    Unified dataset class for drift detection experiments.

    This class provides a consistent interface for loading data from:
    - Original task JSON files (using baseline data_adapters)
    - Preprocessed concatenated images from cache

    Args:
        task_ids: List of task IDs to load, or None for all tasks
        use_concat_cache: Whether to prioritize concatenated cache images
        split: Data split ("test" or "train")
        limit: Maximum number of samples to load per task (None for all)

    Example:
        # Load all ID tasks
        dataset = DriftDataset(task_ids=ID_TASKS)

        # Load with concatenated cache
        dataset = DriftDataset(task_ids=[7, 8], use_concat_cache=True)

        # Access samples
        for sample in dataset:
            print(sample.uid, sample.task_id)
    """

    def __init__(
        self,
        task_ids: Optional[List[int]] = None,
        use_concat_cache: bool = False,
        split: str = "test",
        limit: Optional[int] = None,
    ):
        self.task_ids = task_ids or ALL_TASKS
        self.use_concat_cache = use_concat_cache
        self.split = split
        self.limit = limit

        # Validate task IDs
        for tid in self.task_ids:
            if tid not in ALL_TASKS:
                raise ValueError(f"Invalid task_id: {tid}. Must be in {ALL_TASKS}")

        # Load samples
        self._samples: List[DriftSample] = []
        self._load_all_tasks()

        # Build index for fast lookup
        self._uid_to_idx: Dict[str, int] = {
            s.uid: i for i, s in enumerate(self._samples)
        }

    def _load_all_tasks(self) -> None:
        """Load samples from all specified tasks."""
        for task_id in self.task_ids:
            task_samples = self._load_task(task_id)
            if self.limit is not None:
                task_samples = task_samples[:self.limit]
            self._samples.extend(task_samples)

    def _load_task(self, task_id: int) -> List[DriftSample]:
        """
        Load samples for a single task.

        Priority:
        1. If use_concat_cache=True and cache exists, use cached data
        2. Otherwise, load from original JSON using data_adapters
        """
        if self.use_concat_cache and self._has_concat_cache():
            return self._load_from_concat_cache(task_id)
        else:
            return self._load_from_json(task_id)

    def _has_concat_cache(self) -> bool:
        """Check if concatenated cache directory exists and has files."""
        cache_path = Path(CONCAT_CACHE_DIR)
        if not cache_path.exists():
            return False
        # Check if there are any image files
        for ext in IMAGE_EXTENSIONS:
            if list(cache_path.glob(f"*{ext}")):
                return True
        return False

    def _load_from_json(self, task_id: int) -> List[DriftSample]:
        """Load samples from original task JSON files."""
        # Use baseline data_adapters to load and adapt the data
        json_dir = str(BASELINE_JSON_TEST_DIR) if self.split == "test" else str(
            BASELINE_ROOT / "dataset" / "json_train_taskall"
        )
        dataset_root = str(BASELINE_DATASET_DIR)

        try:
            # For train split, construct JSON filename directly since load_and_adapt
            # hardcodes "test_task" prefix and won't work with train data
            if self.split == "train":
                json_file = Path(json_dir) / f"train_task{task_id:02d}_all.json"
                if not json_file.exists():
                    logger.warning(f"Train JSON file not found: {json_file}")
                    return []

                # Load and parse JSON directly
                with open(json_file, 'r') as f:
                    raw_data = json.load(f)

                if not raw_data:
                    logger.warning(f"Empty JSON file: {json_file}")
                    return []

                # Process raw data - convert to expected format
                # The JSON structure matches the baseline format
                raw_samples = raw_data if isinstance(raw_data, list) else raw_data.get('data', [])
            else:
                # For test split, use load_and_adapt
                raw_samples = load_and_adapt(dataset_root, json_dir, task_id)
        except Exception as e:
            logger.warning(f"Failed to load task {task_id} from JSON: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

        # Convert to DriftSample format
        samples = []
        for idx, item in enumerate(raw_samples):
            # Handle different image path formats
            # For train data, use item keys directly; for test data, load_and_adapt converts them
            image_path = item.get("image_path", item.get("image", ""))  # Task 10 uses 'image'
            image_paths = []

            # Extract metadata based on split and task
            if self.split == "train":
                # Train JSON has different key names
                gt = item.get("ground_truth", item.get("gt", ""))
                # Try to get option label too
                gt_option = item.get("ground_truth_option", "")
                if gt_option:
                    gt = gt_option  # Use option letter (A, B, C, D, etc.)
                uid = item.get("question_id", item.get("id", item.get("uid", f"train_{task_id}_{idx}")))

                # Extract question and options from train JSON
                # Different tasks have different metadata structures
                if task_id == 10:
                    # Task 10 (VQA) has qa_pairs
                    qa_pairs = item.get("qa_pairs", [])
                    if qa_pairs:
                        # Use first QA pair
                        question = qa_pairs[0].get("question", "")
                        answer = qa_pairs[0].get("answer", "")
                        gt = answer  # Ground truth is the answer
                    else:
                        question = ""
                        answer = ""
                    meta = {
                        "question": question,
                        "answer": answer,
                        "qa_pairs": qa_pairs,
                    }
                else:
                    # Tasks 7-9 have standard structure
                    meta = {
                        "question": item.get("prompts", item.get("question", "")),
                        "options": item.get("options", item.get("options_list", "")),
                        "options_list": item.get("options_list", []),
                    }

                # For train data, add full path to image_path if it's not already a full path
                if image_path and not image_path.startswith("/"):
                    image_path = str(Path(dataset_root) / image_path)
            else:
                # Test data uses load_and_adapt which already converts keys and provides full paths
                gt = item.get("gt", "")
                uid = item.get("uid", f"test_{task_id}_{idx}")
                meta = item.get("meta", {})

            # Task 2 has pre/post image paths
            if task_id == 2:
                pre_path = item.get("pre_image_path", "")
                post_path = item.get("post_image_path", "")
                # Add full paths for Task 2 if train
                if self.split == "train" and pre_path and not pre_path.startswith("/"):
                    pre_path = str(Path(dataset_root) / pre_path)
                if self.split == "train" and post_path and not post_path.startswith("/"):
                    post_path = str(Path(dataset_root) / post_path)
                image_paths = [pre_path, post_path]
                image_path = pre_path  # Use pre-image as primary
            elif image_path:
                image_paths = [image_path]

            sample = DriftSample(
                uid=str(uid),
                task_id=item.get("task_id", task_id),
                image_path=image_path,
                image_paths=image_paths,
                meta=meta,
                gt=gt,
                source=DataSource.ORIGINAL_JSON,
            )
            samples.append(sample)

        return samples

    def _load_from_concat_cache(self, task_id: int) -> List[DriftSample]:
        """
        Load samples from concatenated cache directory.

        The cache contains preprocessed images with naming convention:
        concat_{hash}.png

        We need to match these to the original task samples to get metadata.
        """
        # First, load the original samples to get metadata
        original_samples = self._load_from_json(task_id)

        # Get list of cached images
        cache_path = Path(CONCAT_CACHE_DIR)
        cached_images = set()
        for ext in IMAGE_EXTENSIONS:
            for img_path in cache_path.glob(f"concat_*{ext}"):
                cached_images.add(img_path.stem)  # e.g., "concat_abc123"

        # For now, we return original samples but mark those that have cached versions
        # This is a simplified implementation - in practice, you'd need a mapping file
        samples = []
        for orig_sample in original_samples:
            # Create a copy with updated source if cached version exists
            # In real implementation, you'd have a proper hash -> sample mapping
            sample = DriftSample(
                uid=orig_sample.uid,
                task_id=orig_sample.task_id,
                image_path=orig_sample.image_path,
                image_paths=orig_sample.image_paths,
                meta=orig_sample.meta,
                gt=orig_sample.gt,
                source=DataSource.CONCAT_CACHE if cached_images else DataSource.ORIGINAL_JSON,
            )
            samples.append(sample)

        return samples

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self._samples)

    def __getitem__(self, idx: Union[int, str]) -> DriftSample:
        """
        Get a sample by index or UID.

        Args:
            idx: Integer index or string UID

        Returns:
            DriftSample at the specified index or with the specified UID
        """
        if isinstance(idx, str):
            # Lookup by UID
            if idx not in self._uid_to_idx:
                raise KeyError(f"Sample with UID '{idx}' not found")
            return self._samples[self._uid_to_idx[idx]]
        else:
            # Lookup by index
            return self._samples[idx]

    def __iter__(self):
        """Iterate over all samples."""
        return iter(self._samples)

    def get_task_samples(self, task_id: int) -> List[DriftSample]:
        """Get all samples for a specific task."""
        return [s for s in self._samples if s.task_id == task_id]

    def get_id_samples(self) -> List[DriftSample]:
        """Get all samples from in-distribution tasks."""
        return [s for s in self._samples if s.task_id in ID_TASKS]

    def get_ood_samples(self) -> List[DriftSample]:
        """Get all samples from out-of-distribution tasks."""
        return [s for s in self._samples if s.task_id in OOD_TASKS]

    def sample_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded samples."""
        stats = {
            "total_samples": len(self._samples),
            "tasks_loaded": list(set(s.task_id for s in self._samples)),
            "samples_per_task": {},
            "sources": {"original_json": 0, "concat_cache": 0},
        }

        for task_id in self.task_ids:
            count = len([s for s in self._samples if s.task_id == task_id])
            stats["samples_per_task"][task_id] = count

        for sample in self._samples:
            stats["sources"][sample.source.value] += 1

        return stats

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert all samples to a list of dictionaries."""
        return [s.to_dict() for s in self._samples]


# ============================================================================
# Convenience Functions
# ============================================================================

def load_id_dataset(
    use_concat_cache: bool = False,
    split: str = "test",
    limit: Optional[int] = None,
) -> DriftDataset:
    """
    Load in-distribution (ID) dataset.

    Args:
        use_concat_cache: Whether to use cached concatenated images
        split: Data split ("test" or "train")
        limit: Maximum samples per task

    Returns:
        DriftDataset containing only ID task samples
    """
    return DriftDataset(
        task_ids=ID_TASKS,
        use_concat_cache=use_concat_cache,
        split=split,
        limit=limit,
    )


def load_ood_dataset(
    use_concat_cache: bool = False,
    split: str = "test",
    limit: Optional[int] = None,
) -> DriftDataset:
    """
    Load out-of-distribution (OOD) dataset.

    Args:
        use_concat_cache: Whether to use cached concatenated images
        split: Data split ("test" or "train")
        limit: Maximum samples per task

    Returns:
        DriftDataset containing only OOD task samples
    """
    return DriftDataset(
        task_ids=OOD_TASKS,
        use_concat_cache=use_concat_cache,
        split=split,
        limit=limit,
    )


def load_all_tasks(
    use_concat_cache: bool = False,
    split: str = "test",
    limit: Optional[int] = None,
) -> DriftDataset:
    """
    Load all tasks.

    Args:
        use_concat_cache: Whether to use cached concatenated images
        split: Data split ("test" or "train")
        limit: Maximum samples per task

    Returns:
        DriftDataset containing all task samples
    """
    try:
        return DriftDataset(
            task_ids=ALL_TASKS,
            use_concat_cache=use_concat_cache,
            split=split,
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Failed to load tasks: {e}")
        raise

# Add error handling for individual samples
def process_sample(sample: DriftSample):
    try:
        # Process the sample
        ... # existing processing logic
    except Exception as e:
        logger.warning(f"Skipping bad sample {sample.uid}: {e}")
        return None


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing DriftDataset")
    print("=" * 60)

    # Test loading ID tasks
    print("\n1. Loading ID tasks (7, 8, 9, 10)...")
    try:
        id_dataset = load_id_dataset(limit=5)
        print(f"   Loaded {len(id_dataset)} samples")
        stats = id_dataset.sample_statistics()
        print(f"   Statistics: {stats}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test loading a single task
    print("\n2. Loading single task (Task 7)...")
    try:
        single_task = DriftDataset(task_ids=[7], limit=3)
        print(f"   Loaded {len(single_task)} samples")
        for sample in single_task:
            print(f"   - {sample.uid}: {sample.gt[:30]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Test OOD tasks
    print("\n3. Loading OOD tasks...")
    try:
        ood_dataset = load_ood_dataset(limit=2)
        print(f"   Loaded {len(ood_dataset)} samples")
        print(f"   Tasks: {ood_dataset.sample_statistics()['tasks_loaded']}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")
