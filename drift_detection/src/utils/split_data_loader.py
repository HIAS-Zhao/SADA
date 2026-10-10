# -*- coding: utf-8 -*-
"""
Data loader for pre-split dataset.

This module loads data from a pre-split dataset structure:
- 0_Reference_Fit/
- 1_Threshold_Cal/
- 2_Stream_Sim_NoDrift/
- 2_Stream_Sim_Drift/

Each split directory contains:
- data.json: Sample metadata
- images/: Image files
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SplitSample:
    """Sample structure for pre-split dataset."""
    uid: str
    task_id: int
    image_path: str
    image_paths: List[str]  # For multi-image tasks
    question: str
    options: Optional[Dict[str, str]]
    type_id: Optional[int]
    ground_truth: str
    meta: Dict[str, Any]  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            'uid': self.uid,
            'task_id': self.task_id,
            'image_path': self.image_path,
            'image_paths': self.image_paths,
            'question': self.question,
            'options': self.options,
            'type_id': self.type_id,
            'ground_truth': self.ground_truth,
            'meta': self.meta,
        }


class SplitDataset:
    """
    Dataset loader for pre-split data.

    Loads samples from a specific split directory.
    """

    def __init__(self, split_dir: Path):
        """
        Initialize dataset loader.

        Args:
            split_dir: Path to split directory (e.g., 0_Reference_Fit)
        """
        self.split_dir = Path(split_dir)
        self.data_file = self.split_dir / "data.json"
        self.images_dir = self.split_dir / "images"

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from {self.split_dir.name}")

    def _load_samples(self) -> List[SplitSample]:
        """Load samples from data.json."""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = self._parse_sample(item)
            if sample:
                samples.append(sample)

        return samples

    def _parse_sample(self, item: Dict[str, Any]) -> Optional[SplitSample]:
        """
        Parse a single sample from JSON.

        Handles different task formats:
        - Task 1 type_id=3: Options contain image paths (4 images)
        - Task 2: May have pre_image_path and post_image_path
        - Other tasks: Single image_path
        """
        try:
            uid = item.get('uid', '')
            task_id = item.get('task_id', 0)
            question = item.get('question', '')
            options = item.get('options', None)
            type_id = item.get('type_id', None)
            ground_truth = item.get('ground_truth', '')

            # Determine image paths based on task type
            image_paths = []
            main_image_path = None

            # Task 1 with image options (type_id=1 or type_id=3)
            if task_id == 1 and isinstance(options, dict):
                # Check if options contain image paths
                first_option = next(iter(options.values()), '')
                if first_option and ('./images/' in first_option or first_option.endswith(('.jpg', '.png'))):
                    for key in sorted(options.keys()):
                        img_path = options[key]
                        if img_path.startswith('./images/'):
                            img_path = str(self.images_dir / img_path.replace('./images/', ''))
                        elif not img_path.startswith('/'):
                            img_path = str(self.images_dir / img_path)
                        image_paths.append(img_path)
                    main_image_path = image_paths[0] if image_paths else None

            # Task 2: Pre/post disaster images
            elif task_id == 2:
                pre_path = item.get('pre_image_path')
                post_path = item.get('post_image_path')

                if pre_path and post_path:
                    if pre_path.startswith('./images/'):
                        pre_path = str(self.images_dir / pre_path.replace('./images/', ''))
                    if post_path.startswith('./images/'):
                        post_path = str(self.images_dir / post_path.replace('./images/', ''))

                    image_paths = [pre_path, post_path]
                    main_image_path = pre_path

            # Other tasks: Single image
            else:
                img_path = item.get('image_path', '')
                if img_path:
                    if img_path.startswith('./images/'):
                        img_path = str(self.images_dir / img_path.replace('./images/', ''))
                    main_image_path = img_path
                    image_paths = [img_path]

            if not main_image_path:
                logger.warning(f"No image path found for sample: {uid}")
                return None

            meta = {
                'question': question,
                'options': options,
                'type_id': type_id,
                'task_id': task_id,
            }

            if task_id == 2 and len(image_paths) >= 2:
                meta['pre_image_path'] = image_paths[0]
                meta['post_image_path'] = image_paths[1]

            return SplitSample(
                uid=uid,
                task_id=task_id,
                image_path=main_image_path,
                image_paths=image_paths,
                question=question,
                options=options,
                type_id=type_id,
                ground_truth=ground_truth,
                meta=meta,
            )

        except Exception as e:
            logger.error(f"Error parsing sample: {e}")
            logger.error(f"Sample data: {item}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SplitSample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def get_task_ids(self) -> List[int]:
        """Get unique task IDs in this dataset."""
        return sorted(list(set(s.task_id for s in self.samples)))

    def filter_by_task(self, task_id: int) -> List[SplitSample]:
        """Filter samples by task ID."""
        return [s for s in self.samples if s.task_id == task_id]


def load_split_dataset(dataset_dir: Path, split_name: str) -> SplitDataset:
    """
    Load a specific split from the dataset.

    Args:
        dataset_dir: Root dataset directory
        split_name: Split name (e.g., '0_Reference_Fit')

    Returns:
        SplitDataset instance
    """
    split_dir = dataset_dir / split_name
    return SplitDataset(split_dir)
