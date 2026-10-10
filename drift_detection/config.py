# -*- coding: utf-8 -*-
"""
Configuration for Drift Detection Experiment.

This module defines all constants and configuration parameters for the
drift detection experiment, including paths, task definitions, and
experiment settings.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# ============================================================================
# Path Configuration
# ============================================================================

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = PROJECT_ROOT.parent

# Dataset directory for the four-way drift detection split
DATASET_DIR = PROJECTS_DIR / "dataset"

# Baseline project paths (legacy, for compatibility)
BASELINE_ROOT = PROJECTS_DIR / "qwen_eval"
BASELINE_CODE_DIR = BASELINE_ROOT / "code"
BASELINE_DATASET_DIR = BASELINE_ROOT / "dataset"
BASELINE_JSON_TEST_DIR = BASELINE_DATASET_DIR / "json_test_taskall"
BASELINE_JSON_TRAIN_DIR = BASELINE_DATASET_DIR / "json_train_taskall"

# Model and run directories
MODEL_DIR = BASELINE_ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
RUNS_DIR = BASELINE_ROOT / "runs" / "Qwen2.5-VL-3B-Instruct"

# Concatenated cache directory (for preprocessed multi-image data)
CONCAT_CACHE_DIR = RUNS_DIR / "_concat_cache"

# Output directories for this experiment
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ============================================================================
# Task Configuration
# ============================================================================

# All available tasks (1-11)
ALL_TASKS: List[int] = list(range(1, 12))

# In-Distribution (ID) Tasks - used as reference/historical distribution
# These tasks represent the "known" distribution the model was trained on
ID_TASKS: List[int] = [7, 8, 9, 10]

# Out-of-Distribution (OOD) Tasks - used as drift/target tasks
# These tasks represent potential distribution shift scenarios
OOD_TASKS: List[int] = [1, 2, 3, 4, 5, 6, 11]

# Task name mapping for logging and visualization
TASK_NAMES: Dict[int, str] = {
    1: "AR (Anomaly Recognition)",
    2: "DR (Disaster Recognition)",
    3: "RRD (Road Damage Detection)",
    4: "STC (Scene Text Classification)",
    5: "ATC (Aerial Target Classification)",
    6: "LUC (Land Use Classification)",
    7: "SRC (Scene Recognition Classification)",
    8: "SC (Ship Classification)",
    9: "IC (Image Captioning)",
    10: "VQA (Visual Question Answering)",
    11: "POPE (Polling-based Object Probing Evaluation)",
}

# Task type mapping
TASK_TYPES: Dict[int, str] = {
    1: "classification",
    2: "classification",
    3: "classification",
    4: "classification",
    5: "classification",
    6: "classification",
    7: "classification",
    8: "classification",
    9: "captioning",
    10: "vqa",
    11: "binary_classification",
}

# ============================================================================
# Data Configuration
# ============================================================================

# Whether to use cached concatenated images (for multi-image tasks)
USE_CONCAT_CACHE: bool = True

# Supported image extensions
IMAGE_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

# ============================================================================
# Experiment Configuration
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Batch size for data loading
BATCH_SIZE: int = 8

# Number of workers for data loading
NUM_WORKERS: int = 0

# Device configuration
DEVICE: str = "cuda"  # or "cpu"

# ============================================================================
# Drift Detection Configuration
# ============================================================================

# Number of reference samples to use for distribution estimation
N_REFERENCE_SAMPLES: int = 500

# Significance level for drift detection
ALPHA: float = 0.05

# Window size for online drift detection
WINDOW_SIZE: int = 200

# ============================================================================
# Fixed-length feature drift detection configuration
# ============================================================================

# PCA Information Bottleneck
PCA_N_COMPONENTS: int = 192  # Default PCA projection dimension

# Data Split Configuration (for the four-split detection pipeline)
# Following the 4-split strategy:
# - 0_Reference_Fit: Fit PCA model
# - 1_Threshold_Cal: Calibrate detection threshold
# - 2_Stream_Sim_NoDrift: Evaluate FPR (ID test set)
# - 2_Stream_Sim_Drift: Evaluate Recall (OOD test set)

DATA_SPLIT_CONFIG: Dict[str, Dict[str, Any]] = {
    'reference': {
        'name': '0_Reference_Fit',
        'purpose': 'Fit PCA model on ID data',
        'data_file': 'data.json',
        'images_dir': 'images',
    },
    'calibration': {
        'name': '1_Threshold_Cal',
        'purpose': 'Calibrate detection threshold',
        'data_file': 'data.json',
        'images_dir': 'images',
    },
    'test_id': {
        'name': '2_Stream_Sim_NoDrift',
        'purpose': 'Evaluate FPR on held-out ID data',
        'data_file': 'data.json',
        'images_dir': 'images',
    },
    'test_ood': {
        'name': '2_Stream_Sim_Drift',
        'purpose': 'Evaluate Recall on OOD data',
        'data_file': 'data.json',
        'images_dir': 'images',
    },
}

# ============================================================================
# Utility Functions
# ============================================================================

def get_task_json_path(task_id: int, split: str = "test") -> Path:
    """
    Get the path to a task's JSON file.

    Args:
        task_id: Task ID (1-11)
        split: Data split ("test" or "train")

    Returns:
        Path to the JSON file
    """
    if split == "test":
        return BASELINE_JSON_TEST_DIR / f"test_task{task_id:02d}_all.json"
    elif split == "train":
        return BASELINE_JSON_TRAIN_DIR / f"train_task{task_id:02d}_all.json"
    else:
        raise ValueError(f"Unknown split: {split}. Use 'test' or 'train'.")


def get_task_type(task_id: int) -> str:
    """Get the type of a task."""
    return TASK_TYPES.get(task_id, "unknown")


def get_task_name(task_id: int) -> str:
    """Get the name of a task."""
    return TASK_NAMES.get(task_id, f"Task {task_id}")


def is_id_task(task_id: int) -> bool:
    """Check if a task is an in-distribution (ID) task."""
    return task_id in ID_TASKS


def is_ood_task(task_id: int) -> bool:
    """Check if a task is an out-of-distribution (OOD) task."""
    return task_id in OOD_TASKS


def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Validation
# ============================================================================

def validate_config() -> bool:
    """
    Validate that all required paths and configurations exist.

    Returns:
        True if all validations pass

    Raises:
        FileNotFoundError: If required paths don't exist
        ValueError: If configuration is invalid
    """
    # Check baseline paths
    if not BASELINE_ROOT.exists():
        raise FileNotFoundError(f"Baseline project not found: {BASELINE_ROOT}")

    if not BASELINE_DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {BASELINE_DATASET_DIR}")

    # Check task configuration consistency
    all_defined = set(ID_TASKS + OOD_TASKS)
    if all_defined != set(ALL_TASKS):
        raise ValueError(
            f"ID_TASKS + OOD_TASKS should cover all tasks.\n"
            f"Missing: {set(ALL_TASKS) - all_defined}\n"
            f"Extra: {all_defined - set(ALL_TASKS)}"
        )

    # Check for overlapping tasks
    overlap = set(ID_TASKS) & set(OOD_TASKS)
    if overlap:
        raise ValueError(f"Tasks cannot be both ID and OOD: {overlap}")

    return True


# ============================================================================
# Module Initialization
# ============================================================================

if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("Drift Detection Experiment Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Baseline Root: {BASELINE_ROOT}")
    print(f"Concat Cache Dir: {CONCAT_CACHE_DIR}")
    print(f"\nID Tasks: {ID_TASKS}")
    print(f"OOD Tasks: {OOD_TASKS}")
    print("\nValidating configuration...")

    try:
        validate_config()
        print("✓ Configuration is valid!")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
