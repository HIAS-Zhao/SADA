# -*- coding: utf-8 -*-
"""Utility modules for drift detection experiment."""

from .path_setup import setup_baseline_path, get_baseline_module, BASELINE_CODE_DIR

# Lazy import to avoid circular dependencies
def get_drift_dataset():
    """Get DriftDataset class (lazy import)."""
    from .data_loader import DriftDataset
    return DriftDataset

__all__ = [
    "setup_baseline_path",
    "get_baseline_module",
    "BASELINE_CODE_DIR",
    "get_drift_dataset",
]
