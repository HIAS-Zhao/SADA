# -*- coding: utf-8 -*-
"""
Path Setup Utility for Cross-Directory Imports.

This module handles sys.path manipulation and dynamic module loading to enable
importing modules from the baseline project (qwen_eval) without modifying its code.

Since the baseline project's 'code/' directory doesn't have __init__.py,
we use importlib to load modules dynamically.

Usage:
    # At the top of any script that needs baseline imports:
    from src.utils.path_setup import get_baseline_module

    # Load modules from baseline:
    data_adapters = get_baseline_module("data_adapters")
    io_utils = get_baseline_module("io_utils")

    # Use the loaded modules:
    samples = data_adapters.load_and_adapt(dataset_root, json_dir, task_id)
    data = io_utils.load_json(path)
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, Optional

# ============================================================================
# Path Resolution
# ============================================================================

# Current file: drift_detection_exp/src/utils/path_setup.py
# We need to find: qwen_eval/ (sibling directory of drift_detection_exp)

_CURRENT_FILE = Path(__file__).resolve()
_UTILS_DIR = _CURRENT_FILE.parent          # src/utils/
_SRC_DIR = _UTILS_DIR.parent               # src/
_PROJECT_ROOT = _SRC_DIR.parent            # drift_detection_exp/
_PROJECTS_DIR = _PROJECT_ROOT.parent       # @@WORKSPACE@@/

# Baseline project paths
BASELINE_ROOT = _PROJECTS_DIR / "qwen_eval"
BASELINE_CODE_DIR = BASELINE_ROOT / "code"

# ============================================================================
# Path Validation
# ============================================================================

def _validate_paths() -> None:
    """Validate that the baseline project paths exist."""
    if not BASELINE_ROOT.exists():
        raise FileNotFoundError(
            f"Baseline project not found at: {BASELINE_ROOT}\n"
            f"Expected directory structure:\n"
            f"  @@WORKSPACE@@/\n"
            f"    ├── qwen_eval/        <-- Baseline project\n"
            f"    └── drift_detection_exp/  <-- This project"
        )

    if not BASELINE_CODE_DIR.exists():
        raise FileNotFoundError(
            f"Baseline code directory not found at: {BASELINE_CODE_DIR}\n"
            f"Expected 'code/' subdirectory in qwen_eval."
        )

# ============================================================================
# Dynamic Module Loading
# ============================================================================

# Cache for loaded modules
_module_cache: dict = {}


def get_baseline_module(module_name: str) -> Any:
    """
    Dynamically load a module from the baseline project's code directory.

    This function uses importlib to load Python modules from qwen_eval/code/
    without requiring __init__.py files.

    Args:
        module_name: Name of the module to load (e.g., "data_adapters", "io_utils")

    Returns:
        The loaded module object

    Raises:
        FileNotFoundError: If the module file doesn't exist
        ImportError: If the module fails to load

    Example:
        data_adapters = get_baseline_module("data_adapters")
        samples = data_adapters.load_and_adapt(dataset_root, json_dir, task_id)
    """
    # Check cache first
    if module_name in _module_cache:
        return _module_cache[module_name]

    _validate_paths()

    # Construct module file path
    module_path = BASELINE_CODE_DIR / f"{module_name}.py"

    if not module_path.exists():
        raise FileNotFoundError(
            f"Module '{module_name}' not found at: {module_path}"
        )

    # First, ensure io_utils is loaded if we're loading data_adapters
    # (since data_adapters imports from io_utils)
    if module_name == "data_adapters" and "io_utils" not in _module_cache:
        get_baseline_module("io_utils")

    # Add the code directory to sys.path temporarily for relative imports
    code_dir_str = str(BASELINE_CODE_DIR)
    if code_dir_str not in sys.path:
        sys.path.insert(0, code_dir_str)

    try:
        # Load the module using importlib
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for module: {module_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Cache the module
        _module_cache[module_name] = module

        return module
    except Exception as e:
        raise ImportError(f"Failed to load module '{module_name}': {e}")


def setup_baseline_path() -> str:
    """
    Add the baseline project's code directory to sys.path.

    This is an alternative approach that adds the path directly.
    Note: This may not work for all imports if modules have internal
    relative imports.

    Returns:
        str: The path that was added to sys.path.

    Raises:
        FileNotFoundError: If baseline project structure is invalid.
    """
    _validate_paths()

    code_dir_str = str(BASELINE_CODE_DIR)

    # Add to sys.path if not already present
    if code_dir_str not in sys.path:
        sys.path.insert(0, code_dir_str)

    return code_dir_str

# ============================================================================
# Auto-Setup on Import
# ============================================================================

# Validate paths when this module is imported
_validate_paths()

# Pre-load commonly used modules
try:
    _io_utils = get_baseline_module("io_utils")
    _data_adapters = get_baseline_module("data_adapters")
except Exception:
    # Don't fail on import if baseline modules aren't accessible yet
    _io_utils = None
    _data_adapters = None

# ============================================================================
# Convenience Exports
# ============================================================================

# Export path constants as strings for convenience
BASELINE_ROOT_STR = str(BASELINE_ROOT)
BASELINE_CODE_DIR_STR = str(BASELINE_CODE_DIR)

__all__ = [
    "setup_baseline_path",
    "get_baseline_module",
    "BASELINE_ROOT",
    "BASELINE_CODE_DIR",
    "BASELINE_ROOT_STR",
    "BASELINE_CODE_DIR_STR",
]

if __name__ == "__main__":
    # Test path setup
    print(f"Project Root: {_PROJECT_ROOT}")
    print(f"Baseline Root: {BASELINE_ROOT}")
    print(f"Baseline Code Dir: {BASELINE_CODE_DIR}")

    print("\nTesting dynamic module loading...")

    try:
        io_utils = get_baseline_module("io_utils")
        print(f"✓ Loaded io_utils: {io_utils}")
        print(f"  - load_json: {io_utils.load_json}")
    except Exception as e:
        print(f"✗ Failed to load io_utils: {e}")

    try:
        data_adapters = get_baseline_module("data_adapters")
        print(f"✓ Loaded data_adapters: {data_adapters}")
        print(f"  - load_and_adapt: {data_adapters.load_and_adapt}")
    except Exception as e:
        print(f"✗ Failed to load data_adapters: {e}")
