"""Shared runtime setup, device selection, and run logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def prepare_runtime(output_dir: Path) -> Path:
    """Route third-party caches to a writable, experiment-local directory."""
    del output_dir  # Kept in the public signature so callers need no global runtime knowledge.
    runtime_dir = Path(os.environ.get("IDI_RUNTIME_DIR", "artifacts/runtime")).resolve()
    ultralytics_dir = runtime_dir / "ultralytics"
    matplotlib_dir = runtime_dir / "matplotlib"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    return runtime_dir


def resolve_device(requested: str) -> str | int:
    """Resolve ``auto`` to the first CUDA device or CPU."""
    if requested.casefold() != "auto":
        return int(requested) if requested.isdigit() else requested
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def configure_run_logger(name: str, output_dir: Path) -> logging.Logger:
    """Create a timestamped console and file logger without global side effects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger
