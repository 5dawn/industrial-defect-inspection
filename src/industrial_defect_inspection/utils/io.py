"""Small, deterministic file helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": git_commit(),
        "pid": os.getpid(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            snapshot["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        snapshot["torch"] = None
    try:
        import ultralytics

        snapshot["ultralytics"] = ultralytics.__version__
    except ImportError:
        snapshot["ultralytics"] = None
    return snapshot
