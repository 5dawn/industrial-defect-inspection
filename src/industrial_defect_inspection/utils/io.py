"""Small, deterministic file helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
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


def git_state() -> dict[str, Any]:
    """Return commit and dirty-state provenance without storing the diff itself."""
    commit = git_commit()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": commit, "dirty": None, "diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": git_commit(),
        "pid": os.getpid(),
        "captured_at": datetime.now(UTC).isoformat(),
        "git": git_state(),
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


def file_record(path: Path) -> dict[str, Any]:
    """Return a portable fingerprint for an experiment input."""
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
