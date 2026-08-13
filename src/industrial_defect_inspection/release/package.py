"""Build a deterministic source-only evidence bundle for GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from industrial_defect_inspection.utils.io import write_json

VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"\b[A-Za-z]:[\\/]"),
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
    re.compile(r"/(?:mnt|tmp|workspace|var/tmp)/[^\s]*"),
    re.compile(r"\\\\[^\\\s]+\\[^\\\s]+"),
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".onnx", ".jpg", ".jpeg", ".png", ".xml"}
SOURCE_ALLOWLIST = {
    "experiment_summary.json": Path("reports/metrics/published/experiment/experiment_summary.json"),
    "test_per_class.csv": Path("reports/metrics/published/experiment/test_per_class.csv"),
    "dataset_summary.json": Path("reports/metrics/published/data/dataset_summary.json"),
    "model_card.md": Path("reports/model_card.md"),
    "dataset_card.md": Path("reports/dataset_card.md"),
    "requirements-lock-cu130.txt": Path("requirements-lock-cu130.txt"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_text(path: Path) -> None:
    if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Forbidden release artifact type: {path.name}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Release artifact must be UTF-8 text: {path}") from exc
    for pattern in ABSOLUTE_PATH_PATTERNS:
        match = pattern.search(text)
        if match:
            raise ValueError(f"Absolute local path found in release artifact {path.name}")


def _environment_payload(experiment_summary: Path) -> dict[str, Any]:
    payload = json.loads(experiment_summary.read_text(encoding="utf-8"))
    training = payload.get("training", {})
    return {
        "environment": training.get("environment", {}),
        "training_duration_seconds": training.get("duration_seconds"),
        "checkpoint_sha256": training.get("checkpoint_sha256"),
        "weights_included": False,
    }


def _release_notes(version: str) -> str:
    return f"""# Industrial Defect Inspection {version}

This source-only release publishes aggregate, dataset-pixel-free evidence from
the formal YOLO26n experiment. It does not include NEU-DET images, annotations,
PyTorch checkpoints, or ONNX weights.

The upstream NEU-DET page provides downloads and citation guidance but does not
state a recognized standard dataset license. Train locally after reviewing the
upstream terms:
https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm

The included checksums make every attached evidence file independently
verifiable. Reported metrics are research and portfolio results, not a
production quality-control claim.
"""


def _write_deterministic_zip(bundle_dir: Path, destination: Path) -> None:
    members = sorted(path for path in bundle_dir.iterdir() if path.is_file())
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def package_release(
    version: str,
    output_dir: Path,
    *,
    root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Version must use vMAJOR.MINOR.PATCH format")
    root = (root or Path.cwd()).resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Release output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_dir / f"industrial-defect-inspection-{version}-evidence"
    if bundle_dir.exists():
        if bundle_dir.is_symlink():
            raise ValueError(f"Refusing to replace symlinked release bundle: {bundle_dir}")
        if bundle_dir.parent != output_dir:
            raise ValueError(f"Unsafe release bundle path: {bundle_dir}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir()

    copied: list[Path] = []
    for destination_name, relative_source in SOURCE_ALLOWLIST.items():
        source = root / relative_source
        if not source.is_file():
            raise FileNotFoundError(f"Required release source not found: {source}")
        _validate_text(source)
        destination = bundle_dir / destination_name
        shutil.copyfile(source, destination)
        copied.append(destination)

    environment_path = bundle_dir / "environment.json"
    write_json(
        environment_path,
        _environment_payload(bundle_dir / "experiment_summary.json"),
    )
    notes_path = bundle_dir / "RELEASE_NOTES.md"
    notes_path.write_text(_release_notes(version), encoding="utf-8")
    copied.extend((environment_path, notes_path))
    for path in copied:
        _validate_text(path)

    artifacts = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(copied)
    ]
    manifest_path = bundle_dir / "package_manifest.json"
    write_json(
        manifest_path,
        {
            "version": version,
            "release_type": "source-only-evidence",
            "weights_included": False,
            "dataset_pixels_included": False,
            "artifacts": artifacts,
        },
    )
    checksummed = [*copied, manifest_path]
    checksum_path = bundle_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(checksummed)),
        encoding="utf-8",
    )
    zip_path = output_dir / f"industrial-defect-inspection-{version}-evidence.zip"
    _write_deterministic_zip(bundle_dir, zip_path)
    archive_checksum_path = output_dir / f"{zip_path.name}.sha256"
    archive_checksum_path.write_text(
        f"{sha256_file(zip_path)}  {zip_path.name}\n",
        encoding="utf-8",
    )
    return {
        "bundle": bundle_dir,
        "zip": zip_path,
        "zip_checksum": archive_checksum_path,
        "checksums": checksum_path,
        "manifest": manifest_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package a source-only evidence release.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = package_release(args.version, Path(args.output), overwrite=args.overwrite)
    print(f"Release archive: {result['zip']}")


if __name__ == "__main__":
    main()
