"""Package trained VisA PatchCore assets without redistributing dataset pixels."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from industrial_defect_inspection.config import load_anomaly_inference_config
from industrial_defect_inspection.utils.io import sha256_file, write_json


def _portable_metadata(source: Path, category: str) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("category") != category:
        raise ValueError(
            f"Metadata category mismatch for {source}: expected {category}, "
            f"found {payload.get('category')}"
        )
    payload.pop("checkpoint", None)
    payload["checkpoint_file"] = f"{category}.ckpt"
    return payload


def _write_zip(bundle: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in bundle.iterdir() if item.is_file()):
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def package_anomaly_release(
    version: str,
    config_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Build a deterministic checkpoint bundle from an explicit inference config."""
    config = load_anomaly_inference_config(config_path)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Release output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / f"industrial-defect-inspection-{version}-visa-patchcore"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir()

    artifacts: list[dict[str, object]] = []
    for category in sorted(config.checkpoints):
        checkpoint = config.checkpoints[category]
        metadata = config.metadata[category]
        if not checkpoint.is_file() or not metadata.is_file():
            raise FileNotFoundError(
                f"Release inputs missing for '{category}': {checkpoint}, {metadata}"
            )
        checkpoint_destination = bundle / f"{category}.ckpt"
        metadata_destination = bundle / f"{category}.json"
        shutil.copyfile(checkpoint, checkpoint_destination)
        write_json(metadata_destination, _portable_metadata(metadata, category))
        for path in (checkpoint_destination, metadata_destination):
            artifacts.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )

    attribution = bundle / "VISADATA_ATTRIBUTION.md"
    attribution.write_text(
        "# VisA attribution\n\n"
        "These PatchCore checkpoints were trained on VisA, distributed under CC BY 4.0.\n"
        "Source: https://github.com/amazon-science/spot-diff\n\n"
        "No VisA source image or mask is included. Review the repository model card and all "
        "applicable software licenses before redistribution or deployment.\n",
        encoding="utf-8",
    )
    artifacts.append(
        {
            "name": attribution.name,
            "bytes": attribution.stat().st_size,
            "sha256": sha256_file(attribution),
        }
    )
    write_json(
        bundle / "package_manifest.json",
        {
            "version": version,
            "release_type": "visa-patchcore-checkpoints",
            "dataset_pixels_included": False,
            "categories": sorted(config.checkpoints),
            "artifacts": artifacts,
        },
    )
    checksum_files = sorted(path for path in bundle.iterdir() if path.is_file())
    (bundle / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_files),
        encoding="utf-8",
    )
    archive = output_dir / f"industrial-defect-inspection-{version}-visa-patchcore.zip"
    _write_zip(bundle, archive)
    return archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package trained VisA PatchCore checkpoints.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--config", default="configs/anomaly/infer.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    archive = package_anomaly_release(
        args.version, Path(args.config), Path(args.output), overwrite=args.overwrite
    )
    print(f"Anomaly release archive: {archive}")


if __name__ == "__main__":
    main()
