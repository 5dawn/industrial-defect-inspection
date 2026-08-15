"""Validate and prepare selected VisA categories for one-class localization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from PIL import Image

from industrial_defect_inspection.anomaly.preprocessing import (
    letterbox_image,
    letterbox_mask,
)
from industrial_defect_inspection.config import VisaDataConfig, load_visa_data_config
from industrial_defect_inspection.utils.io import sha256_file, write_json

REQUIRED_COLUMNS = {"object", "split", "label", "image", "mask"}
NORMAL_LABELS = {"normal", "good"}
ANOMALOUS_LABELS = {"anomaly", "anomalous", "bad"}


@dataclass(frozen=True, slots=True)
class VisaSample:
    category: str
    official_split: str
    label: str
    image_relative: str
    mask_relative: str | None


def _safe_source(root: Path, relative: str) -> Path:
    normalized = PurePosixPath(relative)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe VisA path in split CSV: {relative}")
    path = root.joinpath(*normalized.parts)
    if not path.is_file():
        raise FileNotFoundError(f"VisA source file not found: {path}")
    return path


def read_official_split(config: VisaDataConfig) -> list[VisaSample]:
    """Read selected categories from the official ``1cls.csv`` file."""
    if not config.raw_root.is_dir():
        raise FileNotFoundError(
            f"VisA root not found: {config.raw_root}. Follow data/README.md and extract VisA first."
        )
    if not config.split_csv.is_file():
        raise FileNotFoundError(
            f"Official VisA split file not found: {config.split_csv}. Download split_csv/1cls.csv "
            "from https://github.com/amazon-science/spot-diff."
        )
    with config.split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"VisA split CSV is missing columns: {sorted(missing)}")
        rows: list[VisaSample] = []
        selected = set(config.categories)
        for line_number, row in enumerate(reader, start=2):
            category = row["object"].strip()
            if category not in selected:
                continue
            split = row["split"].strip().casefold()
            raw_label = row["label"].strip().casefold()
            if split not in {"train", "test"}:
                raise ValueError(f"Invalid split '{split}' at CSV line {line_number}")
            if raw_label in NORMAL_LABELS:
                label = "normal"
            elif raw_label in ANOMALOUS_LABELS:
                label = "anomaly"
            else:
                raise ValueError(f"Invalid label '{raw_label}' at CSV line {line_number}")
            mask = row["mask"].strip() or None
            if label == "anomaly" and not mask:
                raise ValueError(f"Anomalous sample has no mask at CSV line {line_number}")
            if split == "train" and label != "normal":
                raise ValueError(f"One-class training row is not normal at CSV line {line_number}")
            rows.append(
                VisaSample(
                    category=category,
                    official_split=split,
                    label=label,
                    image_relative=row["image"].strip(),
                    mask_relative=mask,
                )
            )
    present = {sample.category for sample in rows}
    missing_categories = set(config.categories) - present
    if missing_categories:
        raise ValueError(
            f"Selected VisA categories are absent from the CSV: {sorted(missing_categories)}"
        )
    return rows


def assign_validation(
    samples: list[VisaSample], categories: list[str], ratio: float, seed: int
) -> dict[VisaSample, str]:
    """Reserve normal official-train images without changing official test rows."""
    assignments: dict[VisaSample, str] = {}
    grouped: dict[str, list[VisaSample]] = defaultdict(list)
    for sample in samples:
        if sample.official_split == "train":
            grouped[sample.category].append(sample)
        else:
            assignments[sample] = "test"
    for category_index, category in enumerate(categories):
        values = sorted(grouped[category], key=lambda item: item.image_relative.casefold())
        if len(values) < 2:
            raise ValueError(
                f"VisA category '{category}' needs at least two normal training images"
            )
        rng = random.Random(seed + category_index)
        rng.shuffle(values)
        validation_count = max(1, min(len(values) - 1, round(len(values) * ratio)))
        validation = set(values[:validation_count])
        for sample in values:
            assignments[sample] = "val" if sample in validation else "train"
    return assignments


def _destination(sample: VisaSample, assigned_split: str, output: Path) -> tuple[Path, Path | None]:
    name = f"{Path(sample.image_relative).stem}.png"
    if assigned_split == "train":
        return output / sample.category / "train" / "good" / name, None
    if assigned_split == "val":
        return output / sample.category / "val" / "good" / name, None
    kind = "good" if sample.label == "normal" else "bad"
    mask = (
        output / sample.category / "ground_truth" / "bad" / name
        if sample.label == "anomaly"
        else None
    )
    return output / sample.category / "test" / kind / name, mask


def _manifest_digest(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(
        "|".join(row[key] for key in sorted(row))
        for row in sorted(rows, key=lambda item: item["source_image"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_visa(config: VisaDataConfig, overwrite: bool = False) -> dict[str, object]:
    """Validate VisA and write padded, leakage-safe category folders."""
    samples = read_official_split(config)
    assignments = assign_validation(
        samples, list(config.categories), config.validation_ratio, config.seed
    )
    output = config.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, list[dict[str, str]]] = defaultdict(list)
    used_destinations: set[Path] = set()
    for sample in samples:
        source_image = _safe_source(config.raw_root, sample.image_relative)
        assigned_split = assignments[sample]
        destination, mask_destination = _destination(sample, assigned_split, output)
        if destination in used_destinations:
            raise ValueError(f"VisA filename collision after conversion: {destination}")
        used_destinations.add(destination)
        with Image.open(source_image) as opened:
            opened.verify()
        with Image.open(source_image) as opened:
            padded, metadata = letterbox_image(opened, config.image_size)
        destination.parent.mkdir(parents=True, exist_ok=True)
        padded.save(destination, format="PNG")

        source_mask = None
        if sample.mask_relative:
            source_mask = _safe_source(config.raw_root, sample.mask_relative)
            with Image.open(source_mask) as opened_mask:
                opened_mask.verify()
            with Image.open(source_mask) as opened_mask:
                padded_mask = letterbox_mask(opened_mask, metadata)
            if not padded_mask.getbbox():
                raise ValueError(f"Anomalous VisA mask is empty: {source_mask}")
            assert mask_destination is not None
            mask_destination.parent.mkdir(parents=True, exist_ok=True)
            padded_mask.save(mask_destination, format="PNG")

        manifests[sample.category].append(
            {
                "category": sample.category,
                "split": assigned_split,
                "official_split": sample.official_split,
                "label": sample.label,
                "source_image": sample.image_relative,
                "source_mask": sample.mask_relative or "",
                "processed_image": destination.relative_to(output).as_posix(),
                "processed_mask": (
                    mask_destination.relative_to(output).as_posix() if mask_destination else ""
                ),
                "image_sha256": sha256_file(source_image),
                "mask_sha256": sha256_file(source_mask) if source_mask else "",
            }
        )

    category_reports: dict[str, object] = {}
    for category, rows in manifests.items():
        manifest_path = output / category / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        counts = Counter((row["split"], row["label"]) for row in rows)
        category_reports[category] = {
            "counts": {
                f"{split}_{label}": count for (split, label), count in sorted(counts.items())
            },
            "manifest_sha256": sha256_file(manifest_path),
            "official_test_digest": _manifest_digest(
                [row for row in rows if row["official_split"] == "test"]
            ),
        }

    report: dict[str, object] = {
        "dataset": config.dataset_name,
        "source": "https://github.com/amazon-science/spot-diff",
        "license": "CC BY 4.0",
        "created_at": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "validation_ratio": config.validation_ratio,
        "image_size": config.image_size,
        "official_split_sha256": sha256_file(config.split_csv),
        "categories": category_reports,
        "test_policy": (
            "Official test rows are unchanged; no anomalous test data is used for calibration."
        ),
    }
    write_json(output / "metadata.json", report)
    write_json(config.report_dir / "visa_dataset_summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare selected official VisA one-class splits.")
    parser.add_argument("--config", default="configs/data/visa.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_visa_data_config(args.config)
    report = prepare_visa(config, overwrite=args.overwrite)
    print(f"Prepared VisA categories {', '.join(report['categories'])} in {config.output_dir}")


if __name__ == "__main__":
    main()
