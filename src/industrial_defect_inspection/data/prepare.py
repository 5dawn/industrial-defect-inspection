"""Prepare NEU-DET style VOC data for reproducible YOLO training."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from industrial_defect_inspection.config import DataConfig, load_data_config
from industrial_defect_inspection.data.voc import VocAnnotation, parse_voc_annotation
from industrial_defect_inspection.utils.io import sha256_file, write_json

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class Sample:
    stem: str
    image_path: Path
    annotation_path: Path
    annotation: VocAnnotation
    image_sha256: str

    @property
    def stratum(self) -> str:
        return "|".join(sorted({box.class_name for box in self.annotation.boxes}))

    def split_stratum(self, strategy: str) -> str:
        if strategy == "filename_prefix":
            prefix, separator, suffix = self.stem.rpartition("_")
            if not separator or not prefix or not suffix:
                raise ValueError(
                    f"Cannot derive filename-prefix stratum from sample stem '{self.stem}'"
                )
            return prefix
        return self.stratum


def _index_by_stem(paths: Iterable[Path], kind: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in paths:
        key = path.stem.casefold()
        if key in index:
            raise ValueError(f"Duplicate {kind} stem '{path.stem}': {index[key]} and {path}")
        index[key] = path
    return index


def discover_pairs(config: DataConfig) -> tuple[dict[str, Path], dict[str, Path]]:
    if not config.raw_images_dir.is_dir() or not config.raw_annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Raw data for '{config.dataset_name}' is incomplete. Expected image directory: "
            f"{config.raw_images_dir}; expected Pascal VOC XML directory: "
            f"{config.raw_annotations_dir}. Follow data/README.md, then run "
            "'idi-prepare --config configs/data/neu_det.yaml'."
        )
    images = _index_by_stem(
        (
            path
            for path in config.raw_images_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        "image",
    )
    annotations = _index_by_stem(
        (path for path in config.raw_annotations_dir.rglob("*.xml") if path.is_file()),
        "annotation",
    )
    missing_annotations = sorted(set(images) - set(annotations))
    missing_images = sorted(set(annotations) - set(images))
    if missing_annotations or missing_images:
        parts = []
        if missing_annotations:
            parts.append(f"images without XML: {missing_annotations[:10]}")
        if missing_images:
            parts.append(f"XML without images: {missing_images[:10]}")
        raise ValueError("Unpaired dataset files; " + "; ".join(parts))
    if not images:
        raise ValueError("No supported images found")
    return images, annotations


def load_samples(config: DataConfig) -> list[Sample]:
    images, annotations = discover_pairs(config)
    samples: list[Sample] = []
    for key in sorted(images):
        annotation = parse_voc_annotation(
            annotations[key], config.class_names, config.class_aliases
        )
        with Image.open(images[key]) as image:
            actual_size = image.size
            image.verify()
        if config.strict_image_size and actual_size != (annotation.width, annotation.height):
            raise ValueError(
                f"Image/XML size mismatch for {images[key]}: image={actual_size}, "
                f"xml={(annotation.width, annotation.height)}"
            )
        digest = sha256_file(images[key])
        samples.append(
            Sample(
                stem=images[key].stem,
                image_path=images[key],
                annotation_path=annotations[key],
                annotation=annotation,
                image_sha256=digest,
            )
        )
    return samples


def _counts_for_group(size: int, train: float, val: float) -> tuple[int, int, int]:
    train_count = round(size * train)
    val_count = round(size * val)
    if train_count + val_count > size:
        val_count = size - train_count
    return train_count, val_count, size - train_count - val_count


def stratified_split(
    samples: list[Sample],
    train: float,
    val: float,
    seed: int,
    stratify_by: str = "annotation_set",
) -> dict[str, list[Sample]]:
    """Split by label set while keeping exact duplicate images in one split."""

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.split_stratum(stratify_by)].append(sample)

    rng = random.Random(seed)
    split: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for group_name in sorted(groups):
        group = sorted(groups[group_name], key=lambda item: item.stem.casefold())
        content_groups: dict[str, list[Sample]] = defaultdict(list)
        for sample in group:
            content_groups[sample.image_sha256].append(sample)
        units = list(content_groups.values())
        rng.shuffle(units)
        units.sort(key=len, reverse=True)

        train_count, val_count, test_count = _counts_for_group(len(group), train, val)
        remaining = {"train": train_count, "val": val_count, "test": test_count}
        for unit in units:
            candidates = [name for name, capacity in remaining.items() if capacity >= len(unit)]
            if not candidates:
                raise RuntimeError(
                    f"Cannot keep duplicate group of {len(unit)} samples together in stratum "
                    f"'{group_name}' with remaining capacities {remaining}"
                )
            destination = max(candidates, key=lambda name: remaining[name])
            split[destination].extend(unit)
            remaining[destination] -= len(unit)
        if any(remaining.values()):
            raise RuntimeError(
                f"Split allocation failed for stratum '{group_name}': remaining {remaining}"
            )

    for name in split:
        split[name].sort(key=lambda item: item.stem.casefold())
    all_stems = [sample.stem.casefold() for values in split.values() for sample in values]
    if len(all_stems) != len(set(all_stems)) or len(all_stems) != len(samples):
        raise RuntimeError("Split invariant failed: samples overlap or are missing")
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, values in split.items():
        for sample in values:
            hash_splits[sample.image_sha256].add(split_name)
    leaked_hashes = [digest for digest, names in hash_splits.items() if len(names) > 1]
    if leaked_hashes:
        raise RuntimeError(f"Duplicate-content leakage across splits: {leaked_hashes[:5]}")
    return split


def _duplicate_content_groups(samples: list[Sample]) -> list[dict[str, object]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.image_sha256].append(sample)
    result: list[dict[str, object]] = []
    for digest, values in sorted(groups.items()):
        if len(values) < 2:
            continue
        annotation_sets = {sample.annotation.boxes for sample in values}
        result.append(
            {
                "sha256": digest,
                "stems": sorted(sample.stem for sample in values),
                "annotations_identical": len(annotation_sets) == 1,
            }
        )
    return result


def _annotation_audit(samples: list[Sample]) -> dict[str, object]:
    affected = [sample for sample in samples if sample.annotation.duplicate_boxes_removed]
    return {
        "duplicate_boxes_removed": sum(
            sample.annotation.duplicate_boxes_removed for sample in affected
        ),
        "files_with_duplicate_boxes": [sample.annotation_path.name for sample in affected],
    }


def _render_preview(sample: Sample, destination: Path) -> None:
    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box in sample.annotation.boxes:
        coords = (box.xmin, box.ymin, box.xmax, box.ymax)
        draw.rectangle(coords, outline=(255, 55, 55), width=2)
        draw.text(
            (box.xmin + 2, max(0, box.ymin - 12)),
            box.class_name,
            fill=(255, 220, 0),
            font=font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=92)


def _dataset_statistics(samples: list[Sample], class_names: list[str]) -> dict[str, object]:
    image_classes = Counter()
    box_classes = Counter()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    for sample in samples:
        image_classes.update({box.class_name for box in sample.annotation.boxes})
        for box in sample.annotation.boxes:
            box_classes[box.class_name] += 1
            width = box.width / sample.annotation.width
            height = box.height / sample.annotation.height
            widths.append(width)
            heights.append(height)
            areas.append(width * height)

    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        if not ordered:
            return {"min": 0.0, "median": 0.0, "max": 0.0}
        return {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
        }

    return {
        "images": len(samples),
        "boxes": sum(box_classes.values()),
        "images_per_class": {name: image_classes[name] for name in class_names},
        "boxes_per_class": {name: box_classes[name] for name in class_names},
        "normalized_box_width": summary(widths),
        "normalized_box_height": summary(heights),
        "normalized_box_area": summary(areas),
    }


def prepare_dataset(config: DataConfig, overwrite: bool = False) -> dict[str, object]:
    samples = load_samples(config)
    output = config.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    split = stratified_split(
        samples,
        train=config.split.train,
        val=config.split.val,
        seed=config.seed,
        stratify_by=config.stratify_by,
    )
    class_ids = {name: index for index, name in enumerate(config.class_names)}
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, object] = {}
    for split_name, split_samples in split.items():
        image_dir = output / "images" / split_name
        label_dir = output / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        manifest_lines: list[str] = []
        for sample in split_samples:
            destination = image_dir / sample.image_path.name
            shutil.copy2(sample.image_path, destination)
            label_lines = [
                box.to_yolo(
                    class_ids[box.class_name],
                    sample.annotation.width,
                    sample.annotation.height,
                )
                for box in sample.annotation.boxes
            ]
            (label_dir / f"{sample.stem}.txt").write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8"
            )
            manifest_lines.append(destination.resolve().as_posix())
        (manifest_dir / f"{split_name}.txt").write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )
        split_stats[split_name] = _dataset_statistics(split_samples, config.class_names)

    dataset_yaml = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(config.class_names)},
    }
    with (output / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, allow_unicode=True, sort_keys=False)

    for index, sample in enumerate(samples[: config.preview_count]):
        _render_preview(sample, config.report_dir / f"{index:03d}_{sample.stem}.jpg")

    metadata: dict[str, object] = {
        "dataset": config.dataset_name,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "class_names": config.class_names,
        "source_statistics": _dataset_statistics(samples, config.class_names),
        "duplicate_content_groups": _duplicate_content_groups(samples),
        "annotation_audit": _annotation_audit(samples),
        "splits": split_stats,
        "files": [
            {
                "stem": sample.stem,
                "sha256": sample.image_sha256,
                "stratum": sample.split_stratum(config.stratify_by),
                "annotation_classes": sample.stratum.split("|"),
            }
            for sample in samples
        ],
    }
    write_json(output / "metadata.json", metadata)
    write_json(config.report_dir / "dataset_report.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Pascal VOC annotations and prepare a YOLO dataset."
    )
    parser.add_argument("--config", required=True, help="Path to the data YAML configuration.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the configured processed-data directory if it is non-empty.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_data_config(args.config)
    metadata = prepare_dataset(config, overwrite=args.overwrite)
    print(
        f"Prepared {metadata['source_statistics']['images']} images in "
        f"{config.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
