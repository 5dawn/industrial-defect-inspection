"""Dataset access helpers shared by evaluation and benchmark commands."""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from industrial_defect_inspection.data.prepare import IMAGE_SUFFIXES
from industrial_defect_inspection.evaluation.metrics import GroundTruth


def dataset_images(dataset_yaml: Path, split: str) -> list[Path]:
    with dataset_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if split not in data:
        raise KeyError(f"Split '{split}' is not present in {dataset_yaml}")
    root = Path(data.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    values = data[split] if isinstance(data[split], list) else [data[split]]
    images: list[Path] = []
    for item in values:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            images.extend(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.casefold() in IMAGE_SUFFIXES
            )
        elif path.suffix.casefold() == ".txt" and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    candidate = Path(line.strip())
                    images.append(candidate if candidate.is_absolute() else root / candidate)
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"Dataset split path not found: {path}")
    return sorted(set(image.resolve() for image in images))


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indices = [index for index, part in enumerate(parts) if part.casefold() == "images"]
    if not indices:
        raise ValueError(f"Cannot infer label path from image path: {image_path}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_ground_truth(image_path: Path) -> tuple[GroundTruth, ...]:
    annotation_path = label_path(image_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Label file not found: {annotation_path}")
    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[GroundTruth] = []
    for line_number, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label at {annotation_path}:{line_number}")
        class_id = int(values[0])
        x_center, y_center, box_width, box_height = (float(value) for value in values[1:])
        boxes.append(
            GroundTruth(
                class_id=class_id,
                bbox=(
                    (x_center - box_width / 2) * width,
                    (y_center - box_height / 2) * height,
                    (x_center + box_width / 2) * width,
                    (y_center + box_height / 2) * height,
                ),
            )
        )
    return tuple(boxes)
