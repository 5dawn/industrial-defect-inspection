"""Evaluate a frozen checkpoint on a named dataset split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

from industrial_defect_inspection.config import EvaluationConfig, load_evaluation_config
from industrial_defect_inspection.data.prepare import IMAGE_SUFFIXES
from industrial_defect_inspection.training.train import resolve_device
from industrial_defect_inspection.utils.io import environment_snapshot, write_json


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dataset_images(dataset_yaml: Path, split: str) -> list[Path]:
    with dataset_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if split not in data:
        raise KeyError(f"Split '{split}' is not present in {dataset_yaml}")
    root = Path(data.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    value = data[split]
    values = value if isinstance(value, list) else [value]
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
        elif path.suffix.casefold() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                if line.strip():
                    images.append(candidate if candidate.is_absolute() else root / candidate)
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"Dataset split path not found: {path}")
    return sorted(set(image.resolve() for image in images))


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indices = [index for index, part in enumerate(parts) if part.casefold() == "images"]
    if not indices:
        raise ValueError(f"Cannot infer label path from image path: {image_path}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_ground_truth(image_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    label_path = _label_path(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        class_id = int(values[0])
        x_center, y_center, box_width, box_height = (float(value) for value in values[1:])
        xmin = (x_center - box_width / 2) * width
        ymin = (y_center - box_height / 2) * height
        xmax = (x_center + box_width / 2) * width
        ymax = (y_center + box_height / 2) * height
        boxes.append((class_id, (xmin, ymin, xmax, ymax)))
    return boxes


def box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    ground_truth: list[tuple[int, tuple[float, float, float, float]]],
    predictions: list[tuple[int, float, tuple[float, float, float, float]]],
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Return unmatched prediction and ground-truth indices."""

    matched_gt: set[int] = set()
    unmatched_predictions: list[int] = []
    for prediction_index in sorted(
        range(len(predictions)), key=lambda index: predictions[index][1], reverse=True
    ):
        class_id, _, predicted_box = predictions[prediction_index]
        candidates = [
            (gt_index, box_iou(predicted_box, gt_box))
            for gt_index, (gt_class, gt_box) in enumerate(ground_truth)
            if gt_index not in matched_gt and gt_class == class_id
        ]
        if candidates:
            best_index, best_iou = max(candidates, key=lambda item: item[1])
            if best_iou >= iou_threshold:
                matched_gt.add(best_index)
                continue
        unmatched_predictions.append(prediction_index)
    unmatched_ground_truth = [
        index for index in range(len(ground_truth)) if index not in matched_gt
    ]
    return unmatched_predictions, unmatched_ground_truth


def _draw_error_sample(
    image_path: Path,
    ground_truth: list[tuple[int, tuple[float, float, float, float]]],
    predictions: list[tuple[int, float, tuple[float, float, float, float]]],
    names: dict[int, str],
    destination: Path,
) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for class_id, box in ground_truth:
        draw.rectangle(box, outline=(255, 210, 0), width=2)
        draw.text(
            (box[0] + 2, box[1] + 2),
            f"GT {names.get(class_id, class_id)}",
            fill=(255, 210, 0),
            font=font,
        )
    for class_id, score, box in predictions:
        draw.rectangle(box, outline=(255, 55, 55), width=2)
        draw.text(
            (box[0] + 2, max(0, box[1] - 12)),
            f"P {names.get(class_id, class_id)} {score:.2f}",
            fill=(255, 55, 55),
            font=font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=92)


def error_analysis(
    model: Any,
    images: list[Path],
    output_dir: Path,
    device: str | int,
    confidence: float,
    image_size: int,
    sample_limit: int,
) -> dict[str, int]:
    false_positive_images = 0
    false_negative_images = 0
    false_positive_boxes = 0
    false_negative_boxes = 0
    for image_path in images:
        raw = model.predict(
            source=str(image_path),
            conf=confidence,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]
        ground_truth = load_ground_truth(image_path)
        predictions: list[tuple[int, float, tuple[float, float, float, float]]] = []
        if raw.boxes is not None:
            for cls, conf, xyxy in zip(
                raw.boxes.cls.detach().cpu().tolist(),
                raw.boxes.conf.detach().cpu().tolist(),
                raw.boxes.xyxy.detach().cpu().tolist(),
                strict=True,
            ):
                predictions.append((int(cls), float(conf), tuple(float(v) for v in xyxy)))
        unmatched_predictions, unmatched_ground_truth = match_detections(ground_truth, predictions)
        false_positive_boxes += len(unmatched_predictions)
        false_negative_boxes += len(unmatched_ground_truth)
        names = raw.names if isinstance(raw.names, dict) else {}
        if unmatched_predictions and false_positive_images < sample_limit:
            _draw_error_sample(
                image_path,
                ground_truth,
                predictions,
                names,
                output_dir / "false_positives" / f"{image_path.stem}.jpg",
            )
            false_positive_images += 1
        if unmatched_ground_truth and false_negative_images < sample_limit:
            _draw_error_sample(
                image_path,
                ground_truth,
                predictions,
                names,
                output_dir / "false_negatives" / f"{image_path.stem}.jpg",
            )
            false_negative_images += 1
    return {
        "false_positive_boxes": false_positive_boxes,
        "false_negative_boxes": false_negative_boxes,
        "saved_false_positive_images": false_positive_images,
        "saved_false_negative_images": false_negative_images,
    }


def latency_benchmark(
    model: Any,
    images: list[Path],
    device: str | int,
    confidence: float,
    image_size: int,
    count: int,
) -> dict[str, Any]:
    selected = images[:count]
    if not selected:
        return {"count": 0}
    for image_path in selected[: min(3, len(selected))]:
        model.predict(str(image_path), device=device, imgsz=image_size, verbose=False)
    totals: list[float] = []
    components = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
    for image_path in selected:
        raw = model.predict(
            str(image_path),
            conf=confidence,
            device=device,
            imgsz=image_size,
            verbose=False,
        )[0]
        speed = raw.speed or {}
        values = [
            max(0.0, float(speed.get("preprocess", 0.0))),
            max(0.0, float(speed.get("inference", 0.0))),
            max(0.0, float(speed.get("postprocess", 0.0))),
        ]
        components["preprocess_ms"].append(values[0])
        components["inference_ms"].append(values[1])
        components["postprocess_ms"].append(values[2])
        totals.append(sum(values))
    ordered = sorted(totals)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(selected),
        "p50_total_ms": median(ordered),
        "p95_total_ms": ordered[p95_index],
        "mean_preprocess_ms": sum(components["preprocess_ms"]) / len(selected),
        "mean_inference_ms": sum(components["inference_ms"]) / len(selected),
        "mean_postprocess_ms": sum(components["postprocess_ms"]) / len(selected),
    }


def evaluate(
    model_path: Path,
    dataset_yaml: Path,
    split: str,
    output_dir: Path,
    device: str,
    confidence: float,
    image_size: int,
    error_samples: int,
    benchmark_count: int,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before evaluation") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(dataset_yaml.resolve()),
        split=split,
        imgsz=image_size,
        conf=confidence,
        device=resolved_device,
        project=str(output_dir.parent.resolve()),
        name=output_dir.name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    images = dataset_images(dataset_yaml, split)
    names = metrics.names if isinstance(metrics.names, dict) else {}
    maps = metrics.box.maps.tolist() if hasattr(metrics.box.maps, "tolist") else metrics.box.maps
    report: dict[str, Any] = {
        "model": str(model_path.resolve()),
        "dataset": str(dataset_yaml.resolve()),
        "split": split,
        "confidence": confidence,
        "image_size": image_size,
        "metrics": _json_value(metrics.results_dict),
        "per_class_map50_95": {
            str(names.get(index, index)): float(value) for index, value in enumerate(maps)
        },
        "error_analysis": error_analysis(
            model,
            images,
            output_dir / "error_samples",
            resolved_device,
            confidence,
            image_size,
            error_samples,
        ),
        "latency": latency_benchmark(
            model,
            images,
            resolved_device,
            confidence,
            image_size,
            benchmark_count,
        ),
        "environment": environment_snapshot(),
    }
    write_json(output_dir / "evaluation.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen detector checkpoint.")
    parser.add_argument(
        "--config",
        default="configs/eval/default.yaml",
        help="Path to the evaluation YAML configuration.",
    )
    parser.add_argument("--model", help="Override the model path from the configuration.")
    parser.add_argument("--data", help="Override the dataset YAML path.")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--output", help="Override the evaluation output directory.")
    parser.add_argument("--device")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--error-samples", type=int)
    parser.add_argument("--benchmark-count", type=int)
    return parser


def apply_cli_overrides(config: EvaluationConfig, args: argparse.Namespace) -> EvaluationConfig:
    overrides = {
        "model": Path(args.model) if args.model else None,
        "dataset": Path(args.data) if args.data else None,
        "output_dir": Path(args.output) if args.output else None,
        "split": args.split,
        "device": args.device,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "error_samples": args.error_samples,
        "benchmark_count": args.benchmark_count,
    }
    payload = config.model_dump()
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return EvaluationConfig.model_validate(payload)


def main() -> None:
    args = build_parser().parse_args()
    config = apply_cli_overrides(load_evaluation_config(args.config), args)
    report = evaluate(
        model_path=config.model,
        dataset_yaml=config.dataset,
        split=config.split,
        output_dir=config.output_dir,
        device=config.device,
        confidence=config.confidence,
        image_size=config.image_size,
        error_samples=config.error_samples,
        benchmark_count=config.benchmark_count,
    )
    metrics = report["metrics"]
    print(f"Evaluation complete ({config.split}). Metrics: {metrics}")


if __name__ == "__main__":
    main()
