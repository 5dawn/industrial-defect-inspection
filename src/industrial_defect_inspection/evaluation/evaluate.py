"""Evaluate a frozen detector with traceable metrics and error analysis."""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from industrial_defect_inspection.config import EvaluationConfig, load_evaluation_config
from industrial_defect_inspection.evaluation.benchmark import benchmark_model
from industrial_defect_inspection.evaluation.dataset import dataset_images, load_ground_truth
from industrial_defect_inspection.evaluation.metrics import (
    BBox,
    ImageRecord,
    Prediction,
    average_precision_by_class,
    box_iou,
    match_record,
    merge_per_class_metrics,
    operating_metrics,
    select_operating_confidence,
)
from industrial_defect_inspection.utils.io import (
    environment_snapshot,
    file_record,
    write_json,
)
from industrial_defect_inspection.utils.runtime import (
    configure_run_logger,
    prepare_runtime,
    resolve_device,
)


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


def match_detections(
    ground_truth: list[tuple[int, BBox]],
    predictions: list[tuple[int, float, BBox]],
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Backward-compatible matching helper used by existing unit tests."""
    from industrial_defect_inspection.evaluation.metrics import GroundTruth

    record = ImageRecord(
        Path("fixture.jpg"),
        tuple(GroundTruth(class_id, bbox) for class_id, bbox in ground_truth),
        tuple(Prediction(class_id, confidence, bbox) for class_id, confidence, bbox in predictions),
    )
    result = match_record(record, 0.0, iou_threshold)
    return list(result.unmatched_predictions), list(result.unmatched_ground_truth)


def _extract_predictions(raw: Any) -> tuple[Prediction, ...]:
    if raw.boxes is None:
        return ()
    return tuple(
        Prediction(int(class_id), float(confidence), tuple(float(value) for value in bbox))
        for class_id, confidence, bbox in zip(
            raw.boxes.cls.detach().cpu().tolist(),
            raw.boxes.conf.detach().cpu().tolist(),
            raw.boxes.xyxy.detach().cpu().tolist(),
            strict=True,
        )
    )


def collect_records(
    model: Any,
    images: list[Path],
    device: str | int,
    metric_confidence: float,
    image_size: int,
) -> tuple[list[ImageRecord], dict[int, str]]:
    records: list[ImageRecord] = []
    names: dict[int, str] = {}
    for image_path in images:
        raw = model.predict(
            str(image_path),
            conf=metric_confidence,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]
        if isinstance(raw.names, dict):
            names = {int(key): str(value) for key, value in raw.names.items()}
        records.append(
            ImageRecord(image_path, load_ground_truth(image_path), _extract_predictions(raw))
        )
    return records, names


def _best_iou(box: BBox, candidates: list[BBox]) -> float:
    return max((box_iou(box, candidate) for candidate in candidates), default=0.0)


def _draw_errors(
    record: ImageRecord,
    unmatched_predictions: tuple[int, ...],
    unmatched_ground_truth: tuple[int, ...],
    names: dict[int, str],
    destination: Path,
) -> None:
    with Image.open(record.image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index in unmatched_ground_truth:
        ground_truth = record.ground_truth[index]
        draw.rectangle(ground_truth.bbox, outline=(255, 210, 0), width=3)
        draw.text(
            (ground_truth.bbox[0] + 2, ground_truth.bbox[1] + 2),
            f"FN {names.get(ground_truth.class_id, ground_truth.class_id)}",
            fill=(255, 210, 0),
            font=font,
        )
    for index in unmatched_predictions:
        prediction = record.predictions[index]
        draw.rectangle(prediction.bbox, outline=(255, 55, 55), width=3)
        draw.text(
            (prediction.bbox[0] + 2, max(0, prediction.bbox[1] - 12)),
            f"FP {names.get(prediction.class_id, prediction.class_id)} {prediction.confidence:.2f}",
            fill=(255, 55, 55),
            font=font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=92)


def write_error_analysis(
    records: list[ImageRecord],
    output_dir: Path,
    confidence: float,
    iou_threshold: float,
    names: dict[int, str],
    sample_limit: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    saved_fp = 0
    saved_fn = 0
    for record in records:
        result = match_record(record, confidence, iou_threshold)
        eligible_predictions = [
            prediction for prediction in record.predictions if prediction.confidence >= confidence
        ]
        for index in result.unmatched_predictions:
            prediction = record.predictions[index]
            rows.append(
                {
                    "image": record.image_path.name,
                    "error_type": "false_positive",
                    "class_id": prediction.class_id,
                    "class_name": names.get(prediction.class_id, str(prediction.class_id)),
                    "confidence": prediction.confidence,
                    "bbox_xyxy": list(prediction.bbox),
                    "best_iou": _best_iou(
                        prediction.bbox, [ground_truth.bbox for ground_truth in record.ground_truth]
                    ),
                }
            )
        for index in result.unmatched_ground_truth:
            ground_truth = record.ground_truth[index]
            rows.append(
                {
                    "image": record.image_path.name,
                    "error_type": "false_negative",
                    "class_id": ground_truth.class_id,
                    "class_name": names.get(ground_truth.class_id, str(ground_truth.class_id)),
                    "confidence": None,
                    "bbox_xyxy": list(ground_truth.bbox),
                    "best_iou": _best_iou(
                        ground_truth.bbox, [prediction.bbox for prediction in eligible_predictions]
                    ),
                }
            )
        if result.unmatched_predictions and saved_fp < sample_limit:
            _draw_errors(
                record,
                result.unmatched_predictions,
                result.unmatched_ground_truth,
                names,
                output_dir / "false_positives" / f"{record.image_path.stem}.jpg",
            )
            saved_fp += 1
        if result.unmatched_ground_truth and saved_fn < sample_limit:
            _draw_errors(
                record,
                result.unmatched_predictions,
                result.unmatched_ground_truth,
                names,
                output_dir / "false_negatives" / f"{record.image_path.stem}.jpg",
            )
            saved_fn += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image",
                "error_type",
                "class_id",
                "class_name",
                "confidence",
                "bbox_xyxy",
                "best_iou",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "manifest": str((output_dir / "errors.csv").resolve()),
        "false_positive_boxes": sum(row["error_type"] == "false_positive" for row in rows),
        "false_negative_boxes": sum(row["error_type"] == "false_negative" for row in rows),
        "saved_false_positive_images": saved_fp,
        "saved_false_negative_images": saved_fn,
    }


def write_threshold_sweep(
    sweep: list[dict[str, float | int]], output_dir: Path
) -> tuple[Path, Path]:
    csv_path = output_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["confidence"] for row in sweep], [row["f1"] for row in sweep], label="F1")
    axis.plot(
        [row["confidence"] for row in sweep],
        [row["precision"] for row in sweep],
        label="Precision",
    )
    axis.plot(
        [row["confidence"] for row in sweep],
        [row["recall"] for row in sweep],
        label="Recall",
    )
    axis.set(xlabel="confidence", ylabel="score", ylim=(0, 1), title="Validation threshold sweep")
    axis.legend()
    figure.tight_layout()
    figure_path = output_dir / "threshold_sweep.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return csv_path, figure_path


def _result_metric(results: dict[str, Any], name: str) -> float:
    for key, value in results.items():
        if key.casefold().replace("-", "").replace("_", "").endswith(name):
            return float(value)
    return 0.0


def evaluate(
    model_path: Path,
    dataset_yaml: Path,
    split: str,
    output_dir: Path,
    device: str,
    metric_confidence: float,
    operating_confidence: float | None,
    iou_threshold: float,
    image_size: int,
    error_samples: int,
    benchmark_count: int,
    benchmark_warmup: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    if split == "test" and operating_confidence is None:
        raise ValueError("Test evaluation requires a frozen operating confidence")
    report_path = output_dir / "evaluation.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"Evaluation exists: {report_path}; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_runtime(output_dir)
    logger = configure_run_logger("industrial_defect_inspection.evaluation", output_dir)
    started_clock = time.perf_counter()
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "stage": "evaluation",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "model": file_record(model_path),
        "dataset": file_record(dataset_yaml),
        "environment": environment_snapshot(),
    }
    write_json(manifest_path, manifest)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before evaluation") from exc

    try:
        resolved_device = resolve_device(device)
        logger.info("Collecting mAP metrics on %s with conf=%s", resolved_device, metric_confidence)
        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(dataset_yaml.resolve()),
            split=split,
            imgsz=image_size,
            conf=metric_confidence,
            device=resolved_device,
            project=str(output_dir.parent.resolve()),
            name=output_dir.name,
            exist_ok=True,
            plots=True,
            verbose=False,
        )
        images = dataset_images(dataset_yaml, split)
        records, names = collect_records(
            model, images, resolved_device, metric_confidence, image_size
        )
        selected_by_validation = operating_confidence is None
        sweep: list[dict[str, float | int]] | None = None
        if operating_confidence is None:
            operating_confidence, sweep = select_operating_confidence(records, iou_threshold, names)
            write_threshold_sweep(sweep, output_dir)
            logger.info("Selected validation operating confidence %.2f", operating_confidence)
        operating = operating_metrics(records, operating_confidence, iou_threshold, names)
        per_class = merge_per_class_metrics(operating, average_precision_by_class(metrics, names))
        errors = write_error_analysis(
            records,
            output_dir / "error_samples",
            operating_confidence,
            iou_threshold,
            names,
            error_samples,
        )
        latency = benchmark_model(
            model_path,
            images,
            str(resolved_device),
            operating_confidence,
            image_size,
            benchmark_count,
            benchmark_warmup,
        )
        raw_results = _json_value(metrics.results_dict)
        report: dict[str, Any] = {
            "model": file_record(model_path),
            "dataset": file_record(dataset_yaml),
            "split": split,
            "image_count": len(images),
            "metric_confidence": metric_confidence,
            "operating_confidence": operating_confidence,
            "operating_confidence_selected_on_this_split": selected_by_validation,
            "iou_threshold": iou_threshold,
            "image_size": image_size,
            "summary": {
                **operating["overall"],
                "map50": _result_metric(raw_results, "map50(b)"),
                "map50_95": _result_metric(raw_results, "map5095(b)"),
            },
            "per_class": per_class,
            "ultralytics_metrics": raw_results,
            "error_analysis": errors,
            "latency": latency,
            "environment": environment_snapshot(),
        }
        write_json(report_path, report)
        manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": time.perf_counter() - started_clock,
                "report": file_record(report_path),
            }
        )
        write_json(manifest_path, manifest)
        logger.info("Evaluation complete: %s", report_path.resolve())
        return report
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": time.perf_counter() - started_clock,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(manifest_path, manifest)
        logger.exception("Evaluation failed")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen detector checkpoint.")
    parser.add_argument("--config", default="configs/eval/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument(
        "--confidence", type=float, help="Frozen operating confidence (legacy-compatible name)."
    )
    parser.add_argument("--metric-confidence", type=float)
    parser.add_argument("--iou-threshold", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--error-samples", type=int)
    parser.add_argument("--benchmark-count", type=int)
    parser.add_argument("--benchmark-warmup", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def apply_cli_overrides(config: EvaluationConfig, args: argparse.Namespace) -> EvaluationConfig:
    overrides = {
        "model": Path(args.model) if args.model else None,
        "dataset": Path(args.data) if args.data else None,
        "output_dir": Path(args.output) if args.output else None,
        "split": args.split,
        "device": args.device,
        "operating_confidence": args.confidence,
        "metric_confidence": args.metric_confidence,
        "iou_threshold": args.iou_threshold,
        "image_size": args.image_size,
        "error_samples": args.error_samples,
        "benchmark_count": args.benchmark_count,
        "benchmark_warmup": args.benchmark_warmup,
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
        metric_confidence=config.metric_confidence,
        operating_confidence=config.operating_confidence,
        iou_threshold=config.iou_threshold,
        image_size=config.image_size,
        error_samples=config.error_samples,
        benchmark_count=config.benchmark_count,
        benchmark_warmup=config.benchmark_warmup,
        overwrite=args.overwrite,
    )
    print(f"Evaluation complete ({config.split}). Summary: {report['summary']}")


if __name__ == "__main__":
    main()
