"""Frozen-test evaluation and CPU resource benchmarking for anomaly localization."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PIL import Image

from industrial_defect_inspection.anomaly.engine import AnomalyEngine
from industrial_defect_inspection.config import (
    AnomalyEvaluationConfig,
    load_anomaly_evaluation_config,
    load_anomaly_inference_config,
)
from industrial_defect_inspection.utils.io import environment_snapshot, file_record, write_json


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute binary AUROC with tie-aware average ranks and no sklearn dependency."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    if len(values) != len(truth):
        raise ValueError("scores and labels must have the same length")
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[truth].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def mask_metrics(predicted: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(predicted, dtype=bool)
    truth = np.asarray(expected, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError(
            f"Mask shape mismatch: predicted={prediction.shape}, expected={truth.shape}"
        )
    intersection = int(np.logical_and(prediction, truth).sum())
    predicted_count = int(prediction.sum())
    expected_count = int(truth.sum())
    union = predicted_count + expected_count - intersection
    dice_denominator = predicted_count + expected_count
    dice = 1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _test_rows(category_root: Path) -> list[dict[str, str]]:
    manifest = category_root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Prepared VisA manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "test"]


def evaluate_category(
    engine: AnomalyEngine,
    dataset_dir: Path,
    category: str,
    benchmark_count: int,
    benchmark_warmup: int,
) -> dict[str, Any]:
    rows = _test_rows(dataset_dir / category)
    if not rows:
        raise ValueError(f"No frozen VisA test rows for category '{category}'")
    image_scores: list[float] = []
    image_labels: list[bool] = []
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    pixel_predictions: list[np.ndarray] = []
    normal_false_positives = 0

    for row in rows:
        image_path = dataset_dir / row["processed_image"]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result, visuals = engine.predict(image, category)
        anomalous = row["label"] == "anomaly"
        if anomalous:
            mask_path = dataset_dir / row["processed_mask"]
            with Image.open(mask_path) as opened_mask:
                expected = np.asarray(opened_mask.convert("L")) > 0
        else:
            expected = np.zeros((image.height, image.width), dtype=bool)
            normal_false_positives += int(result.is_anomalous)
        predicted = np.asarray(visuals.mask) > 0
        image_scores.append(result.anomaly_score)
        image_labels.append(anomalous)
        pixel_scores.append(visuals.anomaly_map)
        pixel_labels.append(expected)
        pixel_predictions.append(predicted)

    benchmark_rows = rows[: min(len(rows), benchmark_count)]
    warmup_row = rows[0]
    with Image.open(dataset_dir / warmup_row["processed_image"]) as opened:
        warmup_image = opened.convert("RGB")
    for _ in range(benchmark_warmup):
        engine.predict(warmup_image, category)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    latencies: list[float] = []
    for row in benchmark_rows:
        with Image.open(dataset_dir / row["processed_image"]) as opened:
            image = opened.convert("RGB")
        started = time.perf_counter()
        engine.predict(image, category)
        latencies.append((time.perf_counter() - started) * 1000.0)
        peak_rss = max(peak_rss, process.memory_info().rss)

    labels = np.asarray(image_labels, dtype=bool)
    normal_count = int((~labels).sum())
    pixel_score_values = np.concatenate([values.reshape(-1) for values in pixel_scores])
    pixel_truth_values = np.concatenate([values.reshape(-1) for values in pixel_labels])
    pixel_prediction_values = np.concatenate([values.reshape(-1) for values in pixel_predictions])
    pixel_dice, pixel_iou = mask_metrics(pixel_prediction_values, pixel_truth_values)
    return {
        "category": category,
        "test_images": len(rows),
        "image_auroc": binary_auroc(np.asarray(image_scores), labels),
        "pixel_auroc": binary_auroc(pixel_score_values, pixel_truth_values),
        "pixel_dice": pixel_dice,
        "pixel_iou": pixel_iou,
        "normal_test_fpr": normal_false_positives / normal_count if normal_count else None,
        "benchmark": {
            "device": str(engine.device),
            "batch_size": 1,
            "images": len(latencies),
            "warmups": benchmark_warmup,
            "mean_ms": statistics.fmean(latencies),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "fps": 1000.0 / statistics.fmean(latencies),
            "peak_process_rss_mb": peak_rss / (1024 * 1024),
        },
    }


def _macro_average(reports: list[dict[str, Any]], key: str) -> float | None:
    values = [float(report[key]) for report in reports if report[key] is not None]
    return statistics.fmean(values) if values else None


def evaluate(config: AnomalyEvaluationConfig) -> dict[str, Any]:
    if not config.dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Prepared VisA directory not found: {config.dataset_dir}. Run idi-prepare-visa first."
        )
    inference_config = load_anomaly_inference_config(config.inference_config)
    engine = AnomalyEngine(inference_config)
    reports = [
        evaluate_category(
            engine,
            config.dataset_dir,
            category,
            config.benchmark_count,
            config.benchmark_warmup,
        )
        for category in config.categories
    ]
    report: dict[str, Any] = {
        "status": "complete",
        "dataset": "VisA",
        "split": "official frozen test",
        "threshold_policy": "Frozen normal-only validation quantiles from checkpoint metadata.",
        "inference_config": file_record(config.inference_config),
        "per_category": reports,
        "macro_average": {
            key: _macro_average(reports, key)
            for key in ("image_auroc", "pixel_auroc", "pixel_dice", "pixel_iou", "normal_test_fpr")
        },
        "environment": environment_snapshot(),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "evaluation.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen VisA anomaly models.")
    parser.add_argument("--config", default="configs/anomaly/eval.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_anomaly_evaluation_config(args.config)
    report = evaluate(config)
    print(json.dumps(report["macro_average"], indent=2))


if __name__ == "__main__":
    main()
