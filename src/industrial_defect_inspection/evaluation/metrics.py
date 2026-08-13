"""Transparent detection matching, threshold selection, and metric extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GroundTruth:
    class_id: int
    bbox: BBox


@dataclass(frozen=True, slots=True)
class Prediction:
    class_id: int
    confidence: float
    bbox: BBox


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_path: Path
    ground_truth: tuple[GroundTruth, ...]
    predictions: tuple[Prediction, ...]


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[tuple[int, int, float], ...]
    unmatched_predictions: tuple[int, ...]
    unmatched_ground_truth: tuple[int, ...]


def box_iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_record(
    record: ImageRecord,
    confidence: float,
    iou_threshold: float,
) -> MatchResult:
    eligible = [
        index
        for index, prediction in enumerate(record.predictions)
        if prediction.confidence >= confidence
    ]
    eligible.sort(key=lambda index: record.predictions[index].confidence, reverse=True)
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    unmatched_predictions: list[int] = []
    for prediction_index in eligible:
        prediction = record.predictions[prediction_index]
        candidates = [
            (gt_index, box_iou(prediction.bbox, ground_truth.bbox))
            for gt_index, ground_truth in enumerate(record.ground_truth)
            if gt_index not in matched_gt and ground_truth.class_id == prediction.class_id
        ]
        if candidates:
            gt_index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= iou_threshold:
                matched_gt.add(gt_index)
                matches.append((prediction_index, gt_index, overlap))
                continue
        unmatched_predictions.append(prediction_index)
    unmatched_ground_truth = tuple(
        index for index in range(len(record.ground_truth)) if index not in matched_gt
    )
    return MatchResult(tuple(matches), tuple(unmatched_predictions), unmatched_ground_truth)


def _scores(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def operating_metrics(
    records: list[ImageRecord],
    confidence: float,
    iou_threshold: float,
    class_names: dict[int, str],
) -> dict[str, Any]:
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in class_names}
    for record in records:
        matched = match_record(record, confidence, iou_threshold)
        for prediction_index, _, _ in matched.matches:
            class_id = record.predictions[prediction_index].class_id
            counts.setdefault(class_id, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for prediction_index in matched.unmatched_predictions:
            class_id = record.predictions[prediction_index].class_id
            counts.setdefault(class_id, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for gt_index in matched.unmatched_ground_truth:
            class_id = record.ground_truth[gt_index].class_id
            counts.setdefault(class_id, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
    per_class = {
        class_names.get(class_id, str(class_id)): _scores(**class_counts)
        for class_id, class_counts in sorted(counts.items())
    }
    totals = {
        key: sum(int(values[key]) for values in per_class.values()) for key in ("tp", "fp", "fn")
    }
    return {
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "overall": _scores(**totals),
        "per_class": per_class,
    }


def select_operating_confidence(
    records: list[ImageRecord],
    iou_threshold: float,
    class_names: dict[int, str],
) -> tuple[float, list[dict[str, float | int]]]:
    sweep: list[dict[str, float | int]] = []
    for index in range(1, 100):
        threshold = index / 100
        overall = operating_metrics(records, threshold, iou_threshold, class_names)["overall"]
        sweep.append({"confidence": threshold, **overall})
    best = max(
        sweep,
        key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["confidence"])),
    )
    return float(best["confidence"]), sweep


def average_precision_by_class(
    metrics: Any, class_names: dict[int, str]
) -> dict[str, dict[str, float]]:
    box = metrics.box
    ap = getattr(box, "ap", None)
    ap50 = getattr(box, "ap50", None)
    class_indices = getattr(box, "ap_class_index", None)
    if ap is None or ap50 is None:
        return {name: {"ap50": 0.0, "map50_95": 0.0} for name in class_names.values()}
    ap_values = ap.tolist() if hasattr(ap, "tolist") else list(ap)
    ap50_values = ap50.tolist() if hasattr(ap50, "tolist") else list(ap50)
    indices = (
        class_indices.tolist()
        if hasattr(class_indices, "tolist")
        else list(class_indices or range(len(ap_values)))
    )
    result = {name: {"ap50": 0.0, "map50_95": 0.0} for name in class_names.values()}
    for class_id, map_value, ap50_value in zip(indices, ap_values, ap50_values, strict=True):
        result[class_names.get(int(class_id), str(class_id))] = {
            "ap50": float(ap50_value),
            "map50_95": float(map_value),
        }
    return result


def merge_per_class_metrics(
    operating: dict[str, Any],
    average_precision: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    return {
        name: {**values, **average_precision.get(name, {"ap50": 0.0, "map50_95": 0.0})}
        for name, values in operating["per_class"].items()
    }
