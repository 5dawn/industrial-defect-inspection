"""Validation-only aggregate error analysis without publishing dataset pixels."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import yaml

from industrial_defect_inspection.evaluation.dataset import dataset_images, load_ground_truth
from industrial_defect_inspection.evaluation.metrics import BBox, box_iou
from industrial_defect_inspection.utils.io import file_record, write_json
from industrial_defect_inspection.utils.runtime import prepare_runtime

WEAK_CLASSES = ("crazing", "rolled-in_scale")
LOCALIZATION_IOU_FLOOR = 0.1
FAILURE_MODES = (
    "background_false_positive",
    "duplicate_prediction",
    "localization_failure",
    "misclassification",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_file_record(path: Path) -> dict[str, Any]:
    record = file_record(path)
    return {key: value for key, value in record.items() if key != "path"}


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    items = list(values)
    if not items:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": len(items),
        "mean": mean(items),
        "median": median(items),
        "p25": float(np.percentile(items, 25)),
        "p75": float(np.percentile(items, 75)),
    }


def _read_errors(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image", "error_type", "class_id", "class_name", "bbox_xyxy", "best_iou"}
    if not rows:
        return []
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Error manifest is missing columns: {sorted(missing)}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            bbox = tuple(float(value) for value in ast.literal_eval(row["bbox_xyxy"]))
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid bbox in error manifest for {row['image']}") from exc
        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox in error manifest for {row['image']}")
        parsed.append(
            {
                **row,
                "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]) if row.get("confidence") else None,
                "bbox_xyxy": bbox,
                "best_iou": float(row["best_iou"]),
            }
        )
    return parsed


def _class_names(dataset_yaml: Path) -> dict[int, str]:
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = payload.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(str(name) for name in names))
    return {int(class_id): str(name) for class_id, name in names.items()}


def _failure_mode(row: dict[str, Any], ground_truth: tuple[Any, ...], threshold: float) -> str:
    same_class = [gt for gt in ground_truth if gt.class_id == row["class_id"]]
    other_class = [gt for gt in ground_truth if gt.class_id != row["class_id"]]
    same_iou = max((box_iou(row["bbox_xyxy"], gt.bbox) for gt in same_class), default=0.0)
    other_iou = max((box_iou(row["bbox_xyxy"], gt.bbox) for gt in other_class), default=0.0)
    if same_iou >= threshold:
        return "duplicate_prediction"
    if other_iou >= threshold:
        return "misclassification"
    if LOCALIZATION_IOU_FLOOR <= same_iou < threshold:
        return "localization_failure"
    return "background_false_positive"


def _box_area(bbox: BBox, image_width: float, image_height: float) -> float:
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    return width * height / (image_width * image_height)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(
    output_dir: Path,
    failure_modes: Counter[str],
    area_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if failure_modes:
        figure, axis = plt.subplots(figsize=(7, 4))
        labels = list(failure_modes)
        axis.bar(labels, [failure_modes[label] for label in labels], color="#ff6b18")
        axis.set(title="Validation false-positive modes", ylabel="count")
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        figure.savefig(output_dir / "failure_modes.png", dpi=160)
        plt.close(figure)
    if area_rows:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(
            [row["area_bin"] for row in area_rows],
            [row["recall"] for row in area_rows],
            color="#138a54",
        )
        axis.set(title="Validation recall by ground-truth box area", ylabel="recall", ylim=(0, 1))
        figure.tight_layout()
        figure.savefig(output_dir / "recall_by_box_area.png", dpi=160)
        plt.close(figure)


def analyze_errors(
    evaluation_path: Path,
    errors_path: Path,
    dataset_yaml: Path,
    output_dir: Path,
) -> Path:
    evaluation = _read_json(evaluation_path)
    if evaluation.get("split") != "val":
        raise ValueError("Error analysis is validation-only; evaluation split must be val")
    if not evaluation.get("operating_confidence_selected_on_this_split"):
        raise ValueError("Validation report must contain a validation-selected threshold")
    threshold = float(evaluation["iou_threshold"])
    names = _class_names(dataset_yaml)
    rows = _read_errors(errors_path)
    images = {path.name: path for path in dataset_images(dataset_yaml, "val")}
    unknown_images = sorted({row["image"] for row in rows} - images.keys())
    if unknown_images:
        raise ValueError(
            f"Error manifest references images outside validation: {unknown_images[:3]}"
        )

    ground_truth_by_image = {name: load_ground_truth(path) for name, path in images.items()}
    failure_modes: Counter[str] = Counter({mode: 0 for mode in FAILURE_MODES})
    failure_modes_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_by_type: dict[str, list[float]] = defaultdict(list)
    iou_by_type: dict[str, list[float]] = defaultdict(list)
    false_negative_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        error_type = str(row["error_type"])
        if row["confidence"] is not None:
            confidence_by_type[error_type].append(float(row["confidence"]))
        iou_by_type[error_type].append(float(row["best_iou"]))
        if error_type == "false_positive":
            mode = _failure_mode(row, ground_truth_by_image[row["image"]], threshold)
            failure_modes[mode] += 1
            failure_modes_by_class[row["class_name"]][mode] += 1
        elif error_type == "false_negative":
            false_negative_rows[row["image"]].append(row)

    ground_truth_rows: list[dict[str, Any]] = []
    for image_name, image_path in images.items():
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        unmatched_fn = list(false_negative_rows.get(image_name, []))
        used_fn: set[int] = set()
        for gt in ground_truth_by_image[image_name]:
            candidates = [
                (index, box_iou(gt.bbox, row["bbox_xyxy"]))
                for index, row in enumerate(unmatched_fn)
                if index not in used_fn and row["class_id"] == gt.class_id
            ]
            fn_match = max(candidates, key=lambda item: item[1]) if candidates else None
            is_fn = bool(fn_match and fn_match[1] >= 0.999)
            if is_fn and fn_match is not None:
                used_fn.add(fn_match[0])
            ground_truth_rows.append(
                {
                    "class_name": names.get(gt.class_id, str(gt.class_id)),
                    "area": _box_area(gt.bbox, width, height),
                    "recalled": not is_fn,
                }
            )

    areas = [row["area"] for row in ground_truth_rows]
    boundaries = [float(value) for value in np.percentile(areas, [25, 50, 75])] if areas else []
    labels = ("Q1-small", "Q2", "Q3", "Q4-large")
    area_counts = {label: {"total": 0, "tp": 0, "fn": 0} for label in labels}
    for row in ground_truth_rows:
        index = int(np.searchsorted(boundaries, row["area"], side="right"))
        bucket = area_counts[labels[index]]
        bucket["total"] += 1
        bucket["tp" if row["recalled"] else "fn"] += 1
    area_rows = [
        {
            "area_bin": label,
            **values,
            "recall": values["tp"] / values["total"] if values["total"] else 0.0,
        }
        for label, values in area_counts.items()
    ]

    per_class_rows: list[dict[str, Any]] = []
    per_class_distributions: dict[str, dict[str, Any]] = {}
    for class_name, metrics in evaluation["per_class"].items():
        for mode in FAILURE_MODES:
            failure_modes_by_class[class_name][mode] += 0
        class_rows = [row for row in rows if row["class_name"] == class_name]
        fp_confidence = [
            row["confidence"]
            for row in class_rows
            if row["error_type"] == "false_positive" and row["confidence"] is not None
        ]
        tp_confidence = [
            row["confidence"]
            for row in class_rows
            if row["error_type"] == "true_positive" and row["confidence"] is not None
        ]
        per_class_rows.append(
            {
                "class_name": class_name,
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "ap50": metrics["ap50"],
                "map50_95": metrics["map50_95"],
                "tp_confidence_mean": mean(tp_confidence) if tp_confidence else None,
                "fp_confidence_mean": mean(fp_confidence) if fp_confidence else None,
            }
        )
        per_class_distributions[class_name] = {
            "confidence": {
                error_type: _summary(
                    float(row["confidence"])
                    for row in class_rows
                    if row["error_type"] == error_type and row["confidence"] is not None
                )
                for error_type in ("true_positive", "false_positive", "false_negative")
            },
            "matching_iou": {
                error_type: _summary(
                    float(row["best_iou"]) for row in class_rows if row["error_type"] == error_type
                )
                for error_type in ("true_positive", "false_positive", "false_negative")
            },
        }

    weak_summary = {
        class_name: next((row for row in per_class_rows if row["class_name"] == class_name), None)
        for class_name in WEAK_CLASSES
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_class.csv", per_class_rows)
    _write_csv(output_dir / "box_area_recall.csv", area_rows)
    failure_rows = [
        {"class_name": class_name, "failure_mode": mode, "count": count}
        for class_name, counts in sorted(failure_modes_by_class.items())
        for mode, count in sorted(counts.items())
    ]
    _write_csv(output_dir / "failure_modes.csv", failure_rows)
    _write_figures(output_dir, failure_modes, area_rows)
    result_path = output_dir / "analysis.json"
    write_json(
        result_path,
        {
            "split": "val",
            "operating_confidence": evaluation["operating_confidence"],
            "iou_threshold": threshold,
            "failure_mode_definition": {
                "localization_iou_floor": LOCALIZATION_IOU_FLOOR,
                "match_iou_threshold": threshold,
            },
            "sources": {
                "evaluation": _portable_file_record(evaluation_path),
                "errors": _portable_file_record(errors_path),
                "dataset": _portable_file_record(dataset_yaml),
            },
            "per_class": {row["class_name"]: row for row in per_class_rows},
            "per_class_distributions": per_class_distributions,
            "confidence_distribution": {
                error_type: _summary(values)
                for error_type, values in sorted(confidence_by_type.items())
            },
            "iou_distribution": {
                error_type: _summary(values) for error_type, values in sorted(iou_by_type.items())
            },
            "failure_modes": dict(sorted(failure_modes.items())),
            "failure_modes_by_class": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(failure_modes_by_class.items())
            },
            "box_area_quartiles": {"boundaries": boundaries, "rows": area_rows},
            "weak_classes": weak_summary,
            "dataset_pixels_published": False,
        },
    )
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze validation detection errors.")
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_runtime(Path(args.output))
    result = analyze_errors(
        Path(args.evaluation),
        Path(args.errors),
        Path(args.dataset),
        Path(args.output),
    )
    print(f"Validation error analysis: {result.resolve()}")


if __name__ == "__main__":
    main()
