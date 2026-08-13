import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from industrial_defect_inspection.evaluation.error_analysis import (
    _failure_mode,
    analyze_errors,
)
from industrial_defect_inspection.evaluation.metrics import GroundTruth


def write_fixture(tmp_path: Path, split: str = "val") -> tuple[Path, Path, Path]:
    root = tmp_path / "dataset"
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("L", (100, 100), 128).save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.25 0.25 0.20 0.20\n1 0.75 0.75 0.20 0.20\n", encoding="utf-8"
    )
    dataset = root / "dataset.yaml"
    dataset.write_text(
        f"path: {root.as_posix()}\nval: images/val\nnames:\n"
        "  0: crazing\n  1: rolled-in_scale\n  2: scratches\n",
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "split": split,
                "operating_confidence": 0.43,
                "operating_confidence_selected_on_this_split": split == "val",
                "iou_threshold": 0.5,
                "per_class": {
                    "crazing": {
                        "tp": 1,
                        "fp": 1,
                        "fn": 0,
                        "precision": 0.5,
                        "recall": 1.0,
                        "f1": 2 / 3,
                        "ap50": 0.8,
                        "map50_95": 0.4,
                    },
                    "rolled-in_scale": {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "ap50": 0.2,
                        "map50_95": 0.1,
                    },
                    "scratches": {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "ap50": 0.0,
                        "map50_95": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    errors = tmp_path / "errors.csv"
    with errors.open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerows(
            [
                {
                    "image": "sample.jpg",
                    "error_type": "true_positive",
                    "class_id": 0,
                    "class_name": "crazing",
                    "confidence": 0.8,
                    "bbox_xyxy": "[15, 15, 35, 35]",
                    "best_iou": 1.0,
                },
                {
                    "image": "sample.jpg",
                    "error_type": "false_positive",
                    "class_id": 0,
                    "class_name": "crazing",
                    "confidence": 0.7,
                    "bbox_xyxy": "[65, 65, 85, 85]",
                    "best_iou": 1.0,
                },
                {
                    "image": "sample.jpg",
                    "error_type": "false_negative",
                    "class_id": 1,
                    "class_name": "rolled-in_scale",
                    "confidence": "",
                    "bbox_xyxy": "[65, 65, 85, 85]",
                    "best_iou": 1.0,
                },
            ]
        )
    return evaluation, errors, dataset


def test_analyze_errors_writes_aggregate_reports(tmp_path: Path) -> None:
    evaluation, errors, dataset = write_fixture(tmp_path)
    result = analyze_errors(evaluation, errors, dataset, tmp_path / "analysis")
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert payload["failure_modes"] == {
        "background_false_positive": 0,
        "duplicate_prediction": 0,
        "localization_failure": 0,
        "misclassification": 1,
    }
    assert payload["confidence_distribution"]["true_positive"]["mean"] == 0.8
    assert (
        payload["per_class_distributions"]["crazing"]["matching_iou"]["false_positive"]["mean"]
        == 1.0
    )
    assert (
        payload["per_class_distributions"]["rolled-in_scale"]["confidence"]["false_positive"][
            "count"
        ]
        == 0
    )
    assert (
        payload["per_class_distributions"]["scratches"]["matching_iou"]["true_positive"]["count"]
        == 0
    )
    assert sum(row["fn"] for row in payload["box_area_quartiles"]["rows"]) == 1
    assert payload["dataset_pixels_published"] is False
    assert all("path" not in record for record in payload["sources"].values())
    assert (tmp_path / "analysis" / "per_class.csv").is_file()
    assert (tmp_path / "analysis" / "failure_modes.png").is_file()


def test_failure_mode_covers_duplicate_localization_and_background() -> None:
    ground_truth = (GroundTruth(0, (0, 0, 10, 10)),)
    assert (
        _failure_mode({"class_id": 0, "bbox_xyxy": (0, 0, 10, 10)}, ground_truth, 0.5)
        == "duplicate_prediction"
    )
    assert (
        _failure_mode({"class_id": 0, "bbox_xyxy": (5, 0, 15, 10)}, ground_truth, 0.5)
        == "localization_failure"
    )
    assert (
        _failure_mode({"class_id": 0, "bbox_xyxy": (20, 20, 30, 30)}, ground_truth, 0.5)
        == "background_false_positive"
    )


def test_analyze_errors_refuses_test_split(tmp_path: Path) -> None:
    evaluation, errors, dataset = write_fixture(tmp_path, split="test")
    with pytest.raises(ValueError, match="validation-only"):
        analyze_errors(evaluation, errors, dataset, tmp_path / "analysis")
