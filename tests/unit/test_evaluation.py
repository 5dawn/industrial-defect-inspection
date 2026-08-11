import csv
from pathlib import Path

import pytest
from PIL import Image

from industrial_defect_inspection.evaluation.evaluate import (
    box_iou,
    match_detections,
    write_error_analysis,
)
from industrial_defect_inspection.evaluation.metrics import GroundTruth, ImageRecord, Prediction


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_match_detections_counts_class_and_location_errors() -> None:
    ground_truth = [(0, (0.0, 0.0, 10.0, 10.0)), (1, (20.0, 20.0, 30.0, 30.0))]
    predictions = [
        (0, 0.9, (0.0, 0.0, 10.0, 10.0)),
        (0, 0.8, (20.0, 20.0, 30.0, 30.0)),
    ]

    false_positives, false_negatives = match_detections(ground_truth, predictions)

    assert false_positives == [1]
    assert false_negatives == [1]


def test_error_analysis_writes_machine_readable_manifest(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (40, 40), "gray").save(image_path)
    record = ImageRecord(
        image_path,
        (GroundTruth(0, (0.0, 0.0, 10.0, 10.0)),),
        (Prediction(0, 0.9, (20.0, 20.0, 30.0, 30.0)),),
    )

    report = write_error_analysis(
        [record], tmp_path / "errors", 0.25, 0.5, {0: "defect"}, sample_limit=1
    )
    with Path(report["manifest"]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert report["false_positive_boxes"] == 1
    assert report["false_negative_boxes"] == 1
    assert {row["error_type"] for row in rows} == {"false_positive", "false_negative"}
    assert (tmp_path / "errors" / "false_positives" / "sample.jpg").is_file()
    assert (tmp_path / "errors" / "false_negatives" / "sample.jpg").is_file()
