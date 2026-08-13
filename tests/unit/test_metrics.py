from pathlib import Path
from types import SimpleNamespace

import pytest

from industrial_defect_inspection.evaluation.metrics import (
    GroundTruth,
    ImageRecord,
    Prediction,
    average_precision_by_class,
    match_record,
    merge_per_class_metrics,
    operating_metrics,
    select_operating_confidence,
)


def record(*predictions: Prediction) -> ImageRecord:
    return ImageRecord(
        Path("sample.jpg"),
        (GroundTruth(0, (0.0, 0.0, 10.0, 10.0)),),
        tuple(predictions),
    )


def test_operating_metrics_count_duplicate_prediction_and_f1() -> None:
    records = [
        record(
            Prediction(0, 0.9, (0.0, 0.0, 10.0, 10.0)),
            Prediction(0, 0.8, (0.0, 0.0, 10.0, 10.0)),
        ),
        record(),
    ]

    metrics = operating_metrics(records, 0.5, 0.5, {0: "defect"})

    assert metrics["overall"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert metrics["per_class"]["defect"]["fp"] == 1


def test_threshold_selection_uses_f1_then_precision_then_higher_threshold() -> None:
    records = [
        record(
            Prediction(0, 0.8, (0.0, 0.0, 10.0, 10.0)),
            Prediction(0, 0.7, (20.0, 20.0, 30.0, 30.0)),
        )
    ]

    threshold, sweep = select_operating_confidence(records, 0.5, {0: "defect"})

    assert threshold == pytest.approx(0.8)
    assert len(sweep) == 99
    assert max(float(row["f1"]) for row in sweep) == 1.0


def test_match_record_rejects_wrong_class_and_low_iou() -> None:
    sample = ImageRecord(
        Path("sample.jpg"),
        (GroundTruth(0, (0.0, 0.0, 10.0, 10.0)),),
        (
            Prediction(1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            Prediction(0, 0.8, (8.0, 8.0, 18.0, 18.0)),
        ),
    )

    result = match_record(sample, 0.25, 0.5)

    assert result.matches == ()
    assert result.unmatched_predictions == (0, 1)
    assert result.unmatched_ground_truth == (0,)


def test_per_class_average_precision_is_merged_with_operating_metrics() -> None:
    box = SimpleNamespace(ap=[0.7, 0.4], ap50=[0.8, 0.5], ap_class_index=[0, 1])
    average_precision = average_precision_by_class(SimpleNamespace(box=box), {0: "a", 1: "b"})
    operating = {
        "per_class": {
            "a": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
            "b": {"tp": 0, "fp": 0, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        }
    }

    merged = merge_per_class_metrics(operating, average_precision)

    assert merged["a"]["ap50"] == pytest.approx(0.8)
    assert merged["a"]["map50_95"] == pytest.approx(0.7)
    assert merged["b"]["ap50"] == pytest.approx(0.5)
