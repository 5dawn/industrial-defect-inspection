import pytest

from industrial_defect_inspection.evaluation.evaluate import box_iou, match_detections


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
