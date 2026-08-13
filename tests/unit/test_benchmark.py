from industrial_defect_inspection.evaluation.benchmark import percentile


def test_percentile_handles_empty_and_uses_nearest_rank() -> None:
    assert percentile([], 0.95) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
