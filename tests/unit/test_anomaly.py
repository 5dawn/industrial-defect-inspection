from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from industrial_defect_inspection.anomaly.evaluate import binary_auroc, mask_metrics
from industrial_defect_inspection.anomaly.fit import calibrate_thresholds
from industrial_defect_inspection.anomaly.schemas import AnomalyResult
from industrial_defect_inspection.config import AnomalyInferenceConfig


def test_anomaly_result_contract_and_total() -> None:
    result = AnomalyResult(
        category="candle",
        image_width=20,
        image_height=10,
        anomaly_score=0.8,
        image_threshold=0.7,
        is_anomalous=True,
        pixel_threshold=0.6,
        anomaly_area_ratio=0.1,
        preprocess_ms=1,
        inference_ms=2,
        postprocess_ms=3,
        model_version="test",
        device="cpu",
    )

    assert result.total_ms == 6
    assert result.is_anomalous is True


def test_anomaly_config_requires_matching_categories() -> None:
    with pytest.raises(ValidationError, match="same categories"):
        AnomalyInferenceConfig(
            checkpoints={"candle": Path("candle.ckpt")},
            metadata={"pcb1": Path("pcb1.json")},
        )


def test_normal_only_threshold_calibration_uses_configured_quantiles() -> None:
    image_threshold, pixel_threshold = calibrate_thresholds(
        [0.1, 0.2, 0.3],
        [np.array([[0.0, 0.1]]), np.array([[0.2, 0.3]])],
        0.5,
        0.5,
    )

    assert image_threshold == pytest.approx(0.2)
    assert pixel_threshold == pytest.approx(0.15)


def test_binary_auroc_handles_ties_and_single_class() -> None:
    assert binary_auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert binary_auroc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert binary_auroc(np.array([0.1, 0.2]), np.array([0, 0])) is None


def test_mask_metrics_cover_empty_and_overlap() -> None:
    empty = np.zeros((2, 2), dtype=bool)
    assert mask_metrics(empty, empty) == (1.0, 1.0)
    predicted = np.array([[1, 1], [0, 0]], dtype=bool)
    expected = np.array([[1, 0], [1, 0]], dtype=bool)
    assert mask_metrics(predicted, expected) == (0.5, 1 / 3)
