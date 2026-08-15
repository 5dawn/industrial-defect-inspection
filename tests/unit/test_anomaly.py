from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from industrial_defect_inspection.anomaly import backend
from industrial_defect_inspection.anomaly.engine import _prediction_engine_settings
from industrial_defect_inspection.anomaly.evaluate import binary_auroc, mask_metrics
from industrial_defect_inspection.anomaly.fit import _engine_settings, calibrate_thresholds
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


def test_anomalib_engine_keeps_checkpointing_enabled(tmp_path: Path) -> None:
    settings = _engine_settings("gpu", 1, tmp_path)

    assert settings["enable_checkpointing"] is True
    assert settings["limit_val_batches"] == 0
    assert settings["enable_progress_bar"] is False
    assert settings["default_root_dir"] == tmp_path


def test_load_patchcore_marks_local_checkpoint_as_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakePatchcore:
        @classmethod
        def load_from_checkpoint(cls, path: str, **kwargs):
            captured.update({"path": path, **kwargs})
            return cls()

    monkeypatch.setattr(backend, "require_anomalib", lambda: (object, object, FakePatchcore))
    checkpoint = tmp_path / "model.ckpt"

    backend.load_patchcore(checkpoint, "cpu")

    assert captured["path"] == str(checkpoint)
    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is False


def test_prediction_engine_keeps_checkpointing_enabled() -> None:
    settings = _prediction_engine_settings("cpu")

    assert settings["accelerator"] == "cpu"
    assert settings["enable_checkpointing"] is True
    assert settings["enable_progress_bar"] is False


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
