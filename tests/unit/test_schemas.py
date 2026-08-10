import pytest
from pydantic import ValidationError

from industrial_defect_inspection.inference.schemas import Detection, InferenceResult


def test_inference_result_contract() -> None:
    result = InferenceResult(
        image_width=200,
        image_height=200,
        detections=[
            Detection(
                class_id=1,
                class_name="inclusion",
                confidence=0.9,
                bbox_xyxy=(1.0, 2.0, 30.0, 40.0),
            )
        ],
        preprocess_ms=1.0,
        inference_ms=5.0,
        postprocess_ms=2.0,
        model_version="test",
        device="cpu",
    )

    assert result.total_ms == 8.0
    assert result.model_dump()["detections"][0]["class_name"] == "inclusion"


def test_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        Detection(
            class_id=0,
            class_name="bad",
            confidence=1.1,
            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        )
