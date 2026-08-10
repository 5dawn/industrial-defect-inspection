from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from industrial_defect_inspection.config import InferenceConfig
from industrial_defect_inspection.inference.engine import InferenceEngine
from industrial_defect_inspection.inference.schemas import Detection, InferenceResult
from industrial_defect_inspection.web.app import (
    DemoError,
    build_parser,
    model_status_markdown,
    run_demo_prediction,
)


class FakeEngine:
    def __init__(self, model: Path) -> None:
        self.config = InferenceConfig(
            model=model,
            output_dir=model.parent / "outputs",
            device="cpu",
            confidence=0.25,
            image_size=640,
            max_detections=100,
            model_version="fake-v1",
            class_names=["crazing"],
        )
        self.device = "cpu"
        self.loaded = True

    def predict(self, image: Image.Image, confidence: float | None = None):
        result = InferenceResult(
            image_width=image.width,
            image_height=image.height,
            detections=[
                Detection(
                    class_id=0,
                    class_name="crazing",
                    confidence=0.8,
                    bbox_xyxy=(1.0, 2.0, 10.0, 11.0),
                )
            ],
            preprocess_ms=1.0,
            inference_ms=2.0,
            postprocess_ms=1.0,
            model_version=self.config.model_version,
            device="cpu",
        )
        return result, image.copy()


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def test_demo_prediction_returns_visual_table_and_timing(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"fixture")
    engine = FakeEngine(model)

    annotated, rows, raw, summary, image_path, json_path = run_demo_prediction(
        png_bytes(), 0.25, engine, tmp_path / "outputs"
    )

    assert annotated.size == (32, 24)
    assert rows == [["crazing", 0.8, 1.0, 2.0, 10.0, 11.0]]
    assert raw["detections"][0]["class_name"] == "crazing"
    assert raw["original_image_width"] == 32
    assert raw["original_image_height"] == 24
    assert "total 4.0 ms" in summary
    assert "inference 2.0 ms" in summary
    assert Path(image_path).is_file()
    assert Path(json_path).is_file()
    assert Path(image_path).is_absolute()
    assert Path(json_path).is_absolute()
    assert "ready" not in model_status_markdown(engine)


def test_demo_reports_missing_model_without_loading_backend(tmp_path: Path) -> None:
    engine = InferenceEngine(InferenceConfig(model=tmp_path / "missing.pt", device="cpu"))

    with pytest.raises(DemoError, match="restart with --model PATH"):
        run_demo_prediction(png_bytes(), 0.25, engine, tmp_path / "outputs")

    status = model_status_markdown(engine)
    assert "Model unavailable" in status
    assert "missing.pt" in status


def test_demo_rejects_corrupt_image(tmp_path: Path) -> None:
    engine = FakeEngine(tmp_path / "best.pt")

    with pytest.raises(DemoError, match="not a valid JPEG, PNG, or WebP"):
        run_demo_prediction(b"not an image", 0.25, engine, tmp_path / "outputs")


def test_web_parser_accepts_explicit_cpu_device() -> None:
    args = build_parser().parse_args(["--device", "cpu"])

    assert args.device == "cpu"
