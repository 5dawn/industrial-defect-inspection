from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from industrial_defect_inspection.anomaly.engine import AnomalyEngine
from industrial_defect_inspection.anomaly.schemas import AnomalyResult, AnomalyVisuals
from industrial_defect_inspection.config import AnomalyInferenceConfig, InferenceConfig
from industrial_defect_inspection.inference.engine import InferenceEngine
from industrial_defect_inspection.inference.schemas import Detection, InferenceResult
from industrial_defect_inspection.web.api import create_app
from industrial_defect_inspection.web.app import build_application, model_unavailable_message


class FakeEngine:
    def __init__(self) -> None:
        self.config = InferenceConfig(
            model=Path("unused.pt"),
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


class FakeAnomalyEngine:
    def __init__(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "candle.ckpt"
        metadata = tmp_path / "candle.json"
        checkpoint.write_bytes(b"fixture")
        metadata.write_text("{}", encoding="utf-8")
        self.config = AnomalyInferenceConfig(
            checkpoints={"candle": checkpoint},
            metadata={"candle": metadata},
            device="cpu",
            image_size=256,
            model_version="fake-anomaly-v1",
        )
        self.device = "cpu"
        self.categories = ["candle"]
        self.loaded = True

    def category_available(self, category: str) -> bool:
        return category == "candle"

    def unavailable_message(self, category: str) -> str:
        return f"Anomaly model for {category} is unavailable"

    def predict(self, image: Image.Image, category: str):
        result = AnomalyResult(
            category=category,
            image_width=image.width,
            image_height=image.height,
            anomaly_score=0.8,
            image_threshold=0.7,
            is_anomalous=True,
            pixel_threshold=0.5,
            anomaly_area_ratio=0.25,
            preprocess_ms=1.0,
            inference_ms=2.0,
            postprocess_ms=1.0,
            model_version=self.config.model_version,
            device="cpu",
        )
        mask = Image.new("L", image.size, 255)
        visuals = AnomalyVisuals(
            heatmap=Image.new("RGB", image.size, "red"),
            mask=mask,
            overlay=image.copy(),
            anomaly_map=np.ones((image.height, image.width), dtype=np.float32),
        )
        return result, visuals


def png_bytes(size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_metadata() -> None:
    client = TestClient(create_app(FakeEngine()))

    assert client.get("/health").json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "fake-v1",
        "device": "cpu",
    }
    assert client.get("/metadata").json()["class_names"] == ["crazing"]


def test_predict_returns_structured_result() -> None:
    client = TestClient(create_app(FakeEngine()))

    response = client.post("/predict", files={"file": ("sample.png", png_bytes(), "image/png")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["original_image_width"] == 32
    assert payload["original_image_height"] == 24
    assert payload["resized"] is False
    assert payload["detections"][0]["class_name"] == "crazing"


def test_predict_rejects_invalid_media() -> None:
    client = TestClient(create_app(FakeEngine()))

    unsupported = client.post(
        "/predict", files={"file": ("sample.txt", b"not an image", "text/plain")}
    )
    corrupt = client.post("/predict", files={"file": ("sample.png", b"not an image", "image/png")})

    assert unsupported.status_code == 415
    assert corrupt.status_code == 400


def test_missing_model_reports_degraded_health_and_unavailable_prediction(tmp_path: Path) -> None:
    engine = InferenceEngine(InferenceConfig(model=tmp_path / "missing.pt", device="cpu"))
    client = TestClient(create_app(engine))

    health = client.get("/health")
    prediction = client.post("/predict", files={"file": ("sample.png", png_bytes(), "image/png")})

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["model_loaded"] is False
    assert prediction.status_code == 503
    assert "Provide trained weights with --model PATH" in prediction.json()["detail"]


def test_mounted_demo_starts_without_model(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    engine = InferenceEngine(InferenceConfig(model=missing, device="cpu"))
    app = build_application(engine, tmp_path / "outputs", model_unavailable_message(missing))

    with TestClient(app) as client:
        response = client.get("/demo/")

    assert response.status_code == 200


def test_anomaly_metadata_and_prediction(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeEngine(), FakeAnomalyEngine(tmp_path)))

    metadata = client.get("/metadata/anomaly")
    prediction = client.post(
        "/predict/anomaly?category=candle",
        files={"file": ("sample.png", png_bytes(), "image/png")},
    )

    assert metadata.status_code == 200
    assert metadata.json()["available_categories"] == ["candle"]
    assert prediction.status_code == 200
    assert prediction.json()["category"] == "candle"
    assert prediction.json()["is_anomalous"] is True


def test_anomaly_api_rejects_unknown_category_and_corrupt_image(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeEngine(), FakeAnomalyEngine(tmp_path)))

    unknown = client.post(
        "/predict/anomaly?category=pcb1",
        files={"file": ("sample.png", png_bytes(), "image/png")},
    )
    corrupt = client.post(
        "/predict/anomaly?category=candle",
        files={"file": ("sample.png", b"not an image", "image/png")},
    )

    assert unknown.status_code == 422
    assert corrupt.status_code == 400


def test_anomaly_api_reports_missing_checkpoint_without_importing_backend(
    tmp_path: Path,
) -> None:
    anomaly = AnomalyEngine(
        AnomalyInferenceConfig(
            checkpoints={"candle": tmp_path / "missing.ckpt"},
            metadata={"candle": tmp_path / "missing.json"},
            device="cpu",
        )
    )
    client = TestClient(create_app(FakeEngine(), anomaly))

    response = client.post(
        "/predict/anomaly?category=candle",
        files={"file": ("sample.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "Run idi-fit-anomaly" in response.json()["detail"]
