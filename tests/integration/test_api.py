from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from industrial_defect_inspection.config import InferenceConfig
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
