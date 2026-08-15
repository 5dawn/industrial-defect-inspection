"""HTTP API wrapping the shared inference engine."""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image

from industrial_defect_inspection.anomaly.schemas import AnomalyMetadata, AnomalyResult
from industrial_defect_inspection.config import AnomalyInferenceConfig, InferenceConfig
from industrial_defect_inspection.inference.schemas import (
    HealthResponse,
    InferenceResult,
    ModelMetadata,
)
from industrial_defect_inspection.web.uploads import (
    MAX_UPLOAD_BYTES,
    UploadTooLargeError,
    UploadValidationError,
    decode_upload,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class EngineProtocol(Protocol):
    config: InferenceConfig
    device: str | int

    @property
    def loaded(self) -> bool: ...

    def predict(
        self, image: Image.Image, confidence: float | None = None
    ) -> tuple[InferenceResult, Image.Image]: ...


class AnomalyEngineProtocol(Protocol):
    config: AnomalyInferenceConfig
    device: str | int
    categories: list[str]

    @property
    def loaded(self) -> bool: ...

    def category_available(self, category: str) -> bool: ...

    def unavailable_message(self, category: str) -> str: ...

    def predict(self, image: Image.Image, category: str) -> tuple[AnomalyResult, object]: ...


def create_app(
    engine: EngineProtocol, anomaly_engine: AnomalyEngineProtocol | None = None
) -> FastAPI:
    app = FastAPI(
        title="Industrial Defect Inspection API",
        version="0.1.0",
        description="Research and portfolio demo; not a production quality-control system.",
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/demo/")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model_available = engine.loaded or engine.config.model.is_file()
        return HealthResponse(
            status="ok" if model_available else "degraded",
            model_loaded=engine.loaded,
            model_version=engine.config.model_version,
            device=str(engine.device),
        )

    @app.get("/metadata", response_model=ModelMetadata)
    def metadata() -> ModelMetadata:
        return ModelMetadata(
            model_version=engine.config.model_version,
            class_names=engine.config.class_names,
            confidence=engine.config.confidence,
            image_size=engine.config.image_size,
            device=str(engine.device),
        )

    @app.get("/metadata/anomaly", response_model=AnomalyMetadata)
    def anomaly_metadata() -> AnomalyMetadata:
        if anomaly_engine is None:
            raise HTTPException(status_code=503, detail="Anomaly localization is not configured")
        return AnomalyMetadata(
            model_version=anomaly_engine.config.model_version,
            categories=anomaly_engine.categories,
            available_categories=[
                category
                for category in anomaly_engine.categories
                if anomaly_engine.category_available(category)
            ],
            image_size=anomaly_engine.config.image_size,
            device=str(anomaly_engine.device),
        )

    @app.post("/predict", response_model=InferenceResult)
    async def predict(file: UploadFile = File(...)) -> InferenceResult:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP are supported")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            image, original_size, resized = decode_upload(payload)
            result, _ = engine.predict(image)
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UploadValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model is unavailable at {engine.config.model}. "
                    "Provide trained weights with --model PATH."
                ),
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503, detail=f"Model inference is unavailable: {exc}"
            ) from exc
        return result.model_copy(
            update={
                "original_image_width": original_size[0],
                "original_image_height": original_size[1],
                "resized": resized,
            }
        )

    @app.post("/predict/anomaly", response_model=AnomalyResult)
    async def predict_anomaly(category: str, file: UploadFile = File(...)) -> AnomalyResult:
        if anomaly_engine is None:
            raise HTTPException(status_code=503, detail="Anomaly localization is not configured")
        if category not in anomaly_engine.categories:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported category '{category}'. Choose from {anomaly_engine.categories}"
                ),
            )
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP are supported")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            image, original_size, resized = decode_upload(payload)
            result, _ = anomaly_engine.predict(image, category)
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UploadValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503, detail=anomaly_engine.unavailable_message(category)
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503, detail=f"Anomaly inference is unavailable: {exc}"
            ) from exc
        return result.model_copy(
            update={
                "original_image_width": original_size[0],
                "original_image_height": original_size[1],
                "resized": resized,
            }
        )

    return app
