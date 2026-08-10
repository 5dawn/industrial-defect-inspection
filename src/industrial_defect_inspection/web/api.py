"""HTTP API wrapping the shared inference engine."""

from __future__ import annotations

from io import BytesIO
from typing import Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from industrial_defect_inspection.config import InferenceConfig
from industrial_defect_inspection.inference.schemas import (
    HealthResponse,
    InferenceResult,
    ModelMetadata,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class EngineProtocol(Protocol):
    config: InferenceConfig
    device: str | int

    @property
    def loaded(self) -> bool: ...

    def predict(
        self, image: Image.Image, confidence: float | None = None
    ) -> tuple[InferenceResult, Image.Image]: ...


def decode_upload(payload: bytes) -> tuple[Image.Image, tuple[int, int], bool]:
    try:
        with Image.open(BytesIO(payload)) as probe:
            probe.verify()
        with Image.open(BytesIO(payload)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The upload is not a valid JPEG, PNG, or WebP image") from exc
    original_size = image.size
    resized = max(image.size) > MAX_IMAGE_SIDE
    if resized:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image, original_size, resized


def create_app(engine: EngineProtocol) -> FastAPI:
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
        return HealthResponse(
            status="ok",
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

    @app.post("/predict", response_model=InferenceResult)
    async def predict(file: UploadFile = File(...)) -> InferenceResult:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP are supported")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload exceeds the 10 MB limit")
        try:
            image, original_size, resized = decode_upload(payload)
            result, _ = engine.predict(image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return result.model_copy(
            update={
                "original_image_width": original_size[0],
                "original_image_height": original_size[1],
                "resized": resized,
            }
        )

    return app
