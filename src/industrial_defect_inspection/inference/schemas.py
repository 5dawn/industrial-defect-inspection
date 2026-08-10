"""Stable inference result contracts used by CLI and web APIs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    original_image_width: int | None = Field(default=None, gt=0)
    original_image_height: int | None = Field(default=None, gt=0)
    resized: bool = False
    detections: list[Detection]
    preprocess_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    postprocess_ms: float = Field(ge=0.0)
    model_version: str
    device: str

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str


class ModelMetadata(BaseModel):
    model_version: str
    class_names: list[str]
    confidence: float
    image_size: int
    device: str
    training_dataset: str = "NEU-DET"
    disclaimer: str = "Research and portfolio demo; not a production quality-control system."
