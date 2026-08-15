"""Public result contracts for anomaly localization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class AnomalyResult(BaseModel):
    """Serializable result returned by CLI and HTTP anomaly inference."""

    model_config = ConfigDict(extra="forbid")

    category: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    original_image_width: int | None = Field(default=None, gt=0)
    original_image_height: int | None = Field(default=None, gt=0)
    resized: bool = False
    anomaly_score: float
    image_threshold: float
    is_anomalous: bool
    pixel_threshold: float
    anomaly_area_ratio: float = Field(ge=0.0, le=1.0)
    preprocess_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    postprocess_ms: float = Field(ge=0.0)
    model_version: str
    device: str

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


class AnomalyMetadata(BaseModel):
    model_version: str
    categories: list[str]
    available_categories: list[str]
    image_size: int
    device: str
    training_dataset: str = "VisA"
    dataset_license: str = "CC BY 4.0"
    calibration_policy: str = "Normal-only validation quantiles; official test is frozen."
    disclaimer: str = "Research and portfolio demo; not a production quality-control system."


@dataclass(frozen=True, slots=True)
class AnomalyVisuals:
    """Non-JSON visual outputs generated from the same anomaly map."""

    heatmap: Image.Image
    mask: Image.Image
    overlay: Image.Image
    anomaly_map: np.ndarray
