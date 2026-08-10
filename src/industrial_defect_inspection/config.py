"""Typed configuration loading shared by command-line applications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitConfig(StrictModel):
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    @model_validator(mode="after")
    def validate_total(self) -> SplitConfig:
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("All split fractions must be positive")
        if abs(self.train + self.val + self.test - 1.0) > 1e-8:
            raise ValueError("Split fractions must sum to 1.0")
        return self


class DataConfig(StrictModel):
    dataset_name: str
    raw_images_dir: Path
    raw_annotations_dir: Path
    output_dir: Path
    report_dir: Path
    class_names: list[str]
    class_aliases: dict[str, str] = Field(default_factory=dict)
    stratify_by: Literal["annotation_set", "filename_prefix"] = "annotation_set"
    split: SplitConfig = Field(default_factory=SplitConfig)
    seed: int = 42
    preview_count: int = Field(default=30, ge=0)
    strict_image_size: bool = True


class AugmentationConfig(StrictModel):
    degrees: float = 0.0
    translate: float = 0.08
    scale: float = 0.20
    fliplr: float = 0.5
    flipud: float = 0.5
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.15
    mosaic: float = 0.25
    mixup: float = 0.0


class TrainConfig(StrictModel):
    model: str
    dataset: Path
    project: Path
    name: str
    epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=20, ge=0)
    image_size: int = Field(default=640, ge=32)
    batch: int | float = 0.60
    workers: int = Field(default=4, ge=0)
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    pretrained: bool = True
    amp: bool = True
    resume: bool | str = False
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


class EvaluationConfig(StrictModel):
    model: Path
    dataset: Path
    output_dir: Path
    split: Literal["val", "test"] = "val"
    device: str = "auto"
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=32)
    error_samples: int = Field(default=20, ge=0)
    benchmark_count: int = Field(default=100, ge=1)


class InferenceConfig(StrictModel):
    model: Path
    output_dir: Path = Path("artifacts/predictions")
    device: str = "auto"
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=32)
    max_detections: int = Field(default=100, ge=1)
    model_version: str = "unknown"
    class_names: list[str] = Field(default_factory=list)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return payload


def load_data_config(path: str | Path) -> DataConfig:
    return DataConfig.model_validate(load_yaml(path))


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig.model_validate(load_yaml(path))


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    return EvaluationConfig.model_validate(load_yaml(path))


def load_inference_config(path: str | Path) -> InferenceConfig:
    return InferenceConfig.model_validate(load_yaml(path))
