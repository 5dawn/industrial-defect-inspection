"""One inference implementation shared by command-line and web surfaces."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from industrial_defect_inspection.config import InferenceConfig
from industrial_defect_inspection.inference.schemas import Detection, InferenceResult
from industrial_defect_inspection.training.train import resolve_device


class InferenceEngine:
    """Lazy, thread-safe Ultralytics engine supporting .pt and .onnx files."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self._model: Any | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.config.model.is_file():
                raise FileNotFoundError(f"Model file not found: {self.config.model}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install the project dependencies before inference") from exc
            self._model = YOLO(str(self.config.model))

    def predict(
        self, image: Image.Image | np.ndarray | str | Path, confidence: float | None = None
    ) -> tuple[InferenceResult, Image.Image]:
        threshold = self.config.confidence if confidence is None else confidence
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.load()
        if isinstance(image, Image.Image):
            source: Any = np.asarray(image.convert("RGB"))
            width, height = image.size
        elif isinstance(image, np.ndarray):
            source = image
            height, width = image.shape[:2]
        else:
            source = str(image)
            with Image.open(source) as opened:
                width, height = opened.size

        with self._lock:
            predictions = self._model.predict(
                source=source,
                conf=threshold,
                imgsz=self.config.image_size,
                max_det=self.config.max_detections,
                device=self.device,
                verbose=False,
            )
        if len(predictions) != 1:
            raise RuntimeError(f"Expected one prediction result, received {len(predictions)}")
        raw = predictions[0]
        detections: list[Detection] = []
        boxes = raw.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            names = raw.names
            for coordinates, score, class_value in zip(xyxy, confidences, classes, strict=True):
                class_id = int(class_value)
                fallback = (
                    self.config.class_names[class_id]
                    if class_id < len(self.config.class_names)
                    else str(class_id)
                )
                class_name = names.get(class_id, fallback) if isinstance(names, dict) else fallback
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(class_name),
                        confidence=float(score),
                        bbox_xyxy=tuple(float(value) for value in coordinates),
                    )
                )

        speed = raw.speed or {}
        result = InferenceResult(
            image_width=width,
            image_height=height,
            detections=detections,
            preprocess_ms=max(0.0, float(speed.get("preprocess", 0.0))),
            inference_ms=max(0.0, float(speed.get("inference", 0.0))),
            postprocess_ms=max(0.0, float(speed.get("postprocess", 0.0))),
            model_version=self.config.model_version,
            device=str(self.device),
        )
        plotted = raw.plot()
        annotated = Image.fromarray(plotted[..., ::-1]).convert("RGB")
        return result, annotated

    def warmup(self) -> None:
        image = Image.new("RGB", (self.config.image_size, self.config.image_size), "gray")
        self.predict(image)
