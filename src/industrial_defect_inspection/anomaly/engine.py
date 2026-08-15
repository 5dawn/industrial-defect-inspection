"""Lazy PatchCore inference shared by anomaly CLI, API, and Gradio."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from industrial_defect_inspection.anomaly.backend import (
    iter_prediction_outputs,
    load_patchcore,
    require_anomalib,
)
from industrial_defect_inspection.anomaly.preprocessing import (
    colorize_anomaly_map,
    letterbox_image,
    restore_anomaly_map,
)
from industrial_defect_inspection.anomaly.schemas import AnomalyResult, AnomalyVisuals
from industrial_defect_inspection.config import AnomalyInferenceConfig
from industrial_defect_inspection.utils.runtime import prepare_runtime, resolve_device


def _prediction_engine_settings(accelerator: str) -> dict[str, Any]:
    """Build inference settings compatible with Anomalib's model callbacks."""
    return {
        "accelerator": accelerator,
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_checkpointing": True,
    }


class AnomalyEngine:
    """Load one model per VisA category on demand and reuse it across requests."""

    def __init__(self, config: AnomalyInferenceConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self._models: dict[str, Any] = {}
        self._engines: dict[str, Any] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def categories(self) -> list[str]:
        return sorted(self.config.checkpoints)

    @property
    def loaded(self) -> bool:
        return bool(self._models)

    def category_available(self, category: str) -> bool:
        self._validate_category(category)
        return (
            self.config.checkpoints[category].is_file() and self.config.metadata[category].is_file()
        )

    def _validate_category(self, category: str) -> None:
        if category not in self.config.checkpoints:
            raise ValueError(
                f"Unsupported anomaly category '{category}'. Choose from: "
                f"{', '.join(self.categories)}"
            )

    def unavailable_message(self, category: str) -> str:
        self._validate_category(category)
        return (
            f"Anomaly model for '{category}' is unavailable. Expected checkpoint "
            f"{self.config.checkpoints[category]} and metadata {self.config.metadata[category]}. "
            "Run idi-fit-anomaly or update configs/anomaly/infer.yaml."
        )

    def load(self, category: str) -> None:
        self._validate_category(category)
        with self._lock:
            if category in self._models:
                return
            if not self.category_available(category):
                raise FileNotFoundError(self.unavailable_message(category))
            prepare_runtime(self.config.output_dir)
            with self.config.metadata[category].open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("category") != category:
                raise ValueError(
                    f"Anomaly metadata category mismatch: expected {category}, "
                    f"found {metadata.get('category')}"
                )
            metadata_size = int(metadata.get("image_size", self.config.image_size))
            if metadata_size != self.config.image_size:
                raise ValueError(
                    f"Anomaly image_size mismatch: config={self.config.image_size}, "
                    f"metadata={metadata_size}"
                )
            _, engine_type, _ = require_anomalib()
            accelerator = "gpu" if isinstance(self.device, int) else "cpu"
            self._models[category] = load_patchcore(self.config.checkpoints[category], self.device)
            self._engines[category] = engine_type(**_prediction_engine_settings(accelerator))
            self._metadata[category] = metadata

    def predict(self, image: Image.Image, category: str) -> tuple[AnomalyResult, AnomalyVisuals]:
        """Return a result and three visuals derived from the same raw anomaly map."""
        started = time.perf_counter()
        padded, transform = letterbox_image(image, self.config.image_size)
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        self.load(category)

        with self._lock, tempfile.TemporaryDirectory(prefix="idi-anomaly-") as directory:
            input_path = Path(directory) / "input.png"
            padded.save(input_path, format="PNG")
            inference_started = time.perf_counter()
            predictions = self._engines[category].predict(
                model=self._models[category],
                data_path=input_path,
                return_predictions=True,
            )
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
        if predictions is None:
            raise RuntimeError("Anomalib returned no prediction")
        outputs = list(iter_prediction_outputs(predictions))
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one anomaly prediction, received {len(outputs)}")

        post_started = time.perf_counter()
        score, padded_map = outputs[0]
        anomaly_map = restore_anomaly_map(padded_map, transform)
        metadata = self._metadata[category]
        image_threshold = float(metadata["image_threshold"])
        pixel_threshold = float(metadata["pixel_threshold"])
        binary = anomaly_map >= pixel_threshold
        mask = Image.fromarray(np.uint8(binary) * 255, mode="L")
        heatmap = colorize_anomaly_map(anomaly_map)
        original = image.convert("RGB")
        overlay = Image.blend(original, heatmap, alpha=0.45)
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0
        result = AnomalyResult(
            category=category,
            image_width=original.width,
            image_height=original.height,
            anomaly_score=score,
            image_threshold=image_threshold,
            is_anomalous=score >= image_threshold,
            pixel_threshold=pixel_threshold,
            anomaly_area_ratio=float(binary.mean()),
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            model_version=str(metadata.get("model_version", self.config.model_version)),
            device=str(self.device),
        )
        return result, AnomalyVisuals(
            heatmap=heatmap,
            mask=mask,
            overlay=overlay,
            anomaly_map=anomaly_map,
        )

    def warmup(self, category: str) -> None:
        image = Image.new("RGB", (self.config.image_size, self.config.image_size), "gray")
        self.predict(image, category)
