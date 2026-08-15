"""Small compatibility boundary around the optional Anomalib dependency."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def require_anomalib() -> tuple[type[Any], type[Any], type[Any]]:
    """Import Anomalib lazily so detection-only installs remain lightweight."""
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as exc:
        raise RuntimeError(
            "Anomaly localization requires the optional dependencies. Install with "
            '`pip install -e ".[anomaly]"`.'
        ) from exc
    return Folder, Engine, Patchcore


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def iter_prediction_outputs(predictions: Iterable[Any]) -> Iterator[tuple[float, np.ndarray]]:
    """Flatten Anomalib prediction batches into score/map pairs."""
    for prediction in predictions:
        if not hasattr(prediction, "pred_score") or not hasattr(prediction, "anomaly_map"):
            raise RuntimeError("Anomalib prediction is missing pred_score or anomaly_map")
        scores = to_numpy(prediction.pred_score).reshape(-1)
        maps = to_numpy(prediction.anomaly_map)
        if maps.ndim >= 4 and maps.shape[1] == 1:
            maps = maps[:, 0]
        elif maps.ndim == 2:
            maps = maps[None, ...]
        elif maps.ndim == 3 and len(scores) == 1 and maps.shape[0] == 1:
            pass
        if len(scores) != len(maps):
            if len(scores) == 1:
                maps = np.asarray(maps).squeeze()[None, ...]
            else:
                raise RuntimeError(
                    f"Anomalib prediction batch mismatch: {len(scores)} scores, {len(maps)} maps"
                )
        for score, anomaly_map in zip(scores, maps, strict=True):
            yield float(score), np.asarray(anomaly_map, dtype=np.float32).squeeze()


def create_patchcore(config: Any) -> Any:
    """Build the configured PatchCore model with a square no-crop preprocessor."""
    _, _, patchcore_type = require_anomalib()
    pre_processor = patchcore_type.configure_pre_processor(
        image_size=(config.image_size, config.image_size), center_crop_size=None
    )
    return patchcore_type(
        backbone=config.backbone,
        layers=config.layers,
        pre_trained=True,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        num_neighbors=config.num_neighbors,
        precision="float32",
        pre_processor=pre_processor,
    )


def load_patchcore(checkpoint: Path, device: str | int) -> Any:
    """Restore a trusted, locally produced Anomalib checkpoint for inference."""
    _, _, patchcore_type = require_anomalib()
    map_location = "cpu" if device == "cpu" else None
    try:
        return patchcore_type.load_from_checkpoint(
            str(checkpoint),
            map_location=map_location,
            strict=False,
            # Anomalib checkpoints contain their PreProcessor object, not tensors only.
            # Callers validate the configured checkpoint path and published SHA-256.
            weights_only=False,
        )
    except TypeError:
        return patchcore_type.load_from_checkpoint(
            str(checkpoint), map_location=map_location, weights_only=False
        )
