"""Aspect-ratio-preserving image and anomaly-map transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

IMAGENET_FILL = (124, 116, 104)


@dataclass(frozen=True, slots=True)
class LetterboxMetadata:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    left: int
    top: int
    size: int


def letterbox_image(
    image: Image.Image, size: int, fill: tuple[int, int, int] = IMAGENET_FILL
) -> tuple[Image.Image, LetterboxMetadata]:
    """Resize an image without cropping and pad it to a square canvas."""
    if size < 1:
        raise ValueError("size must be positive")
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, (left, top))
    return canvas, LetterboxMetadata(
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        left=left,
        top=top,
        size=size,
    )


def letterbox_mask(mask: Image.Image, metadata: LetterboxMetadata) -> Image.Image:
    """Apply image letterboxing metadata to a binary mask."""
    grayscale = mask.convert("L")
    if grayscale.size != (metadata.original_width, metadata.original_height):
        raise ValueError(
            "Image and mask dimensions differ: "
            f"image={(metadata.original_width, metadata.original_height)}, mask={grayscale.size}"
        )
    resized = grayscale.resize(
        (metadata.resized_width, metadata.resized_height), Image.Resampling.NEAREST
    )
    canvas = Image.new("L", (metadata.size, metadata.size), 0)
    canvas.paste(resized, (metadata.left, metadata.top))
    return canvas.point(lambda value: 255 if value > 0 else 0)


def restore_anomaly_map(values: np.ndarray, metadata: LetterboxMetadata) -> np.ndarray:
    """Remove padding and resize a 2-D anomaly map to original coordinates."""
    array = np.asarray(values, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D anomaly map, received shape {array.shape}")
    if array.shape != (metadata.size, metadata.size):
        image = Image.fromarray(array, mode="F")
        image = image.resize((metadata.size, metadata.size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    cropped = array[
        metadata.top : metadata.top + metadata.resized_height,
        metadata.left : metadata.left + metadata.resized_width,
    ]
    restored = Image.fromarray(cropped, mode="F").resize(
        (metadata.original_width, metadata.original_height), Image.Resampling.BILINEAR
    )
    return np.asarray(restored, dtype=np.float32)


def colorize_anomaly_map(values: np.ndarray) -> Image.Image:
    """Render a deterministic blue-to-red heatmap without Matplotlib state."""
    array = np.asarray(values, dtype=np.float32)
    finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    low = float(finite.min(initial=0.0))
    high = float(finite.max(initial=0.0))
    normalized = np.zeros_like(finite) if high <= low else (finite - low) / (high - low)
    red = np.clip(normalized * 2.0, 0.0, 1.0)
    blue = np.clip((1.0 - normalized) * 2.0, 0.0, 1.0)
    green = 1.0 - np.abs(normalized * 2.0 - 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255), mode="RGB")
