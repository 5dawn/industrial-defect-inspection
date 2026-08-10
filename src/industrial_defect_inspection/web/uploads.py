"""Shared upload validation for the API and Gradio demo."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class UploadValidationError(ValueError):
    """Raised when an uploaded file cannot be used as an input image."""


class UploadTooLargeError(UploadValidationError):
    """Raised when an upload exceeds the byte limit."""


def decode_upload(payload: bytes) -> tuple[Image.Image, tuple[int, int], bool]:
    """Validate image bytes, normalize orientation, and limit large dimensions."""
    if not payload:
        raise UploadValidationError("The uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError("Upload exceeds the 10 MB limit")

    try:
        with Image.open(BytesIO(payload)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise UploadValidationError("Only JPEG, PNG, and WebP images are supported")
            probe.verify()
        with Image.open(BytesIO(payload)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except UploadValidationError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise UploadValidationError("The upload is not a valid JPEG, PNG, or WebP image") from exc

    original_size = image.size
    resized = max(image.size) > MAX_IMAGE_SIDE
    if resized:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image, original_size, resized
