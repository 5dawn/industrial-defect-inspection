from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from industrial_defect_inspection.web.uploads import (
    MAX_IMAGE_SIDE,
    MAX_UPLOAD_BYTES,
    UploadTooLargeError,
    UploadValidationError,
    decode_upload,
)


def image_bytes(image_format: str, size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "gray").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_decode_upload_accepts_supported_images(image_format: str) -> None:
    image, original_size, resized = decode_upload(image_bytes(image_format))

    assert image.mode == "RGB"
    assert original_size == (32, 24)
    assert resized is False


def test_decode_upload_rejects_empty_corrupt_and_unsupported_files() -> None:
    with pytest.raises(UploadValidationError, match="empty"):
        decode_upload(b"")
    with pytest.raises(UploadValidationError, match="not a valid"):
        decode_upload(b"not an image")
    with pytest.raises(UploadValidationError, match="Only JPEG, PNG, and WebP"):
        decode_upload(image_bytes("GIF"))


def test_decode_upload_rejects_oversized_payload() -> None:
    with pytest.raises(UploadTooLargeError, match="10 MB"):
        decode_upload(b"x" * (MAX_UPLOAD_BYTES + 1))


def test_decode_upload_resizes_long_side() -> None:
    image, original_size, resized = decode_upload(image_bytes("PNG", (MAX_IMAGE_SIDE + 1, 1)))

    assert original_size == (MAX_IMAGE_SIDE + 1, 1)
    assert max(image.size) == MAX_IMAGE_SIDE
    assert resized is True
