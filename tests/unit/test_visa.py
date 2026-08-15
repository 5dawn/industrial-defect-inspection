from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_defect_inspection.anomaly.preprocessing import (
    letterbox_image,
    letterbox_mask,
    restore_anomaly_map,
)
from industrial_defect_inspection.config import VisaDataConfig
from industrial_defect_inspection.data.visa import prepare_visa


def _image(path: Path, size: tuple[int, int] = (20, 10), color: str = "gray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_letterbox_round_trip_preserves_coordinates() -> None:
    image = Image.new("RGB", (20, 10), "gray")
    mask = Image.new("L", (20, 10), 0)
    mask.putpixel((5, 5), 255)

    padded, metadata = letterbox_image(image, 32)
    padded_mask = letterbox_mask(mask, metadata)
    restored = restore_anomaly_map(np.asarray(padded_mask, dtype=np.float32), metadata)

    assert padded.size == (32, 32)
    assert padded_mask.size == (32, 32)
    assert restored.shape == (10, 20)
    assert restored[5, 5] > 0


def test_letterbox_rejects_misaligned_mask() -> None:
    _, metadata = letterbox_image(Image.new("RGB", (20, 10)), 32)

    with pytest.raises(ValueError, match="dimensions differ"):
        letterbox_mask(Image.new("L", (10, 10)), metadata)


def test_prepare_visa_uses_normal_validation_and_frozen_test(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    rows = []
    for index in range(3):
        relative = f"candle/Data/Images/Normal/{index:04d}.JPG"
        _image(raw / Path(relative))
        rows.append(["candle", "train", "normal", relative, ""])
    normal_test = "candle/Data/Images/Normal/1000.JPG"
    anomaly_test = "candle/Data/Images/Anomaly/2000.JPG"
    anomaly_mask = "candle/Data/Masks/Anomaly/2000.png"
    _image(raw / Path(normal_test))
    _image(raw / Path(anomaly_test), color="red")
    mask_path = raw / Path(anomaly_mask)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (20, 10), 0)
    mask.putpixel((5, 5), 255)
    mask.save(mask_path)
    rows.extend(
        [
            ["candle", "test", "normal", normal_test, ""],
            ["candle", "test", "anomaly", anomaly_test, anomaly_mask],
        ]
    )
    split_csv = tmp_path / "1cls.csv"
    with split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerows(rows)
    config = VisaDataConfig(
        raw_root=raw,
        split_csv=split_csv,
        output_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
        categories=["candle"],
        validation_ratio=1 / 3,
        image_size=32,
    )

    report = prepare_visa(config)

    counts = report["categories"]["candle"]["counts"]
    assert counts == {
        "test_anomaly": 1,
        "test_normal": 1,
        "train_normal": 2,
        "val_normal": 1,
    }
    assert len(list((config.output_dir / "candle" / "test" / "bad").glob("*.png"))) == 1
    assert len(list((config.output_dir / "candle" / "test" / "good").glob("*.png"))) == 1
    assert report["test_policy"].startswith("Official test rows are unchanged")
