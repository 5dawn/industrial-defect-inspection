from pathlib import Path

import pytest
from PIL import Image

from industrial_defect_inspection.config import DataConfig
from industrial_defect_inspection.data.prepare import prepare_dataset


def make_dataset(root: Path) -> DataConfig:
    image_dir = root / "raw" / "images"
    annotation_dir = root / "raw" / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    for index in range(20):
        stem = f"sample_{index:02d}"
        class_name = "a" if index < 10 else "b"
        color = 0 if index in (0, 1) else index * 10
        xmax = 21 if index == 1 else 20
        Image.new("L", (40, 30), color=color).save(image_dir / f"{stem}.bmp")
        (annotation_dir / f"{stem}.xml").write_text(
            f"""<annotation>
<filename>{stem}.bmp</filename>
<size><width>40</width><height>30</height><depth>1</depth></size>
<object><name>{class_name}</name><bndbox>
<xmin>2</xmin><ymin>3</ymin><xmax>{xmax}</xmax><ymax>25</ymax>
</bndbox></object></annotation>""",
            encoding="utf-8",
        )
    return DataConfig(
        dataset_name="synthetic",
        raw_images_dir=image_dir,
        raw_annotations_dir=annotation_dir,
        output_dir=root / "processed",
        report_dir=root / "reports",
        class_names=["a", "b"],
        seed=42,
        preview_count=2,
    )


def test_prepare_dataset_end_to_end(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)
    metadata = prepare_dataset(config)

    assert metadata["source_statistics"]["images"] == 20
    assert metadata["splits"]["train"]["images"] == 14
    assert metadata["splits"]["val"]["images"] == 4
    assert metadata["splits"]["test"]["images"] == 2
    assert len(list((config.output_dir / "labels" / "train").glob("*.txt"))) == 14
    assert (config.output_dir / "dataset.yaml").is_file()
    assert (config.output_dir / "metadata.json").is_file()
    assert len(list(config.report_dir.glob("*.jpg"))) == 2
    assert metadata["duplicate_content_groups"] == [
        {
            "sha256": metadata["files"][0]["sha256"],
            "stems": ["sample_00", "sample_01"],
            "annotations_identical": False,
        }
    ]
    duplicate_splits = [
        split_name
        for split_name in ("train", "val", "test")
        if (config.output_dir / "images" / split_name / "sample_00.bmp").exists()
        or (config.output_dir / "images" / split_name / "sample_01.bmp").exists()
    ]
    assert len(duplicate_splits) == 1


def test_prepare_refuses_to_overwrite(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)
    prepare_dataset(config)

    try:
        prepare_dataset(config)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Expected processed data overwrite protection")


def test_prepare_can_fail_on_conflicting_duplicate_annotations(tmp_path: Path) -> None:
    config = make_dataset(tmp_path).model_copy(update={"duplicate_annotation_policy": "error"})

    with pytest.raises(ValueError, match="duplicate-content group"):
        prepare_dataset(config)
