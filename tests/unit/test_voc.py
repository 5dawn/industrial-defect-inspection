from pathlib import Path

import pytest

from industrial_defect_inspection.data.voc import AnnotationError, parse_voc_annotation


def write_xml(path: Path, class_name: str = "Cr", xmax: int = 90) -> None:
    path.write_text(
        f"""<?xml version="1.0"?>
<annotation>
  <filename>sample.bmp</filename>
  <size><width>100</width><height>80</height><depth>1</depth></size>
  <object>
    <name>{class_name}</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>{xmax}</xmax><ymax>70</ymax></bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )


def test_parse_voc_alias_and_yolo_conversion(tmp_path: Path) -> None:
    source = tmp_path / "sample.xml"
    write_xml(source)
    annotation = parse_voc_annotation(source, ["crazing"], {"Cr": "crazing"})

    assert annotation.width == 100
    assert annotation.height == 80
    assert annotation.boxes[0].class_name == "crazing"
    values = annotation.boxes[0].to_yolo(0, 100, 80).split()
    assert values[0] == "0"
    assert [float(value) for value in values[1:]] == pytest.approx([0.5, 0.5625, 0.8, 0.625])


def test_parse_voc_rejects_unknown_class(tmp_path: Path) -> None:
    source = tmp_path / "sample.xml"
    write_xml(source, class_name="unexpected")

    with pytest.raises(AnnotationError, match="Unknown class"):
        parse_voc_annotation(source, ["crazing"])


def test_parse_voc_rejects_out_of_bounds_box(tmp_path: Path) -> None:
    source = tmp_path / "sample.xml"
    write_xml(source, xmax=101)

    with pytest.raises(AnnotationError, match="exceeds image bounds"):
        parse_voc_annotation(source, ["Cr"])


def test_parse_voc_removes_exact_duplicate_boxes(tmp_path: Path) -> None:
    source = tmp_path / "sample.xml"
    write_xml(source)
    content = source.read_text(encoding="utf-8")
    object_xml = content[content.index("  <object>") : content.index("</annotation>")]
    source.write_text(
        content.replace("</annotation>", object_xml + "</annotation>"), encoding="utf-8"
    )

    annotation = parse_voc_annotation(source, ["Cr"])

    assert len(annotation.boxes) == 1
    assert annotation.duplicate_boxes_removed == 1
