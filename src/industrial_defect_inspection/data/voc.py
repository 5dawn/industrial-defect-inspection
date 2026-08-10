"""Pascal VOC annotation parsing and YOLO conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


class AnnotationError(ValueError):
    """Raised when an annotation is missing required or valid values."""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    def validate(self, image_width: int, image_height: int) -> None:
        if self.width <= 0 or self.height <= 0:
            raise AnnotationError(f"Zero or negative box area: {self}")
        if self.xmin < 0 or self.ymin < 0:
            raise AnnotationError(f"Negative box coordinate: {self}")
        if self.xmax > image_width or self.ymax > image_height:
            raise AnnotationError(
                f"Box exceeds image bounds ({image_width}x{image_height}): {self}"
            )

    def to_yolo(self, class_id: int, image_width: int, image_height: int) -> str:
        self.validate(image_width, image_height)
        x_center = (self.xmin + self.xmax) / 2.0 / image_width
        y_center = (self.ymin + self.ymax) / 2.0 / image_height
        width = self.width / image_width
        height = self.height / image_height
        return f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"


@dataclass(frozen=True, slots=True)
class VocAnnotation:
    filename: str | None
    width: int
    height: int
    boxes: tuple[BoundingBox, ...]
    duplicate_boxes_removed: int = 0


def _required_text(element: ElementTree.Element | None, field: str, source: Path) -> str:
    if element is None or element.text is None or not element.text.strip():
        raise AnnotationError(f"Missing {field} in {source}")
    return element.text.strip()


def parse_voc_annotation(
    path: str | Path,
    valid_classes: list[str],
    class_aliases: dict[str, str] | None = None,
) -> VocAnnotation:
    source = Path(path)
    try:
        root = ElementTree.parse(source).getroot()
    except ElementTree.ParseError as exc:
        raise AnnotationError(f"Invalid XML in {source}: {exc}") from exc

    size = root.find("size")
    if size is None:
        raise AnnotationError(f"Missing size in {source}")
    try:
        width = int(_required_text(size.find("width"), "size/width", source))
        height = int(_required_text(size.find("height"), "size/height", source))
    except ValueError as exc:
        raise AnnotationError(f"Non-integer image size in {source}") from exc
    if width <= 0 or height <= 0:
        raise AnnotationError(f"Invalid image size {width}x{height} in {source}")

    aliases = class_aliases or {}
    boxes: list[BoundingBox] = []
    seen_boxes: set[BoundingBox] = set()
    duplicate_boxes_removed = 0
    for obj in root.findall("object"):
        raw_name = _required_text(obj.find("name"), "object/name", source)
        class_name = aliases.get(raw_name, raw_name)
        if class_name not in valid_classes:
            raise AnnotationError(f"Unknown class '{raw_name}' in {source}")
        box = obj.find("bndbox")
        if box is None:
            raise AnnotationError(f"Missing bndbox for '{raw_name}' in {source}")
        try:
            parsed_box = BoundingBox(
                class_name=class_name,
                xmin=float(_required_text(box.find("xmin"), "bndbox/xmin", source)),
                ymin=float(_required_text(box.find("ymin"), "bndbox/ymin", source)),
                xmax=float(_required_text(box.find("xmax"), "bndbox/xmax", source)),
                ymax=float(_required_text(box.find("ymax"), "bndbox/ymax", source)),
            )
        except ValueError as exc:
            raise AnnotationError(f"Non-numeric box in {source}") from exc
        parsed_box.validate(width, height)
        if parsed_box in seen_boxes:
            duplicate_boxes_removed += 1
            continue
        seen_boxes.add(parsed_box)
        boxes.append(parsed_box)

    if not boxes:
        raise AnnotationError(f"Annotation contains no objects: {source}")
    filename_node = root.find("filename")
    filename = (
        filename_node.text.strip() if filename_node is not None and filename_node.text else None
    )
    return VocAnnotation(
        filename=filename,
        width=width,
        height=height,
        boxes=tuple(boxes),
        duplicate_boxes_removed=duplicate_boxes_removed,
    )
