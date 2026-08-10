"""Export a trained checkpoint to a portable ONNX model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from industrial_defect_inspection.training.train import resolve_device


def export_onnx(
    model_path: Path,
    image_size: int,
    device: str,
    output: Path | None = None,
    overwrite: bool = False,
) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before export") from exc
    exported = Path(
        YOLO(str(model_path)).export(
            format="onnx",
            imgsz=image_size,
            device=resolve_device(device),
            half=False,
            dynamic=False,
            simplify=False,
        )
    )
    if output is None:
        return exported
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a YOLO checkpoint to ONNX.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_onnx(
        model_path=Path(args.model),
        image_size=args.image_size,
        device=args.device,
        output=Path(args.output) if args.output else None,
        overwrite=args.overwrite,
    )
    print(f"Exported ONNX model: {result.resolve()}")


if __name__ == "__main__":
    main()
