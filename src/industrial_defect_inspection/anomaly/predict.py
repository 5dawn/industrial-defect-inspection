"""Single-image anomaly localization command."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from industrial_defect_inspection.anomaly.engine import AnomalyEngine
from industrial_defect_inspection.config import load_anomaly_inference_config
from industrial_defect_inspection.data.prepare import IMAGE_SUFFIXES
from industrial_defect_inspection.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Localize anomalies in one VisA-style image.")
    parser.add_argument("--config", default="configs/anomaly/infer.yaml")
    parser.add_argument("--category", required=True, choices=("candle", "capsules", "pcb1"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_anomaly_inference_config(args.config)
    if args.output:
        config = config.model_copy(update={"output_dir": Path(args.output)})
    if args.device:
        config = config.model_copy(update={"device": args.device})
    source = Path(args.source)
    if not source.is_file() or source.suffix.casefold() not in IMAGE_SUFFIXES:
        raise FileNotFoundError(f"Supported input image not found: {source}")
    engine = AnomalyEngine(config)
    if not engine.category_available(args.category):
        raise FileNotFoundError(engine.unavailable_message(args.category))
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    result, visuals = engine.predict(image, args.category)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{source.stem}_{args.category}"
    visuals.heatmap.save(config.output_dir / f"{prefix}_heatmap.png")
    visuals.mask.save(config.output_dir / f"{prefix}_mask.png")
    visuals.overlay.save(config.output_dir / f"{prefix}_overlay.jpg", quality=92)
    write_json(config.output_dir / f"{prefix}.json", result.model_dump(mode="json"))
    print(f"Anomaly result: {result.is_anomalous}; outputs: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
