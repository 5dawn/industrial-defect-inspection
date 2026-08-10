"""Single-image and directory inference CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_defect_inspection.config import load_inference_config
from industrial_defect_inspection.data.prepare import IMAGE_SUFFIXES
from industrial_defect_inspection.inference.engine import InferenceEngine
from industrial_defect_inspection.utils.io import write_json


def collect_sources(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.casefold() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(f"Image or directory not found: {source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run industrial defect inference.")
    parser.add_argument("--config", default="configs/infer/default.yaml")
    parser.add_argument("--model", help="Override model path from the inference configuration.")
    parser.add_argument("--source", required=True, help="Image or directory to process.")
    parser.add_argument("--output", help="Override the output directory from the configuration.")
    parser.add_argument("--confidence", type=float, help="Override confidence threshold.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_inference_config(args.config)
    if args.model:
        config = config.model_copy(update={"model": Path(args.model)})
    if args.output:
        config = config.model_copy(update={"output_dir": Path(args.output)})
    if not config.model.is_file():
        raise FileNotFoundError(
            f"Model file not found: {config.model}. Train a model first or pass --model."
        )
    engine = InferenceEngine(config)
    sources = collect_sources(Path(args.source))
    if not sources:
        raise ValueError(f"No supported images found under {args.source}")
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        result, annotated = engine.predict(source, confidence=args.confidence)
        annotated.save(output / f"{source.stem}_annotated.jpg", quality=92)
        write_json(output / f"{source.stem}.json", result.model_dump(mode="json"))
    print(f"Processed {len(sources)} image(s); results: {output.resolve()}")


if __name__ == "__main__":
    main()
