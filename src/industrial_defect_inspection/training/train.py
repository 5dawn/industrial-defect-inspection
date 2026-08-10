"""Configuration-driven Ultralytics training."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_defect_inspection.config import TrainConfig, load_train_config
from industrial_defect_inspection.utils.io import environment_snapshot, write_json


def resolve_device(requested: str) -> str | int:
    if requested.casefold() != "auto":
        return requested
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def train(config: TrainConfig) -> Path:
    if not config.dataset.is_file():
        raise FileNotFoundError(
            f"Prepared dataset YAML not found: {config.dataset}. Run idi-prepare first."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before training") from exc

    run_dir = config.project / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "environment.json", environment_snapshot())
    write_json(run_dir / "resolved_config.json", config.model_dump(mode="json"))

    if isinstance(config.resume, str) and config.resume:
        model = YOLO(config.resume)
        resume: bool | str = config.resume
    else:
        model = YOLO(config.model)
        resume = config.resume

    augmentation = config.augmentation.model_dump()
    model.train(
        data=str(config.dataset.resolve()),
        project=str(config.project.resolve()),
        name=config.name,
        epochs=config.epochs,
        patience=config.patience,
        imgsz=config.image_size,
        batch=config.batch,
        workers=config.workers,
        device=resolve_device(config.device),
        seed=config.seed,
        deterministic=config.deterministic,
        pretrained=config.pretrained,
        amp=config.amp and resolve_device(config.device) != "cpu",
        resume=resume,
        exist_ok=True,
        plots=True,
        **augmentation,
    )
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training completed but best checkpoint was not found: {best}")
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an industrial defect detector.")
    parser.add_argument("--config", required=True, help="Path to a training YAML configuration.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    best = train(load_train_config(args.config))
    print(f"Best checkpoint: {best.resolve()}")


if __name__ == "__main__":
    main()
