"""Configuration-driven Ultralytics training."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from industrial_defect_inspection.config import TrainConfig, load_train_config
from industrial_defect_inspection.utils.io import (
    environment_snapshot,
    file_record,
    write_json,
)
from industrial_defect_inspection.utils.runtime import (
    configure_run_logger,
    prepare_runtime,
    resolve_device,
)


def train(config: TrainConfig) -> Path:
    if not config.dataset.is_file():
        raise FileNotFoundError(
            f"Prepared dataset YAML not found: {config.dataset}. Run idi-prepare first."
        )
    run_dir = config.project / config.name
    if run_dir.exists() and any(run_dir.iterdir()) and not config.resume:
        raise FileExistsError(
            f"Training run directory is not empty: {run_dir}. "
            "Use a unique run name or an explicit resume checkpoint."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_runtime(run_dir)
    logger = configure_run_logger("industrial_defect_inspection.training", run_dir)
    started_at = datetime.now(UTC)
    started_clock = time.perf_counter()
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "stage": "training",
        "status": "running",
        "started_at": started_at.isoformat(),
        "resolved_config": config.model_dump(mode="json"),
        "dataset": file_record(config.dataset),
        "environment": environment_snapshot(),
    }
    write_json(manifest_path, manifest)
    write_json(run_dir / "environment.json", environment_snapshot())
    write_json(run_dir / "resolved_config.json", config.model_dump(mode="json"))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before training") from exc

    try:
        if isinstance(config.resume, str) and config.resume:
            model = YOLO(config.resume)
            resume: bool | str = config.resume
        else:
            model = YOLO(config.model)
            resume = config.resume

        device = resolve_device(config.device)
        logger.info("Starting training on device %s", device)
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
            device=device,
            seed=config.seed,
            deterministic=config.deterministic,
            pretrained=config.pretrained,
            amp=config.amp and device != "cpu",
            resume=resume,
            exist_ok=True,
            plots=True,
            **augmentation,
        )
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError(f"Training completed but best checkpoint was not found: {best}")
        manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": time.perf_counter() - started_clock,
                "best_checkpoint": file_record(best),
            }
        )
        write_json(manifest_path, manifest)
        logger.info("Training complete; best checkpoint: %s", best.resolve())
        return best
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": time.perf_counter() - started_clock,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(manifest_path, manifest)
        logger.exception("Training failed")
        raise


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
