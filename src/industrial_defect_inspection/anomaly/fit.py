"""Fit and calibrate PatchCore memory banks using normal-only validation data."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from industrial_defect_inspection.anomaly.backend import (
    create_patchcore,
    iter_prediction_outputs,
    require_anomalib,
)
from industrial_defect_inspection.config import PatchCoreConfig, load_patchcore_config
from industrial_defect_inspection.utils.io import environment_snapshot, file_record, write_json
from industrial_defect_inspection.utils.runtime import (
    configure_run_logger,
    prepare_runtime,
    resolve_device,
)


def calibrate_thresholds(
    image_scores: list[float],
    pixel_scores: list[np.ndarray],
    image_quantile: float,
    pixel_quantile: float,
) -> tuple[float, float]:
    """Calibrate conservative thresholds from normal validation scores only."""
    if not image_scores or not pixel_scores:
        raise ValueError("Normal validation predictions are required for threshold calibration")
    image_threshold = float(np.quantile(np.asarray(image_scores), image_quantile))
    pixels = np.concatenate(
        [np.asarray(values, dtype=np.float32).reshape(-1) for values in pixel_scores]
    )
    pixel_threshold = float(np.quantile(pixels, pixel_quantile))
    return image_threshold, pixel_threshold


def _engine_device(device: str | int) -> tuple[str, int]:
    return ("gpu", 1) if isinstance(device, int) else ("cpu", 1)


def fit_category(config: PatchCoreConfig, category: str) -> dict[str, Any]:
    category_root = config.dataset_dir / category
    train_dir = category_root / "train" / "good"
    validation_dir = category_root / "val" / "good"
    manifest_path = category_root / "manifest.csv"
    for path, description in (
        (train_dir, "normal training directory"),
        (validation_dir, "normal validation directory"),
        (manifest_path, "prepared manifest"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"VisA {description} not found for '{category}': {path}. "
                "Run idi-prepare-visa first."
            )

    output = config.output_dir / category
    output.mkdir(parents=True, exist_ok=True)
    prepare_runtime(output)
    logger = configure_run_logger(f"industrial_defect_inspection.anomaly.fit.{category}", output)
    started = time.perf_counter()
    checkpoint = output / "model.ckpt"
    manifest_file = output / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "stage": "anomaly_fit",
        "status": "running",
        "category": category,
        "started_at": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "dataset_manifest": file_record(manifest_path),
        "environment": environment_snapshot(),
        "calibration_policy": "Normal-only validation quantiles; official test remains untouched.",
    }
    write_json(manifest_file, run_manifest)
    try:
        folder_type, engine_type, _ = require_anomalib()
        resolved_device = resolve_device(config.device)
        accelerator, devices = _engine_device(resolved_device)
        try:
            import torch

            torch.manual_seed(config.seed)
        except ImportError:
            pass
        datamodule = folder_type(
            name=f"visa_{category}",
            root=category_root,
            normal_dir="train/good",
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            num_workers=config.workers,
            test_split_mode="none",
            val_split_mode="none",
            seed=config.seed,
        )
        model = create_patchcore(config)
        engine = engine_type(
            accelerator=accelerator,
            devices=devices,
            default_root_dir=output,
            logger=False,
            max_epochs=1,
            enable_checkpointing=False,
        )
        logger.info("Fitting PatchCore for %s on %s", category, resolved_device)
        engine.fit(model=model, datamodule=datamodule)
        engine.trainer.save_checkpoint(str(checkpoint))
        predictions = engine.predict(
            model=model,
            data_path=validation_dir,
            return_predictions=True,
        )
        if predictions is None:
            raise RuntimeError("Anomalib did not return normal validation predictions")
        outputs = list(iter_prediction_outputs(predictions))
        image_threshold, pixel_threshold = calibrate_thresholds(
            [item[0] for item in outputs],
            [item[1] for item in outputs],
            config.image_threshold_quantile,
            config.pixel_threshold_quantile,
        )
        metadata: dict[str, Any] = {
            "model_version": "visa-patchcore-resnet18-v1",
            "category": category,
            "algorithm": "PatchCore",
            "backbone": config.backbone,
            "layers": config.layers,
            "image_size": config.image_size,
            "coreset_sampling_ratio": config.coreset_sampling_ratio,
            "num_neighbors": config.num_neighbors,
            "precision": "float32",
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
            "image_threshold_quantile": config.image_threshold_quantile,
            "pixel_threshold_quantile": config.pixel_threshold_quantile,
            "calibration_images": len(outputs),
            "calibration_data": "normal-only validation split",
            "dataset": "VisA",
            "dataset_license": "CC BY 4.0",
            "checkpoint": file_record(checkpoint),
        }
        write_json(output / "metadata.json", metadata)
        run_manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": time.perf_counter() - started,
                "checkpoint": file_record(checkpoint),
                "metadata": file_record(output / "metadata.json"),
            }
        )
        write_json(manifest_file, run_manifest)
        return metadata
    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(manifest_file, run_manifest)
        logger.exception("PatchCore fit failed")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit PatchCore models for prepared VisA data.")
    parser.add_argument("--config", default="configs/anomaly/patchcore_resnet18.yaml")
    parser.add_argument("--category", choices=("candle", "capsules", "pcb1"), action="append")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_patchcore_config(args.config)
    categories = args.category or config.categories
    for category in categories:
        metadata = fit_category(config, category)
        print(
            f"Fitted {category}: image_threshold={metadata['image_threshold']:.6f}, "
            f"pixel_threshold={metadata['pixel_threshold']:.6f}"
        )


if __name__ == "__main__":
    main()
