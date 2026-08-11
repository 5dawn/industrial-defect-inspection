"""Build a license-safe public report from completed local experiments."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from industrial_defect_inspection.utils.io import write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_summary(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json")
    with (run_dir / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    checkpoint = manifest["best_checkpoint"]
    environment = manifest["environment"]
    return {
        "epochs_completed": len(rows),
        "best_epoch": int(best["epoch"]),
        "duration_seconds": manifest["duration_seconds"],
        "checkpoint_bytes": checkpoint["bytes"],
        "checkpoint_sha256": checkpoint["sha256"],
        "environment": {
            key: environment.get(key)
            for key in (
                "python",
                "platform",
                "git",
                "torch",
                "cuda_available",
                "cuda_version",
                "cuda_device",
                "ultralytics",
            )
        },
    }


def _public_evaluation(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": report["split"],
        "image_count": report["image_count"],
        "metric_confidence": report["metric_confidence"],
        "operating_confidence": report["operating_confidence"],
        "iou_threshold": report["iou_threshold"],
        "image_size": report["image_size"],
        "summary": report["summary"],
        "per_class": report["per_class"],
        "error_analysis": {
            key: value for key, value in report["error_analysis"].items() if key != "manifest"
        },
    }


def _public_benchmark(report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"parity": report["parity"]}
    for name in ("pt_cpu", "onnx_cpu", "pt_gpu"):
        if name not in report:
            continue
        values = report[name]
        result[name] = {
            "model_bytes": values["model"]["bytes"],
            "model_sha256": values["model"]["sha256"],
            "device": values["device"],
            "image_size": values["image_size"],
            "confidence": values["confidence"],
            "count": values["count"],
            "warmup": values["warmup"],
            "wall_ms": values["wall_ms"],
            "throughput_fps": values["throughput_fps"],
            "components_mean_ms": values["components_mean_ms"],
            "resources": values["resources"],
        }
    return result


def publish_report(
    run_dir: Path,
    validation_report: Path,
    test_report: Path,
    benchmark_report: Path,
    output_dir: Path,
    figure_dir: Path,
) -> Path:
    validation = _read_json(validation_report)
    test = _read_json(test_report)
    benchmark = _read_json(benchmark_report)
    if test["operating_confidence_selected_on_this_split"]:
        raise ValueError("Refusing to publish a test report whose threshold was selected on test")
    if validation["operating_confidence"] != test["operating_confidence"]:
        raise ValueError("Validation and test operating confidence do not match")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "YOLO26n",
        "dataset": "NEU-DET",
        "training": _training_summary(run_dir),
        "validation": _public_evaluation(validation),
        "test": _public_evaluation(test),
        "benchmark": _public_benchmark(benchmark),
        "disclaimer": "Research and portfolio evaluation; not a production quality-control claim.",
    }
    summary_path = output_dir / "experiment_summary.json"
    write_json(summary_path, summary)

    with (output_dir / "test_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["class_name", *next(iter(test["per_class"].values())).keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, values in test["per_class"].items():
            writer.writerow({"class_name": class_name, **values})

    figure_sources = {
        run_dir / "results.png": figure_dir / "training_curves.png",
        validation_report.parent / "threshold_sweep.png": figure_dir / "threshold_sweep.png",
        test_report.parent / "confusion_matrix_normalized.png": figure_dir
        / "confusion_matrix_normalized.png",
        test_report.parent / "BoxPR_curve.png": figure_dir / "precision_recall_curve.png",
    }
    for source, destination in figure_sources.items():
        if source.is_file():
            shutil.copy2(source, destination)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish aggregate experiment evidence.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", default="reports/metrics/published/experiment")
    parser.add_argument("--figures", default="reports/figures/published/experiment")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = publish_report(
        Path(args.run),
        Path(args.validation),
        Path(args.test),
        Path(args.benchmark),
        Path(args.output),
        Path(args.figures),
    )
    print(f"Published aggregate report: {result.resolve()}")


if __name__ == "__main__":
    main()
