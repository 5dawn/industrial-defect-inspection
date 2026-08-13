from pathlib import Path

import pytest

from industrial_defect_inspection.config import DataConfig, InferenceConfig, TrainConfig
from industrial_defect_inspection.data.prepare import discover_pairs
from industrial_defect_inspection.evaluation.evaluate import evaluate
from industrial_defect_inspection.inference.engine import InferenceEngine
from industrial_defect_inspection.inference.predict import collect_sources
from industrial_defect_inspection.training.train import train


def test_prepare_missing_data_explains_expected_layout(tmp_path: Path) -> None:
    config = DataConfig(
        dataset_name="neu_det",
        raw_images_dir=tmp_path / "images",
        raw_annotations_dir=tmp_path / "annotations_xml",
        output_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
        class_names=["crazing"],
    )

    with pytest.raises(FileNotFoundError, match="idi-prepare --config") as error:
        discover_pairs(config)

    assert str(config.raw_images_dir) in str(error.value)
    assert str(config.raw_annotations_dir) in str(error.value)


def test_train_reports_missing_prepared_dataset(tmp_path: Path) -> None:
    missing = tmp_path / "dataset.yaml"
    config = TrainConfig(
        model="yolo26n.pt",
        dataset=missing,
        project=tmp_path / "runs",
        name="smoke",
    )

    with pytest.raises(FileNotFoundError, match="Run idi-prepare first"):
        train(config)


def test_train_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {0: defect}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "registered-run"
    run_dir.mkdir(parents=True)
    (run_dir / "user-result.txt").write_text("keep", encoding="utf-8")
    config = TrainConfig(
        model="yolo26n.pt",
        dataset=dataset,
        project=tmp_path / "runs",
        name="registered-run",
    )

    with pytest.raises(FileExistsError, match="not empty"):
        train(config)


def test_evaluate_reports_missing_model_before_loading_backend(tmp_path: Path) -> None:
    missing = tmp_path / "best.pt"

    with pytest.raises(FileNotFoundError, match="Model not found"):
        evaluate(
            model_path=missing,
            dataset_yaml=tmp_path / "dataset.yaml",
            split="val",
            output_dir=tmp_path / "metrics",
            device="cpu",
            metric_confidence=0.001,
            operating_confidence=0.25,
            iou_threshold=0.5,
            image_size=320,
            error_samples=0,
            benchmark_count=1,
            benchmark_warmup=0,
        )


def test_evaluate_reports_missing_dataset_before_loading_backend(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"fixture")

    with pytest.raises(FileNotFoundError, match="Dataset YAML not found"):
        evaluate(
            model_path=model,
            dataset_yaml=tmp_path / "dataset.yaml",
            split="val",
            output_dir=tmp_path / "metrics",
            device="cpu",
            metric_confidence=0.001,
            operating_confidence=0.25,
            iou_threshold=0.5,
            image_size=320,
            error_samples=0,
            benchmark_count=1,
            benchmark_warmup=0,
        )


def test_predict_reports_missing_input_image(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError, match="Image or directory not found"):
        collect_sources(missing)


def test_inference_engine_reports_missing_model_before_import(tmp_path: Path) -> None:
    engine = InferenceEngine(InferenceConfig(model=tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        engine.load()
