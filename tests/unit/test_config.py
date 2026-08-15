from pathlib import Path

import pytest
from pydantic import ValidationError

from industrial_defect_inspection.config import (
    AnomalyEvaluationConfig,
    EvaluationConfig,
    InferenceConfig,
    PatchCoreConfig,
    load_anomaly_inference_config,
    load_evaluation_config,
    load_visa_data_config,
)
from industrial_defect_inspection.evaluation.evaluate import apply_cli_overrides, build_parser


def write_evaluation_config(path: Path, extra: str = "") -> None:
    path.write_text(
        "\n".join(
            [
                "model: artifacts/model.pt",
                "dataset: data/processed/dataset.yaml",
                "output_dir: reports/metrics/validation",
                "split: val",
                "metric_confidence: 0.001",
                "operating_confidence: 0.25",
                extra,
            ]
        ),
        encoding="utf-8",
    )


def test_load_evaluation_config(tmp_path: Path) -> None:
    source = tmp_path / "evaluation.yaml"
    write_evaluation_config(source)

    config = load_evaluation_config(source)

    assert config.model == Path("artifacts/model.pt")
    assert config.dataset == Path("data/processed/dataset.yaml")
    assert config.output_dir == Path("reports/metrics/validation")
    assert config.split == "val"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("unexpected: true", "Extra inputs are not permitted"),
        ("split: train", "Input should be 'val' or 'test'"),
        ("operating_confidence: 1.1", "less than or equal to 1"),
    ],
)
def test_evaluation_config_rejects_invalid_values(tmp_path: Path, extra: str, message: str) -> None:
    source = tmp_path / "evaluation.yaml"
    write_evaluation_config(source, extra)

    with pytest.raises(ValidationError, match=message):
        load_evaluation_config(source)


def test_load_evaluation_config_reports_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match=str(missing).replace("\\", "\\\\")):
        load_evaluation_config(missing)


def test_inference_config_has_configured_output_directory() -> None:
    config = InferenceConfig(
        model=Path("artifacts/model.pt"),
        output_dir=Path("artifacts/custom-predictions"),
    )

    assert config.output_dir == Path("artifacts/custom-predictions")


def test_evaluation_config_rejects_invalid_cli_override() -> None:
    config = EvaluationConfig(
        model=Path("artifacts/model.pt"),
        dataset=Path("data/dataset.yaml"),
        output_dir=Path("reports/metrics"),
    )

    args = build_parser().parse_args(["--model", "artifacts/override.pt", "--confidence", "-0.1"])

    with pytest.raises(ValidationError):
        apply_cli_overrides(config, args)


def test_test_config_requires_frozen_threshold() -> None:
    with pytest.raises(ValidationError, match="frozen operating_confidence"):
        EvaluationConfig(
            model=Path("artifacts/model.pt"),
            dataset=Path("data/dataset.yaml"),
            output_dir=Path("reports/metrics"),
            split="test",
        )


def test_metric_confidence_must_be_below_operating_confidence() -> None:
    with pytest.raises(ValidationError, match="below operating_confidence"):
        EvaluationConfig(
            model=Path("artifacts/model.pt"),
            dataset=Path("data/dataset.yaml"),
            output_dir=Path("reports/metrics"),
            metric_confidence=0.5,
            operating_confidence=0.25,
        )


def test_evaluation_cli_overrides_configured_paths() -> None:
    config = EvaluationConfig(
        model=Path("artifacts/model.pt"),
        dataset=Path("data/dataset.yaml"),
        output_dir=Path("reports/metrics"),
    )
    args = build_parser().parse_args(
        [
            "--model",
            "artifacts/override.pt",
            "--output",
            "reports/override",
            "--split",
            "test",
            "--confidence",
            "0.25",
        ]
    )

    updated = apply_cli_overrides(config, args)

    assert updated.model == Path("artifacts/override.pt")
    assert updated.output_dir == Path("reports/override")
    assert updated.split == "test"
    assert updated.operating_confidence == 0.25


def test_committed_anomaly_configs_are_valid() -> None:
    root = Path(__file__).resolve().parents[2]

    data_config = load_visa_data_config(root / "configs/data/visa.yaml")
    inference_config = load_anomaly_inference_config(root / "configs/anomaly/infer.yaml")

    assert data_config.categories == ["candle", "capsules", "pcb1"]
    assert set(inference_config.checkpoints) == {"candle", "capsules", "pcb1"}


def test_anomaly_configs_reject_duplicate_categories() -> None:
    with pytest.raises(ValidationError, match="non-empty and unique"):
        PatchCoreConfig(
            dataset_dir=Path("data/processed/visa"),
            output_dir=Path("artifacts/anomaly"),
            categories=["candle", "candle"],
        )
    with pytest.raises(ValidationError, match="non-empty and unique"):
        AnomalyEvaluationConfig(
            dataset_dir=Path("data/processed/visa"),
            inference_config=Path("configs/anomaly/infer.yaml"),
            output_dir=Path("artifacts/eval"),
            categories=["pcb1", "pcb1"],
        )
