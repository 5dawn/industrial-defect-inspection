from pathlib import Path

import pytest
from pydantic import ValidationError

from industrial_defect_inspection.config import (
    EvaluationConfig,
    InferenceConfig,
    load_evaluation_config,
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
                "confidence: 0.25",
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
        ("confidence: 1.1", "less than or equal to 1"),
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
        ]
    )

    updated = apply_cli_overrides(config, args)

    assert updated.model == Path("artifacts/override.pt")
    assert updated.output_dir == Path("reports/override")
    assert updated.split == "test"
