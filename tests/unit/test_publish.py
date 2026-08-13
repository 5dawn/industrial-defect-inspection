import csv
import json
from pathlib import Path

import pytest

from industrial_defect_inspection.evaluation.publish import publish_report


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def evaluation_payload(split: str, selected: bool) -> dict:
    return {
        "split": split,
        "image_count": 2,
        "metric_confidence": 0.001,
        "operating_confidence": 0.43,
        "operating_confidence_selected_on_this_split": selected,
        "iou_threshold": 0.5,
        "image_size": 640,
        "summary": {"tp": 1, "fp": 0, "fn": 0, "precision": 1, "recall": 1, "f1": 1},
        "per_class": {
            "defect": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "precision": 1,
                "recall": 1,
                "f1": 1,
                "ap50": 1,
                "map50_95": 1,
            }
        },
        "error_analysis": {"manifest": "private.csv", "false_positive_boxes": 0},
    }


def make_run(run_dir: Path) -> None:
    run_dir.mkdir()
    write_json(
        run_dir / "run_manifest.json",
        {
            "duration_seconds": 10,
            "best_checkpoint": {"bytes": 5, "sha256": "abc"},
            "environment": {"python": "3.11"},
        },
    )
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP50-95(B)"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "metrics/mAP50-95(B)": 0.5})


def test_publish_report_keeps_raw_error_manifest_private(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    validation = tmp_path / "val" / "evaluation.json"
    test = tmp_path / "test" / "evaluation.json"
    benchmark = tmp_path / "benchmark.json"
    write_json(validation, evaluation_payload("val", True))
    write_json(test, evaluation_payload("test", False))
    write_json(benchmark, {"parity": {"passed": True}})

    result = publish_report(
        run_dir, validation, test, benchmark, tmp_path / "published", tmp_path / "figures"
    )
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert payload["test"]["error_analysis"] == {"false_positive_boxes": 0}
    assert (tmp_path / "published" / "test_per_class.csv").is_file()


def test_publish_refuses_threshold_selected_on_test(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    validation = tmp_path / "val.json"
    test = tmp_path / "test.json"
    benchmark = tmp_path / "benchmark.json"
    write_json(validation, evaluation_payload("val", True))
    write_json(test, evaluation_payload("test", True))
    write_json(benchmark, {"parity": {"passed": True}})

    with pytest.raises(ValueError, match="selected on test"):
        publish_report(
            run_dir, validation, test, benchmark, tmp_path / "published", tmp_path / "figures"
        )
