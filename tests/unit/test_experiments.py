import json
from pathlib import Path

import pytest

from industrial_defect_inspection.evaluation.experiments import (
    ExperimentObservation,
    aggregate_configurations,
    compare_experiments,
    load_observation,
    promotion_decision,
    rank_screen,
)


def observation(
    name: str,
    seed: int,
    score: float,
    weak: float,
    latency: float | None = None,
) -> ExperimentObservation:
    return ExperimentObservation(name, seed, score, weak, latency, 100, f"{name}.json")


def test_screen_ties_use_weak_score_latency_and_size() -> None:
    ranked = rank_screen(
        [
            observation("slow", 42, 0.5, 0.3, 30),
            observation("weak", 42, 0.5, 0.2, 10),
            observation("winner", 42, 0.5, 0.3, 20),
        ]
    )
    assert [item.name for item in ranked] == ["winner", "slow", "weak"]


def test_aggregate_three_seeds_and_promotion_gates() -> None:
    values = [
        observation("candidate", 42, 0.43, 0.20),
        observation("candidate", 43, 0.44, 0.22),
        observation("candidate", 44, 0.45, 0.24),
    ]
    aggregate = aggregate_configurations(values)[0]
    baseline = observation("baseline", 42, 0.42, 0.19)
    decision = promotion_decision(baseline, aggregate)

    assert aggregate["seeds"] == [42, 43, 44]
    assert aggregate["map50_95_mean"] == pytest.approx(0.44)
    assert decision["promoted"] is True
    assert decision["passed_overall_gate"] is True


def test_weak_gate_allows_small_bounded_overall_drop() -> None:
    baseline = observation("baseline", 42, 0.42, 0.18)
    candidate = {"name": "weak", "map50_95_mean": 0.417, "weak_map50_95_mean": 0.21}
    decision = promotion_decision(baseline, candidate)
    assert decision["promoted"] is True
    assert decision["passed_weak_class_gate"] is True


def test_load_observation_rejects_test_metrics(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text(
        json.dumps(
            {
                "split": "test",
                "summary": {"map50_95": 0.5},
                "per_class": {},
                "model": {"bytes": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation-only"):
        load_observation("bad", 42, path)


def test_duplicate_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate seed"):
        aggregate_configurations(
            [observation("same", 42, 0.4, 0.2), observation("same", 42, 0.5, 0.3)]
        )


def test_screen_rejects_non_registered_seed() -> None:
    with pytest.raises(ValueError, match="requires seed 42"):
        rank_screen([observation("candidate", 43, 0.4, 0.2)])


def test_final_comparison_rejects_incomplete_seed_set(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "split": "val",
                "summary": {"map50_95": 0.4},
                "per_class": {
                    "crazing": {"map50_95": 0.1},
                    "rolled-in_scale": {"map50_95": 0.2},
                },
                "model": {"bytes": 100, "sha256": "model-hash"},
                "image_size": 640,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "baseline": {"name": "baseline", "seed": 42, "evaluation": str(evaluation)},
                "observations": [{"name": "candidate", "seed": 42, "evaluation": str(evaluation)}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires seeds"):
        compare_experiments(manifest, tmp_path / "comparison")


def test_screen_comparison_writes_top_two_without_test_metrics(tmp_path: Path) -> None:
    evaluation = tmp_path / "validation.json"
    evaluation.write_text(
        json.dumps(
            {
                "split": "val",
                "summary": {"map50_95": 0.4},
                "per_class": {
                    "crazing": {"map50_95": 0.1},
                    "rolled-in_scale": {"map50_95": 0.2},
                },
                "model": {"bytes": 100, "sha256": "model-hash"},
                "image_size": 640,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "screen.json"
    manifest.write_text(
        json.dumps(
            {
                "stage": "screen",
                "observations": [
                    {"name": name, "seed": 42, "evaluation": str(evaluation)}
                    for name in ("c", "a", "b")
                ],
            }
        ),
        encoding="utf-8",
    )

    result = compare_experiments(manifest, tmp_path / "screen-output")
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["selected_for_confirmation"] == ["a", "b"]
    assert payload["test_metrics_used_for_selection"] is False
    assert (result.parent / "screen_ranking.csv").is_file()
