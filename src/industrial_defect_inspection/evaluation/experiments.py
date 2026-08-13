"""Validation-only experiment aggregation, ranking, and promotion gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from industrial_defect_inspection.utils.io import file_record, write_json

DEFAULT_WEAK_CLASSES = ("crazing", "rolled-in_scale")


@dataclass(frozen=True, slots=True)
class ExperimentObservation:
    name: str
    seed: int
    map50_95: float
    weak_map50_95: float
    cpu_p95_ms: float | None
    model_bytes: int
    evaluation: str
    evaluation_sha256: str = ""
    model_sha256: str = ""
    image_size: int = 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cpu_p95(benchmark_path: Path | None) -> float | None:
    if benchmark_path is None:
        return None
    benchmark = _read_json(benchmark_path)
    values = benchmark.get("pt_cpu")
    if not values:
        return None
    return float(values["wall_ms"]["p95"])


def load_observation(
    name: str,
    seed: int,
    evaluation_path: Path,
    benchmark_path: Path | None = None,
    weak_classes: tuple[str, ...] = DEFAULT_WEAK_CLASSES,
) -> ExperimentObservation:
    evaluation = _read_json(evaluation_path)
    if evaluation.get("split") != "val":
        raise ValueError(f"Experiment selection is validation-only: {evaluation_path}")
    per_class = evaluation["per_class"]
    missing = [class_name for class_name in weak_classes if class_name not in per_class]
    if missing:
        raise ValueError(f"Weak classes missing from evaluation: {missing}")
    model = evaluation["model"]
    evaluation_reference = evaluation_path.as_posix()
    if evaluation_path.is_absolute():
        evaluation_reference = evaluation_path.name
    return ExperimentObservation(
        name=name,
        seed=seed,
        map50_95=float(evaluation["summary"]["map50_95"]),
        weak_map50_95=mean(float(per_class[name]["map50_95"]) for name in weak_classes),
        cpu_p95_ms=_cpu_p95(benchmark_path),
        model_bytes=int(model["bytes"]),
        evaluation=evaluation_reference,
        evaluation_sha256=str(file_record(evaluation_path)["sha256"]),
        model_sha256=str(model["sha256"]),
        image_size=int(evaluation["image_size"]),
    )


def rank_screen(observations: list[ExperimentObservation]) -> list[ExperimentObservation]:
    seeds = {observation.seed for observation in observations}
    if seeds != {42}:
        raise ValueError(f"Screening requires seed 42 for every configuration, got {sorted(seeds)}")
    names = [observation.name for observation in observations]
    if len(names) != len(set(names)):
        raise ValueError("Screening observations must contain one seed per configuration")
    return sorted(
        observations,
        key=lambda item: (
            -item.map50_95,
            -item.weak_map50_95,
            item.cpu_p95_ms if item.cpu_p95_ms is not None else math.inf,
            item.model_bytes,
            item.name,
        ),
    )


def aggregate_configurations(
    observations: list[ExperimentObservation],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ExperimentObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.name].append(observation)
    aggregates: list[dict[str, Any]] = []
    for name, values in grouped.items():
        seeds = sorted(observation.seed for observation in values)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed for experiment {name}: {seeds}")
        maps = [observation.map50_95 for observation in values]
        weak_maps = [observation.weak_map50_95 for observation in values]
        latencies = [
            observation.cpu_p95_ms for observation in values if observation.cpu_p95_ms is not None
        ]
        aggregates.append(
            {
                "name": name,
                "seeds": seeds,
                "runs": len(values),
                "map50_95_mean": mean(maps),
                "map50_95_std": pstdev(maps),
                "weak_map50_95_mean": mean(weak_maps),
                "weak_map50_95_std": pstdev(weak_maps),
                "cpu_p95_ms_mean": mean(latencies) if latencies else None,
                "model_bytes_mean": mean(observation.model_bytes for observation in values),
            }
        )
    return sorted(
        aggregates,
        key=lambda item: (
            -item["map50_95_mean"],
            -item["weak_map50_95_mean"],
            item["cpu_p95_ms_mean"] if item["cpu_p95_ms_mean"] is not None else math.inf,
            item["model_bytes_mean"],
            item["name"],
        ),
    )


def promotion_decision(
    baseline: ExperimentObservation,
    candidate: dict[str, Any],
    *,
    overall_delta_required: float = 0.010,
    weak_delta_required: float = 0.025,
    maximum_overall_drop: float = 0.005,
) -> dict[str, Any]:
    overall_delta = float(candidate["map50_95_mean"]) - baseline.map50_95
    weak_delta = float(candidate["weak_map50_95_mean"]) - baseline.weak_map50_95
    overall_gate = overall_delta >= overall_delta_required
    weak_gate = weak_delta >= weak_delta_required and overall_delta >= -maximum_overall_drop
    return {
        "promoted": overall_gate or weak_gate,
        "candidate": candidate["name"],
        "overall_delta": overall_delta,
        "weak_class_delta": weak_delta,
        "passed_overall_gate": overall_gate,
        "passed_weak_class_gate": weak_gate,
        "thresholds": {
            "overall_delta_required": overall_delta_required,
            "weak_delta_required": weak_delta_required,
            "maximum_overall_drop": maximum_overall_drop,
        },
    }


def compare_experiments(manifest_path: Path, output_dir: Path) -> Path:
    manifest = _read_json(manifest_path)
    weak_classes = tuple(manifest.get("weak_classes", DEFAULT_WEAK_CLASSES))
    observations = [
        load_observation(
            item["name"],
            int(item["seed"]),
            Path(item["evaluation"]),
            Path(item["benchmark"]) if item.get("benchmark") else None,
            weak_classes,
        )
        for item in manifest["observations"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest.get("stage") == "screen":
        ranked = rank_screen(observations)
        result_path = output_dir / "screen_ranking.json"
        write_json(
            result_path,
            {
                "selection_split": "val",
                "screen_seed": 42,
                "weak_classes": list(weak_classes),
                "ranking": [asdict(observation) for observation in ranked],
                "selected_for_confirmation": [item.name for item in ranked[:2]],
                "test_metrics_used_for_selection": False,
            },
        )
        with (output_dir / "screen_ranking.csv").open("w", encoding="utf-8", newline="") as handle:
            screen_rows = [asdict(observation) for observation in ranked]
            writer = csv.DictWriter(handle, fieldnames=list(screen_rows[0]))
            writer.writeheader()
            writer.writerows(screen_rows)
        return result_path

    baseline_spec = manifest["baseline"]
    baseline = load_observation(
        baseline_spec["name"],
        int(baseline_spec["seed"]),
        Path(baseline_spec["evaluation"]),
        Path(baseline_spec["benchmark"]) if baseline_spec.get("benchmark") else None,
        weak_classes,
    )
    aggregates = aggregate_configurations(observations)
    if not aggregates:
        raise ValueError("Experiment manifest contains no observations")
    required_seeds = sorted(int(seed) for seed in manifest.get("required_seeds", [42, 43, 44]))
    incomplete = {
        item["name"]: item["seeds"] for item in aggregates if item["seeds"] != required_seeds
    }
    if incomplete:
        raise ValueError(
            f"Final comparison requires seeds {required_seeds} for every candidate: {incomplete}"
        )
    decision = promotion_decision(baseline, aggregates[0])
    result_path = output_dir / "experiment_comparison.json"
    write_json(
        result_path,
        {
            "selection_split": "val",
            "weak_classes": list(weak_classes),
            "standard_deviation": "population (ddof=0)",
            "baseline": asdict(baseline),
            "observations": [asdict(observation) for observation in observations],
            "configurations": aggregates,
            "promotion": decision,
            "test_metrics_used_for_selection": False,
        },
    )
    with (output_dir / "experiment_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare validation-only training experiments.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compare_experiments(Path(args.manifest), Path(args.output))
    print(f"Experiment comparison: {result.resolve()}")


if __name__ == "__main__":
    main()
