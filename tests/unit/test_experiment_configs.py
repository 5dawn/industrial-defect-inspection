from pathlib import Path

from industrial_defect_inspection.config import load_train_config


def test_registered_v2_experiment_configs_are_unique_and_complete() -> None:
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "configs" / "train" / "experiments" / "v2"
    configs = [load_train_config(path) for path in sorted(config_dir.glob("*.yaml"))]

    assert len(configs) == 9
    assert len({config.name for config in configs}) == 9
    by_family: dict[str, set[int]] = {}
    for config in configs:
        family = config.name.rsplit("-seed", maxsplit=1)[0]
        by_family.setdefault(family, set()).add(config.seed)

    assert by_family == {
        "efficient-512": {42, 43, 44},
        "weak-640": {42, 43, 44},
        "weak-no-flip-640": {42, 43, 44},
    }
