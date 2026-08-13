import tomllib
from pathlib import Path

import yaml

import industrial_defect_inspection


def test_package_and_release_metadata_versions_match() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))

    expected = pyproject["project"]["version"]
    assert industrial_defect_inspection.__version__ == expected
    assert str(citation["version"]) == expected
