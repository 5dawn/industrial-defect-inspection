import json
import zipfile
from pathlib import Path

import pytest

from industrial_defect_inspection.release.package import (
    SOURCE_ALLOWLIST,
    package_release,
    sha256_file,
)


def make_release_sources(root: Path) -> None:
    for destination, relative in SOURCE_ALLOWLIST.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if destination == "experiment_summary.json":
            path.write_text(
                json.dumps(
                    {
                        "training": {
                            "duration_seconds": 10,
                            "checkpoint_sha256": "abc",
                            "environment": {"python": "3.11", "torch": "2.12"},
                        }
                    }
                ),
                encoding="utf-8",
            )
        elif path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix == ".csv":
            path.write_text("class_name,map50_95\ndefect,0.5\n", encoding="utf-8")
        else:
            path.write_text(f"safe {destination}\n", encoding="utf-8")


def test_package_release_is_source_only_and_deterministic(tmp_path: Path) -> None:
    make_release_sources(tmp_path)
    first = package_release("v0.1.1", tmp_path / "out-one", root=tmp_path)
    second = package_release("v0.1.1", tmp_path / "out-two", root=tmp_path)

    assert sha256_file(first["zip"]) == sha256_file(second["zip"])
    assert first["zip_checksum"].read_text(encoding="utf-8") == (
        f"{sha256_file(first['zip'])}  {first['zip'].name}\n"
    )
    with zipfile.ZipFile(first["zip"]) as archive:
        names = archive.namelist()
        assert "package_manifest.json" in names
        assert "SHA256SUMS.txt" in names
        assert not any(Path(name).suffix in {".pt", ".onnx", ".jpg", ".xml"} for name in names)
        manifest = json.loads(archive.read("package_manifest.json"))
    assert manifest["weights_included"] is False
    assert manifest["dataset_pixels_included"] is False


def test_package_release_rejects_absolute_local_path(tmp_path: Path) -> None:
    make_release_sources(tmp_path)
    (tmp_path / SOURCE_ALLOWLIST["model_card.md"]).write_text(
        "checkpoint: C:\\Users\\person\\best.pt", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Absolute local path"):
        package_release("v0.1.1", tmp_path / "out", root=tmp_path)


def test_package_release_rejects_unix_temporary_path(tmp_path: Path) -> None:
    make_release_sources(tmp_path)
    (tmp_path / SOURCE_ALLOWLIST["dataset_card.md"]).write_text(
        "generated from /tmp/private/data.xml", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Absolute local path"):
        package_release("v0.1.1", tmp_path / "out", root=tmp_path)


def test_package_release_refuses_nonempty_output(tmp_path: Path) -> None:
    make_release_sources(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("user file", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        package_release("v0.1.1", output, root=tmp_path)
