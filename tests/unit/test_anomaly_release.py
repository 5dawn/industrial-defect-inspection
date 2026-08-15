from __future__ import annotations

import json
import zipfile
from pathlib import Path

from industrial_defect_inspection.release.anomaly import package_anomaly_release


def test_anomaly_release_contains_only_models_metadata_and_attribution(tmp_path: Path) -> None:
    checkpoints: dict[str, Path] = {}
    metadata: dict[str, Path] = {}
    for category in ("candle", "capsules", "pcb1"):
        checkpoint = tmp_path / "models" / category / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{category}".encode())
        metadata_path = checkpoint.parent / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "category": category,
                    "model_version": "test",
                    "checkpoint": {"path": f"C:/private/{category}.ckpt"},
                }
            ),
            encoding="utf-8",
        )
        checkpoints[category] = checkpoint
        metadata[category] = metadata_path
    config = tmp_path / "infer.yaml"
    config.write_text(
        "checkpoints:\n"
        + "".join(f"  {key}: {value.as_posix()}\n" for key, value in checkpoints.items())
        + "metadata:\n"
        + "".join(f"  {key}: {value.as_posix()}\n" for key, value in metadata.items()),
        encoding="utf-8",
    )

    archive = package_anomaly_release("v0.3.0", config, tmp_path / "release")

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert "candle.ckpt" in names
        assert "VISADATA_ATTRIBUTION.md" in names
        assert "SHA256SUMS.txt" in names
        assert not any(Path(name).suffix in {".png", ".jpg", ".xml"} for name in names)
        portable = json.loads(bundle.read("candle.json"))
    assert portable["checkpoint_file"] == "candle.ckpt"
    assert "checkpoint" not in portable
