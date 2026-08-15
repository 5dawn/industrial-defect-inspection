import os
from pathlib import Path

from industrial_defect_inspection.utils.runtime import prepare_runtime, resolve_device


def test_prepare_runtime_uses_writable_configured_root(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("IDI_RUNTIME_DIR", str(runtime_root))
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)

    result = prepare_runtime(tmp_path / "outputs")

    assert result == runtime_root
    assert (runtime_root / "ultralytics").is_dir()
    assert (runtime_root / "matplotlib").is_dir()
    assert (runtime_root / "huggingface").is_dir()
    assert Path(os.environ["HF_HOME"]) == runtime_root / "huggingface"
    assert resolve_device("cpu") == "cpu"
