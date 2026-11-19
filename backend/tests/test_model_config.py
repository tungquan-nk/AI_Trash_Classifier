"""Model configuration tests, without loading a model or importing PyTorch."""

import importlib
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import config


def reload_config(monkeypatch, **values):
    for name in (
        "MODEL_CHECKPOINT_PATH",
        "MODEL_CLASSES_PATH",
        "MODEL_DEVICE",
        "MODEL_TOP_K",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))
    return importlib.reload(config)


def test_model_defaults_point_to_existing_v2_files(monkeypatch) -> None:
    cfg = reload_config(monkeypatch)
    assert cfg.MODEL_CHECKPOINT_PATH == cfg.PROJECT_ROOT / "checkpoints/resnet18_v2_best.pth"
    assert cfg.MODEL_CLASSES_PATH == cfg.PROJECT_ROOT / "artifacts/resnet18_v2_classes.json"
    assert cfg.MODEL_CHECKPOINT_PATH.is_file()
    assert cfg.MODEL_CLASSES_PATH.is_file()
    assert cfg.MODEL_DEVICE == "auto"
    assert cfg.MODEL_TOP_K == 3


def test_model_paths_can_be_overridden_by_relative_environment_paths(monkeypatch) -> None:
    cfg = reload_config(
        monkeypatch,
        MODEL_CHECKPOINT_PATH="custom/model.pth",
        MODEL_CLASSES_PATH="custom/classes.json",
    )
    assert cfg.MODEL_CHECKPOINT_PATH == cfg.PROJECT_ROOT / "custom/model.pth"
    assert cfg.MODEL_CLASSES_PATH == cfg.PROJECT_ROOT / "custom/classes.json"


def test_absolute_model_paths_are_kept(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.pth"
    classes = tmp_path / "classes.json"
    cfg = reload_config(
        monkeypatch,
        MODEL_CHECKPOINT_PATH=checkpoint,
        MODEL_CLASSES_PATH=classes,
    )
    assert cfg.MODEL_CHECKPOINT_PATH == checkpoint.resolve()
    assert cfg.MODEL_CLASSES_PATH == classes.resolve()


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_supported_model_devices(monkeypatch, device: str) -> None:
    cfg = reload_config(monkeypatch, MODEL_DEVICE=device)
    assert cfg.MODEL_DEVICE == device


def test_invalid_model_device_is_a_clear_configuration_error(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="MODEL_DEVICE must be one of"):
        reload_config(monkeypatch, MODEL_DEVICE="gpu")


def test_device_resolution_handles_auto_cpu_and_required_cuda(monkeypatch) -> None:
    cfg = reload_config(monkeypatch, MODEL_DEVICE="auto")
    assert cfg.resolve_model_device(True) == "cuda"
    assert cfg.resolve_model_device(False) == "cpu"
    cfg = reload_config(monkeypatch, MODEL_DEVICE="cpu")
    assert cfg.resolve_model_device(True) == "cpu"
    cfg = reload_config(monkeypatch, MODEL_DEVICE="cuda")
    assert cfg.resolve_model_device(True) == "cuda"
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        cfg.resolve_model_device(False)


@pytest.mark.parametrize("raw_value", ["0", "-1", "1.5", "three", ""])
def test_model_top_k_must_be_a_positive_integer(monkeypatch, raw_value: str) -> None:
    with pytest.raises(RuntimeError, match="MODEL_TOP_K must be a positive integer"):
        reload_config(monkeypatch, MODEL_TOP_K=raw_value)


def test_model_top_k_can_be_overridden(monkeypatch) -> None:
    cfg = reload_config(monkeypatch, MODEL_TOP_K="5")
    assert cfg.MODEL_TOP_K == 5


def test_relative_paths_do_not_depend_on_current_working_directory(monkeypatch, tmp_path) -> None:
    cfg = reload_config(monkeypatch)
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert cfg.MODEL_CHECKPOINT_PATH == cfg.PROJECT_ROOT / "checkpoints/resnet18_v2_best.pth"
        assert cfg.MODEL_CLASSES_PATH == cfg.PROJECT_ROOT / "artifacts/resnet18_v2_classes.json"
    finally:
        os.chdir(original_cwd)
