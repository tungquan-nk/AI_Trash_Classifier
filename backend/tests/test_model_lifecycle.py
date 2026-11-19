"""Model startup and cleanup tests using small PyTorch/Torchvision doubles."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config
from app.main import app
from app.services import classifier as classifier_module


class FakeModel:
    def __init__(self) -> None:
        self.fc = SimpleNamespace(in_features=512)
        self.loaded_state = None
        self.to_device = None
        self.eval_called = False

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict

    def to(self, device: str):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def setup_files(monkeypatch, tmp_path: Path, classes: object = None, checkpoint: bool = True):
    classes_path = tmp_path / "classes.json"
    checkpoint_path = tmp_path / "model.pth"
    if classes is not None:
        classes_path.write_text(json.dumps(classes), encoding="utf-8")
    monkeypatch.setattr(config, "MODEL_CLASSES_PATH", classes_path)
    monkeypatch.setattr(config, "MODEL_CHECKPOINT_PATH", checkpoint_path)
    if checkpoint:
        checkpoint_path.write_bytes(b"test checkpoint placeholder")
    return classes_path, checkpoint_path


def install_torch_stubs(monkeypatch, *, cuda_available: bool = False, load_error=None):
    fake_model = FakeModel()
    created = []
    linear_calls = []
    load_calls = []

    def make_model(*, weights):
        created.append(weights)
        return fake_model

    def make_linear(in_features: int, out_features: int):
        linear_calls.append((in_features, out_features))
        return SimpleNamespace(in_features=in_features, out_features=out_features)

    def load(checkpoint_path, *, map_location, weights_only):
        load_calls.append((checkpoint_path, map_location, weights_only))
        if load_error is not None:
            raise load_error
        return {"weight": "state"}

    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    torch_stub.nn = SimpleNamespace(Linear=make_linear)
    torch_stub.load = load
    torchvision_stub = ModuleType("torchvision")
    torchvision_stub.models = SimpleNamespace(resnet18=make_model)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision_stub)
    return fake_model, created, linear_calls, load_calls


def test_service_loads_classes_model_and_checkpoint(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, ["cardboard", "glass", "metal"])
    model, created, linear_calls, load_calls = install_torch_stubs(monkeypatch)
    resolve_device = Mock(return_value="cpu")
    monkeypatch.setattr(config, "resolve_model_device", resolve_device)
    service = classifier_module.ClassifierService()

    service.load()

    assert service.classes == ["cardboard", "glass", "metal"]
    assert service.model is model
    assert service.device == "cpu"
    assert created == [None]
    assert linear_calls == [(512, 3)]
    assert model.fc.in_features == 512
    assert model.fc.out_features == 3
    resolve_device.assert_called_once_with(False)
    assert load_calls == [(config.MODEL_CHECKPOINT_PATH, "cpu", True)]
    assert model.loaded_state == {"weight": "state"}
    assert model.to_device == "cpu"
    assert model.eval_called


def test_service_load_is_idempotent_once_model_exists(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, ["paper"])
    _, _, _, load_calls = install_torch_stubs(monkeypatch)
    service = classifier_module.ClassifierService()

    service.load()
    service.load()

    assert len(load_calls) == 1


def test_missing_checkpoint_is_reported(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, ["paper"], checkpoint=False)
    service = classifier_module.ClassifierService()
    with pytest.raises(classifier_module.ModelFileError, match="checkpoint file not found"):
        service.load()


def test_missing_classes_file_is_reported(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, checkpoint=True)
    service = classifier_module.ClassifierService()
    with pytest.raises(classifier_module.ModelFileError, match="classes file not found"):
        service.load()


def test_invalid_classes_json_is_reported(monkeypatch, tmp_path: Path) -> None:
    classes_path, _ = setup_files(monkeypatch, tmp_path, ["paper"])
    classes_path.write_text("{invalid", encoding="utf-8")
    service = classifier_module.ClassifierService()
    with pytest.raises(classifier_module.ModelConfigurationError, match="JSON is invalid"):
        service.load()


@pytest.mark.parametrize("classes", [[], ["paper", "paper"], ["paper", " "], [1]])
def test_invalid_class_lists_are_rejected(monkeypatch, tmp_path: Path, classes) -> None:
    setup_files(monkeypatch, tmp_path, classes)
    service = classifier_module.ClassifierService()
    with pytest.raises(classifier_module.ModelConfigurationError):
        service.load()


def test_non_list_class_json_is_rejected(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, {"classes": ["paper"]})
    service = classifier_module.ClassifierService()
    with pytest.raises(classifier_module.ModelConfigurationError, match="must contain a list"):
        service.load()


def test_checkpoint_incompatibility_preserves_original_cause(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, ["paper"])
    model, _, _, _ = install_torch_stubs(monkeypatch)
    incompatibility = RuntimeError("size mismatch")
    model.load_state_dict = Mock(side_effect=incompatibility)
    service = classifier_module.ClassifierService()

    with pytest.raises(classifier_module.ModelLoadError, match="checkpoint may be incompatible") as error:
        service.load()

    assert error.value.__cause__ is incompatibility
    assert service.model is None
    assert service.classes is None
    assert service.device is None


def test_torch_load_failure_preserves_cause_and_does_not_commit_state(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, ["paper"])
    original_error = OSError("cannot read state file")
    install_torch_stubs(monkeypatch, load_error=original_error)
    service = classifier_module.ClassifierService()

    with pytest.raises(classifier_module.ModelLoadError, match="Could not load checkpoint") as error:
        service.load()

    assert error.value.__cause__ is original_error
    assert service.model is None
    assert service.classes is None
    assert service.device is None


def test_torch_import_failure_is_reported_as_model_load_error(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, ["paper"])
    monkeypatch.setitem(sys.modules, "torch", None)
    service = classifier_module.ClassifierService()

    with pytest.raises(classifier_module.ModelLoadError, match="Could not import PyTorch") as error:
        service.load()

    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_required_cuda_fails_clearly_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, ["paper"])
    install_torch_stubs(monkeypatch, cuda_available=False)
    monkeypatch.setattr(config, "MODEL_DEVICE", "cuda")
    service = classifier_module.ClassifierService()

    with pytest.raises(classifier_module.ModelConfigurationError, match="CUDA is not available"):
        service.load()


def test_cleanup_clears_loaded_state_and_transform(monkeypatch, tmp_path: Path) -> None:
    setup_files(monkeypatch, tmp_path, ["paper"])
    install_torch_stubs(monkeypatch)
    service = classifier_module.ClassifierService()
    service.load()
    service._transform = object()

    service.cleanup()

    assert service.model is None
    assert service.classes is None
    assert service.device is None
    assert service._transform is None


def test_cleanup_is_safe_before_load() -> None:
    service = classifier_module.ClassifierService()

    service.cleanup()

    assert service.model is None
    assert service.classes is None
    assert service.device is None
    assert service._transform is None


def test_lifespan_loads_once_and_cleans_up_on_shutdown(monkeypatch) -> None:
    load = Mock()
    cleanup = Mock()
    monkeypatch.setattr(classifier_module.classifier_service, "load", load)
    monkeypatch.setattr(classifier_module.classifier_service, "cleanup", cleanup)

    with TestClient(app):
        load.assert_called_once_with()
        cleanup.assert_not_called()

    cleanup.assert_called_once_with()


def test_lifespan_startup_failure_prevents_app_start_and_cleans_up(monkeypatch) -> None:
    load = Mock(side_effect=classifier_module.ModelLoadError("bad checkpoint"))
    cleanup = Mock()
    monkeypatch.setattr(classifier_module.classifier_service, "load", load)
    monkeypatch.setattr(classifier_module.classifier_service, "cleanup", cleanup)

    with pytest.raises(classifier_module.ModelLoadError, match="bad checkpoint"):
        with TestClient(app):
            pytest.fail("Application startup should not complete")

    load.assert_called_once_with()
    cleanup.assert_called_once_with()
