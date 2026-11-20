"""Inference tests using real CPU torch and a deterministic fake model."""

from io import BytesIO
import math
from pathlib import Path
import sys

from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config
from app.services.classifier import ClassifierNotReadyError, ClassifierService


class FixedLogitsModel(torch.nn.Module):
    def __init__(self, logits: list[float] | None = None) -> None:
        super().__init__()
        self.logits = logits or [0.0, 2.0, 1.0]
        self.seen_shape: tuple[int, ...] | None = None
        self.seen_device: torch.device | None = None
        self.inference_mode: bool | None = None
        self.seen_input: torch.Tensor | None = None

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(image_tensor.shape)
        self.seen_device = image_tensor.device
        self.inference_mode = torch.is_inference_mode_enabled()
        self.seen_input = image_tensor.detach().clone()
        return torch.tensor([self.logits], dtype=torch.float32, device=image_tensor.device)


def encode_image(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def ready_service(monkeypatch, *, classes=None, top_k=3, logits=None):
    model = FixedLogitsModel(logits)
    service = ClassifierService()
    service.model = model
    service.classes = classes or ["cardboard", "metal", "glass"]
    service.device = "cpu"
    monkeypatch.setattr(config, "MODEL_TOP_K", top_k)
    return service, model


def test_predictions_are_sorted_and_keep_model_index_class_mapping(monkeypatch) -> None:
    service, _ = ready_service(
        monkeypatch,
        classes=["first-index", "highest-index", "middle-index"],
    )
    predictions = service.predict(encode_image(Image.new("RGB", (8, 8), "white")))

    denominator = sum(math.exp(value) for value in (0.0, 2.0, 1.0))
    assert [entry["label"] for entry in predictions] == [
        "highest-index",
        "middle-index",
        "first-index",
    ]
    assert [entry["confidence"] for entry in predictions] == pytest.approx(
        [math.exp(2.0) / denominator, math.exp(1.0) / denominator, 1.0 / denominator]
    )
    assert sum(entry["confidence"] for entry in predictions) == pytest.approx(1.0)


@pytest.mark.parametrize(("top_k", "class_count", "expected_count"), [(1, 3, 1), (2, 3, 2), (9, 2, 2)])
def test_top_k_is_read_at_prediction_time_and_capped_by_class_count(
    monkeypatch, top_k: int, class_count: int, expected_count: int
) -> None:
    classes = [f"class-{index}" for index in range(class_count)]
    service, _ = ready_service(
        monkeypatch,
        classes=classes,
        top_k=top_k,
        logits=[0.0, 2.0, 1.0][:class_count],
    )
    image_bytes = encode_image(Image.new("RGB", (8, 8), "white"))

    predictions = service.predict(image_bytes)

    assert len(predictions) == expected_count
    assert all(prediction["label"] in classes for prediction in predictions)


def test_predictions_use_plain_python_types(monkeypatch) -> None:
    service, _ = ready_service(monkeypatch)
    prediction = service.predict(encode_image(Image.new("RGB", (8, 8), "white")))[0]

    assert type(prediction["label"]) is str
    assert type(prediction["confidence"]) is float


def test_input_batch_shape_device_and_inference_mode(monkeypatch) -> None:
    service, model = ready_service(monkeypatch)
    service.predict(encode_image(Image.new("RGB", (8, 8), "white")))

    assert model.seen_shape == (1, 3, 224, 224)
    assert model.seen_device == torch.device("cpu")
    assert model.inference_mode is True


def test_preprocessing_matches_rgb_resize_crop_tensor_and_normalize(monkeypatch) -> None:
    service, model = ready_service(monkeypatch)
    pixel = (255, 128, 0)
    service.predict(encode_image(Image.new("RGB", (2, 2), pixel)))

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    normalized = [
        (channel / 255.0 - mean) / std
        for channel, mean, std in zip(pixel, means, stds)
    ]
    expected = torch.tensor(normalized, dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 224, 224)
    torch.testing.assert_close(model.seen_input, expected, rtol=1e-5, atol=1e-5)


def test_preprocessing_matches_reference_for_non_square_color_pattern(monkeypatch) -> None:
    from torchvision import models, transforms

    service, model = ready_service(monkeypatch)
    image = Image.new("RGB", (341, 257))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = ((x * 3 + y) % 256, (x + y * 5) % 256, (x * 7 + y * 2) % 256)

    service.predict(encode_image(image))

    weight_transform = models.ResNet18_Weights.DEFAULT.transforms()
    # Build the reference independently from the documented evaluation pipeline.
    reference_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=weight_transform.mean, std=weight_transform.std),
        ]
    )
    expected = reference_transform(image).unsqueeze(0)
    assert model.seen_input is not None
    assert model.seen_input.shape == (1, 3, 224, 224)
    torch.testing.assert_close(model.seen_input, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "image",
    [Image.new("RGBA", (12, 9), (20, 100, 200, 128)), Image.new("L", (12, 9), 127)],
    ids=["rgba", "grayscale"],
)
def test_rgba_and_grayscale_images_are_supported(monkeypatch, image: Image.Image) -> None:
    service, model = ready_service(monkeypatch)

    predictions = service.predict(encode_image(image))

    assert len(predictions) == 3
    assert model.seen_shape == (1, 3, 224, 224)


@pytest.mark.parametrize("missing_attribute", ["model", "classes", "device"])
def test_predict_rejects_service_without_loaded_state(monkeypatch, missing_attribute: str) -> None:
    service, _ = ready_service(monkeypatch)
    setattr(service, missing_attribute, None)

    with pytest.raises(
        ClassifierNotReadyError,
        match="Prediction is unavailable: the model is not loaded\\.",
    ):
        service.predict(b"not decoded because service is not ready")


def test_transform_is_cached_across_requests(monkeypatch) -> None:
    from torchvision import transforms

    original_compose = transforms.Compose
    compose_calls = []

    def counting_compose(operations):
        compose_calls.append(operations)
        return original_compose(operations)

    monkeypatch.setattr(transforms, "Compose", counting_compose)
    service, _ = ready_service(monkeypatch)
    image_bytes = encode_image(Image.new("RGB", (8, 8), "white"))

    service.predict(image_bytes)
    service.predict(image_bytes)

    assert len(compose_calls) == 1
