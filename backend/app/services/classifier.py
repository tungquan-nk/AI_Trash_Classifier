"""Model lifecycle boundary for the backend classifier."""

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from app.core import config


class ClassifierNotReadyError(RuntimeError):
    """Raised when prediction is requested before the model is loaded."""


class ModelConfigurationError(RuntimeError):
    """Raised when configured model metadata or device settings are invalid."""


class ModelFileError(RuntimeError):
    """Raised when a configured model file is missing or unreadable."""


class ModelLoadError(RuntimeError):
    """Raised when the model architecture or checkpoint cannot be loaded."""


class ClassifierService:
    """Load and retain the model once for the lifetime of the application."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.classes: list[str] | None = None
        self.device: str | None = None
        self._transform: Any | None = None

    def load(self) -> None:
        """Read class metadata and load the configured ResNet18 checkpoint."""
        if self.model is not None:
            return

        self.cleanup()
        classes = self._read_classes(config.MODEL_CLASSES_PATH)
        checkpoint_path = Path(config.MODEL_CHECKPOINT_PATH)
        if not checkpoint_path.is_file():
            raise ModelFileError(f"Model checkpoint file not found: {checkpoint_path}")

        try:
            import torch
            from torchvision import models
        except Exception as exc:
            raise ModelLoadError("Could not import PyTorch or Torchvision for model loading") from exc

        try:
            device = config.resolve_model_device(torch.cuda.is_available())
        except RuntimeError as exc:
            raise ModelConfigurationError(f"Could not resolve model device: {exc}") from exc

        try:
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        except Exception as exc:
            raise ModelLoadError("Could not initialize the ResNet18 model") from exc

        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load checkpoint into ResNet18 from {checkpoint_path}; "
                "the checkpoint may be incompatible with this architecture"
            ) from exc

        self.model = model
        self.classes = classes
        self.device = device

    @staticmethod
    def _read_classes(classes_path: Path) -> list[str]:
        """Load and validate the ordered class labels from the configured JSON."""
        if not Path(classes_path).is_file():
            raise ModelFileError(f"Model classes file not found: {classes_path}")
        try:
            with Path(classes_path).open("r", encoding="utf-8") as classes_file:
                classes = json.load(classes_file)
        except (OSError, UnicodeError) as exc:
            raise ModelFileError(f"Could not read model classes file: {classes_path}") from exc
        except json.JSONDecodeError as exc:
            raise ModelConfigurationError(f"Model classes JSON is invalid: {classes_path}") from exc

        if not isinstance(classes, list):
            raise ModelConfigurationError("Model classes JSON must contain a list")
        if not classes:
            raise ModelConfigurationError("Model classes list must not be empty")
        if any(not isinstance(label, str) or not label.strip() for label in classes):
            raise ModelConfigurationError("Every model class label must be a non-empty string")
        if len(set(classes)) != len(classes):
            raise ModelConfigurationError("Model classes list must not contain duplicate labels")
        return classes

    def cleanup(self) -> None:
        """Drop references to loaded model state during application shutdown."""
        self.model = None
        self.classes = None
        self.device = None
        self._transform = None

    def _get_transform(self) -> Any:
        """Build and cache the model's deterministic evaluation transform."""
        if self._transform is None:
            from torchvision import transforms

            self._transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        return self._transform

    def predict(self, image_bytes: bytes) -> list[dict[str, str | float]]:
        """Preprocess image bytes and return the configured top class predictions."""
        if self.model is None or self.classes is None or self.device is None:
            raise ClassifierNotReadyError("Prediction is unavailable: the model is not loaded.")

        import torch
        from PIL import Image

        transform = self._get_transform()
        with Image.open(BytesIO(image_bytes)) as image:
            input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            k = min(config.MODEL_TOP_K, len(self.classes))
            values, indices = torch.topk(probabilities, k=k, dim=1)

        return [
            {"label": self.classes[int(index)], "confidence": float(value)}
            for value, index in zip(values[0].tolist(), indices[0].tolist())
        ]


classifier_service = ClassifierService()
