"""Classifier integration boundary.

The project model has not yet been confirmed for backend use. This service
intentionally does not load checkpoints, preprocess images, or invent results.
"""


class ClassifierNotReadyError(RuntimeError):
    """Raised while a confirmed model implementation is not configured."""


class ClassifierService:
    """Placeholder service to be implemented after the AI contract is confirmed."""

    def predict(self, image_bytes: bytes) -> list[dict[str, str | float]]:
        """Classify image bytes once a real model is integrated."""
        del image_bytes
        raise ClassifierNotReadyError(
            "Prediction is unavailable: the AI model has not been integrated."
        )


classifier_service = ClassifierService()
