"""Configuration for the standalone FastAPI backend."""

import os
from pathlib import Path


DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
REQUEST_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
MAX_IMAGE_WIDTH = 10_000
MAX_IMAGE_HEIGHT = 10_000
MAX_IMAGE_PIXELS = 25_000_000

# config.py is under <project>/backend/app/core, so this remains stable even
# when the application is launched with a different working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CHECKPOINT_PATH = "checkpoints/resnet18_v2_best.pth"
DEFAULT_MODEL_CLASSES_PATH = "artifacts/resnet18_v2_classes.json"
DEFAULT_MODEL_DEVICE = "auto"
DEFAULT_MODEL_TOP_K = 3


def _resolve_project_path(raw_path: str) -> Path:
    """Resolve absolute paths as given and relative paths from project root."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _get_model_device() -> str:
    device = os.getenv("MODEL_DEVICE", DEFAULT_MODEL_DEVICE)
    if device not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("MODEL_DEVICE must be one of: auto, cpu, cuda")
    return device


def _get_model_top_k() -> int:
    raw_value = os.getenv("MODEL_TOP_K", str(DEFAULT_MODEL_TOP_K))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("MODEL_TOP_K must be a positive integer") from exc
    if str(value) != raw_value.strip() or value <= 0:
        raise RuntimeError("MODEL_TOP_K must be a positive integer")
    return value


MODEL_CHECKPOINT_PATH = _resolve_project_path(
    os.getenv("MODEL_CHECKPOINT_PATH", DEFAULT_MODEL_CHECKPOINT_PATH)
)
MODEL_CLASSES_PATH = _resolve_project_path(
    os.getenv("MODEL_CLASSES_PATH", DEFAULT_MODEL_CLASSES_PATH)
)
MODEL_DEVICE = _get_model_device()
MODEL_TOP_K = _get_model_top_k()


def resolve_model_device(cuda_available: bool) -> str:
    """Resolve configured device against runtime CUDA availability.

    The caller supplies CUDA availability (for example from torch.cuda) so
    configuration does not add a PyTorch import or model-loading behavior.
    """
    if MODEL_DEVICE == "cpu":
        return "cpu"
    if MODEL_DEVICE == "cuda":
        if not cuda_available:
            raise RuntimeError("MODEL_DEVICE is 'cuda' but CUDA is not available")
        return "cuda"
    return "cuda" if cuda_available else "cpu"


def get_max_upload_bytes() -> int:
    """Read the upload limit from the environment, defaulting to 10 MiB."""
    raw_value = os.getenv("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("MAX_UPLOAD_BYTES must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError("MAX_UPLOAD_BYTES must be a positive integer")
    return value


MAX_UPLOAD_BYTES = get_max_upload_bytes()
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + REQUEST_MULTIPART_OVERHEAD_BYTES
