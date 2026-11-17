"""Configuration for the standalone FastAPI backend."""

import os


DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
REQUEST_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
MAX_IMAGE_WIDTH = 10_000
MAX_IMAGE_HEIGHT = 10_000
MAX_IMAGE_PIXELS = 25_000_000


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
