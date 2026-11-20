"""API and upload safety tests."""

from io import BytesIO
import asyncio
import math
from pathlib import Path
import struct
import sys
import threading
import zlib
from unittest.mock import Mock

import httpx
from fastapi.testclient import TestClient
from PIL import Image
import pytest

# Keep the backend's ``app`` package distinct from the repository-root app.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import MAX_IMAGE_PIXELS, MAX_IMAGE_WIDTH, MAX_REQUEST_BYTES, MAX_UPLOAD_BYTES
from app.core import config
from app.api.routes import predictions as prediction_routes
from app.main import app
from conftest import FixedLogitsModel

client = TestClient(app)


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="green").save(buffer, format="PNG")
    return buffer.getvalue()


def make_png_header_with_dimensions(width: int, height: int) -> bytes:
    """Keep a valid PNG header while setting dimensions beyond policy."""
    data = bytearray(make_png())
    data[16:24] = struct.pack(">II", width, height)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]) & 0xFFFFFFFF)
    return bytes(data)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_empty_file_is_rejected() -> None:
    response = client.post("/api/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is empty."}


def test_missing_file_field_has_consistent_error_shape() -> None:
    response = client.post("/api/predict", files={})
    assert response.status_code == 422
    assert response.json() == {"detail": "Request is missing required fields or is malformed."}


def test_fake_image_is_rejected() -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("not-image.png", b"plain text", "image/png")},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid image."}


def test_corrupt_image_that_cannot_be_decoded_is_rejected() -> None:
    corrupt_png = make_png()[:30]
    response = client.post(
        "/api/predict",
        files={"file": ("corrupt.png", corrupt_png, "image/png")},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid image."}


def test_image_over_pixel_limit_is_rejected_before_classifier(monkeypatch) -> None:
    width = MAX_IMAGE_WIDTH + 1
    height = max(1, (MAX_IMAGE_PIXELS // width) + 1)
    oversized_dimensions = make_png_header_with_dimensions(width, height)
    predict = Mock()
    monkeypatch.setattr(prediction_routes.classifier_service, "predict", predict)
    response = client.post(
        "/api/predict",
        files={"file": ("large-dimensions.png", oversized_dimensions, "image/png")},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Image dimensions exceed the supported limits."}
    predict.assert_not_called()


def test_file_over_limit_is_rejected(monkeypatch) -> None:
    predict = Mock()
    monkeypatch.setattr(prediction_routes.classifier_service, "predict", predict)
    response = client.post(
        "/api/predict",
        files={"file": ("too-large.png", b"x" * (MAX_UPLOAD_BYTES + 1), "image/png")},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Image exceeds the upload size limit."}
    predict.assert_not_called()


def test_total_request_limit_without_content_length_is_rejected() -> None:
    boundary = "request-limit-test"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="large.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii")
    body = prefix + (b"x" * (MAX_REQUEST_BYTES + 1)) + f"\r\n--{boundary}--\r\n".encode("ascii")
    chunks = (body[offset : offset + 64 * 1024] for offset in range(0, len(body), 64 * 1024))
    response = client.post(
        "/api/predict",
        content=chunks,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the total upload size limit."}


def test_valid_image_reports_model_not_ready() -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("sample.png", make_png(), "image/png")},
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Prediction is unavailable: the model is not loaded."
    }


def test_valid_image_with_model_returns_prediction_response(monkeypatch) -> None:
    service = prediction_routes.classifier_service
    monkeypatch.setattr(config, "MODEL_TOP_K", 3)
    monkeypatch.setattr(service, "model", FixedLogitsModel())
    monkeypatch.setattr(service, "classes", ["cardboard", "metal", "glass"])
    monkeypatch.setattr(service, "device", "cpu")
    monkeypatch.setattr(service, "_transform", None)

    response = client.post(
        "/api/predict",
        files={"file": ("sample.png", make_png(), "image/png")},
    )

    exponential_sum = sum(math.exp(value) for value in (0.0, 2.0, 1.0))
    assert response.status_code == 200
    assert response.json()["filename"] == "sample.png"
    predictions = response.json()["predictions"]
    assert [item["label"] for item in predictions] == ["metal", "glass", "cardboard"]
    assert [item["confidence"] for item in predictions] == pytest.approx(
        [math.exp(2.0) / exponential_sum, math.exp(1.0) / exponential_sum, 1.0 / exponential_sum]
    )
    assert set(response.json()) == {"filename", "predictions"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_remains_responsive_while_prediction_is_running(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()
    result_state: dict[str, bool | None] = {"released": None}

    def blocking_predict(image_bytes: bytes) -> list[dict[str, str | float]]:
        started.set()
        result_state["released"] = release.wait(timeout=5)
        return [{"label": "cardboard", "confidence": 1.0}]

    monkeypatch.setattr(prediction_routes.classifier_service, "predict", blocking_predict)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        prediction_task = asyncio.create_task(
            async_client.post(
                "/api/predict",
                files={"file": ("sample.png", make_png(), "image/png")},
            )
        )
        try:
            started_in_time = await asyncio.get_running_loop().run_in_executor(
                None, started.wait, 5
            )
            assert started_in_time is True
            health_response = await async_client.get("/health")
            assert health_response.status_code == 200
            assert prediction_task.done() is False
        finally:
            release.set()

        prediction_response = await asyncio.wait_for(prediction_task, timeout=5)

    assert result_state["released"] is True
    assert prediction_response.status_code == 200


def test_unexpected_classifier_error_does_not_leak_internal_details(monkeypatch) -> None:
    def fail_with_internal_path(image_bytes: bytes) -> list[dict[str, str | float]]:
        raise RuntimeError("D:/internal/checkpoints/model.pth failed")

    monkeypatch.setattr(prediction_routes.classifier_service, "predict", fail_with_internal_path)
    safe_client = TestClient(app, raise_server_exceptions=False)
    response = safe_client.post(
        "/api/predict",
        files={"file": ("sample.png", make_png(), "image/png")},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}


def test_error_responses_are_documented_with_error_schema() -> None:
    openapi = client.get("/openapi.json").json()
    responses = openapi["paths"]["/api/predict"]["post"]["responses"]
    for status_code in ("400", "413", "422", "500", "503"):
        assert responses[status_code]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/ErrorResponse"
        )
