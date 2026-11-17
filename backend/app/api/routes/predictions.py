"""Health and image prediction endpoints."""

from io import BytesIO
import warnings

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import (
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_WIDTH,
    MAX_UPLOAD_BYTES,
)
from app.schemas.prediction import ErrorResponse, PredictionResponse
from app.services.classifier import ClassifierNotReadyError, classifier_service

router = APIRouter()


def validate_image(data: bytes) -> None:
    """Reject empty, oversized, or undecodable image uploads."""
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="Image exceeds the upload size limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                if (
                    width > MAX_IMAGE_WIDTH
                    or height > MAX_IMAGE_HEIGHT
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="Image dimensions exceed the supported limits.",
                    )
                image.load()
    except HTTPException:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Image dimensions exceed the supported limits.",
        ) from None
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        EOFError,
        ValueError,
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is not a valid image.") from None


@router.get("/health")
def health_check() -> dict[str, str]:
    """Report that the API process is responsive."""
    return {"status": "ok"}


@router.post(
    "/api/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image upload."},
        413: {"model": ErrorResponse, "description": "Image or request exceeds a configured size limit."},
        422: {"model": ErrorResponse, "description": "The request is missing the required file field."},
        503: {"model": ErrorResponse, "description": "AI prediction is not configured."},
        500: {"model": ErrorResponse, "description": "Unexpected server error."},
    },
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Validate an uploaded image and forward it to the classifier service."""
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    validate_image(data)
    try:
        predictions = classifier_service.predict(data)
    except ClassifierNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return PredictionResponse(filename=file.filename or "upload", predictions=predictions)
