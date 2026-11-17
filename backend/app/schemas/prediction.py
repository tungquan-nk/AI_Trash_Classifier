"""Pydantic schemas for prediction API responses."""

from pydantic import BaseModel, ConfigDict


class PredictionItem(BaseModel):
    """One classified label and its model-produced confidence."""

    label: str
    confidence: float


class PredictionResponse(BaseModel):
    """Successful prediction response."""

    filename: str
    predictions: list[PredictionItem]


class ErrorResponse(BaseModel):
    """Consistent error response body."""

    model_config = ConfigDict(json_schema_extra={"example": {"detail": "Image is invalid."}})

    detail: str
