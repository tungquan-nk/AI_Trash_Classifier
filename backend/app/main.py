"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes.predictions import router
from app.core.config import MAX_REQUEST_BYTES
from app.core.middleware import RequestSizeLimitMiddleware
from app.schemas.prediction import ErrorResponse
from app.services.classifier import classifier_service


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the classifier before serving requests and clean it up at shutdown."""
    try:
        classifier_service.load()
        yield
    finally:
        classifier_service.cleanup()

api = FastAPI(
    title="AI Trash Classifier Backend",
    version="0.1.0",
    description="Image upload and trash classification API powered by the configured ResNet18 model.",
    lifespan=lifespan,
)
api.include_router(router)


@api.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Keep HTTP errors in the documented, non-sensitive response shape."""
    detail = exc.detail if isinstance(exc.detail, str) else "Request could not be processed."
    response = ErrorResponse(detail=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
        headers=exc.headers,
    )


@api.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Do not reflect parser internals or user-supplied values to clients."""
    response = ErrorResponse(detail="Request is missing required fields or is malformed.")
    return JSONResponse(status_code=422, content=response.model_dump())


@api.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures server-side and return a generic client error."""
    logging.getLogger(__name__).error(
        "Unhandled backend error",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    response = ErrorResponse(detail="Internal server error.")
    return JSONResponse(status_code=500, content=response.model_dump())


# This pure ASGI middleware runs before routing and multipart form parsing while
# keeping the exported ``app`` as a regular FastAPI instance.
api.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
app = api
