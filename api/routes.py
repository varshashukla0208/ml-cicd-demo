"""
==========================================================
File : routes.py

Description
-----------
Defines all REST API endpoints.

Responsibilities
----------------
- Receive HTTP requests
- Validate uploaded files
- Call inference pipeline (non-blocking)
- Return structured JSON responses

Author:
    Varsha Shukla
==========================================================
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from api.schemas import (
    RootResponse,
    HealthResponse,
    VersionResponse,
    PredictionResponse,
)

from configs.settings import settings

# ==========================================================
# Router
# ==========================================================

router = APIRouter()


# ==========================================================
# Root Endpoint
# ==========================================================


@router.get(
    "/",
    response_model=RootResponse,
    tags=["Root"],
)
async def root():
    """
    Root endpoint.
    """

    return RootResponse(
        application=settings.APP_NAME,
        status="running",
        version=settings.APP_VERSION,
    )


# ==========================================================
# Health Endpoint
# ==========================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
async def health(
    request: Request,
):
    """
    Health check endpoint with readiness check and uptime calculation.
    """

    model_loaded = (
        hasattr(request.app.state, "model") and request.app.state.model is not None
    )

    health_status = "healthy" if model_loaded else "unhealthy"

    start_time: Optional[float] = getattr(request.app.state, "start_time", None)

    if start_time:
        uptime_seconds = int(time.time() - start_time)
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        seconds = uptime_seconds % 60
        uptime_str = f"{hours}h {minutes}m {seconds}s"
    else:
        uptime_str = "unknown"

    return HealthResponse(
        status=health_status,
        uptime=uptime_str,
    )


# ==========================================================
# Version Endpoint
# ==========================================================


@router.get(
    "/version",
    response_model=VersionResponse,
    tags=["Version"],
)
async def version(
    request: Request,
):
    """
    API version endpoint.
    """

    return VersionResponse(
        api_version=settings.APP_VERSION,
        model_version=settings.MODEL_VERSION,
        model_name=settings.MODEL_NAME,
    )


# ==========================================================
# Prediction Endpoint
# ==========================================================


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Predict image class.
    """

    # ------------------------------------------------------
    # Validate filename
    # ------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    # ------------------------------------------------------
    # Validate extension
    # ------------------------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    filename = file.filename.lower()

    if not any(filename.endswith(ext) for ext in allowed_extensions):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported image format. "
                "Supported formats: "
                ".jpg, .jpeg, .png, .bmp"
            ),
        )

    # ------------------------------------------------------
    # Read uploaded image payload
    # ------------------------------------------------------

    image_bytes = await file.read()

    if len(image_bytes) == 0:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ------------------------------------------------------
    # Validate file size limit
    # ------------------------------------------------------

    max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if len(image_bytes) > max_size_bytes:

        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    # ------------------------------------------------------
    # Retrieve Predictor Singleton from app.state
    # ------------------------------------------------------

    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model predictor is not initialized or ready.",
        )

    # ------------------------------------------------------
    # Run non-blocking inference via thread pool
    # ------------------------------------------------------

    try:

        prediction = await asyncio.to_thread(
            predictor.predict,
            image_bytes=image_bytes,
        )

    except ValueError as ve:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(ve)}",
        )

    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    return prediction
