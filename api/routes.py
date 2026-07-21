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
- Call inference pipeline
- Return structured JSON responses

Author:
    Varsha Shukla
==========================================================
"""

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from api.schemas import (
    RootResponse,
    HealthResponse,
    VersionResponse,
    PredictionResponse,
)

from inference.predictor import Predictor
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
        application="Image Classification API",
        status="running",
        version="1.0.0",
    )


# ==========================================================
# Health Endpoint
# ==========================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
async def health():
    """
    Health check endpoint.
    """

    return HealthResponse(
        status="healthy",
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
        model_version="1.0.0",
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
            status_code=400,
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
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Supported formats: "
                ".jpg, .jpeg, .png, .bmp"
            ),
        )

    # ------------------------------------------------------
    # Read uploaded image
    # ------------------------------------------------------

    image_bytes = await file.read()

    if len(image_bytes) == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # ------------------------------------------------------
    # Run inference
    # ------------------------------------------------------

    predictor = Predictor(
        model=request.app.state.model,
        device=request.app.state.device,
        config=request.app.state.config,
    )

    try:

        prediction = predictor.predict(
            image_bytes=image_bytes,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )

    return prediction
