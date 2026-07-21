"""
=============================================================
File: schemas.py

Description
-----------
Pydantic schemas used by the FastAPI application.

Responsibilities
----------------
- Request validation
- Response serialization
- Swagger documentation
- API contract definition

Author:
    Varsha Shukla
=============================================================
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# ============================================================
# Root Response
# ============================================================


class RootResponse(BaseModel):
    """
    Root endpoint response.
    """

    application: str = Field(
        ...,
        description="Application name",
        examples=["Image Classification API"],
    )

    version: str = Field(
        ...,
        description="API version",
        examples=["1.0.0"],
    )

    status: str = Field(
        ...,
        description="Application status",
        examples=["running"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "application": "Image Classification API",
                "version": "1.0.0",
                "status": "running",
            }
        }
    )


# ============================================================
# Health Response
# ============================================================


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str = Field(
        ...,
        description="Health status",
        examples=["healthy"],
    )

    uptime: Optional[str] = Field(
        default=None,
        description="Server uptime",
        examples=["3h 42m"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "uptime": "3h 42m",
            }
        }
    )


# ============================================================
# Version Response
# ============================================================


class VersionResponse(BaseModel):
    """
    Version information.
    """

    api_version: str = Field(
        ...,
        examples=["1.0.0"],
    )

    model_version: str = Field(
        ...,
        examples=["1.0.0"],
    )

    model_name: str = Field(
        ...,
        examples=["EfficientNetV2-S"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_version": "1.0.0",
                "model_version": "1.0.0",
                "model_name": "EfficientNetV2-S",
            }
        }
    )


# ============================================================
# Prediction Response
# ============================================================


class PredictionResponse(BaseModel):
    """
    Prediction result.
    """

    predicted_class: str = Field(
        ...,
        description="Predicted class",
        examples=["Dog"],
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence",
        examples=[0.9873],
    )

    inference_time_ms: float = Field(
        ...,
        ge=0,
        description="Inference latency",
        examples=[18.52],
    )

    model_version: str = Field(
        ...,
        examples=["1.0.0"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_class": "Dog",
                "confidence": 0.9873,
                "inference_time_ms": 18.52,
                "model_version": "1.0.0",
            }
        }
    )


# ============================================================
# Error Response
# ============================================================


class ErrorResponse(BaseModel):
    """
    Standard API error response.
    """

    detail: str = Field(
        ...,
        examples=["Unsupported image format."],
    )

    error_code: Optional[str] = Field(
        default=None,
        examples=["INVALID_FILE_TYPE"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Unsupported image format.",
                "error_code": "INVALID_FILE_TYPE",
            }
        }
    )
