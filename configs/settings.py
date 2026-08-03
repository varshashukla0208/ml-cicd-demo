"""
=============================================================
Application Configuration

Author      : Varsha Shukla
Project     : Image Classification API

Description
-----------
Centralized configuration management for the entire application.

Configuration Priority
----------------------
1. Environment Variables
2. .env File
3. Default Values

Supported Environments
----------------------
- development
- testing
- production

Used By
-------
- FastAPI
- Render
- Docker
- MLflow
- Model Registry
- Inference
=============================================================
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "artifacts"

LOG_DIR = PROJECT_ROOT / "logs"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


# ============================================================
# Settings
# ============================================================


class Settings(BaseSettings):
    """
    Global application settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================
    # Application
    # ========================================================

    APP_NAME: str = "Image Classification API"

    APP_VERSION: str = "1.0.0"

    APP_DESCRIPTION: str = (
        "Production-ready FastAPI service " "for Image Classification."
    )

    APP_ENV: str = "development"

    DEBUG: bool = True

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in {"release", "prod", "production"}:
                return False

            if normalized in {"dev", "development"}:
                return True

        return value

    # ========================================================
    # Server
    # ========================================================

    HOST: str = "0.0.0.0"

    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    RELOAD: bool = True

    # ========================================================
    # API
    # ========================================================

    API_PREFIX: str = "/"

    DOCS_URL: str = "/docs"

    REDOC_URL: str = "/redoc"

    OPENAPI_URL: str = "/openapi.json"

    MAX_UPLOAD_SIZE_MB: int = 10

    ALLOWED_ORIGINS: list[str] = ["*"]

    # ========================================================
    # Model Loading
    # ========================================================

    MODEL_SOURCE: str = "local"
    # local | mlflow | registry

    MODEL_NAME: str = "ImageClassifier"

    MODEL_ALIAS: str = "production"

    MODEL_VERSION: str = "latest"

    LOCAL_MODEL_PATH: str = "artifacts/model"

    MLFLOW_MODEL_URI: str = ""

    REGISTRY_MODEL_URI: str = "models:/ImageClassifier@production"

    # ========================================================
    # MLflow
    # ========================================================

    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

    MLFLOW_EXPERIMENT_NAME: str = "ImageClassification"

    # ========================================================
    # Logging
    # ========================================================

    LOG_LEVEL: str = "INFO"

    LOG_FORMAT: str = "%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"

    # ========================================================
    # Render
    # ========================================================

    RENDER: bool = False

    # ========================================================
    # Health Check
    # ========================================================

    HEALTH_ENDPOINT: str = "/health"

    # ========================================================
    # Project Paths
    # ========================================================

    PROJECT_ROOT: Path = PROJECT_ROOT

    MODEL_DIR: Path = MODEL_DIR

    LOG_DIR: Path = LOG_DIR

    CHECKPOINT_DIR: Path = CHECKPOINT_DIR

    # ========================================================
    # Environment Helpers
    # ========================================================

    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.APP_ENV.lower() == "development"

    @property
    def IS_PRODUCTION(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def IS_TESTING(self) -> bool:
        return self.APP_ENV.lower() == "testing"


# ============================================================
# Cached Singleton
# ============================================================


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    The configuration is loaded only once during the
    application's lifetime.
    """
    return Settings()


settings = get_settings()
