"""
=============================================================
File : app.py

Description
-----------
FastAPI application entry point.

Responsibilities
----------------
- Create FastAPI application
- Configure middleware
- Load application settings
- Load trained model at startup
- Register API routes
- Handle startup/shutdown lifecycle
- Prepare application for Render/Docker deployment

Author:
    Varsha Shukla
=============================================================
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from configs.settings import settings
from inference.model_loader import ModelLoader
from training.train import load_config

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Application Lifespan
# ==========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI startup and shutdown lifecycle.
    """

    logger.info("=" * 70)
    logger.info("Starting %s", settings.APP_NAME)
    logger.info("Environment : %s", settings.APP_ENV)
    logger.info("=" * 70)

    try:

        # --------------------------------------------------
        # Load training configuration
        # --------------------------------------------------

        config = load_config()

        logger.info("Configuration loaded successfully.")

        # --------------------------------------------------
        # Build and load model
        # --------------------------------------------------

        loader = ModelLoader(config)

        model = loader.load()

        logger.info("Inference model loaded.")

        # --------------------------------------------------
        # Store objects inside FastAPI state
        # --------------------------------------------------

        app.state.config = config
        app.state.model = model
        app.state.device = loader.inference_device

        logger.info(
            "Inference Device : %s",
            loader.inference_device,
        )

        logger.info("Application startup completed.")

    except Exception:

        logger.exception("Application startup failed.")

        raise

    yield

    logger.info("=" * 70)
    logger.info("Stopping Application...")
    logger.info("=" * 70)


# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
    debug=settings.DEBUG,
    lifespan=lifespan,
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Routers
# ==========================================================

app.include_router(
    router,
    prefix=settings.API_PREFIX,
)


# ==========================================================
# Startup Log
# ==========================================================

logger.info(
    "%s initialized successfully.",
    settings.APP_NAME,
)
