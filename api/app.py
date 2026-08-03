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
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from configs.settings import settings
from inference.model_loader import ModelLoader
from inference.predictor import Predictor
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

    app.state.start_time = time.time()

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
        # Build Predictor Singleton
        # --------------------------------------------------

        predictor = Predictor(
            model=model,
            device=loader.inference_device,
            config=config,
        )

        # --------------------------------------------------
        # Store objects inside FastAPI state
        # --------------------------------------------------

        app.state.config = config
        app.state.model = model
        app.state.device = loader.inference_device
        app.state.predictor = predictor

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
# Request ID Middleware
# ==========================================================


@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    """
    Attach X-Request-ID to incoming requests and outgoing responses.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Routers
# ==========================================================

app.include_router(
    router,
)


# ==========================================================
# Startup Log
# ==========================================================

logger.info(
    "%s initialized successfully.",
    settings.APP_NAME,
)
