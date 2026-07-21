"""
Central configuration for the MLflow Model Registry.
"""

from pathlib import Path
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow


# ---------------------------------------------------
# MLflow Tracking
# ---------------------------------------------------

TRACKING_URI = "sqlite:///mlflow.db"

# Registry URI defaults to the tracking backend
REGISTRY_URI = TRACKING_URI


# ---------------------------------------------------
# Experiment
# ---------------------------------------------------

EXPERIMENT_NAME = "Image Classification"


# ---------------------------------------------------
# Model Registry
# ---------------------------------------------------

MODEL_NAME = "ImageClassifier"

DEFAULT_STAGE = "None"


# ---------------------------------------------------
# Artifact Directories
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "saved_models"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

CONFIG_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------
# Initialize MLflow
# ---------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(REGISTRY_URI)
