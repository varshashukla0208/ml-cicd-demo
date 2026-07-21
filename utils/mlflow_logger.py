"""
Utils/mlflow_logger.py

Centralized MLflow logger for experiment tracking.

Responsibilities
----------------
- Configure MLflow tracking
- Start/End runs
- Log parameters
- Log metrics
- Log artifacts
- Log models (optional)

Author: Varsha Shukla
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch

import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class MLFlowLogger:
    """
     Wrapper around MLflow APIs.

     Example
     -------
    logger = MLFlowLogger(config["mlflow"])

     logger.start_run(run_name="CNN_Run_1")
     logger.log_params({...})
     logger.log_metric("train_loss", 0.35, step=1)
     logger.end_run()
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "sqlite:///mlflow.db",
    ) -> None:

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    ####################################################################
    # Run Management
    ####################################################################

    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Start a new MLflow run.
        """
        mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        """
        End current MLflow run.
        """
        mlflow.end_run()

    ####################################################################
    # Parameters
    ####################################################################

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    ####################################################################
    # Metrics
    ####################################################################

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        mlflow.log_metrics(metrics, step=step)

    ####################################################################
    # Tags
    ####################################################################

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]) -> None:
        mlflow.set_tags(tags)

    ####################################################################
    # Artifacts
    ####################################################################

    def log_artifact(
        self,
        artifact_path: str,
        artifact_dir: Optional[str] = None,
    ) -> None:
        """
        Log a single file.

        Example
        -------
        checkpoints/best_model.pth
        """
        mlflow.log_artifact(
            artifact_path,
            artifact_path=artifact_dir,
        )

    def log_artifacts(
        self,
        directory: str,
        artifact_dir: Optional[str] = None,
    ) -> None:
        """
        Log an entire directory.
        """
        mlflow.log_artifacts(
            directory,
            artifact_path=artifact_dir,
        )

    ####################################################################
    # Model Logging
    ####################################################################

    def log_pytorch_model(
        self,
        model,
        artifact_path: str = "model",
    ) -> None:
        """
        Store complete PyTorch model.

        Optional.
        """
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
        )

    ####################################################################
    # Utility
    ####################################################################

    def get_artifact_uri(self) -> str:
        """
        Return artifact storage URI.
        """
        return mlflow.get_artifact_uri()

    def active_run(self):
        """
        Return active MLflow run.
        """
        return mlflow.active_run()

    def is_active(self) -> bool:
        """
        Check whether a run is active.
        """
        return mlflow.active_run() is not None

    def log_file(self, file_path: str, artifact_dir: str | None = None):
        path = Path(file_path)

        if path.exists():
            mlflow.log_artifact(
                str(path),
                artifact_path=artifact_dir,
            )
