"""
MLflow Model Registry utilities.

This module registers model artifacts from MLflow runs into the MLflow Model
Registry. It supports both MLflow model directories, such as ``runs:/<run_id>/model``,
and plain checkpoint artifacts, such as ``runs:/<run_id>/checkpoints/best_model.pth``.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities.model_registry import RegisteredModel
from mlflow.exceptions import MlflowException

from registry.exceptions import DuplicateModelError
from registry.exceptions import ModelNotFoundError
from registry.exceptions import VersionNotFoundError
from registry.registry_config import EXPERIMENT_NAME
from registry.registry_config import MODEL_NAME
from registry.registry_config import REGISTRY_URI
from registry.registry_config import TRACKING_URI

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Wrapper around MLflow Model Registry operations.

    Examples
    --------
    registry = ModelRegistry()
    version = registry.register_model_from_run(
        run_id="abc123",
        artifact_path="checkpoints/best_model.pth",
        alias="champion",
    )
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        tracking_uri: str = TRACKING_URI,
        registry_uri: str = REGISTRY_URI,
    ) -> None:
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)

        self.client = MlflowClient(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
        )

        self._validate_model_name()

    def _validate_model_name(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Model name must be a non-empty string.")

    def _model_uri(self, run_id: str, artifact_path: str) -> str:
        artifact_path = artifact_path.strip().strip("/\\")

        if not run_id:
            raise ValueError("run_id cannot be empty.")

        if not artifact_path:
            raise ValueError("artifact_path cannot be empty.")

        return f"runs:/{run_id}/{artifact_path}"

    def _run_artifact_exists(
        self,
        run_id: str,
        artifact_path: str,
    ) -> bool:
        artifact_path = artifact_path.strip().strip("/\\")
        parent, _, child_name = artifact_path.replace("\\", "/").rpartition("/")

        try:
            artifacts = self.client.list_artifacts(run_id, parent or None)
        except Exception:
            return False

        return any(
            artifact.path.replace("\\", "/") == artifact_path for artifact in artifacts
        )

    def _get_run_metric(
        self,
        run: Run,
        metric_name: str,
    ) -> float:
        metric = run.data.metrics.get(metric_name)

        if metric is None:
            return float("-inf")

        return float(metric)

    def health_check(self) -> bool:
        """
        Return True when the registry backend is reachable.
        """
        try:
            list(self.client.search_registered_models())
            return True
        except Exception:
            logger.exception("MLflow registry health check failed.")
            return False

    def registered_model_exists(self) -> bool:
        """
        Check whether the configured registered model already exists.
        """
        try:
            self.client.get_registered_model(self.model_name)
            return True
        except Exception:
            return False

    def create_registered_model(
        self,
        description: str | None = None,
        tags: dict[str, Any] | None = None,
        exist_ok: bool = False,
    ) -> RegisteredModel:
        """
        Create the registered model container.
        """
        if self.registered_model_exists():
            if exist_ok:
                return self.get_registered_model()

            raise DuplicateModelError(f"'{self.model_name}' already exists.")

        model = self.client.create_registered_model(
            name=self.model_name,
            tags=tags,
            description=description,
        )

        logger.info("Created registered model '%s'.", self.model_name)
        return model

    def ensure_registered_model(
        self,
        description: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> RegisteredModel:
        """
        Create the registered model if it does not already exist.
        """
        return self.create_registered_model(
            description=description,
            tags=tags,
            exist_ok=True,
        )

    def get_registered_model(self) -> RegisteredModel:
        """
        Return registered model metadata.
        """
        try:
            return self.client.get_registered_model(self.model_name)
        except Exception as exc:
            raise ModelNotFoundError(
                f"Registered model '{self.model_name}' does not exist."
            ) from exc

    def list_registered_models(self) -> list[RegisteredModel]:
        """
        Return all registered models.
        """
        return list(self.client.search_registered_models())

    def delete_registered_model(self) -> None:
        """
        Permanently delete the registered model.
        """
        if not self.registered_model_exists():
            raise ModelNotFoundError(
                f"Registered model '{self.model_name}' does not exist."
            )

        self.client.delete_registered_model(self.model_name)
        logger.warning("Deleted registered model '%s'.", self.model_name)

    def get_run(self, run_id: str) -> Run:
        """
        Return an MLflow run by id.
        """
        try:
            return self.client.get_run(run_id)
        except Exception as exc:
            raise ModelNotFoundError(f"MLflow run '{run_id}' was not found.") from exc

    def search_runs(
        self,
        experiment_name: str = EXPERIMENT_NAME,
        filter_string: str | None = None,
        max_results: int = 100,
    ) -> list[Run]:
        """
        Search runs for an experiment.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise ModelNotFoundError(
                f"MLflow experiment '{experiment_name}' was not found."
            )

        return self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string or "",
            max_results=max_results,
        )

    def find_best_run(
        self,
        experiment_name: str = EXPERIMENT_NAME,
        metric_name: str = "validation_accuracy",
        artifact_path: str | None = None,
        filter_string: str | None = None,
        max_results: int = 100,
    ) -> Run:
        """
        Return the run with the highest metric value.
        """
        runs = self.search_runs(
            experiment_name=experiment_name,
            filter_string=filter_string,
            max_results=max_results,
        )

        runs_with_metric = [run for run in runs if metric_name in run.data.metrics]

        if artifact_path:
            runs_with_metric = [
                run
                for run in runs_with_metric
                if self._run_artifact_exists(run.info.run_id, artifact_path)
            ]

        if not runs_with_metric:
            raise ModelNotFoundError(
                f"No runs in '{experiment_name}' contain metric "
                f"'{metric_name}' and artifact '{artifact_path}'."
            )

        return max(
            runs_with_metric,
            key=lambda run: self._get_run_metric(run, metric_name),
        )

    def latest_version(self) -> ModelVersion | None:
        """
        Return the latest registered version, or None if no version exists.
        """
        versions = self.client.search_model_versions(
            filter_string=f"name='{self.model_name}'"
        )

        if not versions:
            return None

        return max(versions, key=lambda version: int(version.version))

    def get_model_version(self, version: int | str) -> ModelVersion:
        """
        Return one registered model version.
        """
        try:
            return self.client.get_model_version(
                name=self.model_name,
                version=str(version),
            )
        except Exception as exc:
            raise VersionNotFoundError(
                f"Version {version} of '{self.model_name}' was not found."
            ) from exc

    def register_model_from_run(
        self,
        run_id: str,
        artifact_path: str = "checkpoints/best_model.pth",
        description: str | None = None,
        tags: dict[str, Any] | None = None,
        alias: str | None = None,
        await_creation_for: int = 300,
        use_register_model: bool = False,
    ) -> ModelVersion:
        """
        Register a model artifact from an MLflow run.

        Parameters
        ----------
        run_id:
            Source MLflow run id.
        artifact_path:
            Artifact path inside the run, for example ``model`` or
            ``checkpoints/best_model.pth``.
        use_register_model:
            Use ``mlflow.register_model``. This is best for MLflow model
            directories that contain an ``MLmodel`` file. The default uses
            ``MlflowClient.create_model_version`` so checkpoint files can be
            registered too.
        """
        self.ensure_registered_model()
        self.get_run(run_id)

        if not self._run_artifact_exists(run_id, artifact_path):
            raise ModelNotFoundError(
                f"Artifact '{artifact_path}' was not found in run '{run_id}'."
            )

        model_uri = self._model_uri(
            run_id=run_id,
            artifact_path=artifact_path,
        )

        version_tags = {
            "source_run_id": run_id,
            "source_artifact_path": artifact_path,
        }

        if tags:
            version_tags.update(tags)

        if use_register_model:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name,
                await_registration_for=await_creation_for,
                tags=version_tags,
            )
        else:
            model_version = self.client.create_model_version(
                name=self.model_name,
                source=model_uri,
                run_id=run_id,
                tags=version_tags,
                description=description,
                await_creation_for=await_creation_for,
            )

        if description and use_register_model:
            self.client.update_model_version(
                name=self.model_name,
                version=model_version.version,
                description=description,
            )

        if alias:
            self.set_alias(
                alias=alias,
                version=model_version.version,
            )

        logger.info(
            "Registered '%s' version %s from %s.",
            self.model_name,
            model_version.version,
            model_uri,
        )

        return model_version

    def register_best_run(
        self,
        experiment_name: str = EXPERIMENT_NAME,
        metric_name: str = "validation_accuracy",
        artifact_path: str = "checkpoints/best_model.pth",
        filter_string: str | None = None,
        alias: str | None = "champion",
        description: str | None = None,
        tags: dict[str, Any] | None = None,
        max_results: int = 100,
    ) -> ModelVersion:
        """
        Find the best run by metric and register its model artifact.
        """
        run = self.find_best_run(
            experiment_name=experiment_name,
            metric_name=metric_name,
            artifact_path=artifact_path,
            filter_string=filter_string,
            max_results=max_results,
        )

        run_tags = {
            "selection_metric": metric_name,
            "selection_metric_value": self._get_run_metric(run, metric_name),
        }

        if tags:
            run_tags.update(tags)

        return self.register_model_from_run(
            run_id=run.info.run_id,
            artifact_path=artifact_path,
            description=description,
            tags=run_tags,
            alias=alias,
        )

    def set_alias(
        self,
        alias: str,
        version: int | str,
    ) -> None:
        """
        Point an alias, such as champion or staging, to a version.
        """
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=alias,
            version=str(version),
        )

    def get_version_by_alias(self, alias: str) -> ModelVersion:
        """
        Return the model version assigned to an alias.
        """
        try:
            return self.client.get_model_version_by_alias(
                name=self.model_name,
                alias=alias,
            )
        except MlflowException as exc:
            raise VersionNotFoundError(
                f"Alias '{alias}' was not found for '{self.model_name}'."
            ) from exc

    def set_model_description(self, description: str) -> None:
        """
        Update the registered model description.
        """
        self.client.update_registered_model(
            name=self.model_name,
            description=description,
        )

    def set_version_description(
        self,
        version: int | str,
        description: str,
    ) -> None:
        """
        Update a model version description.
        """
        self.client.update_model_version(
            name=self.model_name,
            version=str(version),
            description=description,
        )

    def set_version_tag(
        self,
        version: int | str,
        key: str,
        value: Any,
    ) -> None:
        """
        Add or update a tag on a model version.
        """
        self.client.set_model_version_tag(
            name=self.model_name,
            version=str(version),
            key=key,
            value=value,
        )

    def registry_summary(self) -> dict[str, Any]:
        """
        Return a compact registry summary.
        """
        exists = self.registered_model_exists()

        if not exists:
            return {
                "healthy": self.health_check(),
                "exists": False,
                "model_name": self.model_name,
                "total_models": len(self.list_registered_models()),
                "latest_version": None,
            }

        latest = self.latest_version()

        return {
            "healthy": self.health_check(),
            "exists": True,
            "model_name": self.model_name,
            "total_models": len(self.list_registered_models()),
            "latest_version": None if latest is None else latest.version,
        }

    def print_registry_summary(self) -> None:
        """
        Print a readable registry summary.
        """
        summary = self.registry_summary()

        print("\n" + "=" * 60)
        print("MLFLOW MODEL REGISTRY")
        print("=" * 60)
        print(f"Healthy        : {summary['healthy']}")
        print(f"Model Name     : {summary['model_name']}")
        print(f"Exists         : {summary['exists']}")
        print(f"Total Models   : {summary['total_models']}")
        print(f"Latest Version : {summary['latest_version']}")
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register an MLflow run artifact in the model registry."
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="MLflow run id to register.",
    )
    parser.add_argument(
        "--best-run",
        action="store_true",
        help="Register the best run from an experiment instead of a run id.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=EXPERIMENT_NAME,
        help="Experiment used with --best-run.",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default="validation_accuracy",
        help="Metric used to select --best-run.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="checkpoints/best_model.pth",
        help="Artifact path inside the MLflow run.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Registered model name.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="champion",
        help="Alias to assign to the new version. Use empty string for none.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description for the model version.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=TRACKING_URI,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--registry-uri",
        type=str,
        default=REGISTRY_URI,
        help="MLflow registry URI.",
    )
    parser.add_argument(
        "--use-register-model",
        action="store_true",
        help="Use mlflow.register_model for MLflow model directories.",
    )

    return parser.parse_args()


def main() -> ModelVersion:
    args = parse_args()

    registry = ModelRegistry(
        model_name=args.model_name,
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
    )

    alias = args.alias or None

    if args.best_run:
        version = registry.register_best_run(
            experiment_name=args.experiment_name,
            metric_name=args.metric_name,
            artifact_path=args.artifact_path,
            alias=alias,
            description=args.description,
        )
    else:
        if not args.run_id:
            raise ValueError("--run-id is required unless --best-run is used.")

        version = registry.register_model_from_run(
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            alias=alias,
            description=args.description,
            use_register_model=args.use_register_model,
        )

    print(
        f"Registered model '{version.name}' "
        f"version {version.version} from run {version.run_id}."
    )

    if alias:
        print(f"Alias '{alias}' now points to version {version.version}.")

    return version


if __name__ == "__main__":
    main()
