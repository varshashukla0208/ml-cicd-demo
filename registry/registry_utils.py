"""
registry_utils.py

Utility functions for interacting with the MLflow Model Registry.

This module contains only read/query operations.
It never creates, deletes, or modifies registry resources.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mlflow import MlflowClient
from mlflow.entities.model_registry import (
    RegisteredModel,
    ModelVersion,
)

from registry.registry_config import MODEL_NAME
from registry.exceptions import (
    ModelNotFoundError,
    VersionNotFoundError,
)

# ---------------------------------------------------------------------
# Singleton MLflow Client
# ---------------------------------------------------------------------

_client = MlflowClient()


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------


def get_client() -> MlflowClient:
    """
    Return the singleton MLflow client.

    Returns
    -------
    MlflowClient
        Configured MLflow client.
    """
    return _client


# ---------------------------------------------------------------------
# Model Existence
# ---------------------------------------------------------------------


def registered_model_exists(model_name: str = MODEL_NAME) -> bool:
    """
    Check whether a registered model exists.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    bool
    """

    try:
        _client.get_registered_model(model_name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# Registered Model
# ---------------------------------------------------------------------


def get_registered_model(
    model_name: str = MODEL_NAME,
) -> RegisteredModel:
    """
    Return registered model metadata.

    Raises
    ------
    ModelNotFoundError
    """

    try:
        return _client.get_registered_model(model_name)

    except Exception as e:
        raise ModelNotFoundError(f"Registered model '{model_name}' not found.") from e


# ---------------------------------------------------------------------
# List Registered Models
# ---------------------------------------------------------------------


def list_registered_models() -> List[RegisteredModel]:
    """
    Return all registered models.

    Returns
    -------
    List[RegisteredModel]
    """

    return list(_client.search_registered_models())


# ---------------------------------------------------------------------
# Search Models
# ---------------------------------------------------------------------


def search_registered_models(
    filter_string: Optional[str] = None,
) -> List[RegisteredModel]:
    """
    Search registered models.

    Parameters
    ----------
    filter_string : str | None

    Returns
    -------
    List[RegisteredModel]
    """

    if filter_string is None:
        return list(_client.search_registered_models())

    return list(_client.search_registered_models(filter_string=filter_string))


# ---------------------------------------------------------------------
# Model Versions
# ---------------------------------------------------------------------


def get_model_versions(
    model_name: str = MODEL_NAME,
) -> List[ModelVersion]:
    """
    Return all versions of a registered model.

    Raises
    ------
    ModelNotFoundError
    """

    if not registered_model_exists(model_name):
        raise ModelNotFoundError(f"Model '{model_name}' does not exist.")

    versions = _client.search_model_versions(f"name='{model_name}'")

    return sorted(
        versions,
        key=lambda x: int(x.version),
    )


# ---------------------------------------------------------------------
# Latest Version
# ---------------------------------------------------------------------


def get_latest_version(
    model_name: str = MODEL_NAME,
) -> Optional[ModelVersion]:
    """
    Return latest version irrespective of stage.

    Returns
    -------
    ModelVersion | None
    """

    versions = get_model_versions(model_name)

    if len(versions) == 0:
        return None

    return versions[-1]


# ---------------------------------------------------------------------
# Specific Version
# ---------------------------------------------------------------------


def get_model_version(
    model_name: str,
    version: int,
) -> ModelVersion:
    """
    Return a specific version.

    Raises
    ------
    VersionNotFoundError
    """

    try:
        return _client.get_model_version(
            model_name,
            str(version),
        )

    except Exception as e:
        raise VersionNotFoundError(f"Version {version} not found.") from e


# ---------------------------------------------------------------------
# Latest Versions Per Stage
# ---------------------------------------------------------------------


def get_latest_versions(
    model_name: str = MODEL_NAME,
) -> List[ModelVersion]:
    """
    Return MLflow latest versions.

    Returns
    -------
    List[ModelVersion]
    """

    if not registered_model_exists(model_name):
        return []

    return _client.get_latest_versions(model_name)


# ---------------------------------------------------------------------
# Get Version By Stage
# ---------------------------------------------------------------------


def get_stage_version(
    model_name: str,
    stage: str,
) -> Optional[ModelVersion]:
    """
    Return the version assigned to a stage.

    Parameters
    ----------
    stage:
        Production
        Staging
        Archived
        None

    Returns
    -------
    ModelVersion | None
    """

    latest = _client.get_latest_versions(
        model_name,
        stages=[stage],
    )

    if len(latest) == 0:
        return None

    return latest[0]


# ---------------------------------------------------------------------
# Count Versions
# ---------------------------------------------------------------------


def count_versions(
    model_name: str = MODEL_NAME,
) -> int:
    """
    Number of registered versions.
    """

    return len(get_model_versions(model_name))


# ---------------------------------------------------------------------
# Registry Summary
# ---------------------------------------------------------------------


def registry_summary(
    model_name: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    Return a registry summary dictionary.
    """

    if not registered_model_exists(model_name):
        return {
            "exists": False,
            "model_name": model_name,
        }

    model = get_registered_model(model_name)

    latest = get_latest_version(model_name)

    production = get_stage_version(
        model_name,
        "Production",
    )

    staging = get_stage_version(
        model_name,
        "Staging",
    )

    archived = [
        v.version
        for v in get_model_versions(model_name)
        if v.current_stage == "Archived"
    ]

    return {
        "exists": True,
        "model_name": model.name,
        "description": model.description,
        "creation_timestamp": model.creation_timestamp,
        "last_updated_timestamp": model.last_updated_timestamp,
        "latest_version": None if latest is None else latest.version,
        "production_version": None if production is None else production.version,
        "staging_version": None if staging is None else staging.version,
        "archived_versions": archived,
        "total_versions": count_versions(model_name),
    }


# ---------------------------------------------------------------------
# Pretty Print
# ---------------------------------------------------------------------


def print_registry_summary(
    model_name: str = MODEL_NAME,
) -> None:
    """
    Print registry summary.
    """

    summary = registry_summary(model_name)

    print("=" * 60)
    print("MLFLOW MODEL REGISTRY")
    print("=" * 60)

    if not summary["exists"]:
        print(f"Registered model '{model_name}' not found.")
        print("=" * 60)
        return

    print(f"Model Name         : {summary['model_name']}")
    print(f"Total Versions     : {summary['total_versions']}")
    print(f"Latest Version     : {summary['latest_version']}")
    print(f"Production Version : {summary['production_version']}")
    print(f"Staging Version    : {summary['staging_version']}")
    print(f"Archived Versions  : {summary['archived_versions']}")
    print("=" * 60)
