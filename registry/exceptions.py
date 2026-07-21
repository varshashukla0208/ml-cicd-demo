"""
Custom exceptions for Model Registry.
"""


class RegistryError(Exception):
    """Base Registry Exception."""


class ModelNotFoundError(RegistryError):
    """Registered model not found."""


class VersionNotFoundError(RegistryError):
    """Requested model version not found."""


class StageTransitionError(RegistryError):
    """Failed to transition model stage."""


class DuplicateModelError(RegistryError):
    """Model already exists."""