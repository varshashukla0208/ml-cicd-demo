from .registry_config import ARTIFACT_DIR
from .registry_config import CHECKPOINT_DIR
from .registry_config import CONFIG_DIR
from .registry_config import DEFAULT_STAGE
from .registry_config import EXPERIMENT_NAME
from .registry_config import MODEL_NAME
from .registry_config import PROJECT_ROOT
from .registry_config import REGISTRY_URI
from .registry_config import TRACKING_URI
from .registry_utils import count_versions
from .registry_utils import get_client
from .registry_utils import get_latest_version
from .registry_utils import get_latest_versions
from .registry_utils import get_model_version
from .registry_utils import get_model_versions
from .registry_utils import get_registered_model
from .registry_utils import get_stage_version
from .registry_utils import list_registered_models
from .registry_utils import print_registry_summary
from .registry_utils import registered_model_exists
from .registry_utils import registry_summary
from .registry_utils import search_registered_models


def __getattr__(name):
    if name == "ModelRegistry":
        from .model_registry import ModelRegistry

        return ModelRegistry

    raise AttributeError(f"module 'registry' has no attribute {name!r}")


__all__ = [
    "ARTIFACT_DIR",
    "CHECKPOINT_DIR",
    "CONFIG_DIR",
    "DEFAULT_STAGE",
    "EXPERIMENT_NAME",
    "MODEL_NAME",
    "ModelRegistry",
    "PROJECT_ROOT",
    "REGISTRY_URI",
    "TRACKING_URI",
    "count_versions",
    "get_client",
    "get_latest_version",
    "get_latest_versions",
    "get_model_version",
    "get_model_versions",
    "get_registered_model",
    "get_stage_version",
    "list_registered_models",
    "print_registry_summary",
    "registered_model_exists",
    "registry_summary",
    "search_registered_models",
]
