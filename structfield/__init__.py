"""Stable top-level API for StructField-Vision."""

from importlib.metadata import PackageNotFoundError, version

from ._version import VERSION
from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    LoadedApplicationCheckpoint,
    load_application_checkpoint,
    save_application_checkpoint,
)
from .configs import available_configs, read_config_text, read_estimator_config
from .core import FieldMetadata, OrientationModes, StructuralField
from .estimators import (
    AnalyticEstimatorConfig,
    AnalyticStructuralFieldEstimator,
    BaseStructuralFieldEstimator,
    StructuralFieldEstimator,
    estimate_structural_field,
)
from .validation import ValidationMode


def _distribution_version() -> str:
    try:
        installed = version("structfield-vision")
    except PackageNotFoundError:
        return f"{VERSION}.dev0"
    return installed or f"{VERSION}.dev0"


__version__ = _distribution_version()

__all__ = [
    "__version__",
    "AnalyticEstimatorConfig",
    "AnalyticStructuralFieldEstimator",
    "BaseStructuralFieldEstimator",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "FieldMetadata",
    "LoadedApplicationCheckpoint",
    "OrientationModes",
    "StructuralField",
    "StructuralFieldEstimator",
    "ValidationMode",
    "available_configs",
    "estimate_structural_field",
    "load_application_checkpoint",
    "read_config_text",
    "read_estimator_config",
    "save_application_checkpoint",
]
