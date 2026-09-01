"""Frozen v1 public boundary and experimental-label contracts."""

from dataclasses import fields

import structfield
import structfield.experimental as experimental
from structfield import FieldMetadata, StructuralField, ValidationMode
from structfield._version import VERSION


def test_top_level_api_is_exactly_the_frozen_v1_surface() -> None:
    assert structfield.__all__ == [
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
    assert VERSION == "1.0.0"


def test_field_v1_has_one_orientation_source_and_v1_metadata() -> None:
    assert "orientation_topk" not in {item.name for item in fields(StructuralField)}
    assert FieldMetadata(8).schema_version == "1.0"


def test_research_components_are_explicitly_experimental() -> None:
    assert experimental.HeuristicStructuralUncertainty.__module__.startswith("structfield.uncertainty")
    assert experimental.applications.__name__ == "structfield.applications"
    assert ValidationMode.strict.value == "strict"
