"""Read-only access to active packaged StructField configuration resources."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, cast

from structfield.estimators import AnalyticEstimatorConfig

_CONFIGS: tuple[str, ...] = (
    "analytic_default.yaml",
    "analytic_8dir.yaml",
    "analytic_16dir.yaml",
    "analytic_32dir.yaml",
    "baseline_matrix.yaml",
    "failure_prediction.yaml",
    "corruption_sweep.yaml",
    "equivariance_smoke.yaml",
)

_ESTIMATOR_CONFIGS = frozenset(
    {"analytic_default.yaml", "analytic_8dir.yaml", "analytic_16dir.yaml", "analytic_32dir.yaml"}
)


def available_configs() -> tuple[str, ...]:
    """Return canonical, stable IDs for active packaged YAML resources."""

    return _CONFIGS


def read_config_text(resource_id: str) -> str:
    """Return one canonical resource without exposing a persistent file path."""

    if not isinstance(resource_id, str):
        raise TypeError("resource_id must be a string")
    if resource_id not in _CONFIGS:
        raise KeyError(f"unknown config resource_id: {resource_id!r}")
    return resources.files(__package__).joinpath(resource_id).read_text(encoding="utf-8")


def _parse_scalar(value: str) -> object:
    """Parse the deliberately small JSON-compatible YAML subset used by v1 configs."""

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"unsupported config scalar: {value!r}") from exc


def read_estimator_config(resource_id: str = "analytic_default.yaml") -> AnalyticEstimatorConfig:
    """Load one packaged analytic config without importing a general YAML parser.

    Only the four immutable estimator resources are accepted. The parser supports
    the restricted mapping/scalar/list subset authored by this project; aliases,
    tags, anchors, duplicate keys, and arbitrary objects are rejected.
    """

    if resource_id not in _ESTIMATOR_CONFIGS:
        raise KeyError(f"unknown estimator config resource_id: {resource_id!r}")
    top_level: dict[str, object] = {}
    parameters: dict[str, object] = {}
    in_parameters = False
    for line_number, raw_line in enumerate(read_config_text(resource_id).splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"invalid config line {line_number}")
        key, raw_value = (part.strip() for part in stripped.split(":", 1))
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"invalid config key on line {line_number}")
        target = parameters if indentation == 2 and in_parameters else top_level
        if indentation not in {0, 2} or (indentation == 2 and not in_parameters):
            raise ValueError(f"unsupported config nesting on line {line_number}")
        if key in target:
            raise ValueError(f"duplicate config key {key!r}")
        if indentation == 0 and key == "parameters":
            if raw_value:
                raise ValueError("parameters must be a mapping")
            in_parameters = True
            continue
        if indentation == 0:
            in_parameters = False
        target[key] = _parse_scalar(raw_value)
    expected = {
        "schema_version",
        "kind",
        "implementation",
        "config_dataclass",
        "contains_results",
    }
    if set(top_level) != expected:
        raise ValueError("estimator config top-level fields are invalid")
    if top_level["schema_version"] != "1.0":
        raise ValueError("unsupported estimator config schema")
    if top_level["kind"] != "analytic-structural-field-estimator":
        raise ValueError("unsupported estimator config kind")
    for sequence_name in ("frequencies", "tensor_scales", "hessian_scales"):
        value = parameters.get(sequence_name)
        if not isinstance(value, list):
            raise ValueError(f"{sequence_name} must be a list")
        parameters[sequence_name] = tuple(value)
    try:
        return AnalyticEstimatorConfig(**cast(Any, parameters))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid estimator parameters in {resource_id}") from exc


__all__ = ["available_configs", "read_config_text", "read_estimator_config"]
