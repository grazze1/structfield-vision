"""Dependency-free static contracts for the Stage 13 YAML configurations."""

from __future__ import annotations

import importlib
import inspect
import json
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, get_args

import pytest

from structfield.corruption.metadata import CorruptionName
from structfield.estimators import AnalyticEstimatorConfig
from structfield.losses import ReliabilityLossConfig
from structfield.transforms import GeometricTransform

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STRUCTFIELD_CONFIG_ROOT = REPOSITORY_ROOT / "structfield" / "configs"
EXPERIMENT_CONFIG_ROOT = STRUCTFIELD_CONFIG_ROOT

STRUCTFIELD_CONFIGS = (
    STRUCTFIELD_CONFIG_ROOT / "analytic_default.yaml",
    STRUCTFIELD_CONFIG_ROOT / "analytic_8dir.yaml",
    STRUCTFIELD_CONFIG_ROOT / "analytic_16dir.yaml",
    STRUCTFIELD_CONFIG_ROOT / "analytic_32dir.yaml",
)
EXPERIMENT_CONFIGS = (
    EXPERIMENT_CONFIG_ROOT / "baseline_matrix.yaml",
    EXPERIMENT_CONFIG_ROOT / "failure_prediction.yaml",
    EXPERIMENT_CONFIG_ROOT / "corruption_sweep.yaml",
    EXPERIMENT_CONFIG_ROOT / "equivariance_smoke.yaml",
)
STAGE13_CONFIGS = STRUCTFIELD_CONFIGS + EXPERIMENT_CONFIGS

_KEY = r"[A-Za-z_][A-Za-z0-9_-]*"
_MAPPING_LINE = re.compile(rf"^(?P<indent> *)(?P<key>{_KEY}):(?: (?P<value>.+))?$")
_SEQUENCE_MAPPING_LINE = re.compile(rf"^(?P<indent> *)- (?P<key>{_KEY}):(?: (?P<value>.+))?$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_scalar(value: str) -> Any:
    if value.startswith(('"', "[", "{")):
        return json.loads(value)
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        return float(value)


def _direct_child_mapping(text: str, key: str) -> dict[str, Any]:
    lines = text.splitlines()
    parent_index = next(
        index
        for index, line in enumerate(lines)
        if (match := _MAPPING_LINE.fullmatch(line)) and match.group("key") == key and match.group("value") is None
    )
    parent_indent = len(lines[parent_index]) - len(lines[parent_index].lstrip())
    child_indent = parent_indent + 2
    result: dict[str, Any] = {}
    for line in lines[parent_index + 1 :]:
        indent = len(line) - len(line.lstrip())
        if line and indent <= parent_indent:
            break
        if not line:
            continue
        match = _MAPPING_LINE.fullmatch(line)
        if match is None or indent != child_indent or match.group("value") is None:
            raise AssertionError(f"{key!r} must be a flat scalar mapping: {line!r}")
        result[match.group("key")] = _parse_scalar(match.group("value"))
    return result


def _sequence_records(text: str, item_key: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = _SEQUENCE_MAPPING_LINE.fullmatch(line)
        if match is None or match.group("key") != item_key:
            continue
        item_indent = len(match.group("indent"))
        value = match.group("value")
        assert value is not None
        record = {item_key: _parse_scalar(value)}
        for child in lines[index + 1 :]:
            indent = len(child) - len(child.lstrip())
            if child and indent <= item_indent:
                break
            if not child:
                continue
            child_match = _MAPPING_LINE.fullmatch(child)
            if child_match is None or indent != item_indent + 2:
                break
            child_value = child_match.group("value")
            if child_value is None:
                break
            record[child_match.group("key")] = _parse_scalar(child_value)
        records.append(record)
    return records


def _resolve_dotted_path(value: str) -> object:
    module_name, attribute_name = value.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute_name)


@pytest.mark.parametrize("path", STAGE13_CONFIGS, ids=lambda path: path.name)
def test_yaml_files_use_a_safe_dependency_free_static_subset(path: Path) -> None:
    """Check the authored subset without importing or requiring PyYAML."""
    text = _read(path)
    assert text.endswith("\n")
    assert "\t" not in text
    assert not re.search(r"[ \t]+$", text, flags=re.MULTILINE)
    assert not any(token in text for token in ("!!", "!<", "{{", "${"))

    for line in text.splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        indentation = len(line) - len(line.lstrip())
        assert indentation % 2 == 0
        assert _MAPPING_LINE.fullmatch(line) or _SEQUENCE_MAPPING_LINE.fullmatch(line), line

    assert re.search(r'^schema_version: "1\.0"$', text, flags=re.MULTILINE)
    assert re.search(r"^contains_results: false$", text, flags=re.MULTILINE)
    assert not re.search(r"^\s*(?:results|training_results|metric_results):", text, flags=re.MULTILINE)


@pytest.mark.parametrize(
    ("path", "num_orientations"),
    (
        (STRUCTFIELD_CONFIG_ROOT / "analytic_default.yaml", 16),
        (STRUCTFIELD_CONFIG_ROOT / "analytic_8dir.yaml", 8),
        (STRUCTFIELD_CONFIG_ROOT / "analytic_16dir.yaml", 16),
        (STRUCTFIELD_CONFIG_ROOT / "analytic_32dir.yaml", 32),
    ),
    ids=("default", "8dir", "16dir", "32dir"),
)
def test_analytic_configs_exactly_match_the_validated_dataclass(
    path: Path,
    num_orientations: int,
) -> None:
    parameters = _direct_child_mapping(_read(path), "parameters")
    assert set(parameters) == {field.name for field in fields(AnalyticEstimatorConfig)}
    assert parameters["num_orientations"] == num_orientations

    config = AnalyticEstimatorConfig(**parameters)
    assert config.num_orientations == num_orientations
    if num_orientations == 16:
        assert asdict(config) == asdict(AnalyticEstimatorConfig())


def test_all_declared_python_components_are_real_public_objects() -> None:
    path_keys = (
        "implementation",
        "config_dataclass",
        "adapter_python_class",
        "python_class",
        "python_function",
        "transform_dataclass",
        "tensor_transform_function",
        "field_transform_function",
    )
    pattern = re.compile(
        rf'^\s*(?:{"|".join(path_keys)}): "(?P<path>[A-Za-z_][A-Za-z0-9_.]+)"$',
        flags=re.MULTILINE,
    )
    declared = [match.group("path") for path in STAGE13_CONFIGS for match in pattern.finditer(_read(path))]
    assert declared
    for dotted_path in declared:
        assert callable(_resolve_dotted_path(dotted_path)), dotted_path


def test_all_internal_config_references_exist() -> None:
    pattern = re.compile(
        r"^\s*(?:estimator_config|baseline_matrix|corruption_config|equivariance_config): "
        r'"(?P<path>[A-Za-z0-9_]+\.yaml)"$',
        flags=re.MULTILINE,
    )
    references = [match.group("path") for path in STAGE13_CONFIGS for match in pattern.finditer(_read(path))]
    assert references
    for relative_path in references:
        assert (STRUCTFIELD_CONFIG_ROOT / relative_path).is_file(), relative_path


def test_v1_protocol_and_future_matrix_are_frozen_before_results() -> None:
    text = _read(EXPERIMENT_CONFIG_ROOT / "baseline_matrix.yaml")
    assert 'definition_status: "frozen-a-priori"' in text
    assert "frozen: true" in text

    protocol_records = _sequence_records(text, "protocol_id")
    assert [record["protocol_id"] for record in protocol_records] == [
        "no_prior",
        "fixed_prior",
        "probabilistic_prior",
        "reliability_gated_prior",
        "reliability_gated_prior_corruption",
        "reliability_gated_prior_corruption_equivariance",
    ]
    assert [record["label"] for record in protocol_records] == [
        "NoPrior",
        "FixedPrior",
        "ProbabilisticPrior",
        "ReliabilityGatedPrior",
        "ReliabilityGatedPrior + Corruption",
        "ReliabilityGatedPrior + Corruption + Equivariance",
    ]

    matrix_records = _sequence_records(text, "matrix_id")
    observed_matrix = [
        (
            record["model"],
            record["prior"],
            record["adaptive"],
            record["uq"],
            record["equivariance"],
        )
        for record in matrix_records
    ]
    assert observed_matrix == [
        ("CNN", "no", "no", "no", "no"),
        ("CNN + Fixed", "yes", "no", "no", "no"),
        ("CNN + Prob", "yes", "no", "yes", "no"),
        ("CNN + Gate", "yes", "yes", "yes", "no"),
        ("CNN + Gate + EQ", "yes", "yes", "yes", "yes"),
        ("escnn", "implicit", "not-applicable", "no", "yes"),
        ("escnn + APSF", "yes", "yes", "yes", "yes"),
    ]
    assert all(record["implementation_status"] == "planned-optional" for record in matrix_records[-2:])


def test_failure_prediction_parameters_match_reliability_loss_api() -> None:
    text = _read(EXPERIMENT_CONFIG_ROOT / "failure_prediction.yaml")
    parameters = _direct_child_mapping(text, "parameters")
    assert set(parameters) == {field.name for field in fields(ReliabilityLossConfig)}
    config = ReliabilityLossConfig(**parameters)
    assert asdict(config) == asdict(ReliabilityLossConfig())
    assert "implicit_label_generation: false" in text
    assert "calibration_claim_allowed: false" in text


def test_corruption_sweep_covers_the_public_api_with_exact_parameter_names() -> None:
    text = _read(EXPERIMENT_CONFIG_ROOT / "corruption_sweep.yaml")
    records = _sequence_records(text, "name")
    assert tuple(record["name"] for record in records) == get_args(CorruptionName)

    for record in records:
        function = _resolve_dotted_path(record["python_function"])
        signature = inspect.signature(function)
        parameters = set(signature.parameters) - {"field", "seed"}
        levels = record["levels"]
        assert levels
        assert all(set(level) == parameters for level in levels)

        seed_parameter = signature.parameters.get("seed")
        requires_seed = seed_parameter is not None
        assert record["requires_seed"] is requires_seed
        if seed_parameter is not None:
            assert seed_parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert seed_parameter.default is inspect.Parameter.empty


def test_equivariance_smoke_uses_valid_transforms_and_all_reference_losses() -> None:
    text = _read(EXPERIMENT_CONFIG_ROOT / "equivariance_smoke.yaml")
    transforms = _sequence_records(text, "name")
    configured = [GeometricTransform(**record["parameters"]) for record in transforms]
    assert configured[0] == GeometricTransform()
    assert any(transform.rotation_rad != 0.0 for transform in configured)
    assert any(transform.scale != 1.0 for transform in configured)

    losses = _sequence_records(text, "signal")
    assert [record["signal"] for record in losses] == [
        "orientation_prob",
        "frequency_mean",
        "structural_uncertainty",
    ]
    assert 'valid_mask_policy: "intersection-of-observed-and-expected-strict-valid-masks"' in text
    assert "claim_exact_group_equivariance: false" in text
