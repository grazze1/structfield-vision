"""StructuralField schema, validation, and device/dtype contracts."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from structfield import FieldMetadata, OrientationModes, StructuralField, ValidationMode
from structfield.core import StructuralFieldDiagnostics


def make_field(
    num_orientations: int = 16,
    *,
    batch: int = 2,
    height: int = 9,
    width: int = 11,
    with_scale: bool = True,
) -> StructuralField:
    dtype = torch.float32
    probabilities = torch.zeros(batch, num_orientations, height, width, dtype=dtype)
    probabilities[:, 0] = 0.6
    probabilities[:, num_orientations // 2] = 0.4
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(1, keepdim=True)
    entropy = entropy / math.log(num_orientations)
    metadata = FieldMetadata(
        num_orientations=num_orientations,
        frequency_candidates=(0.05, 0.10, 0.15),
        scale_candidates=(1.0, 2.0) if with_scale else (),
        extras={"fixture": "stage-2"},
    )
    scalar = torch.ones(batch, 1, height, width, dtype=dtype)
    scale_prob = None
    selected_scale = None
    if with_scale:
        scale_prob = torch.empty(batch, 2, height, width, dtype=dtype)
        scale_prob[:, 0] = 0.25
        scale_prob[:, 1] = 0.75
        selected_scale = scalar * 1.75
    return StructuralField(
        orientation_prob=probabilities,
        frequency_mean=scalar * 0.10,
        frequency_var=scalar * 0.001,
        coherence=scalar * 0.70,
        orientation_entropy=entropy,
        structural_uncertainty=scalar * 0.25,
        metadata=metadata,
        scale_prob=scale_prob,
        selected_scale=selected_scale,
    )


@pytest.mark.parametrize("num_orientations", [8, 16, 32])
def test_field_shapes_and_probability_normalization(num_orientations: int) -> None:
    field = make_field(num_orientations, batch=3)
    assert field.orientation_prob.shape == (3, num_orientations, 9, 11)
    assert field.topk_orientations(k=2).angles.shape == (3, 2, 9, 11)
    assert field.frequency_mean.shape == (3, 1, 9, 11)
    assert field.scale_prob is not None and field.scale_prob.shape == (3, 2, 9, 11)
    assert torch.allclose(field.orientation_prob.sum(1), torch.ones(3, 9, 11))
    assert field.batch_size == 3
    assert field.spatial_shape == (9, 11)
    assert field.dtype == torch.float32
    assert field.device.type == "cpu"


def test_field_without_scale_is_valid() -> None:
    field = make_field(with_scale=False)
    assert field.scale_prob is None
    assert field.selected_scale is None


def test_trusted_field_skips_values_but_retains_structural_contracts() -> None:
    field = make_field()
    nan_coherence = field.coherence.clone()
    nan_coherence[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        replace(field, coherence=nan_coherence)
    trusted = replace(field, coherence=nan_coherence, validation_mode=ValidationMode.trusted)
    assert trusted.validation_mode is ValidationMode.trusted
    with pytest.raises(ValueError, match="shape"):
        replace(trusted, frequency_mean=trusted.frequency_mean[..., :-1])


def test_field_to_returns_new_object_without_mutation() -> None:
    field = make_field()
    converted = field.to(dtype=torch.float64)
    assert converted is not field
    assert field.dtype == torch.float32
    assert converted.dtype == torch.float64
    assert converted.metadata is field.metadata
    assert converted.orientation_prob.data_ptr() != field.orientation_prob.data_ptr()
    with pytest.raises(FrozenInstanceError):
        field.frequency_mean = torch.zeros_like(field.frequency_mean)  # type: ignore[misc]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_field_to_cuda_moves_every_tensor() -> None:
    field = make_field().to("cuda")
    tensor_values = [value for value in vars(field).values() if isinstance(value, torch.Tensor)]
    assert tensor_values and all(value.device.type == "cuda" for value in tensor_values)


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("orientation_prob", lambda field: field.orientation_prob * 0.5, "sum to one"),
        ("frequency_mean", lambda field: -torch.ones_like(field.frequency_mean), "non-negative"),
        ("frequency_var", lambda field: -torch.ones_like(field.frequency_var), "non-negative"),
        ("coherence", lambda field: torch.ones_like(field.coherence) * 1.1, r"in \[0, 1\]"),
        (
            "orientation_entropy",
            lambda field: torch.full_like(field.orientation_entropy, float("nan")),
            "NaN or Inf",
        ),
        (
            "structural_uncertainty",
            lambda field: torch.ones_like(field.structural_uncertainty, dtype=torch.float64),
            "share orientation_prob dtype",
        ),
    ],
)
def test_field_rejects_invalid_values(field_name: str, replacement: object, message: str) -> None:
    field = make_field()
    value = replacement(field)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=message):
        replace(field, **{field_name: value})


def test_field_rejects_invalid_scale_pair_and_metadata_mismatch() -> None:
    field = make_field()
    with pytest.raises(ValueError, match="both be present"):
        replace(field, selected_scale=None)
    with pytest.raises(ValueError, match="metadata.num_orientations"):
        replace(field, metadata=FieldMetadata(num_orientations=8, scale_candidates=(1.0, 2.0)))

    empty_scale_prob = torch.empty(field.batch_size, 0, *field.spatial_shape)
    empty_scale_metadata = replace(field.metadata, scale_candidates=())
    with pytest.raises(ValueError, match="at least one scale channel"):
        replace(field, scale_prob=empty_scale_prob, metadata=empty_scale_metadata)


def test_field_rejects_nonfloating_and_empty_inputs() -> None:
    field = make_field()
    with pytest.raises(TypeError, match="floating-point"):
        replace(field, orientation_prob=field.orientation_prob.to(torch.int64))
    empty = torch.empty(0, 16, 9, 11)
    with pytest.raises(ValueError, match="non-empty"):
        replace(field, orientation_prob=empty)
    with pytest.raises(TypeError, match="unexpected"):
        StructuralField(**{**vars(field), "orientation_topk": torch.zeros(1)})


def test_core_objects_reject_unsupported_float8_schema_and_conversion() -> None:
    float8_dtype = getattr(torch, "float8_e4m3fn", None)
    if float8_dtype is None:
        pytest.skip("float8_e4m3fn is unavailable")
    field = make_field()
    with pytest.raises(TypeError, match="supported floating-point"):
        replace(field, orientation_prob=field.orientation_prob.to(dtype=float8_dtype))
    with pytest.raises(TypeError, match="supported floating-point"):
        field.to(dtype=float8_dtype)

    modes = field.topk_orientations(k=2)
    with pytest.raises(TypeError, match="supported floating-point"):
        modes.to(dtype=float8_dtype)

    diagnostics = StructuralFieldDiagnostics(
        metadata=field.metadata,
        normalized_image=torch.zeros(1, 1, *field.spatial_shape),
    )
    with pytest.raises(TypeError, match="supported floating-point"):
        diagnostics.to(dtype=float8_dtype)


def test_metadata_validation_and_serialization() -> None:
    metadata = FieldMetadata(8, frequency_candidates=(0.1,), extras={"source": "unit-test"})
    payload = metadata.to_dict()
    assert payload["num_orientations"] == 8
    assert payload["angle_convention"] == "axial-[0,pi)"
    assert payload["extras"] == {"source": "unit-test"}
    with pytest.raises(ValueError, match="at least 2"):
        FieldMetadata(1)
    with pytest.raises(ValueError, match="strictly increasing"):
        FieldMetadata(3, orientation_bin_centers=(0.0, 0.5, 0.4))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("angle_convention", "directed-[0,2pi)", "angle_convention"),
        ("coordinate_convention", "x-right,y-up", "coordinate_convention"),
        ("positive_rotation", "clockwise", "positive_rotation"),
    ],
)
def test_metadata_rejects_noncanonical_conventions(field_name: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FieldMetadata(8, **{field_name: value})


def test_metadata_rejects_non_json_extras() -> None:
    with pytest.raises(TypeError, match="JSON-compatible"):
        FieldMetadata(8, extras={"tensor": torch.tensor(1.0)})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        FieldMetadata(8, extras={"score": float("nan")})
    with pytest.raises(TypeError, match="estimator must be a non-empty string"):
        FieldMetadata(8, estimator="")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"orientation_bin_centers": (0.0,)}, "length"),
        ({"orientation_bin_centers": (0.0, math.pi)}, r"\[0, pi\)"),
        ({"frequency_candidates": (0.0,)}, "frequency candidates"),
        ({"scale_candidates": (float("inf"),)}, "scale candidates"),
        ({"default_min_separation_rad": math.pi}, "separation"),
        ({"angle_unit": "degrees"}, "units"),
        ({"schema_version": "0.2"}, "schema_version"),
        ({"extras": (("duplicate", 1), ("duplicate", 2))}, "unique"),
    ],
)
def test_metadata_rejects_all_schema_boundary_violations(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FieldMetadata(2, **kwargs)  # type: ignore[arg-type]


def test_diagnostics_shape_probability_and_to_contract() -> None:
    metadata = FieldMetadata(
        8,
        frequency_candidates=(0.05, 0.1),
        scale_candidates=(1.0, 2.0),
    )
    joint = torch.full((2, 2, 8, 7, 9), 1.0 / 16.0)
    frequency_prob = joint.sum(2)
    diagnostics = StructuralFieldDiagnostics(
        metadata=metadata,
        normalized_image=torch.zeros(2, 1, 7, 9),
        orientation_frequency_response=torch.ones(2, 2, 8, 7, 9),
        orientation_frequency_prob=joint,
        orientation_response=torch.ones(2, 8, 7, 9),
        frequency_prob=frequency_prob,
        scale_response=torch.ones(2, 2, 7, 9),
        tensor_orientation=torch.zeros(2, 1, 7, 9),
    )
    converted = diagnostics.to(dtype=torch.float64)
    assert converted.normalized_image is not None
    assert converted.normalized_image.dtype == torch.float64
    assert diagnostics.normalized_image is not None
    assert diagnostics.normalized_image.dtype == torch.float32
    with pytest.raises(ValueError, match="sum to one"):
        replace(diagnostics, orientation_frequency_prob=joint * 0.5)


def test_diagnostics_rejects_low_dimensional_and_empty_channel_tensors() -> None:
    metadata = FieldMetadata(8)
    with pytest.raises(TypeError, match="normalized_image must be a torch.Tensor"):
        StructuralFieldDiagnostics(metadata=metadata, normalized_image=[1.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch and two spatial dimensions"):
        StructuralFieldDiagnostics(metadata=metadata, normalized_image=torch.ones(2))
    with pytest.raises(ValueError, match="at least one frequency channel"):
        StructuralFieldDiagnostics(metadata=metadata, frequency_prob=torch.empty(1, 0, 3, 3))


@pytest.mark.parametrize("field_name", ["tensor_orientation", "hessian_orientation"])
def test_diagnostics_rejects_exact_pi_orientation(field_name: str) -> None:
    with pytest.raises(ValueError, match=r"\[0, pi\)"):
        StructuralFieldDiagnostics(
            metadata=FieldMetadata(8),
            **{field_name: torch.full((1, 1, 3, 3), math.pi)},
        )


def test_orientation_modes_rejects_non_tensor_indices() -> None:
    with pytest.raises(TypeError, match="indices must be a torch.Tensor"):
        OrientationModes(
            angles=torch.zeros(1, 1, 2, 2),
            probabilities=torch.ones(1, 1, 2, 2),
            indices=None,  # type: ignore[arg-type]
        )


def test_orientation_modes_rejects_empty_batch_or_spatial_dimensions() -> None:
    for shape in ((0, 1, 2, 2), (1, 1, 0, 2), (1, 1, 2, 0)):
        with pytest.raises(ValueError, match="must be non-empty"):
            OrientationModes(
                angles=torch.empty(shape),
                probabilities=torch.empty(shape),
                indices=torch.empty(shape, dtype=torch.long),
            )
