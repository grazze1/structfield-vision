"""Geometry-aware transforms for tensors and structural fields.

The implementation is project-authored from the coordinate conventions frozen
by :mod:`structfield.core`: tensor rows increase downward, while positive
angles and rotations are counter-clockwise on the displayed image.  Sampling
uses one output-to-input affine grid for every component of a field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from structfield.core import FieldMetadata, StructuralField
from structfield.uncertainty import normalized_categorical_entropy


@dataclass(frozen=True)
class GeometricTransform:
    """A same-canvas rotation and isotropic scaling about the pixel center.

    ``rotation_rad`` follows the public displayed-image counter-clockwise
    convention. ``scale`` is a forward spatial scale: values greater than one
    magnify the source, so structural frequencies are divided by ``scale``.
    """

    rotation_rad: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.rotation_rad, bool) or isinstance(self.scale, bool):
            raise TypeError("rotation_rad and scale must be real numbers, not bool")
        try:
            rotation_rad = float(self.rotation_rad)
            scale = float(self.scale)
        except (TypeError, ValueError) as error:
            raise TypeError("rotation_rad and scale must be real numbers") from error
        if not math.isfinite(rotation_rad):
            raise ValueError("rotation_rad must be finite")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        object.__setattr__(self, "rotation_rad", rotation_rad)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True)
class TensorTransformResult:
    """A transformed BCHW tensor and its strict bilinear-valid region."""

    tensor: torch.Tensor
    equivariance_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        _validate_result_mask(self.tensor, self.equivariance_valid_mask)


@dataclass(frozen=True)
class StructuralFieldTransformResult:
    """A transformed structural field and its strict bilinear-valid region."""

    field: StructuralField
    equivariance_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        _validate_result_mask(self.field.orientation_prob, self.equivariance_valid_mask)


def _validate_bchw(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating-point dtype")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W]")
    if any(size < 1 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_source_mask(reference: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    expected_shape = (reference.shape[0], 1, reference.shape[-2], reference.shape[-1])
    if mask is None:
        return torch.ones(expected_shape, device=reference.device, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor):
        raise TypeError("source_valid_mask must be a torch.Tensor")
    if mask.dtype != torch.bool:
        raise TypeError("source_valid_mask must use torch.bool")
    if mask.shape != expected_shape:
        raise ValueError("source_valid_mask must have shape [B, 1, H, W]")
    if mask.device != reference.device:
        raise ValueError("source_valid_mask must share the input device")
    return mask


def _validate_result_mask(reference: torch.Tensor, mask: torch.Tensor) -> None:
    expected_shape = (reference.shape[0], 1, reference.shape[-2], reference.shape[-1])
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("equivariance_valid_mask must be a bool tensor")
    if mask.shape != expected_shape:
        raise ValueError("equivariance_valid_mask must have shape [B, 1, H, W]")
    if mask.device != reference.device:
        raise ValueError("equivariance_valid_mask must share the transformed value device")


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype


def _sampling_grid(reference: torch.Tensor, transform: GeometricTransform) -> torch.Tensor:
    """Build the output-to-input grid with pixel-correct rectangular factors."""
    batch, _, height, width = reference.shape
    dtype = _working_dtype(reference.dtype)
    cosine = math.cos(transform.rotation_rad) / transform.scale
    sine = math.sin(transform.rotation_rad) / transform.scale
    # align_corners=False maps normalized coordinates to centered pixel offsets
    # as x_n=2*x_offset/W and y_n=2*y_offset/H.  The H/W and W/H terms
    # therefore preserve a true pixel-space rotation on rectangular canvases.
    theta = torch.tensor(
        (
            (cosine, -sine * height / width, 0.0),
            (sine * width / height, cosine, 0.0),
        ),
        device=reference.device,
        dtype=dtype,
    ).expand(batch, -1, -1)
    return F.affine_grid(theta, list(reference.shape), align_corners=False)


def _warp(tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    working = tensor.to(dtype=grid.dtype)
    return F.grid_sample(
        working,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _valid_mask(source_valid_mask: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Require the complete bilinear footprint to come from valid source pixels."""
    support = F.grid_sample(
        source_valid_mask.to(dtype=grid.dtype),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    tolerance = 4.0 * torch.finfo(grid.dtype).eps
    return support >= 1.0 - tolerance


def _normalize_probability(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp_min(0.0)
    total = probability.sum(dim=1, keepdim=True)
    tiny = torch.finfo(probability.dtype).tiny
    normalized = probability / total.clamp_min(tiny)
    uniform = torch.full_like(probability, 1.0 / probability.shape[1])
    return torch.where(total > tiny, normalized, uniform)


def _orientation_transport_matrix(
    metadata: FieldMetadata,
    rotation_rad: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a column-stochastic axial linear mass-transport matrix."""
    centers = metadata.orientation_bin_centers
    num_bins = len(centers)
    matrix = torch.zeros((num_bins, num_bins), device=reference.device, dtype=reference.dtype)
    period = math.pi
    snap_tolerance = 1e-12

    for source_index, center in enumerate(centers):
        shifted = (center + rotation_rad) % period
        distances = tuple(abs(shifted - destination) for destination in centers)
        exact_index = min(range(num_bins), key=distances.__getitem__)
        circular_exact_distance = min(distances[exact_index], period - distances[exact_index])
        if circular_exact_distance <= snap_tolerance:
            matrix[exact_index, source_index] = 1.0
            continue

        upper = next((index for index, destination in enumerate(centers) if destination > shifted), 0)
        lower = (upper - 1) % num_bins
        lower_angle = centers[lower]
        upper_angle = centers[upper]
        unwrapped_shifted = shifted
        if shifted < centers[0]:
            lower_angle -= period
        elif upper == 0:
            upper_angle += period
        interval = upper_angle - lower_angle
        upper_weight = (unwrapped_shifted - lower_angle) / interval
        matrix[lower, source_index] = 1.0 - upper_weight
        matrix[upper, source_index] = upper_weight
    return matrix


def _transport_orientation_probability(
    probability: torch.Tensor,
    metadata: FieldMetadata,
    rotation_rad: float,
) -> torch.Tensor:
    matrix = _orientation_transport_matrix(metadata, rotation_rad, probability)
    transported = torch.einsum("ij,bjhw->bihw", matrix, probability)
    return _normalize_probability(transported)


def _transformed_metadata(metadata: FieldMetadata, scale: float) -> FieldMetadata:
    if scale == 1.0:
        return metadata
    return replace(
        metadata,
        frequency_candidates=tuple(value / scale for value in metadata.frequency_candidates),
        scale_candidates=tuple(value * scale for value in metadata.scale_candidates),
    )


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def transform_tensor(
    tensor: torch.Tensor,
    transform: GeometricTransform,
    *,
    source_valid_mask: torch.Tensor | None = None,
) -> TensorTransformResult:
    """Apply a geometry transform to a BCHW tensor on the same canvas."""
    _validate_bchw("tensor", tensor)
    if not isinstance(transform, GeometricTransform):
        raise TypeError("transform must be a GeometricTransform")
    source_mask = _validate_source_mask(tensor, source_valid_mask)
    grid = _sampling_grid(tensor, transform)
    transformed = _warp(tensor, grid).to(dtype=tensor.dtype)
    valid_mask = _valid_mask(source_mask, grid)
    return TensorTransformResult(tensor=transformed, equivariance_valid_mask=valid_mask)


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def transform_structural_field(
    field: StructuralField,
    transform: GeometricTransform,
    *,
    source_valid_mask: torch.Tensor | None = None,
) -> StructuralFieldTransformResult:
    """Transform all spatial and semantic components of a structural field."""
    if not isinstance(field, StructuralField):
        raise TypeError("field must be a StructuralField")
    if not isinstance(transform, GeometricTransform):
        raise TypeError("transform must be a GeometricTransform")
    source_mask = _validate_source_mask(field.orientation_prob, source_valid_mask)
    grid = _sampling_grid(field.orientation_prob, transform)
    output_dtype = field.dtype

    spatial_orientation_probability = _normalize_probability(_warp(field.orientation_prob, grid))
    orientation_probability = _transport_orientation_probability(
        spatial_orientation_probability,
        field.metadata,
        transform.rotation_rad,
    )
    frequency_mean = (_warp(field.frequency_mean, grid) / transform.scale).clamp_min(0.0)
    frequency_var = (_warp(field.frequency_var, grid) / transform.scale**2).clamp_min(0.0)
    coherence = _warp(field.coherence, grid).clamp(0.0, 1.0)
    structural_uncertainty = _warp(field.structural_uncertainty, grid).clamp(0.0, 1.0)
    orientation_entropy = normalized_categorical_entropy(
        orientation_probability,
        dim=1,
        validation_mode=field.validation_mode,
    )

    scale_probability: torch.Tensor | None = None
    selected_scale: torch.Tensor | None = None
    if field.scale_prob is not None and field.selected_scale is not None:
        scale_probability = _normalize_probability(_warp(field.scale_prob, grid))
        selected_scale = (_warp(field.selected_scale, grid) * transform.scale).clamp_min(0.0)

    transformed_field = StructuralField(
        orientation_prob=orientation_probability.to(dtype=output_dtype),
        frequency_mean=frequency_mean.to(dtype=output_dtype),
        frequency_var=frequency_var.to(dtype=output_dtype),
        coherence=coherence.to(dtype=output_dtype),
        orientation_entropy=orientation_entropy.to(dtype=output_dtype),
        structural_uncertainty=structural_uncertainty.to(dtype=output_dtype),
        metadata=_transformed_metadata(field.metadata, transform.scale),
        scale_prob=None if scale_probability is None else scale_probability.to(dtype=output_dtype),
        selected_scale=None if selected_scale is None else selected_scale.to(dtype=output_dtype),
        validation_mode=field.validation_mode,
    )
    valid_mask = _valid_mask(source_mask, grid)
    return StructuralFieldTransformResult(
        field=transformed_field,
        equivariance_valid_mask=valid_mask,
    )


def rotate_tensor(
    tensor: torch.Tensor,
    rotation_rad: float,
    *,
    source_valid_mask: torch.Tensor | None = None,
) -> TensorTransformResult:
    """Rotate a BCHW tensor counter-clockwise on the displayed image."""
    return transform_tensor(
        tensor,
        GeometricTransform(rotation_rad=rotation_rad),
        source_valid_mask=source_valid_mask,
    )


def scale_tensor(
    tensor: torch.Tensor,
    scale: float,
    *,
    source_valid_mask: torch.Tensor | None = None,
) -> TensorTransformResult:
    """Scale a BCHW tensor isotropically about the same-canvas center."""
    return transform_tensor(
        tensor,
        GeometricTransform(scale=scale),
        source_valid_mask=source_valid_mask,
    )


def rotate_structural_field(
    field: StructuralField,
    rotation_rad: float,
    *,
    source_valid_mask: torch.Tensor | None = None,
) -> StructuralFieldTransformResult:
    """Rotate a field spatially and shift its axial orientations."""
    return transform_structural_field(
        field,
        GeometricTransform(rotation_rad=rotation_rad),
        source_valid_mask=source_valid_mask,
    )


def scale_structural_field(
    field: StructuralField,
    scale: float,
    *,
    source_valid_mask: torch.Tensor | None = None,
) -> StructuralFieldTransformResult:
    """Scale a field spatially and update frequency/scale units."""
    return transform_structural_field(
        field,
        GeometricTransform(scale=scale),
        source_valid_mask=source_valid_mask,
    )


__all__ = [
    "GeometricTransform",
    "StructuralFieldTransformResult",
    "TensorTransformResult",
    "rotate_structural_field",
    "rotate_tensor",
    "scale_structural_field",
    "scale_tensor",
    "transform_structural_field",
    "transform_tensor",
]
