"""Validated, task-neutral data structures for probabilistic structural fields."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, cast

import torch

from structfield.validation import ValidationMode, is_strict_validation

from .conventions import axial_angle_distance, uniform_orientation_bins
from .types import MetadataItems, MetadataValue

_SUPPORTED_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _probability_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 2e-2
    if dtype == torch.float16:
        return 5e-3
    if dtype == torch.float32:
        return 2e-5
    # A field converted from float32 to float64 retains the source rounding
    # error.  Keep validation stable across ``field.to(dtype=...)`` calls.
    return 1e-7


def _require_floating(name: str, value: torch.Tensor, *, finite: bool = True) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(f"{name} must use a supported floating-point dtype: float16, bfloat16, float32, or float64")
    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_probability(name: str, value: torch.Tensor, *, dim: int) -> None:
    tolerance = _probability_tolerance(value.dtype)
    if float(value.detach().min()) < -tolerance:
        raise ValueError(f"{name} contains negative probability mass")
    total = value.sum(dim=dim)
    if not torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=tolerance):
        raise ValueError(f"{name} must sum to one along dimension {dim}")


def _validate_unit_interval(name: str, value: torch.Tensor) -> None:
    tolerance = _probability_tolerance(value.dtype)
    if float(value.detach().min()) < -tolerance or float(value.detach().max()) > 1.0 + tolerance:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_metadata_value(name: str, value: MetadataValue) -> None:
    scalar_types = (str, int, float, bool, type(None))
    values = value if isinstance(value, tuple) else (value,)
    if not isinstance(value, scalar_types + (tuple,)) or not all(isinstance(item, scalar_types) for item in values):
        raise TypeError(f"metadata extra {name!r} must contain only JSON-compatible scalar values")
    if any(isinstance(item, float) and not math.isfinite(item) for item in values):
        raise ValueError(f"metadata extra {name!r} contains a non-finite float")


@dataclass(frozen=True)
class FieldMetadata:
    """Non-tensor conventions and candidate values associated with a field."""

    num_orientations: int
    orientation_bin_centers: tuple[float, ...] = ()
    frequency_candidates: tuple[float, ...] = ()
    scale_candidates: tuple[float, ...] = ()
    default_min_separation_rad: float = math.pi / 8.0
    estimator: str = "unspecified"
    schema_version: str = "1.0"
    angle_unit: str = "radians"
    angle_convention: str = "axial-[0,pi)"
    frequency_unit: str = "cycles/pixel"
    scale_unit: str = "pixels"
    coordinate_convention: str = "x-right,y-down"
    positive_rotation: str = "counter-clockwise-in-image-coordinates"
    extras: MetadataItems | Mapping[str, MetadataValue] = ()

    def __post_init__(self) -> None:
        bins = self.orientation_bin_centers or uniform_orientation_bins(self.num_orientations)
        bins = tuple(float(value) for value in bins)
        frequencies = tuple(float(value) for value in self.frequency_candidates)
        scales = tuple(float(value) for value in self.scale_candidates)
        extra_items = self.extras.items() if isinstance(self.extras, Mapping) else self.extras
        extras = tuple((str(key), value) for key, value in extra_items)
        object.__setattr__(self, "orientation_bin_centers", bins)
        object.__setattr__(self, "frequency_candidates", frequencies)
        object.__setattr__(self, "scale_candidates", scales)
        object.__setattr__(self, "extras", extras)
        self.validate()

    def validate(self) -> FieldMetadata:
        if isinstance(self.num_orientations, bool) or not isinstance(self.num_orientations, int):
            raise TypeError("num_orientations must be an integer")
        if self.num_orientations < 2:
            raise ValueError("num_orientations must be at least 2")
        if len(self.orientation_bin_centers) != self.num_orientations:
            raise ValueError("orientation_bin_centers length must match num_orientations")
        if not all(math.isfinite(value) and 0.0 <= value < math.pi for value in self.orientation_bin_centers):
            raise ValueError("orientation bin centers must be finite and lie in [0, pi)")
        if any(right <= left for left, right in zip(self.orientation_bin_centers, self.orientation_bin_centers[1:])):
            raise ValueError("orientation bin centers must be strictly increasing")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.frequency_candidates):
            raise ValueError("frequency candidates must be finite and positive")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.scale_candidates):
            raise ValueError("scale candidates must be finite and positive")
        if not math.isfinite(self.default_min_separation_rad) or not 0.0 <= self.default_min_separation_rad < math.pi:
            raise ValueError("default_min_separation_rad must lie in [0, pi)")
        if self.angle_unit != "radians" or self.frequency_unit != "cycles/pixel" or self.scale_unit != "pixels":
            raise ValueError("unsupported field units")
        if not isinstance(self.estimator, str) or not self.estimator:
            raise TypeError("estimator must be a non-empty string")
        if self.schema_version != "1.0":
            raise ValueError("schema_version must be '1.0'")
        if self.angle_convention != "axial-[0,pi)":
            raise ValueError("angle_convention must be axial-[0,pi)")
        if self.coordinate_convention != "x-right,y-down":
            raise ValueError("coordinate_convention must be x-right,y-down")
        if self.positive_rotation != "counter-clockwise-in-image-coordinates":
            raise ValueError("positive_rotation must be counter-clockwise-in-image-coordinates")
        extras = cast(MetadataItems, self.extras)
        if len({key for key, _ in extras}) != len(extras):
            raise ValueError("metadata extras keys must be unique")
        for key, value in extras:
            _validate_metadata_value(key, value)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "num_orientations": self.num_orientations,
            "orientation_bin_centers": list(self.orientation_bin_centers),
            "frequency_candidates": list(self.frequency_candidates),
            "scale_candidates": list(self.scale_candidates),
            "default_min_separation_rad": self.default_min_separation_rad,
            "estimator": self.estimator,
            "schema_version": self.schema_version,
            "angle_unit": self.angle_unit,
            "angle_convention": self.angle_convention,
            "frequency_unit": self.frequency_unit,
            "scale_unit": self.scale_unit,
            "coordinate_convention": self.coordinate_convention,
            "positive_rotation": self.positive_rotation,
            "extras": dict(self.extras),
        }


@dataclass(frozen=True)
class OrientationModes:
    """Top orientation modes selected from an axial probability distribution."""

    angles: torch.Tensor
    probabilities: torch.Tensor
    indices: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> OrientationModes:
        _require_floating("angles", self.angles)
        _require_floating("probabilities", self.probabilities)
        if not isinstance(self.indices, torch.Tensor):
            raise TypeError("orientation mode indices must be a torch.Tensor")
        if self.indices.dtype != torch.long:
            raise TypeError("orientation mode indices must use torch.long")
        if self.angles.ndim != 4 or self.angles.shape[1] < 1:
            raise ValueError("orientation modes must have shape [B, T, H, W] with T >= 1")
        if self.angles.shape[0] < 1 or self.angles.shape[-2] < 1 or self.angles.shape[-1] < 1:
            raise ValueError("orientation mode batch and spatial dimensions must be non-empty")
        if self.angles.shape != self.probabilities.shape or self.angles.shape != self.indices.shape:
            raise ValueError("angles, probabilities, and indices must have identical shapes")
        if self.angles.device != self.probabilities.device or self.angles.device != self.indices.device:
            raise ValueError("orientation mode tensors must share a device")
        if self.angles.dtype != self.probabilities.dtype:
            raise ValueError("orientation mode angles and probabilities must share a dtype")
        if float(self.angles.detach().min()) < 0.0 or float(self.angles.detach().max()) >= math.pi:
            raise ValueError("orientation mode angles must lie in [0, pi)")
        tolerance = _probability_tolerance(self.angles.dtype)
        if (
            float(self.probabilities.detach().min()) < -tolerance
            or float(self.probabilities.detach().max()) > 1.0 + tolerance
        ):
            raise ValueError("orientation mode probabilities must lie in [0, 1]")
        if int(self.indices.min()) < 0:
            raise ValueError("orientation mode indices must be non-negative")
        return self

    def to(self, *args: Any, **kwargs: Any) -> OrientationModes:
        """Return a new mode object on the requested device/dtype."""
        angles = self.angles.to(*args, **kwargs)
        probabilities = self.probabilities.to(*args, **kwargs)
        indices = self.indices.to(device=angles.device)
        return replace(self, angles=angles, probabilities=probabilities, indices=indices)


@dataclass(frozen=True)
class StructuralField:
    """Batch-aware probabilistic structural-field representation.

    All spatial tensors share ``B``, ``H``, ``W``, a supported floating dtype,
    and device.
    The dataclass is structurally immutable and all transformation methods
    return new objects; contained tensors retain standard PyTorch mutability.
    """

    orientation_prob: torch.Tensor
    frequency_mean: torch.Tensor
    frequency_var: torch.Tensor
    coherence: torch.Tensor
    orientation_entropy: torch.Tensor
    structural_uncertainty: torch.Tensor
    metadata: FieldMetadata
    scale_prob: torch.Tensor | None = None
    selected_scale: torch.Tensor | None = None
    validation_mode: ValidationMode | str = ValidationMode.strict

    def __post_init__(self) -> None:
        object.__setattr__(self, "validation_mode", ValidationMode.normalize(self.validation_mode))
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.orientation_prob.shape[0])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.orientation_prob.shape[-2]), int(self.orientation_prob.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.orientation_prob.device

    @property
    def dtype(self) -> torch.dtype:
        return self.orientation_prob.dtype

    def validate(self) -> StructuralField:
        strict = is_strict_validation(self.validation_mode)
        if not isinstance(self.metadata, FieldMetadata):
            raise TypeError("metadata must be FieldMetadata")
        self.metadata.validate()
        _require_floating("orientation_prob", self.orientation_prob, finite=strict)
        if self.orientation_prob.ndim != 4:
            raise ValueError("orientation_prob must have shape [B, K, H, W]")
        batch, bins, height, width = self.orientation_prob.shape
        if batch < 1 or height < 1 or width < 1:
            raise ValueError("StructuralField dimensions must be non-empty")
        if bins != self.metadata.num_orientations:
            raise ValueError("orientation_prob channels must match metadata.num_orientations")
        if strict:
            _validate_probability("orientation_prob", self.orientation_prob, dim=1)

        scalar_maps = {
            "frequency_mean": self.frequency_mean,
            "frequency_var": self.frequency_var,
            "coherence": self.coherence,
            "orientation_entropy": self.orientation_entropy,
            "structural_uncertainty": self.structural_uncertainty,
        }
        all_tensors = scalar_maps
        for name, value in scalar_maps.items():
            _require_floating(name, value, finite=strict)
            if value.shape != (batch, 1, height, width):
                raise ValueError(f"{name} must have shape [B, 1, H, W]")
        for name, value in all_tensors.items():
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError(f"{name} must share orientation_prob dtype and device")

        tolerance = _probability_tolerance(self.dtype)
        if strict:
            if float(self.frequency_mean.detach().min()) < -tolerance:
                raise ValueError("frequency_mean must be non-negative")
            if float(self.frequency_var.detach().min()) < -tolerance:
                raise ValueError("frequency_var must be non-negative")
            _validate_unit_interval("coherence", self.coherence)
            _validate_unit_interval("orientation_entropy", self.orientation_entropy)
            _validate_unit_interval("structural_uncertainty", self.structural_uncertainty)

        if (self.scale_prob is None) != (self.selected_scale is None):
            raise ValueError("scale_prob and selected_scale must either both be present or both be absent")
        if self.scale_prob is not None and self.selected_scale is not None:
            _require_floating("scale_prob", self.scale_prob, finite=strict)
            _require_floating("selected_scale", self.selected_scale, finite=strict)
            if self.scale_prob.ndim != 4 or self.scale_prob.shape[0] != batch:
                raise ValueError("scale_prob must have shape [B, S, H, W]")
            if self.scale_prob.shape[1] < 1:
                raise ValueError("scale_prob must have at least one scale channel")
            if self.scale_prob.shape[-2:] != (height, width):
                raise ValueError("scale_prob must share field spatial dimensions")
            if self.selected_scale.shape != (batch, 1, height, width):
                raise ValueError("selected_scale must have shape [B, 1, H, W]")
            if self.scale_prob.device != self.device or self.selected_scale.device != self.device:
                raise ValueError("scale tensors must share field device")
            if self.scale_prob.dtype != self.dtype or self.selected_scale.dtype != self.dtype:
                raise ValueError("scale tensors must share field dtype")
            if len(self.metadata.scale_candidates) != self.scale_prob.shape[1]:
                raise ValueError("scale_prob channels must match metadata.scale_candidates")
            if strict:
                _validate_probability("scale_prob", self.scale_prob, dim=1)
                if float(self.selected_scale.detach().min()) < -tolerance:
                    raise ValueError("selected_scale must be non-negative")
        return self

    def topk_orientations(
        self,
        k: int = 2,
        min_separation: float | None = None,
    ) -> OrientationModes:
        """Select axial modes with circular non-maximum suppression."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 1 or k > self.metadata.num_orientations:
            raise ValueError("k must lie in [1, num_orientations]")
        separation = self.metadata.default_min_separation_rad if min_separation is None else float(min_separation)
        if not math.isfinite(separation) or not 0.0 <= separation < math.pi:
            raise ValueError("min_separation must lie in [0, pi)")
        if k > 1 and separation > 0.0 and k * separation > math.pi + 1e-7:
            raise ValueError("requested modes cannot satisfy the minimum axial separation")

        centers = self.orientation_prob.new_tensor(self.metadata.orientation_bin_centers)
        centers_map = centers.view(1, -1, 1, 1)
        suppressed = torch.zeros_like(self.orientation_prob, dtype=torch.bool)
        selected_indices: list[torch.Tensor] = []
        selected_angles: list[torch.Tensor] = []
        selected_probabilities: list[torch.Tensor] = []
        tolerance = _probability_tolerance(self.dtype)
        fallback_modes: OrientationModes | None = None
        fallback_mask = torch.zeros(
            (self.batch_size, *self.spatial_shape),
            device=self.device,
            dtype=torch.bool,
        )

        def uniform_fallback() -> OrientationModes | None:
            expected = uniform_orientation_bins(self.metadata.num_orientations)
            if any(
                abs(actual - target) > 1e-10 for actual, target in zip(self.metadata.orientation_bin_centers, expected)
            ):
                return None
            step = math.pi / self.metadata.num_orientations
            required_gap = max(1, math.ceil(max(separation - tolerance, 0.0) / step))
            if k * required_gap > self.metadata.num_orientations:
                return None
            anchor = self.orientation_prob.argmax(dim=1, keepdim=True)
            offsets = torch.arange(k, device=self.device, dtype=torch.long) * required_gap
            indices = torch.remainder(anchor + offsets.view(1, k, 1, 1), self.metadata.num_orientations)
            probabilities = torch.gather(self.orientation_prob, 1, indices)
            order = probabilities.argsort(dim=1, descending=True)
            indices = torch.gather(indices, 1, order)
            probabilities = torch.gather(probabilities, 1, order)
            return OrientationModes(angles=centers[indices], probabilities=probabilities, indices=indices)

        for mode_index in range(k):
            scores = self.orientation_prob.masked_fill(suppressed, -torch.inf)
            probability, index = scores.max(dim=1)
            new_failure = ~torch.isfinite(probability)
            if bool(new_failure.any()):
                fallback_modes = uniform_fallback()
                if fallback_modes is None:
                    raise ValueError("orientation bins cannot satisfy the requested separation")
                fallback_mask = fallback_mask | new_failure
            if fallback_modes is not None:
                fallback_index = fallback_modes.indices[:, mode_index]
                fallback_probability = fallback_modes.probabilities[:, mode_index]
                index = torch.where(fallback_mask, fallback_index, index)
                probability = torch.where(fallback_mask, fallback_probability, probability)
            angle = centers[index]
            selected_indices.append(index)
            selected_angles.append(angle)
            selected_probabilities.append(probability)
            distance = axial_angle_distance(centers_map, angle[:, None])
            suppressed = suppressed | (distance + tolerance < separation)
            suppressed = suppressed.scatter(1, index[:, None], True)

        angles = torch.stack(selected_angles, dim=1)
        probabilities = torch.stack(selected_probabilities, dim=1)
        indices = torch.stack(selected_indices, dim=1)
        if fallback_modes is not None:
            mask = fallback_mask[:, None]
            angles = torch.where(mask, fallback_modes.angles, angles)
            probabilities = torch.where(mask, fallback_modes.probabilities, probabilities)
            indices = torch.where(mask, fallback_modes.indices, indices)
        return OrientationModes(angles=angles, probabilities=probabilities, indices=indices)

    def to(self, *args: Any, **kwargs: Any) -> StructuralField:
        """Return a new field with every tensor moved or cast consistently."""
        updates: dict[str, torch.Tensor | None] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                updates[item.name] = value.to(*args, **kwargs)
        return replace(self, **cast(Any, updates))


@dataclass(frozen=True)
class StructuralFieldDiagnostics:
    """Optional analytic tensors kept outside the task-neutral field schema."""

    metadata: FieldMetadata
    normalized_image: torch.Tensor | None = None
    orientation_frequency_response: torch.Tensor | None = None
    orientation_frequency_prob: torch.Tensor | None = None
    orientation_response: torch.Tensor | None = None
    frequency_prob: torch.Tensor | None = None
    scale_response: torch.Tensor | None = None
    tensor_orientation: torch.Tensor | None = None
    tensor_eigenvalue_max: torch.Tensor | None = None
    tensor_eigenvalue_min: torch.Tensor | None = None
    hessian_vesselness: torch.Tensor | None = None
    hessian_orientation: torch.Tensor | None = None
    hessian_line_confidence: torch.Tensor | None = None
    hessian_blob_response: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def _tensor_items(self) -> dict[str, torch.Tensor]:
        return {
            item.name: value for item in fields(self) if isinstance((value := getattr(self, item.name)), torch.Tensor)
        }

    def validate(self) -> StructuralFieldDiagnostics:
        if not isinstance(self.metadata, FieldMetadata):
            raise TypeError("metadata must be FieldMetadata")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name != "metadata" and value is not None:
                _require_floating(item.name, value)
        tensors = self._tensor_items()
        if not tensors:
            return self
        for name, value in tensors.items():
            if value.ndim < 3:
                raise ValueError(f"{name} must include batch and two spatial dimensions")
        reference = next(iter(tensors.values()))
        batch, height, width = int(reference.shape[0]), int(reference.shape[-2]), int(reference.shape[-1])
        if batch < 1 or height < 1 or width < 1:
            raise ValueError("diagnostic tensors must have non-empty batch and spatial dimensions")
        for name, value in tensors.items():
            if value.shape[0] != batch or value.shape[-2:] != (height, width):
                raise ValueError(f"{name} must share diagnostic batch and spatial dimensions")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must share diagnostic dtype and device")

        scalar_names = {
            "normalized_image",
            "tensor_orientation",
            "tensor_eigenvalue_max",
            "tensor_eigenvalue_min",
            "hessian_vesselness",
            "hessian_orientation",
            "hessian_line_confidence",
            "hessian_blob_response",
        }
        for name in scalar_names:
            value = getattr(self, name)
            if value is not None and value.shape != (batch, 1, height, width):
                raise ValueError(f"{name} must have shape [B, 1, H, W]")

        for name in ("tensor_orientation", "hessian_orientation"):
            value = getattr(self, name)
            if value is not None and (float(value.detach().min()) < 0.0 or float(value.detach().max()) >= math.pi):
                raise ValueError(f"{name} must lie in [0, pi)")

        for name in ("orientation_frequency_response", "orientation_frequency_prob"):
            value = getattr(self, name)
            if value is not None:
                if value.ndim != 5 or value.shape[2] != self.metadata.num_orientations:
                    raise ValueError(f"{name} must have shape [B, F, K, H, W]")
                if value.shape[1] < 1:
                    raise ValueError(f"{name} must have at least one frequency channel")
                if self.metadata.frequency_candidates and value.shape[1] != len(self.metadata.frequency_candidates):
                    raise ValueError(f"{name} frequency channels must match metadata")
        if self.orientation_frequency_prob is not None:
            flattened = self.orientation_frequency_prob.flatten(1, 2)
            _validate_probability("orientation_frequency_prob", flattened, dim=1)
        if self.orientation_response is not None:
            if self.orientation_response.shape != (batch, self.metadata.num_orientations, height, width):
                raise ValueError("orientation_response must have shape [B, K, H, W]")
        if self.frequency_prob is not None:
            if self.frequency_prob.ndim != 4:
                raise ValueError("frequency_prob must have shape [B, F, H, W]")
            if self.frequency_prob.shape[1] < 1:
                raise ValueError("frequency_prob must have at least one frequency channel")
            if self.metadata.frequency_candidates and self.frequency_prob.shape[1] != len(
                self.metadata.frequency_candidates
            ):
                raise ValueError("frequency_prob channels must match metadata")
            _validate_probability("frequency_prob", self.frequency_prob, dim=1)
        if self.scale_response is not None:
            if self.scale_response.ndim != 4:
                raise ValueError("scale_response must have shape [B, S, H, W]")
            if self.scale_response.shape[1] < 1:
                raise ValueError("scale_response must have at least one scale channel")
            if self.metadata.scale_candidates and self.scale_response.shape[1] != len(self.metadata.scale_candidates):
                raise ValueError("scale_response channels must match metadata")
        return self

    def to(self, *args: Any, **kwargs: Any) -> StructuralFieldDiagnostics:
        """Return a new diagnostics object with all present tensors converted."""
        updates = {name: value.to(*args, **kwargs) for name, value in self._tensor_items().items()}
        return replace(self, **cast(Any, updates))
