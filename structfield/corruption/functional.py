"""Pure structural-field corruption functions for robustness experiments.

These are transparent V0.2 reference corruptions. They do not mutate or share
tensor storage with the input field, and stochastic functions use an explicit
function-local generator rather than PyTorch's global RNG.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from typing import Any, cast

import torch

from structfield.core import StructuralField
from structfield.uncertainty import normalized_categorical_entropy

from .metadata import CorruptionMetadata, CorruptionName, CorruptionParameters


def _require_field(field: StructuralField) -> StructuralField:
    if not isinstance(field, StructuralField):
        raise TypeError("field must be a StructuralField")
    return field


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _nonnegative_float(name: str, value: float) -> float:
    converted = _finite_float(name, value)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _seed(value: int) -> int:
    value = _integer("seed", value)
    if not 0 <= value <= 2**63 - 1:
        raise ValueError("seed must lie in [0, 2**63 - 1]")
    return value


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype


def _local_generator(field: StructuralField, seed: int) -> torch.Generator:
    generator = torch.Generator(device=field.device)
    generator.manual_seed(_seed(seed))
    return generator


def _normal_noise(field: StructuralField, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(
        shape,
        device=field.device,
        dtype=_working_dtype(field.dtype),
        generator=_local_generator(field, seed),
    )


def _copy_with(field: StructuralField, **updates: torch.Tensor | None) -> StructuralField:
    """Build an independently stored field while preserving the autograd graph."""
    copied: dict[str, torch.Tensor | None] = {}
    for item in fields(field):
        value = getattr(field, item.name)
        if isinstance(value, torch.Tensor) and item.name not in updates:
            copied[item.name] = value.clone()
    copied.update(updates)
    return replace(field, **cast(Any, copied))


def _metadata(
    name: CorruptionName,
    parameters: CorruptionParameters,
    *,
    seed: int | None = None,
) -> CorruptionMetadata:
    return CorruptionMetadata(name=name, parameters=parameters, seed=seed)


def _resample_orientation(
    field: StructuralField,
    shift_radians: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift a posterior on arbitrary ordered axial bins by linear interpolation."""
    working_dtype = _working_dtype(field.dtype)
    probability = field.orientation_prob.to(dtype=working_dtype)
    shift = shift_radians.to(device=field.device, dtype=working_dtype)
    shift = torch.broadcast_to(
        shift,
        (field.batch_size, 1, *field.spatial_shape),
    )
    centers = probability.new_tensor(field.metadata.orientation_bin_centers)
    centers_map = centers.view(1, -1, 1, 1)
    period = probability.new_tensor(math.pi)
    target = torch.remainder(centers_map + shift, period).contiguous()

    # Transport each source bin's mass to its two neighboring target bins.
    right_raw = torch.searchsorted(centers, target, right=True)
    bins = field.metadata.num_orientations
    left_index = torch.remainder(right_raw - 1, bins)
    right_index = torch.remainder(right_raw, bins)
    expanded_centers = centers_map.expand_as(probability)
    left_coordinate = torch.gather(expanded_centers, 1, left_index)
    right_coordinate = torch.gather(expanded_centers, 1, right_index)
    left_coordinate = torch.where(right_raw == 0, left_coordinate - period, left_coordinate)
    right_coordinate = torch.where(right_raw == bins, right_coordinate + period, right_coordinate)
    weight = (target - left_coordinate) / (right_coordinate - left_coordinate)
    left_mass = (1.0 - weight) * probability
    right_mass = weight * probability
    target_channels = []
    for target_index in range(bins):
        from_left = left_mass.masked_fill(left_index != target_index, 0.0).sum(dim=1, keepdim=True)
        from_right = right_mass.masked_fill(right_index != target_index, 0.0).sum(dim=1, keepdim=True)
        target_channels.append(from_left + from_right)
    shifted_probability = torch.cat(target_channels, dim=1)
    tiny = torch.finfo(working_dtype).tiny
    shifted_probability = shifted_probability / shifted_probability.sum(dim=1, keepdim=True).clamp_min(tiny)
    shifted_probability = shifted_probability.to(dtype=field.dtype)

    shifted_entropy = normalized_categorical_entropy(
        shifted_probability,
        dim=1,
        validation_mode=field.validation_mode,
    ).to(dtype=field.dtype)
    return shifted_probability, shifted_entropy


def orientation_shift(
    field: StructuralField,
    shift_radians: float,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Rotate orientations by a spatially constant axial angle.

    Posterior bins are sampled periodically on ``[0, pi)``. Consequently, any
    integer multiple of ``pi`` is an exact identity apart from independent
    output storage.
    """
    field = _require_field(field)
    requested_shift = _finite_float("shift_radians", shift_radians)
    wrapped_shift = requested_shift % math.pi
    if math.isclose(wrapped_shift, 0.0, abs_tol=1e-12) or math.isclose(
        wrapped_shift,
        math.pi,
        abs_tol=1e-12,
    ):
        wrapped_shift = 0.0
        corrupted = _copy_with(field)
    else:
        shift = field.orientation_prob.new_full((1, 1, 1, 1), wrapped_shift)
        probability, entropy = _resample_orientation(field, shift)
        corrupted = _copy_with(
            field,
            orientation_prob=probability,
            orientation_entropy=entropy,
        )
    metadata = _metadata(
        "orientation_shift",
        (
            ("shift_radians", requested_shift),
            ("wrapped_shift_radians", wrapped_shift),
            ("period_radians", math.pi),
            ("interpolation", "periodic-linear-mass"),
        ),
    )
    return corrupted, metadata


def orientation_noise(
    field: StructuralField,
    std_radians: float,
    *,
    seed: int,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Apply one Gaussian angular displacement per field pixel."""
    field = _require_field(field)
    std = _nonnegative_float("std_radians", std_radians)
    seed = _seed(seed)
    if std == 0.0:
        corrupted = _copy_with(field)
    else:
        noise = (
            _normal_noise(
                field,
                (field.batch_size, 1, *field.spatial_shape),
                seed,
            )
            * std
        )
        probability, entropy = _resample_orientation(field, noise)
        corrupted = _copy_with(
            field,
            orientation_prob=probability,
            orientation_entropy=entropy,
        )
    metadata = _metadata(
        "orientation_noise",
        (
            ("std_radians", std),
            ("distribution", "gaussian"),
            ("granularity", "pixel"),
            ("period_radians", math.pi),
            ("interpolation", "periodic-linear-mass"),
        ),
        seed=seed,
    )
    return corrupted, metadata


def frequency_scale(
    field: StructuralField,
    factor: float,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Scale frequency mean and variance by ``factor`` and ``factor**2``."""
    field = _require_field(field)
    factor = _finite_float("factor", factor)
    if factor <= 0.0:
        raise ValueError("factor must be positive")
    working_dtype = _working_dtype(field.dtype)
    mean = (field.frequency_mean.to(dtype=working_dtype) * factor).to(dtype=field.dtype)
    variance = (field.frequency_var.to(dtype=working_dtype) * factor**2).to(dtype=field.dtype)
    corrupted = _copy_with(field, frequency_mean=mean, frequency_var=variance)
    return corrupted, _metadata(
        "frequency_scale",
        (("factor", factor), ("variance_factor", factor**2)),
    )


def frequency_noise(
    field: StructuralField,
    std: float,
    *,
    seed: int,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Add Gaussian frequency noise in cycles/pixel and clamp at zero."""
    field = _require_field(field)
    std = _nonnegative_float("std", std)
    seed = _seed(seed)
    working_dtype = _working_dtype(field.dtype)
    if std == 0.0:
        mean = field.frequency_mean.clone()
    else:
        noise = _normal_noise(field, tuple(field.frequency_mean.shape), seed) * std
        mean = (field.frequency_mean.to(dtype=working_dtype) + noise).clamp_min(0.0).to(dtype=field.dtype)
    corrupted = _copy_with(field, frequency_mean=mean)
    return corrupted, _metadata(
        "frequency_noise",
        (("std", std), ("unit", "cycles/pixel"), ("distribution", "gaussian"), ("lower_bound", 0.0)),
        seed=seed,
    )


def spatial_shift(
    field: StructuralField,
    shift_y: int,
    shift_x: int,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Circularly shift every spatial field tensor by integer pixels."""
    field = _require_field(field)
    shift_y = _integer("shift_y", shift_y)
    shift_x = _integer("shift_x", shift_x)
    updates: dict[str, torch.Tensor] = {}
    for item in fields(field):
        value = getattr(field, item.name)
        if isinstance(value, torch.Tensor):
            updates[item.name] = torch.roll(value, shifts=(shift_y, shift_x), dims=(-2, -1))
    corrupted = _copy_with(field, **updates)
    return corrupted, _metadata(
        "spatial_shift",
        (("shift_y", shift_y), ("shift_x", shift_x), ("boundary", "circular")),
    )


def scale_shift(
    field: StructuralField,
    bins: int,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Shift scale-posterior mass by discrete bins with endpoint accumulation."""
    field = _require_field(field)
    bins = _integer("bins", bins)
    if field.scale_prob is None or field.selected_scale is None:
        raise ValueError("scale_shift requires scale_prob and selected_scale")
    scale_count = field.scale_prob.shape[1]
    destinations = [min(max(source + bins, 0), scale_count - 1) for source in range(scale_count)]
    shifted_channels = []
    for destination in range(scale_count):
        source_indices = [index for index, target in enumerate(destinations) if target == destination]
        shifted_channels.append(field.scale_prob[:, source_indices].sum(dim=1, keepdim=True))
    shifted_probability = torch.cat(shifted_channels, dim=1)
    working_dtype = _working_dtype(field.dtype)
    scale_values = shifted_probability.new_tensor(
        field.metadata.scale_candidates,
        dtype=working_dtype,
    ).view(1, -1, 1, 1)
    selected_scale = (
        (shifted_probability.to(dtype=working_dtype) * scale_values).sum(dim=1, keepdim=True).to(dtype=field.dtype)
    )
    corrupted = _copy_with(
        field,
        scale_prob=shifted_probability,
        selected_scale=selected_scale,
    )
    return corrupted, _metadata(
        "scale_shift",
        (("bins", bins), ("boundary", "clamp-accumulate")),
    )


def coherence_noise(
    field: StructuralField,
    std: float,
    *,
    seed: int,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Add Gaussian coherence noise and clamp the result to ``[0,1]``."""
    field = _require_field(field)
    std = _nonnegative_float("std", std)
    seed = _seed(seed)
    working_dtype = _working_dtype(field.dtype)
    if std == 0.0:
        coherence = field.coherence.clone()
    else:
        noise = _normal_noise(field, tuple(field.coherence.shape), seed) * std
        coherence = (field.coherence.to(dtype=working_dtype) + noise).clamp(0.0, 1.0).to(dtype=field.dtype)
    corrupted = _copy_with(field, coherence=coherence)
    return corrupted, _metadata(
        "coherence_noise",
        (("std", std), ("distribution", "gaussian"), ("bounds", "[0,1]")),
        seed=seed,
    )


def mode_swap(
    field: StructuralField,
    first_mode: int = 0,
    second_mode: int = 1,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Swap the posterior masses of two dynamically selected orientation modes."""
    field = _require_field(field)
    first_mode = _integer("first_mode", first_mode)
    second_mode = _integer("second_mode", second_mode)
    mode_count = max(first_mode, second_mode) + 1
    if first_mode < 0 or second_mode < 0 or mode_count > field.metadata.num_orientations:
        raise ValueError("mode indices must lie within the orientation distribution")
    if first_mode == second_mode:
        raise ValueError("mode indices must be distinct")

    modes = field.topk_orientations(k=mode_count, min_separation=0.0)
    first_index = modes.indices[:, first_mode : first_mode + 1]
    second_index = modes.indices[:, second_mode : second_mode + 1]
    first_probability = torch.gather(field.orientation_prob, 1, first_index)
    second_probability = torch.gather(field.orientation_prob, 1, second_index)
    swapped_probability = field.orientation_prob.scatter(1, first_index, second_probability)
    swapped_probability = swapped_probability.scatter(1, second_index, first_probability)
    corrupted = _copy_with(
        field,
        orientation_prob=swapped_probability,
    )
    return corrupted, _metadata(
        "mode_swap",
        (("first_mode", first_mode), ("second_mode", second_mode)),
    )


def field_dropout(
    field: StructuralField,
    probability: float = 1.0,
    *,
    seed: int,
) -> tuple[StructuralField, CorruptionMetadata]:
    """Replace complete per-sample fields with a schema-valid neutral prior."""
    field = _require_field(field)
    probability = _finite_float("probability", probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    seed = _seed(seed)
    working_dtype = _working_dtype(field.dtype)
    random_values = torch.rand(
        (field.batch_size, 1, 1, 1),
        device=field.device,
        dtype=working_dtype,
        generator=_local_generator(field, seed),
    )
    dropped = random_values < probability

    orientation_neutral = torch.full_like(
        field.orientation_prob,
        1.0 / field.metadata.num_orientations,
    )

    def choose(neutral: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return torch.where(dropped, neutral, original)

    updates: dict[str, torch.Tensor | None] = {
        "orientation_prob": choose(orientation_neutral, field.orientation_prob),
        "frequency_mean": choose(torch.zeros_like(field.frequency_mean), field.frequency_mean),
        "frequency_var": choose(torch.zeros_like(field.frequency_var), field.frequency_var),
        "coherence": choose(torch.zeros_like(field.coherence), field.coherence),
        "orientation_entropy": choose(torch.ones_like(field.orientation_entropy), field.orientation_entropy),
        "structural_uncertainty": choose(
            torch.ones_like(field.structural_uncertainty),
            field.structural_uncertainty,
        ),
    }
    if field.scale_prob is not None and field.selected_scale is not None:
        uniform_scale = torch.full_like(field.scale_prob, 1.0 / field.scale_prob.shape[1])
        scale_values = field.scale_prob.new_tensor(
            field.metadata.scale_candidates,
            dtype=working_dtype,
        ).view(1, -1, 1, 1)
        neutral_selected_scale = (
            (uniform_scale.to(dtype=working_dtype) * scale_values)
            .sum(
                dim=1,
                keepdim=True,
            )
            .to(dtype=field.dtype)
        )
        updates["scale_prob"] = choose(uniform_scale, field.scale_prob)
        updates["selected_scale"] = choose(neutral_selected_scale, field.selected_scale)
    corrupted = _copy_with(field, **updates)
    return corrupted, _metadata(
        "field_dropout",
        (
            ("probability", probability),
            ("granularity", "sample"),
            ("replacement", "neutral-valid-field"),
        ),
        seed=seed,
    )


__all__ = [
    "coherence_noise",
    "field_dropout",
    "frequency_noise",
    "frequency_scale",
    "mode_swap",
    "orientation_noise",
    "orientation_shift",
    "scale_shift",
    "spatial_shift",
]
