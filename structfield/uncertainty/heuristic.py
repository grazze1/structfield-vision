"""Parameter-free heuristic structural uncertainty.

This module intentionally implements a transparent baseline heuristic, not a
theoretically optimal or learned uncertainty model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from structfield.core import StructuralField
from structfield.validation import ValidationMode, is_strict_validation

DEFAULT_WEIGHTS = (0.25, 0.20, 0.20, 0.25, 0.10)


def _validated_weights(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    weights = tuple(values)
    if len(weights) != len(DEFAULT_WEIGHTS):
        raise ValueError("weights must contain exactly five values")
    if any(isinstance(value, bool) for value in weights):
        raise TypeError("uncertainty weights must be real numbers, not bool")
    try:
        converted = tuple(float(value) for value in weights)
    except (TypeError, ValueError) as error:
        raise TypeError("uncertainty weights must be real numbers") from error
    if any(not math.isfinite(value) or value < 0.0 for value in converted):
        raise ValueError("uncertainty weights must be finite and non-negative")
    if sum(converted[:4]) <= 0.0:
        raise ValueError("at least one non-scale uncertainty weight must be positive")
    return converted[0], converted[1], converted[2], converted[3], converted[4]


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def normalized_categorical_entropy(
    probability: torch.Tensor,
    *,
    dim: int = 1,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> torch.Tensor:
    """Return entropy normalized by ``log(num_categories)``.

    Float16 and bfloat16 inputs produce a float32 working result. A single
    category is defined to have zero entropy.
    """
    if not isinstance(probability, torch.Tensor) or not torch.is_floating_point(probability):
        raise TypeError("probability must be a floating-point torch.Tensor")
    if probability.ndim == 0:
        raise ValueError("probability must have at least one dimension")
    if probability.numel() == 0:
        raise ValueError("probability dimensions must be non-empty")
    strict = is_strict_validation(validation_mode)
    if strict and not bool(torch.isfinite(probability).all()):
        raise ValueError("probability contains NaN or Inf")
    normalized_dim = dim if dim >= 0 else probability.ndim + dim
    if not 0 <= normalized_dim < probability.ndim:
        raise ValueError("entropy dimension is out of range")
    num_categories = probability.shape[normalized_dim]
    if num_categories < 1:
        raise ValueError("entropy dimension must be non-empty")

    working_dtype = torch.float32 if probability.dtype in {torch.float16, torch.bfloat16} else probability.dtype
    tolerance = {
        torch.bfloat16: 2e-2,
        torch.float16: 5e-3,
        torch.float32: 2e-5,
        torch.float64: 1e-7,
    }.get(probability.dtype, 1e-7)
    if strict and float(probability.detach().min()) < -tolerance:
        raise ValueError("probability contains negative mass")
    working = probability.to(dtype=working_dtype).clamp_min(0.0)
    tiny = torch.finfo(working_dtype).tiny
    total = working.sum(dim=normalized_dim, keepdim=True)
    if strict and bool((total <= 0.0).any()):
        raise ValueError("probability must contain positive mass along the entropy dimension")
    working = working / total.clamp_min(tiny)
    if num_categories == 1:
        return torch.zeros_like(total)
    entropy = -(working * working.clamp_min(tiny).log()).sum(dim=normalized_dim, keepdim=True)
    return (entropy / math.log(num_categories)).clamp(0.0, 1.0)


class HeuristicStructuralUncertainty(nn.Module):
    """Combine five bounded field-reliability signals without learned parameters.

    The signals are normalized orientation entropy, relative margin uncertainty
    between two angularly separated orientation modes, normalized frequency
    variance, one minus coherence, and normalized scale entropy. When
    ``scale_prob`` is absent, the scale term is omitted and the remaining
    weights are renormalized automatically. If the configured angular
    separation admits no runner-up mode at a pixel, its margin uncertainty is
    zero; orientation entropy still represents distributional diffuseness.
    """

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        *,
        orientation_entropy_weight: float | None = None,
        orientation_margin_weight: float | None = None,
        frequency_variance_weight: float | None = None,
        coherence_weight: float | None = None,
        scale_entropy_weight: float | None = None,
    ) -> None:
        super().__init__()
        configured = list(DEFAULT_WEIGHTS if weights is None else _validated_weights(weights))
        overrides = (
            orientation_entropy_weight,
            orientation_margin_weight,
            frequency_variance_weight,
            coherence_weight,
            scale_entropy_weight,
        )
        for index, override in enumerate(overrides):
            if override is not None:
                configured[index] = override
        self.weights = _validated_weights(configured)

    @property
    def orientation_entropy_weight(self) -> float:
        return self.weights[0]

    @property
    def orientation_margin_weight(self) -> float:
        return self.weights[1]

    @property
    def frequency_variance_weight(self) -> float:
        return self.weights[2]

    @property
    def coherence_weight(self) -> float:
        return self.weights[3]

    @property
    def scale_entropy_weight(self) -> float:
        return self.weights[4]

    @staticmethod
    def _normalized_frequency_variance(field: StructuralField, dtype: torch.dtype) -> torch.Tensor:
        candidates = field.metadata.frequency_candidates
        if not candidates:
            raise ValueError("frequency_candidates are required when frequency variance is weighted")
        if len(candidates) == 1:
            return torch.zeros_like(field.frequency_var, dtype=dtype)
        frequency_span = max(candidates) - min(candidates)
        if frequency_span <= 0.0:
            raise ValueError("frequency_candidates must span a positive range")
        # Popoviciu's bound for a distribution supported on [f_min, f_max].
        maximum_variance = frequency_span**2 / 4.0
        return (field.frequency_var.to(dtype=dtype) / maximum_variance).clamp(0.0, 1.0)

    @staticmethod
    def _separated_mode_probabilities(
        field: StructuralField,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top mode and separated runner-up, using zero when none exists."""
        probabilities = field.orientation_prob.to(dtype=dtype).clamp_min(0.0)
        tiny = torch.finfo(dtype).tiny
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(tiny)
        first_probability, first_index = probabilities.max(dim=1, keepdim=True)

        centers = probabilities.new_tensor(field.metadata.orientation_bin_centers).view(1, -1, 1, 1)
        first_angle = torch.gather(centers.expand_as(probabilities), 1, first_index)
        delta = (centers - first_angle).abs()
        axial_distance = torch.minimum(delta, math.pi - delta)
        tolerance = max(8.0 * torch.finfo(dtype).eps, 1e-7)
        eligible = axial_distance + tolerance >= field.metadata.default_min_separation_rad
        eligible = eligible.scatter(1, first_index, False)
        second_probability = probabilities.masked_fill(~eligible, 0.0).amax(dim=1, keepdim=True)
        return first_probability, second_probability

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(self, field: StructuralField) -> torch.Tensor:
        """Return a deterministic uncertainty map with shape ``[B,1,H,W]``."""
        if not isinstance(field, StructuralField):
            raise TypeError("field must be a StructuralField")

        output_dtype = field.dtype
        working_dtype = torch.float32 if output_dtype in {torch.float16, torch.bfloat16} else output_dtype
        if self.orientation_margin_weight > 0.0:
            first_mode, second_mode = self._separated_mode_probabilities(field, working_dtype)
            denominator = (first_mode + second_mode).clamp_min(torch.finfo(working_dtype).tiny)
            margin_uncertainty = (1.0 - (first_mode - second_mode) / denominator).clamp(0.0, 1.0)
        else:
            margin_uncertainty = torch.zeros_like(field.orientation_entropy, dtype=working_dtype)

        components = [
            field.orientation_entropy.to(dtype=working_dtype).clamp(0.0, 1.0),
            margin_uncertainty,
            self._normalized_frequency_variance(field, working_dtype)
            if self.frequency_variance_weight > 0.0
            else torch.zeros_like(field.frequency_var, dtype=working_dtype),
            (1.0 - field.coherence.to(dtype=working_dtype).clamp(0.0, 1.0)),
        ]
        active_weights = list(self.weights[:4])
        if field.scale_prob is not None:
            components.append(
                normalized_categorical_entropy(
                    field.scale_prob,
                    dim=1,
                    validation_mode=field.validation_mode,
                ).to(dtype=working_dtype)
            )
            active_weights.append(self.scale_entropy_weight)

        weight_sum = sum(active_weights)
        uncertainty = torch.zeros_like(components[0])
        for component, weight in zip(components, active_weights):
            uncertainty = uncertainty + component * (weight / weight_sum)
        uncertainty = uncertainty.clamp(0.0, 1.0)
        return uncertainty.to(dtype=output_dtype)


__all__ = ["DEFAULT_WEIGHTS", "HeuristicStructuralUncertainty", "normalized_categorical_entropy"]
