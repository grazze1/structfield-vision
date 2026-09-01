"""Masked reference metrics for aligned structural fields.

All metrics are parameter-free ``nn.Module`` instances and require an explicit
boolean validity mask.  They are transparent Stage-9 formulations rather than
claims of final task-specific evaluation definitions.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import (
    mask_invalid,
    masked_mean,
    validate_aligned,
    validate_floating_bchw,
    validate_valid_mask,
    working_dtype,
)


def _masked_channel_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    validate_floating_bchw("prediction", prediction)
    validate_floating_bchw("target", target)
    validate_aligned("prediction", prediction, "target", target)
    mask = validate_valid_mask(valid_mask, reference=prediction)

    # Exclude raw invalid values before subtraction: an extreme invalid pair
    # could otherwise overflow, and multiplying the result by zero is too late.
    safe_prediction = mask_invalid("prediction", prediction, mask)
    safe_target = mask_invalid("target", target, mask)
    dtype = working_dtype(prediction)
    pixel_error = (safe_prediction.to(dtype=dtype) - safe_target.to(dtype=dtype)).abs().mean(dim=1, keepdim=True)
    return masked_mean(pixel_error, mask)


class FrequencyMAE(nn.Module):
    """Masked channel-mean absolute error for aligned frequency fields."""

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return channel-mean frequency error, then mean over valid pixels."""
        return _masked_channel_mae(prediction, target, valid_mask)


class EquivarianceError(nn.Module):
    """Masked channel-mean absolute error between two aligned field branches."""

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a direct reference error after geometric branch alignment."""
        return _masked_channel_mae(prediction, target, valid_mask)


def _probability_tolerance(dtype: torch.dtype) -> float:
    return {
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
        torch.float32: 1e-5,
        torch.float64: 1e-10,
    }[dtype]


def _validate_distribution(
    name: str,
    probability: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    safe_probability = mask_invalid(name, probability, valid_mask)
    if bool((safe_probability < 0.0).any()):
        raise ValueError(f"{name} must be non-negative at valid pixels")
    channel_sum = safe_probability.sum(dim=1, keepdim=True)
    tolerance = _probability_tolerance(probability.dtype)
    normalized = torch.isclose(
        channel_sum,
        torch.ones_like(channel_sum),
        rtol=tolerance,
        atol=tolerance,
    )
    if not bool(normalized[valid_mask].all()):
        raise ValueError(f"{name} must sum to one over channels at valid pixels")
    return safe_probability


class OrientationDistributionJS(nn.Module):
    """Masked natural-log Jensen-Shannon divergence of orientation PMFs.

    Both inputs are strict channel-wise probability distributions at valid
    pixels.  The per-pixel result is mathematically in ``[0, ln(2)]`` and is
    averaged over the explicit validity mask, subject to floating-point
    reduction roundoff.
    """

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean natural-log Jensen-Shannon divergence."""
        validate_floating_bchw("prediction", prediction)
        validate_floating_bchw("target", target)
        validate_aligned("prediction", prediction, "target", target)
        mask = validate_valid_mask(valid_mask, reference=prediction)
        safe_prediction = _validate_distribution("prediction", prediction, mask)
        safe_target = _validate_distribution("target", target, mask)

        dtype = working_dtype(prediction)
        first = safe_prediction.to(dtype=dtype)
        second = safe_target.to(dtype=dtype)
        midpoint = 0.5 * (first + second)
        tiny = torch.finfo(dtype).tiny

        # Clamp only logarithm arguments.  Multiplication by the original PMF
        # retains the continuous convention 0*log(0/q)=0 and avoids NaN
        # gradients that a raw log(0) followed by torch.where can introduce.
        log_midpoint = midpoint.clamp_min(tiny).log()
        first_kl = first * (first.clamp_min(tiny).log() - log_midpoint)
        second_kl = second * (second.clamp_min(tiny).log() - log_midpoint)
        pixel_js = 0.5 * (first_kl + second_kl).sum(dim=1, keepdim=True)
        # The clamp removes per-pixel round-off excursions beyond the
        # mathematical range.  A subsequent masked reduction can still differ
        # from the real-valued bound by ordinary floating-point roundoff.
        # The nearest float32 representation of ln(2) lies slightly above the
        # real value.  One representable step toward zero keeps the advertised
        # closed upper bound true even after ``Tensor.item()`` conversion.
        upper_bound = torch.nextafter(
            pixel_js.new_tensor(math.log(2.0)),
            pixel_js.new_tensor(0.0),
        )
        pixel_js = torch.minimum(pixel_js.clamp_min(0.0), upper_bound)
        return masked_mean(pixel_js, mask)


__all__ = [
    "EquivarianceError",
    "FrequencyMAE",
    "OrientationDistributionJS",
]
