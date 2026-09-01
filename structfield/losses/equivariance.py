"""Masked reference losses for structural equivariance supervision.

These parameter-free objectives are transparent Stage-7 baselines, not a
claim of a final research definition.  They compare two aligned ``BCHW``
branches only where an explicit ``equivariance_valid_mask`` marks the
geometric transform as valid.
"""

from __future__ import annotations

import torch
from torch import nn


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    for name, value in (("prediction", prediction), ("target", target)):
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"{name} must be a floating-point torch.Tensor")
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape [B,C,H,W]")
        if value.numel() == 0:
            raise ValueError(f"{name} dimensions must be non-empty")

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical BCHW shapes")
    if prediction.dtype != target.dtype:
        raise ValueError("prediction and target must use the same dtype")
    if prediction.device != target.device:
        raise ValueError("prediction and target must use the same device")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction contains NaN or Inf")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains NaN or Inf")


def _validate_mask(
    equivariance_valid_mask: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> None:
    if not isinstance(equivariance_valid_mask, torch.Tensor):
        raise TypeError("equivariance_valid_mask must be a torch.Tensor")
    expected_shape = (reference.shape[0], 1, reference.shape[2], reference.shape[3])
    if equivariance_valid_mask.ndim != 4 or tuple(equivariance_valid_mask.shape) != expected_shape:
        raise ValueError("equivariance_valid_mask must have shape [B,1,H,W] aligned with prediction")
    if equivariance_valid_mask.device != reference.device:
        raise ValueError("equivariance_valid_mask and prediction must use the same device")

    if equivariance_valid_mask.dtype is torch.bool:
        has_valid_pixel = bool(equivariance_valid_mask.any())
    elif torch.is_floating_point(equivariance_valid_mask):
        if not bool(torch.isfinite(equivariance_valid_mask).all()):
            raise ValueError("equivariance_valid_mask contains NaN or Inf")
        if bool(((equivariance_valid_mask < 0.0) | (equivariance_valid_mask > 1.0)).any()):
            raise ValueError("floating equivariance_valid_mask must lie in [0,1]")
        has_valid_pixel = bool((equivariance_valid_mask > 0.0).any())
    else:
        raise TypeError("equivariance_valid_mask must have bool or floating-point dtype")

    if not has_valid_pixel:
        raise ValueError("equivariance_valid_mask must contain at least one valid pixel")


def _masked_equivariance_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    equivariance_valid_mask: torch.Tensor,
    *,
    squared: bool,
) -> torch.Tensor:
    _validate_pair(prediction, target)
    _validate_mask(equivariance_valid_mask, reference=prediction)

    working_dtype = torch.float32 if prediction.dtype in {torch.float16, torch.bfloat16} else prediction.dtype
    working_prediction = prediction.to(dtype=working_dtype)
    working_target = target.to(dtype=working_dtype)

    # Remove exactly invalid values before subtraction/squaring.  Multiplying
    # by zero afterwards would not be sufficient: a large finite invalid value
    # can overflow first, and ``inf * 0`` is NaN rather than an exclusion.
    detached_mask = equivariance_valid_mask.detach()
    if torch.is_floating_point(detached_mask):
        # A weighted mean is invariant to a common positive rescaling.  Scale
        # before casting so a valid float64 subnormal mask cannot underflow to
        # an all-zero float32 denominator.
        detached_mask = detached_mask / detached_mask.max()
    mask = detached_mask.to(dtype=working_dtype)
    participates = mask > 0.0
    safe_prediction = torch.where(participates, working_prediction, 0.0)
    safe_target = torch.where(participates, working_target, 0.0)
    difference = safe_prediction - safe_target
    channel_error = difference.square() if squared else difference.abs()
    pixel_error = channel_error.mean(dim=1, keepdim=True)

    # Validity is transform metadata, not a learnable branch.  Fractional masks
    # are interpreted as non-negative pixel weights and intentionally detached.
    return (pixel_error * mask).sum() / mask.sum()


class OrientationEquivarianceLoss(nn.Module):
    """Masked per-pixel MSE between aligned orientation-bin distributions.

    Squared error is averaged across the ``K`` orientation channels first and
    then averaged across valid pixels.  Both prediction and target remain in
    the autograd graph because they commonly represent two estimator branches.
    """

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        equivariance_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar mask-weighted orientation-distribution MSE."""
        return _masked_equivariance_error(
            prediction,
            target,
            equivariance_valid_mask,
            squared=True,
        )


class FrequencyEquivarianceLoss(nn.Module):
    """Masked per-pixel absolute error between aligned frequency fields.

    For a multi-channel input, absolute error is averaged over channels before
    the masked pixel reduction.  Both compared branches retain gradients.
    """

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        equivariance_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar mask-weighted frequency-field L1 error."""
        return _masked_equivariance_error(
            prediction,
            target,
            equivariance_valid_mask,
            squared=False,
        )


class UncertaintyEquivarianceLoss(nn.Module):
    """Masked per-pixel absolute error between aligned uncertainty fields.

    For a multi-channel input, absolute error is averaged over channels before
    the masked pixel reduction.  Both compared branches retain gradients.
    """

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        equivariance_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar mask-weighted uncertainty-field L1 error."""
        return _masked_equivariance_error(
            prediction,
            target,
            equivariance_valid_mask,
            squared=False,
        )


__all__ = [
    "FrequencyEquivarianceLoss",
    "OrientationEquivarianceLoss",
    "UncertaintyEquivarianceLoss",
]
