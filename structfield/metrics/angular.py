"""Transparent reference metrics for axial orientation fields.

Angles represent unoriented axes, so every comparison is periodic modulo
``pi`` rather than ``2*pi``.  These parameter-free modules are auditable
Stage-9 references, not a claim of final task-specific metric definitions.
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


def _axial_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return elementwise axial distance, mathematically in ``[0, pi/2]``.

    Each operand is reduced modulo ``pi`` before subtraction.  Besides making
    the convention explicit, this prevents subtraction overflow for very
    large but finite angles. Returned floating values remain subject to
    roundoff at the mathematical endpoints.
    """
    first_wrapped = torch.remainder(first, math.pi)
    second_wrapped = torch.remainder(second, math.pi)
    signed = torch.remainder(first_wrapped - second_wrapped + math.pi / 2.0, math.pi)
    return (signed - math.pi / 2.0).abs()


class AxialAngularError(nn.Module):
    """Masked mean axial error between single-angle maps, in radians."""

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean ``pi``-periodic error over explicitly valid pixels."""
        validate_floating_bchw("prediction", prediction, channels=1)
        validate_floating_bchw("target", target, channels=1)
        validate_aligned("prediction", prediction, "target", target)
        mask = validate_valid_mask(valid_mask, reference=prediction)

        safe_prediction = mask_invalid("prediction", prediction, mask)
        safe_target = mask_invalid("target", target, mask)
        dtype = working_dtype(prediction)
        pixel_error = _axial_distance(
            safe_prediction.to(dtype=dtype),
            safe_target.to(dtype=dtype),
        )
        return masked_mean(pixel_error, mask)


def _angular_threshold(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("angular_threshold_rad must be a real number, not bool")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("angular_threshold_rad must be a real number") from error
    if not math.isfinite(threshold):
        raise ValueError("angular_threshold_rad must be finite")
    if not 0.0 <= threshold <= math.pi / 2.0:
        raise ValueError("angular_threshold_rad must lie in [0, pi/2]")
    return threshold


def _active_target_modes(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    target_count: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_modes = target.shape[1]
    if target_count is None:
        count = torch.full(
            (target.shape[0], 1, target.shape[2], target.shape[3]),
            target_modes,
            device=target.device,
            dtype=torch.int64,
        )
    else:
        if not isinstance(target_count, torch.Tensor):
            raise TypeError("target_count must be a torch.Tensor when provided")
        expected_shape = (target.shape[0], 1, target.shape[2], target.shape[3])
        if target_count.ndim != 4 or tuple(target_count.shape) != expected_shape:
            raise ValueError("target_count must have shape [B,1,H,W]")
        if target_count.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("target_count must use a supported integer dtype: uint8, int8, int16, int32, or int64")
        if target_count.device != target.device:
            raise ValueError("target_count and target must use the same device")
        count = target_count.to(dtype=torch.int64)

    invalid_count = valid_mask & ((count < 1) | (count > target_modes))
    if bool(invalid_count.any()):
        raise ValueError(f"target_count must lie in [1,{target_modes}] at every valid pixel")
    # Invalid pixels do not participate, so arbitrary integer sentinels there
    # are removed before comparison with the mode index.
    safe_count = torch.where(valid_mask, count, 0)
    mode_index = torch.arange(target_modes, device=target.device).view(1, -1, 1, 1)
    active = valid_mask & (mode_index < safe_count)
    return safe_count, active


class OrientationRecallAtK(nn.Module):
    """Recall of active ground-truth modes against K predicted axial modes.

    A ground-truth mode is recalled when at least one predicted mode is within
    the explicitly configured angular threshold.  Pixel recall is the number
    of recalled active ground-truth modes divided by the number of active
    ground-truth modes; the final result is its boolean-masked spatial mean.

    If ``target_count`` is supplied, its integer value at each pixel declares
    that only the first ``target_count`` channels in ``target`` are active.
    """

    def __init__(self, *, angular_threshold_rad: float) -> None:
        super().__init__()
        self.angular_threshold_rad = _angular_threshold(angular_threshold_rad)

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        target_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return mode-wise recall averaged over explicitly valid pixels."""
        validate_floating_bchw("prediction", prediction)
        validate_floating_bchw("target", target)
        validate_aligned(
            "prediction",
            prediction,
            "target",
            target,
            same_channels=False,
        )
        mask = validate_valid_mask(valid_mask, reference=prediction)
        safe_count, active_target = _active_target_modes(target, mask, target_count)

        safe_prediction = mask_invalid("prediction", prediction, mask)
        safe_target = torch.where(active_target, target, 0.0)
        if not bool(torch.isfinite(safe_target).all()):
            raise ValueError("target contains NaN or Inf in an active mode")

        dtype = working_dtype(prediction)
        predicted_modes = safe_prediction.to(dtype=dtype).unsqueeze(2)
        target_modes = safe_target.to(dtype=dtype).unsqueeze(1)
        pairwise_error = _axial_distance(predicted_modes, target_modes)
        target_recalled = (pairwise_error <= self.angular_threshold_rad).any(dim=1)
        recalled_count = (target_recalled & active_target).sum(dim=1, keepdim=True)
        pixel_recall = recalled_count.to(dtype=dtype) / safe_count.clamp_min(1).to(dtype=dtype)
        return masked_mean(pixel_recall, mask)

    def extra_repr(self) -> str:
        return f"angular_threshold_rad={self.angular_threshold_rad}"


__all__ = ["AxialAngularError", "OrientationRecallAtK"]
