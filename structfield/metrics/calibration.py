"""Transparent orientation-confidence calibration diagnostics.

The metric in this module is a stateless empirical diagnostic.  A low ECE on
one evaluation set is not evidence that a model is calibrated in general and
is not presented as a final research definition.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import (
    mask_invalid,
    validate_aligned,
    validate_floating_bchw,
    validate_valid_mask,
    working_dtype,
)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


class OrientationCalibrationECE(nn.Module):
    """Expected calibration error for axial-orientation correctness.

    A prediction is correct when its axial angular error is less than or equal
    to ``angular_threshold_rad``.  Confidence is partitioned into
    ``num_bins`` equal-width bins over ``[0, 1]``: bins are left-closed and
    right-open except that confidence exactly equal to one belongs to the last
    bin.  The result is

    ``sum_b n_b / N * abs(mean_confidence_b - accuracy_b)``.

    Both constructor arguments are intentionally explicit.  In particular,
    the angular correctness threshold is an evaluation choice and is never
    inferred from the inputs.
    """

    angular_threshold_rad: float
    num_bins: int

    def __init__(self, angular_threshold_rad: float, num_bins: int) -> None:
        super().__init__()
        threshold = _finite_real("angular_threshold_rad", angular_threshold_rad)
        if threshold < 0.0 or threshold > math.pi / 2.0:
            raise ValueError("angular_threshold_rad must lie in [0, pi/2]")
        if isinstance(num_bins, bool) or not isinstance(num_bins, int):
            raise TypeError("num_bins must be an integer")
        if int(num_bins) < 1:
            raise ValueError("num_bins must be positive")

        self.angular_threshold_rad = threshold
        self.num_bins = int(num_bins)

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        predicted_angle: torch.Tensor,
        target_angle: torch.Tensor,
        confidence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar equal-width-bin ECE over explicitly valid pixels."""
        validate_floating_bchw("predicted_angle", predicted_angle, channels=1)
        validate_floating_bchw("target_angle", target_angle, channels=1)
        validate_floating_bchw("confidence", confidence, channels=1)
        validate_aligned("predicted_angle", predicted_angle, "target_angle", target_angle)
        validate_aligned("predicted_angle", predicted_angle, "confidence", confidence)
        mask = validate_valid_mask(valid_mask, reference=predicted_angle)

        # Exclude masked sentinels before subtraction, modulo, or binning.
        safe_prediction = mask_invalid("predicted_angle", predicted_angle, mask)
        safe_target = mask_invalid("target_angle", target_angle, mask)
        safe_confidence = mask_invalid("confidence", confidence, mask)
        valid_confidence = safe_confidence[mask]
        if bool(((valid_confidence < 0.0) | (valid_confidence > 1.0)).any()):
            raise ValueError("confidence must lie in [0,1] on valid pixels")

        dtype = working_dtype(predicted_angle)
        prediction = safe_prediction[mask].to(dtype=dtype)
        target = safe_target[mask].to(dtype=dtype)
        confidence_values = valid_confidence.to(dtype=dtype)

        period = prediction.new_tensor(math.pi)
        # Wrap each finite angle before subtracting so even opposite finite
        # dtype extremes cannot overflow during the difference.
        wrapped_prediction = torch.remainder(prediction, period)
        wrapped_target = torch.remainder(target, period)
        wrapped_difference = torch.remainder(wrapped_prediction - wrapped_target, period)
        axial_error = torch.minimum(wrapped_difference, period - wrapped_difference)
        threshold = prediction.new_tensor(self.angular_threshold_rad)
        correct = (axial_error <= threshold).to(dtype=dtype)

        # floor(c * M) implements the documented half-open bins; clamp maps
        # the closed endpoint c=1 to the final bin.
        bin_index = torch.floor(confidence_values * self.num_bins).to(torch.int64)
        bin_index = bin_index.clamp(max=self.num_bins - 1)
        total = confidence_values.new_tensor(confidence_values.numel())
        ece = confidence_values.new_zeros(())
        for index in range(self.num_bins):
            in_bin = bin_index == index
            count = in_bin.sum()
            if bool(count):
                mean_confidence = confidence_values[in_bin].mean()
                accuracy = correct[in_bin].mean()
                ece = ece + count.to(dtype=dtype) / total * (mean_confidence - accuracy).abs()
        return ece

    def extra_repr(self) -> str:
        return f"angular_threshold_rad={self.angular_threshold_rad}, num_bins={self.num_bins}"


__all__ = ["OrientationCalibrationECE"]
