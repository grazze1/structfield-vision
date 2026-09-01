"""Transparent reference supervision for a predicted reliability map.

This is deliberately a configurable baseline formulation.  It is not a claim
that structural error is calibrated probability, nor that this objective is a
final research definition of reliability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _finite_real(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class ReliabilityLossConfig:
    """All constants used by :class:`ReliabilityLoss`.

    ``threshold`` only partitions the continuous target into two supervision
    regions for weighting; it does not discretize the target itself.
    """

    orientation_error_weight: float = 1.0
    frequency_error_weight: float = 1.0
    reliable_sample_weight: float = 1.0
    unreliable_sample_weight: float = 2.0
    tau: float = 0.25
    threshold: float = 0.5
    eps: float = 1e-8

    def __post_init__(self) -> None:
        names = (
            "orientation_error_weight",
            "frequency_error_weight",
            "reliable_sample_weight",
            "unreliable_sample_weight",
            "tau",
            "threshold",
            "eps",
        )
        for name in names:
            object.__setattr__(self, name, _finite_real(name, getattr(self, name)))

        if self.orientation_error_weight < 0.0 or self.frequency_error_weight < 0.0:
            raise ValueError("error weights must be non-negative")
        if self.orientation_error_weight + self.frequency_error_weight <= 0.0:
            raise ValueError("at least one error weight must be positive")
        if self.reliable_sample_weight <= 0.0 or self.unreliable_sample_weight <= 0.0:
            raise ValueError("sample weights must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0,1]")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class ReliabilityTargets:
    """Inspectable target maps derived from two non-negative error maps."""

    combined_error: torch.Tensor
    soft_reliability: torch.Tensor
    is_reliable: torch.Tensor

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, torch.Tensor) for value in (self.combined_error, self.soft_reliability, self.is_reliable)
        ):
            raise TypeError("reliability target values must be torch.Tensor instances")
        if self.combined_error.shape != self.soft_reliability.shape:
            raise ValueError("combined_error and soft_reliability must have identical shapes")
        if self.is_reliable.shape != self.soft_reliability.shape:
            raise ValueError("is_reliable and soft_reliability must have identical shapes")
        if self.is_reliable.dtype is not torch.bool:
            raise TypeError("is_reliable must have bool dtype")
        if self.combined_error.device != self.soft_reliability.device:
            raise ValueError("target maps must use the same device")
        if self.is_reliable.device != self.soft_reliability.device:
            raise ValueError("target maps must use the same device")


def _validate_error_map(
    name: str,
    value: torch.Tensor,
    *,
    maximum: float | None = None,
) -> None:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W]")
    if value.numel() == 0:
        raise ValueError(f"{name} dimensions must be non-empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    if bool((value < 0.0).any()):
        raise ValueError(f"{name} must be non-negative")
    if maximum is not None:
        tolerance = {
            torch.float16: 2e-3,
            torch.bfloat16: 1e-2,
            torch.float32: 1e-6,
            torch.float64: 1e-12,
        }.get(value.dtype, 1e-12)
        if bool((value > maximum + tolerance).any()):
            raise ValueError(f"{name} must not exceed {maximum:.8f}")


def _validate_prediction(prediction: torch.Tensor) -> None:
    if not isinstance(prediction, torch.Tensor) or not torch.is_floating_point(prediction):
        raise TypeError("prediction must be a floating-point torch.Tensor")
    if prediction.ndim != 4:
        raise ValueError("prediction must have shape [B,C,H,W]")
    if prediction.numel() == 0 or prediction.shape[1] < 1:
        raise ValueError("prediction dimensions must be non-empty")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction contains NaN or Inf")
    if bool(((prediction < 0.0) | (prediction > 1.0)).any()):
        raise ValueError("prediction must lie in [0,1]")


class ReliabilityLoss(nn.Module):
    """Weighted MSE against an exponential soft-reliability reference target.

    Given an axial orientation error in radians and a dimensionless relative
    frequency error, first define

    ``e = (w_orientation * (e_orientation / (pi/2)) + w_frequency * e_frequency) / (w_orientation + w_frequency)``

    and then ``r = exp(-e / tau)``.  Normalizing orientation error to its
    axial maximum and normalizing by the active weights makes the reference
    target invariant to a common rescaling of both error weights.

    Pixels are classified by ``r >= threshold`` solely to select the configured
    reliable/unreliable sample weight.  Error-derived targets are detached: the
    loss supervises the reliability predictor, not the process producing error
    labels.  A predicted reliability map is an auxiliary model output and is
    not automatically the Stage-5 adapter's usage gate.
    """

    def __init__(self, config: ReliabilityLossConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ReliabilityLossConfig()
        if not isinstance(config, ReliabilityLossConfig):
            raise TypeError("config must be a ReliabilityLossConfig")
        self.config = config

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def targets(
        self,
        orientation_error_rad: torch.Tensor,
        frequency_error_relative: torch.Tensor,
    ) -> ReliabilityTargets:
        """Construct detached, inspectable continuous and thresholded targets."""
        _validate_error_map(
            "orientation_error_rad",
            orientation_error_rad,
            maximum=math.pi / 2.0,
        )
        _validate_error_map("frequency_error_relative", frequency_error_relative)
        if orientation_error_rad.shape != frequency_error_relative.shape:
            raise ValueError("orientation_error_rad and frequency_error_relative must have identical shapes")
        if orientation_error_rad.device != frequency_error_relative.device:
            raise ValueError("orientation_error_rad and frequency_error_relative must use the same device")
        if orientation_error_rad.dtype != frequency_error_relative.dtype:
            raise ValueError("orientation_error_rad and frequency_error_relative must use the same dtype")

        working_dtype = (
            torch.float32
            if orientation_error_rad.dtype in {torch.float16, torch.bfloat16}
            else orientation_error_rad.dtype
        )
        orientation = orientation_error_rad.detach().to(dtype=working_dtype).clamp_max(math.pi / 2.0) / (math.pi / 2.0)
        frequency = frequency_error_relative.detach().to(dtype=working_dtype)
        # Scale before normalization so even very large finite configured
        # weights do not overflow merely while computing their sum.
        weight_scale = max(
            self.config.orientation_error_weight,
            self.config.frequency_error_weight,
        )
        orientation_weight = self.config.orientation_error_weight / weight_scale
        frequency_weight = self.config.frequency_error_weight / weight_scale
        weight_sum = orientation_weight + frequency_weight
        combined = (orientation_weight * orientation + frequency_weight * frequency) / weight_sum
        soft = torch.exp(-combined / self.config.tau)
        return ReliabilityTargets(
            combined_error=combined,
            soft_reliability=soft,
            is_reliable=soft >= self.config.threshold,
        )

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        orientation_error_rad: torch.Tensor,
        frequency_error_relative: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar region-weighted mean-squared error."""
        _validate_prediction(prediction)
        targets = self.targets(orientation_error_rad, frequency_error_relative)
        if prediction.shape[0] != orientation_error_rad.shape[0]:
            raise ValueError("prediction and error maps must have the same batch size")
        if prediction.shape[2:] != orientation_error_rad.shape[2:]:
            raise ValueError("prediction and error maps must have the same spatial shape")
        if prediction.device != orientation_error_rad.device:
            raise ValueError("prediction and error maps must use the same device")
        if prediction.dtype != orientation_error_rad.dtype:
            raise ValueError("prediction and error maps must use the same dtype")

        working_prediction = prediction.to(dtype=targets.soft_reliability.dtype)
        soft_target = targets.soft_reliability.expand_as(working_prediction)
        reliable = targets.is_reliable.expand_as(working_prediction)
        sample_weight = torch.where(
            reliable,
            working_prediction.new_tensor(self.config.reliable_sample_weight),
            working_prediction.new_tensor(self.config.unreliable_sample_weight),
        )
        squared_error = (working_prediction - soft_target).square()
        normalizer = sample_weight.sum().clamp_min(self.config.eps)
        return (sample_weight * squared_error).sum() / normalizer


__all__ = ["ReliabilityLoss", "ReliabilityLossConfig", "ReliabilityTargets"]
