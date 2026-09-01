"""Explicit reconstruction and gradient losses for fringe restoration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


@dataclass(frozen=True)
class FringeRestorationLossConfig:
    """Weights for three inspectable image-space loss components."""

    l1_weight: float = 1.0
    axial_gradient_weight: float = 0.02
    gradient_magnitude_weight: float = 0.01
    eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "l1_weight",
            "axial_gradient_weight",
            "gradient_magnitude_weight",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(name, getattr(self, name)),
            )
        eps = _finite_nonnegative("eps", self.eps)
        if eps == 0.0:
            raise ValueError("eps must be positive")
        object.__setattr__(self, "eps", eps)
        if self.l1_weight + self.axial_gradient_weight + self.gradient_magnitude_weight == 0.0:
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True)
class FringeRestorationLossOutput:
    """Scalar total plus detached, explicitly named component values."""

    total: torch.Tensor
    l1: torch.Tensor
    axial_gradient: torch.Tensor
    gradient_magnitude: torch.Tensor


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    for name, value in (("prediction", prediction), ("target", target)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise TypeError(f"{name} must use a supported floating-point dtype")
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,H,W]")
        if value.shape[0] < 1 or value.shape[-2] < 2 or value.shape[-1] < 2:
            raise ValueError(f"{name} dimensions must be non-empty and at least 2x2")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.device != target.device:
        raise ValueError("prediction and target must use the same device")
    if prediction.dtype != target.dtype:
        raise ValueError("prediction and target must use the same dtype")


def _spatial_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independently implemented reflect-padded Sobel derivatives."""
    horizontal = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    vertical = horizontal.transpose(0, 1)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return (
        F.conv2d(padded, horizontal.view(1, 1, 3, 3)),
        F.conv2d(padded, vertical.view(1, 1, 3, 3)),
    )


class FringeRestorationLoss(nn.Module):
    """L1 plus axial gradient-direction and gradient-magnitude consistency.

    The third component is deliberately named ``gradient_magnitude``: it is an
    image-space Sobel-gradient loss, not a Fourier-domain or structural-field
    frequency loss.  The normalized absolute 2-D cross product measures sine
    misalignment, so reversing a gradient does not incur a direction penalty.
    """

    def __init__(self, config: FringeRestorationLossConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = FringeRestorationLossConfig()
        if not isinstance(config, FringeRestorationLossConfig):
            raise TypeError("config must be FringeRestorationLossConfig")
        self.config = config

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> FringeRestorationLossOutput:
        """Return the differentiable total and detached scalar components."""
        _validate_pair(prediction, target)
        work_dtype = torch.float32 if prediction.dtype in {torch.float16, torch.bfloat16} else prediction.dtype
        prediction_work = prediction.to(dtype=work_dtype)
        target_work = target.to(dtype=work_dtype)
        prediction_x, prediction_y = _spatial_gradients(prediction_work)
        target_x, target_y = _spatial_gradients(target_work)

        prediction_magnitude = torch.sqrt(prediction_x.square() + prediction_y.square() + self.config.eps)
        target_magnitude = torch.sqrt(target_x.square() + target_y.square() + self.config.eps)
        axial_cross = (prediction_x * target_y - prediction_y * target_x).abs()

        l1 = (prediction_work - target_work).abs().mean()
        axial_gradient = (axial_cross / (prediction_magnitude * target_magnitude)).clamp(0.0, 1.0).mean()
        gradient_magnitude = (prediction_magnitude - target_magnitude).abs().mean()
        total = (
            self.config.l1_weight * l1
            + self.config.axial_gradient_weight * axial_gradient
            + self.config.gradient_magnitude_weight * gradient_magnitude
        )
        return FringeRestorationLossOutput(
            total=total,
            l1=l1.detach(),
            axial_gradient=axial_gradient.detach(),
            gradient_magnitude=gradient_magnitude.detach(),
        )


__all__ = [
    "FringeRestorationLoss",
    "FringeRestorationLossConfig",
    "FringeRestorationLossOutput",
]
