"""Transparent reference objective for V0.2 texture restoration.

The objective preserves the four independently inspectable terms used by the
historical project, while naming the FFT term accurately as Fourier-amplitude
consistency.  It is a reference formulation rather than a final research loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(name: str, value: object) -> float:
    result = _finite_nonnegative(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class TextureRestorationLossConfig:
    """Explicit weights for the four-term reference restoration objective."""

    l1_weight: float = 1.0
    spatial_gradient_weight: float = 0.1
    axial_direction_weight: float = 0.02
    fourier_amplitude_weight: float = 0.01
    eps: float = 1e-6

    def __post_init__(self) -> None:
        weights = {
            "l1_weight": _finite_nonnegative("l1_weight", self.l1_weight),
            "spatial_gradient_weight": _finite_nonnegative(
                "spatial_gradient_weight",
                self.spatial_gradient_weight,
            ),
            "axial_direction_weight": _finite_nonnegative(
                "axial_direction_weight",
                self.axial_direction_weight,
            ),
            "fourier_amplitude_weight": _finite_nonnegative(
                "fourier_amplitude_weight",
                self.fourier_amplitude_weight,
            ),
        }
        if not any(value > 0.0 for value in weights.values()):
            raise ValueError("at least one texture-restoration loss weight must be positive")
        for name, value in weights.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "eps", _finite_positive("eps", self.eps))


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
            raise ValueError(f"{name} must have non-empty batch and spatial dimensions at least 2x2")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.dtype != target.dtype:
        raise ValueError("prediction and target must use the same dtype")
    if prediction.device != target.device:
        raise ValueError("prediction and target must use the same device")


def _spatial_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    vertical = horizontal.transpose(0, 1)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return (
        F.conv2d(padded, horizontal.view(1, 1, 3, 3)),
        F.conv2d(padded, vertical.view(1, 1, 3, 3)),
    )


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
def texture_restoration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    config: TextureRestorationLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the scalar loss and detached, explicitly named components."""
    _validate_pair(prediction, target)
    resolved = TextureRestorationLossConfig() if config is None else config
    if not isinstance(resolved, TextureRestorationLossConfig):
        raise TypeError("config must be a TextureRestorationLossConfig")

    work_dtype = torch.float32 if prediction.dtype in {torch.float16, torch.bfloat16} else prediction.dtype
    prediction_work = prediction.to(dtype=work_dtype)
    target_work = target.to(dtype=work_dtype)

    l1 = (prediction_work - target_work).abs().mean()
    prediction_x, prediction_y = _spatial_gradients(prediction_work)
    target_x, target_y = _spatial_gradients(target_work)
    spatial_gradient = (prediction_x - target_x).abs().mean() + (prediction_y - target_y).abs().mean()

    axial_cross = (prediction_x * target_y - prediction_y * target_x).abs()
    prediction_norm = torch.sqrt(prediction_x.square() + prediction_y.square() + resolved.eps)
    target_norm = torch.sqrt(target_x.square() + target_y.square() + resolved.eps)
    axial_direction = (axial_cross / (prediction_norm * target_norm)).clamp(0.0, 1.0).mean()

    prediction_amplitude = torch.fft.rfft2(prediction_work.contiguous(), norm="ortho").abs()
    target_amplitude = torch.fft.rfft2(target_work.contiguous(), norm="ortho").abs()
    fourier_amplitude = (prediction_amplitude - target_amplitude).abs().mean()

    total = (
        resolved.l1_weight * l1
        + resolved.spatial_gradient_weight * spatial_gradient
        + resolved.axial_direction_weight * axial_direction
        + resolved.fourier_amplitude_weight * fourier_amplitude
    )
    components = {
        "l1": l1.detach(),
        "spatial_gradient": spatial_gradient.detach(),
        "axial_gradient_direction": axial_direction.detach(),
        "fourier_amplitude": fourier_amplitude.detach(),
    }
    return total, components


class TextureRestorationLoss(nn.Module):
    """Parameter-free module wrapper around :func:`texture_restoration_loss`."""

    def __init__(self, config: TextureRestorationLossConfig | None = None) -> None:
        super().__init__()
        self.config = TextureRestorationLossConfig() if config is None else config
        if not isinstance(self.config, TextureRestorationLossConfig):
            raise TypeError("config must be a TextureRestorationLossConfig")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return scalar loss and detached component diagnostics."""
        return texture_restoration_loss(prediction, target, config=self.config)


__all__ = [
    "TextureRestorationLoss",
    "TextureRestorationLossConfig",
    "texture_restoration_loss",
]
