"""Independent multi-scale structure-tensor implementation.

The implementation follows the standard second-moment-matrix construction.
Angles are converted to the project convention: line tangents increase
counter-clockwise in displayed image coordinates, where image ``y`` points
downward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from structfield.core import wrap_axial_angle


@dataclass(frozen=True)
class StructureTensorMaps:
    """Selected tensor maps and the differentiable scale posterior."""

    orientation: torch.Tensor
    coherence: torch.Tensor
    eigenvalue_max: torch.Tensor
    eigenvalue_min: torch.Tensor
    scale_response: torch.Tensor
    scale_prob: torch.Tensor
    selected_scale: torch.Tensor


def _validate_positive_values(name: str, values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _pad_separable(image: torch.Tensor, radius: int, *, horizontal: bool) -> torch.Tensor:
    dimension = image.shape[-1] if horizontal else image.shape[-2]
    mode = "reflect" if dimension > radius else "replicate"
    padding = (radius, radius, 0, 0) if horizontal else (0, 0, radius, radius)
    return F.pad(image, padding, mode=mode)


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a dtype/device-preserving separable Gaussian blur to BCHW data."""
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Gaussian sigma must be finite and positive")
    if image.ndim != 4 or not torch.is_floating_point(image):
        raise ValueError("gaussian_blur expects a floating-point BCHW tensor")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinate = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    output = F.conv2d(_pad_separable(image, radius, horizontal=True), horizontal, groups=channels)
    return F.conv2d(_pad_separable(output, radius, horizontal=False), vertical, groups=channels)


def spatial_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Sobel ``x`` and downward-image-``y`` derivatives."""
    if image.ndim != 4 or not torch.is_floating_point(image):
        raise ValueError("spatial_gradients expects a floating-point BCHW tensor")
    kernel_x = image.new_tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))) / 8.0
    kernel_y = kernel_x.transpose(0, 1)
    channels = image.shape[1]
    kernel_x = kernel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    mode = "reflect" if image.shape[-2] > 1 and image.shape[-1] > 1 else "replicate"
    padded = F.pad(image, (1, 1, 1, 1), mode=mode)
    return (
        F.conv2d(padded, kernel_x, groups=channels),
        F.conv2d(padded, kernel_y, groups=channels),
    )


def estimate_structure_tensor(
    image: torch.Tensor,
    scales: Sequence[float],
    *,
    scale_temperature: float = 0.1,
    eps: float = 1e-6,
) -> StructureTensorMaps:
    """Estimate canonical tangent orientation and stable coherence over scales."""
    if image.ndim != 4 or image.shape[1] != 1 or not torch.is_floating_point(image):
        raise ValueError("structure tensor expects a floating-point [B,1,H,W] tensor")
    scales = _validate_positive_values("tensor scales", scales)
    if not math.isfinite(scale_temperature) or scale_temperature <= 0.0:
        raise ValueError("scale_temperature must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    gradient_x, gradient_y = spatial_gradients(image)
    orientations: list[torch.Tensor] = []
    coherences: list[torch.Tensor] = []
    eigenvalue_maxima: list[torch.Tensor] = []
    eigenvalue_minima: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []

    for sigma in scales:
        tensor_xx = gaussian_blur(gradient_x.square(), sigma)
        tensor_yy = gaussian_blur(gradient_y.square(), sigma)
        tensor_xy = gaussian_blur(gradient_x * gradient_y, sigma)
        discriminant_squared = ((tensor_xx - tensor_yy).square() + 4.0 * tensor_xy.square()).clamp_min(0.0)
        trace = (tensor_xx + tensor_yy).clamp_min(0.0)
        discriminant = (torch.sqrt(discriminant_squared + eps**2) - eps).clamp_min(0.0)
        discriminant = torch.minimum(discriminant, trace)
        eigenvalue_max = (0.5 * (trace + discriminant)).clamp_min(0.0)
        eigenvalue_min = (0.5 * (trace - discriminant)).clamp_min(0.0)
        anisotropic = discriminant_squared > eps**2
        angle_y = torch.where(anisotropic, 2.0 * tensor_xy, torch.zeros_like(tensor_xy))
        angle_x = torch.where(anisotropic, tensor_xx - tensor_yy, torch.ones_like(tensor_xx))
        normal_clockwise = 0.5 * torch.atan2(angle_y, angle_x)
        tangent_counter_clockwise = wrap_axial_angle(-normal_clockwise + math.pi / 2.0)
        tangent_counter_clockwise = torch.where(
            anisotropic,
            tangent_counter_clockwise,
            torch.zeros_like(tangent_counter_clockwise),
        )
        coherence = (discriminant / (trace + eps)).clamp(0.0, 1.0)
        response = coherence * (torch.sqrt(eigenvalue_max + eps**2) - eps).clamp_min(0.0)
        orientations.append(tangent_counter_clockwise)
        coherences.append(coherence)
        eigenvalue_maxima.append(eigenvalue_max)
        eigenvalue_minima.append(eigenvalue_min)
        responses.append(response)

    response_stack = torch.cat(responses, dim=1)
    scale_prob = torch.softmax(response_stack / scale_temperature, dim=1)
    best_index = response_stack.argmax(dim=1, keepdim=True)[:, :, None]

    def select(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.gather(torch.stack(values, dim=1), 1, best_index).squeeze(1)

    scale_values = image.new_tensor(scales).view(1, -1, 1, 1)
    selected_scale = (scale_prob * scale_values).sum(dim=1, keepdim=True)
    return StructureTensorMaps(
        orientation=select(orientations),
        coherence=select(coherences).clamp(0.0, 1.0),
        eigenvalue_max=select(eigenvalue_maxima).clamp_min(0.0),
        eigenvalue_min=select(eigenvalue_minima).clamp_min(0.0),
        scale_response=response_stack,
        scale_prob=scale_prob,
        selected_scale=selected_scale,
    )
