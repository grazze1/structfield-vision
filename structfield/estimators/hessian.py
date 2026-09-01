"""Independent multi-scale Hessian line-evidence diagnostics.

This is a paper-inspired analytic baseline built from the Hessian eigensystem;
it does not vendor or reproduce a third-party implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from structfield.core import wrap_axial_angle

from .structure_tensor import gaussian_blur, spatial_gradients


@dataclass(frozen=True)
class HessianMaps:
    """Best-response Hessian maps and per-scale line evidence."""

    vesselness: torch.Tensor
    orientation: torch.Tensor
    line_confidence: torch.Tensor
    blob_response: torch.Tensor
    selected_scale: torch.Tensor
    scale_response: torch.Tensor


def estimate_hessian_maps(
    image: torch.Tensor,
    scales: Sequence[float],
    *,
    beta: float = 0.5,
    sensitivity: float = 0.25,
    eps: float = 1e-6,
) -> HessianMaps:
    """Estimate bright-line evidence and canonical tangent orientation."""
    if image.ndim != 4 or image.shape[1] != 1 or not torch.is_floating_point(image):
        raise ValueError("Hessian estimator expects a floating-point [B,1,H,W] tensor")
    scales = tuple(float(value) for value in scales)
    if not scales or any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("Hessian scales must contain finite positive values")
    if not math.isfinite(beta) or beta <= 0.0 or not math.isfinite(sensitivity) or sensitivity <= 0.0:
        raise ValueError("Hessian beta and sensitivity must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    vesselness_values: list[torch.Tensor] = []
    orientations: list[torch.Tensor] = []
    line_confidences: list[torch.Tensor] = []
    blob_responses: list[torch.Tensor] = []
    for sigma in scales:
        smoothed = gaussian_blur(image, sigma)
        gradient_x, gradient_y = spatial_gradients(smoothed)
        derivative_xx, derivative_xy_a = spatial_gradients(gradient_x)
        derivative_xy_b, derivative_yy = spatial_gradients(gradient_y)
        derivative_xy = 0.5 * (derivative_xy_a + derivative_xy_b)
        scale_factor = sigma**2
        derivative_xx = derivative_xx * scale_factor
        derivative_xy = derivative_xy * scale_factor
        derivative_yy = derivative_yy * scale_factor

        discriminant_squared = ((derivative_xx - derivative_yy).square() + 4.0 * derivative_xy.square()).clamp_min(0.0)
        discriminant = (torch.sqrt(discriminant_squared + eps**2) - eps).clamp_min(0.0)
        eigenvalue_low = 0.5 * (derivative_xx + derivative_yy - discriminant)
        eigenvalue_high = 0.5 * (derivative_xx + derivative_yy + discriminant)
        swap = eigenvalue_low.abs() > eigenvalue_high.abs()
        eigenvalue_small = torch.where(swap, eigenvalue_high, eigenvalue_low)
        eigenvalue_large = torch.where(swap, eigenvalue_low, eigenvalue_high)
        anisotropy_ratio = eigenvalue_small.abs() / (eigenvalue_large.abs() + eps)
        structure_energy = (torch.sqrt(eigenvalue_small.square() + eigenvalue_large.square() + eps**2) - eps).clamp_min(
            0.0
        )
        vesselness = torch.exp(-anisotropy_ratio.square() / (2.0 * beta**2))
        vesselness = vesselness * (1.0 - torch.exp(-structure_energy.square() / (2.0 * sensitivity**2)))
        vesselness = vesselness * (eigenvalue_large < 0.0).to(vesselness.dtype)

        anisotropic = discriminant_squared > eps**2
        angle_y = torch.where(anisotropic, 2.0 * derivative_xy, torch.zeros_like(derivative_xy))
        angle_x = torch.where(
            anisotropic,
            derivative_xx - derivative_yy,
            torch.ones_like(derivative_xx),
        )
        tangent_clockwise = 0.5 * torch.atan2(angle_y, angle_x)
        tangent_counter_clockwise = wrap_axial_angle(-tangent_clockwise)
        tangent_counter_clockwise = torch.where(
            anisotropic,
            tangent_counter_clockwise,
            torch.zeros_like(tangent_counter_clockwise),
        )
        line_confidence = (
            (eigenvalue_large.abs() - eigenvalue_small.abs()).clamp_min(0.0) / (eigenvalue_large.abs() + eps)
        ).clamp(0.0, 1.0)
        blob_response = (anisotropy_ratio.clamp(0.0, 1.0) * structure_energy).clamp_min(0.0)
        vesselness_values.append(vesselness.clamp(0.0, 1.0))
        orientations.append(tangent_counter_clockwise)
        line_confidences.append(line_confidence)
        blob_responses.append(blob_response)

    vesselness_stack = torch.stack(vesselness_values, dim=1)
    best_index = vesselness_stack.argmax(dim=1, keepdim=True)

    def select(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.gather(torch.stack(values, dim=1), 1, best_index).squeeze(1)

    scale_values = image.new_tensor(scales).view(1, -1, 1, 1, 1).expand_as(vesselness_stack)
    selected_scale = torch.gather(scale_values, 1, best_index).squeeze(1)
    return HessianMaps(
        vesselness=select(vesselness_values),
        orientation=select(orientations),
        line_confidence=select(line_confidences),
        blob_response=select(blob_responses),
        selected_scale=selected_scale,
        scale_response=vesselness_stack.squeeze(2),
    )
