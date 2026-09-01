"""Independent differentiable multi-frequency Gabor evidence bank."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GaborPosterior:
    """Raw quadrature evidence and normalized structural posteriors."""

    joint_response: torch.Tensor
    joint_prob: torch.Tensor
    orientation_response: torch.Tensor
    orientation_prob: torch.Tensor
    frequency_prob: torch.Tensor


def _validate_candidates(frequencies: Sequence[float], num_orientations: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in frequencies)
    if not values or any(not math.isfinite(value) or value <= 0.0 or value >= 0.5 for value in values):
        raise ValueError("Gabor frequencies must be finite and lie in (0, 0.5)")
    if isinstance(num_orientations, bool) or not isinstance(num_orientations, int) or num_orientations < 2:
        raise ValueError("num_orientations must be an integer >= 2")
    return values


def _gabor_kernels(
    reference: torch.Tensor,
    frequencies: tuple[float, ...],
    num_orientations: int,
    sigma: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinate = torch.arange(-radius, radius + 1, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    real_kernels: list[torch.Tensor] = []
    imaginary_kernels: list[torch.Tensor] = []
    for frequency in frequencies:
        for index in range(num_orientations):
            tangent_ccw = index * math.pi / num_orientations
            # In y-down image coordinates this is the normal to the displayed
            # counter-clockwise tangent (cos(theta), -sin(theta)).
            normal = math.pi / 2.0 - tangent_ccw
            across = xx * math.cos(normal) + yy * math.sin(normal)
            along = -xx * math.sin(normal) + yy * math.cos(normal)
            envelope = torch.exp(-(across.square() + 0.5 * along.square()) / (2.0 * sigma**2))
            phase = 2.0 * math.pi * frequency * across
            real = envelope * torch.cos(phase)
            imaginary = envelope * torch.sin(phase)
            real = real - real.mean()
            imaginary = imaginary - imaginary.mean()
            real = real / torch.sqrt(real.square().sum().clamp_min(eps))
            imaginary = imaginary / torch.sqrt(imaginary.square().sum().clamp_min(eps))
            real_kernels.append(real)
            imaginary_kernels.append(imaginary)
    return torch.stack(real_kernels)[:, None], torch.stack(imaginary_kernels)[:, None]


def estimate_gabor_posterior(
    image: torch.Tensor,
    frequencies: Sequence[float],
    num_orientations: int,
    *,
    sigma: float = 3.0,
    orientation_temperature: float = 0.15,
    eps: float = 1e-6,
) -> GaborPosterior:
    """Estimate a coherent joint posterior with canonical orientation bins."""
    if image.ndim != 4 or image.shape[1] != 1 or not torch.is_floating_point(image):
        raise ValueError("Gabor estimator expects a floating-point [B,1,H,W] tensor")
    frequencies = _validate_candidates(frequencies, num_orientations)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Gabor sigma must be finite and positive")
    if not math.isfinite(orientation_temperature) or orientation_temperature <= 0.0:
        raise ValueError("orientation_temperature must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    real_kernel, imaginary_kernel = _gabor_kernels(image, frequencies, num_orientations, sigma, eps)
    radius = real_kernel.shape[-1] // 2
    mode = "reflect" if image.shape[-2] > radius and image.shape[-1] > radius else "replicate"
    padded = F.pad(image, (radius, radius, radius, radius), mode=mode)
    real_response = F.conv2d(padded, real_kernel)
    imaginary_response = F.conv2d(padded, imaginary_kernel)
    energy = real_response.square() + imaginary_response.square()
    magnitude = (torch.sqrt(energy + eps**2) - eps).clamp_min(0.0)
    batch, _, height, width = magnitude.shape
    joint_response = magnitude.view(batch, len(frequencies), num_orientations, height, width)

    orientation_response = joint_response.amax(dim=1)
    response_scale = orientation_response.mean(dim=1, keepdim=True).clamp_min(eps)
    orientation_logits = orientation_response / response_scale
    orientation_prob = torch.softmax(orientation_logits / orientation_temperature, dim=1)

    conditional_frequency = (joint_response + eps) / (joint_response.sum(dim=1, keepdim=True) + len(frequencies) * eps)
    joint_prob = conditional_frequency * orientation_prob[:, None]
    frequency_prob = joint_prob.sum(dim=2)
    return GaborPosterior(
        joint_response=joint_response,
        joint_prob=joint_prob,
        orientation_response=orientation_response,
        orientation_prob=orientation_prob,
        frequency_prob=frequency_prob,
    )
