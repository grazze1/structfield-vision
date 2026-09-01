"""Coordinate and axial-angle conventions used throughout StructField-Vision.

The formulas are project-authored from standard axial-angle mathematics. An
orientation describes an unoriented line, so ``theta`` and ``theta + pi`` are
equivalent and doubled-angle vectors avoid the discontinuity at that boundary.
"""

from __future__ import annotations

import math

import torch

AXIAL_PERIOD_RADIANS = math.pi
ANGLE_UNIT = "radians"
FREQUENCY_UNIT = "cycles/pixel"
SCALE_UNIT = "pixels"


def uniform_orientation_bins(num_orientations: int) -> tuple[float, ...]:
    """Return uniformly spaced axial bin centers in ``[0, pi)``."""
    if isinstance(num_orientations, bool) or not isinstance(num_orientations, int):
        raise TypeError("num_orientations must be an integer")
    if num_orientations < 2:
        raise ValueError("num_orientations must be at least 2")
    return tuple(index * math.pi / num_orientations for index in range(num_orientations))


def wrap_axial_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[0, pi)`` without changing dtype or device."""
    if not torch.is_floating_point(theta):
        raise TypeError("axial angles must use a floating-point tensor")
    period = theta.new_tensor(math.pi)
    wrapped = torch.remainder(theta, period)
    # float32's nearest representation of pi lies slightly above mathematical
    # pi; clamp to the preceding representable value to honor the strict schema.
    upper = torch.nextafter(period, torch.zeros_like(period))
    return torch.minimum(wrapped, upper)


def axial_angle_distance(theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
    """Return the smallest absolute distance between axial angles in radians."""
    if not torch.is_floating_point(theta_a) or not torch.is_floating_point(theta_b):
        raise TypeError("axial angles must use floating-point tensors")
    period = theta_a.new_tensor(math.pi)
    delta = torch.remainder((theta_a - theta_b).abs(), period)
    return torch.minimum(delta, period - delta)


def doubled_angle(theta: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    """Encode an axial angle tensor as concatenated ``cos(2θ), sin(2θ)``."""
    if not torch.is_floating_point(theta):
        raise TypeError("axial angles must use a floating-point tensor")
    return torch.cat((torch.cos(2.0 * theta), torch.sin(2.0 * theta)), dim=dim)
