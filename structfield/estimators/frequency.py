"""Frequency-posterior moments for analytic structural fields."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _probability_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 2e-2
    if dtype == torch.float16:
        return 5e-3
    if dtype == torch.float32:
        return 2e-5
    return 1e-7


def frequency_moments(
    frequency_prob: torch.Tensor,
    candidates: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return posterior mean and variance in cycles/pixel."""
    if not isinstance(frequency_prob, torch.Tensor) or not torch.is_floating_point(frequency_prob):
        raise TypeError("frequency_prob must be a floating-point torch.Tensor")
    if frequency_prob.ndim != 4 or frequency_prob.shape[1] < 1:
        raise ValueError("frequency_prob must have shape [B,F,H,W] with F >= 1")
    if frequency_prob.shape[0] < 1 or frequency_prob.shape[-2] < 1 or frequency_prob.shape[-1] < 1:
        raise ValueError("frequency_prob batch and spatial dimensions must be non-empty")
    if not bool(torch.isfinite(frequency_prob).all()):
        raise ValueError("frequency_prob contains NaN or Inf")
    tolerance = _probability_tolerance(frequency_prob.dtype)
    if float(frequency_prob.detach().min()) < -tolerance:
        raise ValueError("frequency_prob contains negative mass")
    total = frequency_prob.sum(dim=1)
    if not torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=tolerance):
        raise ValueError("frequency_prob must sum to one over frequency channels")
    values = tuple(float(value) for value in candidates)
    if len(values) != frequency_prob.shape[1]:
        raise ValueError("frequency candidate count must match frequency_prob channels")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("frequency candidates must be finite and positive")
    candidates_tensor = frequency_prob.new_tensor(values).view(1, -1, 1, 1)
    mean = (frequency_prob * candidates_tensor).sum(dim=1, keepdim=True)
    variance = (frequency_prob * (candidates_tensor - mean).square()).sum(dim=1, keepdim=True)
    return mean, variance.clamp_min(0.0)
