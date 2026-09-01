"""Estimate a deterministic field from a synthetic grayscale grating."""

import math

import torch

from structfield import AnalyticStructuralFieldEstimator

axis = torch.arange(48, dtype=torch.float32)
yy, xx = torch.meshgrid(axis, axis, indexing="ij")
image = (0.5 + 0.4 * torch.cos(2 * math.pi * 0.1 * yy))[None, None]
field = AnalyticStructuralFieldEstimator()(image)
print(field.orientation_prob.shape, field.frequency_mean.mean().item())
