"""Derive separated orientation modes from the sole probability state."""

import math

import torch

from structfield import AnalyticStructuralFieldEstimator

image = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(7))
field = AnalyticStructuralFieldEstimator()(image)
modes = field.topk_orientations(k=2, min_separation=math.pi / 8)
print(modes.angles.shape, modes.probabilities.shape, modes.indices.dtype)
