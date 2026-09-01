"""Exercise RGB conversion and batch-aware estimation on CPU."""

import torch

from structfield import AnalyticStructuralFieldEstimator

generator = torch.Generator().manual_seed(100)
images = torch.rand(2, 3, 32, 40, generator=generator)
field = AnalyticStructuralFieldEstimator(num_orientations=8)(images)
assert field.batch_size == 2 and field.metadata.num_orientations == 8
print(field.spatial_shape)
