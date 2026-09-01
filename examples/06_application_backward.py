"""Run a synthetic forward/backward pass through an experimental application."""

import torch

from structfield import AnalyticEstimatorConfig, AnalyticStructuralFieldEstimator
from structfield.experimental import applications

image = torch.rand(1, 1, 16, 16, requires_grad=True)
field = AnalyticStructuralFieldEstimator(AnalyticEstimatorConfig(num_orientations=8))(image)
model = applications.CurvilinearSegmenter(
    num_orientations=8,
    width=4,
    adapter_mode="concat",
)
prediction = model(image, field=field)["segmentation"]
prediction.mean().backward()
assert image.grad is not None and torch.isfinite(image.grad).all()
print(prediction.shape)
