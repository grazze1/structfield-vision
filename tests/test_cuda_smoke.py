"""Focused CUDA release smoke path; skipped when the NVIDIA runtime is absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from structfield import load_application_checkpoint, save_application_checkpoint
from structfield.applications import CurvilinearSegmenter, FringeRestorer, TextureRestorer
from structfield.cli.estimate import resolve_current_bundle, run_estimation
from structfield.estimators import AnalyticEstimatorConfig, AnalyticStructuralFieldEstimator
from structfield.transforms import GeometricTransform, transform_structural_field

pytestmark = [
    pytest.mark.cuda_smoke,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="compatible CUDA runtime unavailable"),
]


def test_cuda_estimator_applications_checkpoint_and_cli(tmp_path: Path) -> None:
    config = AnalyticEstimatorConfig(num_orientations=8)
    estimator = AnalyticStructuralFieldEstimator(config).cuda()
    with torch.inference_mode():
        field = estimator(torch.rand(1, 1, 16, 16, device="cuda"))
        transform_structural_field(field, GeometricTransform(rotation_rad=0.1, scale=1.0))

    models_and_calls = [
        (
            CurvilinearSegmenter(num_orientations=8, width=4, adapter_mode="concat").cuda(),
            lambda model, image: model(image, field)["segmentation"],
        ),
        (
            FringeRestorer(num_orientations=8, width=4, noise_dim=2, adapter_mode="concat").cuda(),
            lambda model, image: model(image, field=field, normalized_image=image),
        ),
        (
            TextureRestorer(num_orientations=8, width=4, adapter_mode="concat").cuda(),
            lambda model, image: model(image, field),
        ),
    ]
    for model, call in models_and_calls:
        image = torch.rand(1, 1, 16, 16, device="cuda", requires_grad=True)
        call(model, image).mean().backward()
        assert image.grad is not None

    checkpoint = tmp_path / "checkpoint"
    save_application_checkpoint(
        checkpoint,
        models_and_calls[-1][0],
        estimator=estimator,
        preprocessing={},
        training_provenance={"status": "untrained", "datasets": []},
    )
    loaded = load_application_checkpoint(checkpoint, device="cuda")
    assert next(loaded.model.parameters()).is_cuda

    input_path = tmp_path / "image.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32), allow_pickle=False)
    output = tmp_path / "output"
    run_estimation(input_path, output, device="cuda", num_orientations=8)
    assert resolve_current_bundle(output).is_dir()
