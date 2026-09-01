"""Numerical contracts for analytic estimator components."""

from __future__ import annotations

import math

import pytest
import torch

from structfield import AnalyticEstimatorConfig
from structfield.core import axial_angle_distance
from structfield.estimators.frequency import frequency_moments
from structfield.estimators.gabor import estimate_gabor_posterior
from structfield.estimators.hessian import estimate_hessian_maps
from structfield.estimators.structure_tensor import estimate_structure_tensor, gaussian_blur
from tests.structfield.test_analytic_estimator import synthetic_grating


def test_frequency_moments_match_direct_statistics() -> None:
    probability = torch.tensor([0.2, 0.5, 0.3]).view(1, 3, 1, 1).expand(2, 3, 4, 5)
    mean, variance = frequency_moments(probability, (0.05, 0.10, 0.15))
    assert torch.allclose(mean, torch.full_like(mean, 0.105))
    assert torch.allclose(variance, torch.full_like(variance, 0.001225), atol=1e-8)
    assert float(variance.max()) <= (0.15 - 0.05) ** 2 / 4.0 + 2e-6


def test_gabor_joint_posterior_is_normalized_and_frequency_sensitive() -> None:
    candidates = (0.06, 0.10, 0.14)
    predicted: list[float] = []
    for frequency in candidates:
        posterior = estimate_gabor_posterior(synthetic_grating(0.0, frequency), candidates, 16)
        assert torch.allclose(
            posterior.joint_prob.sum((1, 2)),
            torch.ones(1, 64, 64),
            atol=2e-5,
            rtol=2e-5,
        )
        mean, _ = frequency_moments(posterior.frequency_prob, candidates)
        predicted.append(float(mean[:, :, 16:-16, 16:-16].mean()))
    assert predicted[0] < predicted[1] < predicted[2]
    assert predicted[1] - predicted[0] > 0.005
    assert predicted[2] - predicted[1] > 0.005


def test_structure_tensor_flat_stability_and_canonical_orientation() -> None:
    flat = torch.ones(1, 1, 32, 32, requires_grad=True)
    maps = estimate_structure_tensor(flat, (1.0, 2.0))
    assert torch.equal(maps.coherence, torch.zeros_like(maps.coherence))
    maps.coherence.sum().backward()
    assert flat.grad is not None and torch.isfinite(flat.grad).all()

    tangent = math.radians(37.0)
    grating = synthetic_grating(tangent)
    maps = estimate_structure_tensor(grating, (1.0, 2.0))
    target = torch.full_like(maps.orientation[:, :, 16:-16, 16:-16], tangent)
    error = axial_angle_distance(maps.orientation[:, :, 16:-16, 16:-16], target)
    assert float(error.mean()) <= math.radians(3.0)


def test_hessian_flat_backward_is_finite() -> None:
    flat = torch.ones(1, 1, 24, 24, requires_grad=True)
    maps = estimate_hessian_maps(flat, (1.0, 2.0))
    (maps.vesselness.mean() + maps.blob_response.mean()).backward()
    assert flat.grad is not None and torch.isfinite(flat.grad).all()


def test_hessian_orientation_on_vertical_bright_ridge() -> None:
    image = torch.zeros(1, 1, 64, 64)
    image[..., 30:34] = 1.0
    maps = estimate_hessian_maps(image, (1.0, 2.0, 3.0))
    strong = maps.vesselness > torch.quantile(maps.vesselness, 0.9)
    target = torch.full_like(maps.orientation[strong], math.pi / 2.0)
    error = axial_angle_distance(maps.orientation[strong], target)
    assert float(error.median()) < math.radians(5.0)


def test_small_image_padding_is_supported() -> None:
    image = torch.rand(1, 1, 2, 3)
    blurred = gaussian_blur(image, 2.0)
    posterior = estimate_gabor_posterior(image, (0.05, 0.10), 8, sigma=2.0)
    assert blurred.shape == image.shape
    assert posterior.orientation_prob.shape == (1, 8, 2, 3)
    assert torch.isfinite(blurred).all() and torch.isfinite(posterior.orientation_prob).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequencies": ()},
        {"frequencies": (0.1, 0.1)},
        {"frequencies": (0.1, 0.5)},
        {"tensor_scales": (2.0, 1.0)},
        {"weak_denoise_type": "median"},
        {"topk": 0},
        {"num_orientations": 4, "topk": 3, "min_separation_rad": 1.0},
        {"include_hessian": 1},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        AnalyticEstimatorConfig(**kwargs)  # type: ignore[arg-type]
