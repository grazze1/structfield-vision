"""Integration tests for the Stage 3 analytic structural-field estimator."""

from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch

from structfield import (
    AnalyticEstimatorConfig,
    AnalyticStructuralFieldEstimator,
    StructuralField,
    estimate_structural_field,
)
from structfield.core import StructuralFieldDiagnostics, axial_angle_distance
from structfield.estimators.base import to_luminance
from structfield.uncertainty import HeuristicStructuralUncertainty


def synthetic_grating(
    tangent_rad: float,
    frequency: float = 0.10,
    *,
    size: int = 64,
    amplitude: float = 0.4,
) -> torch.Tensor:
    """Return a line grating using the displayed-image counter-clockwise convention."""
    axis = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    phase = 2.0 * math.pi * frequency * (math.sin(tangent_rad) * xx + math.cos(tangent_rad) * yy)
    return (0.5 + amplitude * torch.cos(phase))[None, None]


def crossing_pattern(size: int = 64) -> torch.Tensor:
    return (
        synthetic_grating(0.0, size=size, amplitude=0.22)
        + synthetic_grating(
            math.pi / 2.0,
            size=size,
            amplitude=0.22,
        )
        - 0.5
    )


def estimator(num_orientations: int = 16, *, include_hessian: bool = False) -> AnalyticStructuralFieldEstimator:
    return AnalyticStructuralFieldEstimator(
        num_orientations=num_orientations,
        frequencies=(0.05, 0.10, 0.15),
        tensor_scales=(1.0, 2.0),
        hessian_scales=(1.0, 2.0),
        weak_denoise_sigma=0.6,
        local_radius=4.0,
        include_hessian=include_hessian,
    )


@pytest.mark.parametrize("num_orientations", [8, 11, 32, 16])
def test_estimator_shapes_joint_marginals_and_arbitrary_orientation_count(num_orientations: int) -> None:
    image = torch.rand(2, 1, 28, 30)
    field, diagnostics = estimator(num_orientations).estimate_with_diagnostics(image)
    assert field.orientation_prob.shape == (2, num_orientations, 28, 30)
    assert field.topk_orientations(k=2).angles.shape == (2, 2, 28, 30)
    assert field.frequency_mean.shape == (2, 1, 28, 30)
    assert field.scale_prob is not None and field.scale_prob.shape == (2, 2, 28, 30)
    assert diagnostics.orientation_frequency_prob is not None
    assert diagnostics.orientation_frequency_prob.shape == (2, 3, num_orientations, 28, 30)
    assert diagnostics.frequency_prob is not None
    joint = diagnostics.orientation_frequency_prob
    assert torch.allclose(joint.sum((1, 2)), torch.ones(2, 28, 30), atol=2e-5, rtol=2e-5)
    assert torch.allclose(joint.sum(1), field.orientation_prob, atol=2e-5, rtol=2e-5)
    assert torch.allclose(joint.sum(2), diagnostics.frequency_prob, atol=2e-5, rtol=2e-5)
    assert torch.allclose(field.scale_prob.sum(1), torch.ones(2, 28, 30), atol=2e-5, rtol=2e-5)
    assert field.selected_scale is not None
    scale_values = field.scale_prob.new_tensor((1.0, 2.0)).view(1, 2, 1, 1)
    expected_scale = (field.scale_prob * scale_values).sum(1, keepdim=True)
    assert torch.allclose(field.selected_scale, expected_scale, atol=2e-5, rtol=2e-5)
    assert field.metadata.num_orientations == num_orientations


def test_float64_dtype_is_preserved() -> None:
    field, diagnostics = estimator(8).estimate_with_diagnostics(torch.rand(1, 1, 24, 26, dtype=torch.float64))
    assert field.dtype == torch.float64
    assert diagnostics.normalized_image is not None and diagnostics.normalized_image.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_dtype_is_preserved(dtype: torch.dtype) -> None:
    image = torch.rand(1, 1, 24, 26, dtype=dtype, requires_grad=True)
    field, diagnostics = estimator(8, include_hessian=True).estimate_with_diagnostics(image)
    assert isinstance(field, StructuralField)
    assert field.dtype == dtype
    assert diagnostics.hessian_vesselness is not None and diagnostics.hessian_vesselness.dtype == dtype
    assert torch.isfinite(field.orientation_prob).all()
    loss = (
        field.frequency_mean.mean()
        + field.coherence.mean()
        + field.structural_uncertainty.mean()
        + diagnostics.hessian_vesselness.mean()
    )
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()

    flat = torch.full((1, 1, 24, 26), 0.5, dtype=dtype, requires_grad=True)
    flat_field, flat_diagnostics = estimator(8, include_hessian=True).estimate_with_diagnostics(flat)
    assert flat_diagnostics.hessian_vesselness is not None
    assert torch.equal(flat_field.orientation_prob, torch.full_like(flat_field.orientation_prob, 1.0 / 8.0))
    assert torch.equal(flat_field.orientation_entropy, torch.ones_like(flat_field.orientation_entropy))
    assert flat_diagnostics.orientation_frequency_response is not None
    assert torch.count_nonzero(flat_diagnostics.orientation_frequency_response) == 0
    flat_loss = (
        flat_field.orientation_prob[:, 0].mean()
        + flat_field.orientation_entropy.mean()
        + flat_field.coherence.mean()
        + flat_field.structural_uncertainty.mean()
        + flat_diagnostics.hessian_vesselness.mean()
    )
    flat_loss.backward()
    assert flat.grad is not None and torch.isfinite(flat.grad).all()


def test_cpu_autocast_does_not_mix_diagnostic_dtypes() -> None:
    image = torch.rand(1, 1, 24, 26, dtype=torch.float32, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        field, diagnostics = estimator(8, include_hessian=True).estimate_with_diagnostics(image)
        assert diagnostics.tensor_orientation is not None
        assert diagnostics.hessian_vesselness is not None
        loss = field.frequency_mean.mean() + field.structural_uncertainty.mean() + diagnostics.hessian_vesselness.mean()
    assert field.dtype == torch.float32
    assert diagnostics.tensor_orientation.dtype == torch.float32
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_flat_field_has_uniform_orientation_zero_coherence_and_finite_statistics() -> None:
    image = torch.full((2, 1, 48, 48), 0.5)
    field, diagnostics = estimator().estimate_with_diagnostics(image)
    assert torch.allclose(field.orientation_prob, torch.full_like(field.orientation_prob, 1.0 / 16.0))
    assert float(field.coherence.max()) <= 1e-6
    assert torch.allclose(field.frequency_mean, torch.full_like(field.frequency_mean, 0.10), atol=2e-6)
    assert torch.allclose(field.orientation_entropy, torch.ones_like(field.orientation_entropy), atol=2e-6)
    assert diagnostics.normalized_image is not None
    assert torch.count_nonzero(diagnostics.normalized_image) == 0
    assert diagnostics.orientation_frequency_response is not None
    assert torch.count_nonzero(diagnostics.orientation_frequency_response) == 0
    assert all(torch.isfinite(value).all() for value in vars(field).values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize("num_orientations", [8, 11, 16, 32])
def test_single_orientation_and_frequency_are_recovered(num_orientations: int) -> None:
    tangent = math.radians(37.0)
    field, diagnostics = estimator(num_orientations).estimate_with_diagnostics(synthetic_grating(tangent))
    interior = (slice(None), slice(None), slice(16, -16), slice(16, -16))
    top1 = field.topk_orientations(k=1).angles
    target = torch.full_like(top1[interior], tangent)
    gabor_error = axial_angle_distance(top1[interior], target)
    assert float(gabor_error.mean()) <= math.pi / (2.0 * num_orientations) + math.radians(3.0)
    assert abs(float(field.frequency_mean[interior].mean()) - 0.10) <= 0.015
    assert float(field.coherence[interior].mean()) >= 0.8
    assert diagnostics.tensor_orientation is not None
    tensor_target = torch.full_like(diagnostics.tensor_orientation[interior], tangent)
    tensor_error = axial_angle_distance(diagnostics.tensor_orientation[interior], tensor_target)
    assert float(tensor_error.mean()) <= math.radians(3.0)


def test_crossing_returns_two_orthogonal_unordered_modes() -> None:
    field = estimator()(crossing_pattern())
    assert isinstance(field, StructuralField)
    interior_modes = field.topk_orientations(k=2).angles[:, :, 16:-16, 16:-16]
    targets = interior_modes.new_tensor((0.0, math.pi / 2.0)).view(1, 2, 1, 1)
    direct = axial_angle_distance(interior_modes, targets).sum(dim=1)
    swapped = axial_angle_distance(interior_modes, targets.flip(1)).sum(dim=1)
    matching_error = torch.minimum(direct, swapped) / 2.0
    assert float(matching_error.mean()) <= math.pi / 16.0 + math.radians(3.0)
    separation = axial_angle_distance(interior_modes[:, 0], interior_modes[:, 1])
    assert float(separation.min()) >= math.radians(60.0)


def test_rgb_conversion_and_batch_independence() -> None:
    samples = [torch.full((1, 1, 48, 48), 0.5), synthetic_grating(math.radians(37.0), size=48), crossing_pattern(48)]
    batch = torch.cat(samples)
    module = estimator()
    batch_field = module(batch)
    assert isinstance(batch_field, StructuralField)
    separate = [module(sample) for sample in samples]
    for batch_index, single in enumerate(separate):
        assert isinstance(single, StructuralField)
        assert torch.allclose(batch_field.orientation_prob[batch_index : batch_index + 1], single.orientation_prob)
        assert torch.allclose(batch_field.frequency_mean[batch_index : batch_index + 1], single.frequency_mean)
        assert torch.allclose(batch_field.coherence[batch_index : batch_index + 1], single.coherence)

    grayscale = samples[1]
    rgb = grayscale.repeat(1, 3, 1, 1)
    gray_field = module(grayscale)
    rgb_field = module(rgb)
    assert isinstance(gray_field, StructuralField) and isinstance(rgb_field, StructuralField)
    assert torch.allclose(gray_field.orientation_prob, rgb_field.orientation_prob, atol=2e-6, rtol=2e-5)
    assert torch.allclose(gray_field.frequency_mean, rgb_field.frequency_mean, atol=2e-6, rtol=2e-5)
    assert rgb_field.metadata.to_dict()["extras"]["input_channels"] == 3


def test_rgb_luminance_weights_are_frozen_independently() -> None:
    colors = torch.eye(3, dtype=torch.float32).view(3, 3, 1, 1)
    converted = to_luminance(colors).flatten()
    assert torch.allclose(converted, torch.tensor((0.2126, 0.7152, 0.0722)), atol=1e-7)

    rgb = torch.rand(1, 3, 32, 34)
    manual = 0.2126 * rgb[:, :1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
    rgb_field = estimator(8)(rgb)
    manual_field = estimator(8)(manual)
    assert isinstance(rgb_field, StructuralField) and isinstance(manual_field, StructuralField)
    assert torch.allclose(rgb_field.orientation_prob, manual_field.orientation_prob, atol=2e-6, rtol=2e-5)


def test_gradient_flow_uses_smooth_outputs() -> None:
    image = torch.rand(2, 1, 28, 30, requires_grad=True)
    field = estimator(11)(image)
    assert isinstance(field, StructuralField) and field.scale_prob is not None
    loss = (
        field.orientation_prob[:, 0].mean()
        + field.frequency_mean.mean()
        + field.coherence.mean()
        + field.scale_prob[:, 0].mean()
        + field.structural_uncertainty.mean()
    )
    loss.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert int(torch.count_nonzero(image.grad)) > 0

    flat = torch.full((1, 1, 24, 24), 0.5, requires_grad=True)
    flat_field = estimator(8)(flat)
    assert isinstance(flat_field, StructuralField)
    (
        flat_field.orientation_prob[:, 0].mean()
        + flat_field.orientation_entropy.mean()
        + flat_field.coherence.mean()
        + flat_field.structural_uncertainty.mean()
    ).backward()
    assert flat.grad is not None and torch.isfinite(flat.grad).all()


def test_optional_hessian_only_changes_diagnostics() -> None:
    image = synthetic_grating(0.0, size=40)
    field_without, diagnostics_without = estimator(include_hessian=False).estimate_with_diagnostics(image)
    field_with, diagnostics_with = estimator(include_hessian=True).estimate_with_diagnostics(image)
    for item in fields(field_without):
        first = getattr(field_without, item.name)
        second = getattr(field_with, item.name)
        if isinstance(first, torch.Tensor):
            assert torch.equal(first, second), item.name
    assert diagnostics_without.hessian_vesselness is None
    assert diagnostics_with.hessian_vesselness is not None
    assert diagnostics_with.hessian_orientation is not None
    assert diagnostics_with.hessian_line_confidence is not None
    assert 0.0 <= float(diagnostics_with.hessian_vesselness.min())
    assert float(diagnostics_with.hessian_vesselness.max()) <= 1.0


def test_functional_api_and_invalid_inputs() -> None:
    result = estimate_structural_field(
        torch.rand(1, 1, 24, 24),
        num_orientations=8,
        frequencies=(0.05, 0.10),
        tensor_scales=(1.0, 2.0),
        return_diagnostics=True,
    )
    assert isinstance(result, tuple)
    assert isinstance(result[0], StructuralField)
    assert isinstance(result[1], StructuralFieldDiagnostics)
    module = estimator()
    with pytest.raises(ValueError, match=r"\[B,1,H,W\]"):
        module(torch.rand(1, 2, 24, 24))
    with pytest.raises(TypeError, match="floating-point"):
        module(torch.ones(1, 1, 24, 24, dtype=torch.int64))
    float8_dtype = getattr(torch, "float8_e4m3fn", None)
    if float8_dtype is not None:
        with pytest.raises(TypeError, match="supported floating-point"):
            module(torch.empty(1, 1, 24, 24, dtype=float8_dtype))
    invalid = torch.rand(1, 1, 24, 24)
    invalid[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        module(invalid)
    with pytest.raises(ValueError, match="at least 2"):
        AnalyticEstimatorConfig(num_orientations=1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_estimator_cuda_preserves_device_dtype_and_gradient(dtype: torch.dtype) -> None:
    image = torch.rand(2, 3, 32, 32, device="cuda", dtype=dtype, requires_grad=True)
    field, diagnostics = estimator(11, include_hessian=True).estimate_with_diagnostics(image)
    assert field.device.type == "cuda"
    assert field.dtype == dtype
    assert diagnostics.normalized_image is not None and diagnostics.normalized_image.device.type == "cuda"
    assert torch.equal(field.structural_uncertainty, HeuristicStructuralUncertainty()(field))
    (
        field.orientation_prob[:, 0].mean()
        + field.orientation_entropy.mean()
        + field.frequency_mean.mean()
        + field.coherence.mean()
        + field.structural_uncertainty.mean()
    ).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_autocast_preserves_contract_and_backward(dtype: torch.dtype, flat: bool) -> None:
    if flat:
        image = torch.full((1, 1, 24, 26), 0.5, device="cuda", dtype=dtype, requires_grad=True)
    else:
        image = torch.rand(1, 1, 24, 26, device="cuda", dtype=dtype, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        field, diagnostics = estimator(8, include_hessian=True).estimate_with_diagnostics(image)
        assert diagnostics.tensor_orientation is not None
        assert diagnostics.hessian_vesselness is not None
        loss = (
            field.orientation_prob[:, 0].mean()
            + field.orientation_entropy.mean()
            + field.frequency_mean.mean()
            + field.coherence.mean()
            + field.structural_uncertainty.mean()
            + diagnostics.hessian_vesselness.mean()
        )
    assert field.dtype == dtype
    assert diagnostics.tensor_orientation.dtype == dtype
    assert torch.equal(field.structural_uncertainty, HeuristicStructuralUncertainty()(field))
    if flat:
        assert torch.equal(field.orientation_prob, torch.full_like(field.orientation_prob, 1.0 / 8.0))
        assert torch.equal(field.orientation_entropy, torch.ones_like(field.orientation_entropy))
        assert diagnostics.orientation_frequency_response is not None
        assert torch.count_nonzero(diagnostics.orientation_frequency_response) == 0
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
