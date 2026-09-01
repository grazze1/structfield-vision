"""Tiny contracts for the active V0.2 curvilinear segmentation application."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from structfield import FieldMetadata, StructuralField
from structfield.applications._shared import curvilinear_structural_features
from structfield.applications.curvilinear_segmentation import (
    CurvilinearSegmenter,
    curvilinear_segmentation_loss,
)


def _uniform_field(
    image: torch.Tensor,
    *,
    num_orientations: int = 8,
    include_scale: bool = True,
) -> StructuralField:
    batch, _, height, width = image.shape
    probability = torch.full(
        (batch, num_orientations, height, width),
        1.0 / num_orientations,
        device=image.device,
        dtype=image.dtype,
    )
    metadata = FieldMetadata(
        num_orientations=num_orientations,
        frequency_candidates=(0.05, 0.10, 0.15),
        scale_candidates=(1.0, 2.0) if include_scale else (),
        estimator="curvilinear-test",
    )
    scalar_shape = (batch, 1, height, width)
    scale_prob = None
    selected_scale = None
    if include_scale:
        scale_prob = image.new_tensor((0.25, 0.75)).view(1, 2, 1, 1)
        scale_prob = scale_prob.expand(batch, 2, height, width)
        selected_scale = torch.full(
            scalar_shape,
            1.75,
            device=image.device,
            dtype=image.dtype,
        )
    return StructuralField(
        orientation_prob=probability,
        frequency_mean=torch.full(scalar_shape, 0.10, device=image.device, dtype=image.dtype),
        frequency_var=torch.full(scalar_shape, 0.0004, device=image.device, dtype=image.dtype),
        coherence=torch.full(scalar_shape, 0.6, device=image.device, dtype=image.dtype),
        orientation_entropy=torch.ones(scalar_shape, device=image.device, dtype=image.dtype),
        structural_uncertainty=torch.full(scalar_shape, 0.25, device=image.device, dtype=image.dtype),
        metadata=metadata,
        scale_prob=scale_prob,
        selected_scale=selected_scale,
    )


def test_default_is_the_plain_one_channel_historical_size_baseline() -> None:
    model = CurvilinearSegmenter()
    first_convolution = model.backbone.encoder1.body[0]
    assert model.num_orientations == 16
    assert model.width == 24
    assert model.adapter_mode == "none"
    assert isinstance(first_convolution, torch.nn.Conv2d)
    assert first_convolution.in_channels == 1
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_085_305


@pytest.mark.parametrize(
    "adapter_mode",
    ["none", "concat", "film", "reliability_gated"],
)
def test_all_modes_forward_backward_on_odd_spatial_shape(adapter_mode: str) -> None:
    torch.manual_seed(10)
    image = torch.rand(2, 1, 31, 33, requires_grad=True)
    field = None if adapter_mode == "none" else _uniform_field(image.detach())
    model = CurvilinearSegmenter(
        num_orientations=8,
        width=4,
        adapter_mode=adapter_mode,
    )

    outputs = model(image, field)
    logits = outputs["segmentation"]
    target = (torch.rand_like(logits) > 0.7).to(dtype=logits.dtype)
    loss, components = curvilinear_segmentation_loss(outputs, target)
    loss.backward()

    assert logits.shape == image.shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert set(components) == {"dice_loss", "binary_cross_entropy"}
    assert image.grad is not None and torch.isfinite(image.grad).all()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients)

    if adapter_mode == "reliability_gated":
        gate = outputs["structural_gate"]
        assert gate.shape == image.shape
        assert bool(((0.0 <= gate) & (gate <= 1.0)).all())
        assert outputs["structural_delta"].shape == image.shape
    elif adapter_mode == "film":
        assert "structural_gate" not in outputs
        assert outputs["structural_delta"].shape == image.shape
    else:
        assert "structural_gate" not in outputs
        assert "structural_delta" not in outputs


def test_fixed_eight_features_have_zero_uniform_axial_moment() -> None:
    image = torch.rand(2, 1, 17, 19)
    field = _uniform_field(image, include_scale=False)
    features = curvilinear_structural_features(field)
    assert features.shape == (2, 8, 17, 19)
    assert torch.allclose(features[:, :2], torch.zeros_like(features[:, :2]), atol=2e-7)
    assert torch.equal(features[:, 7:8], torch.zeros_like(features[:, 7:8]))
    assert torch.isfinite(features).all()


def test_fixed_features_do_not_duplicate_explicit_structural_uncertainty() -> None:
    image = torch.rand(1, 1, 17, 19)
    field = _uniform_field(image)
    changed = replace(
        field,
        structural_uncertainty=torch.full_like(field.structural_uncertainty, 0.95),
    )
    torch.testing.assert_close(
        curvilinear_structural_features(field),
        curvilinear_structural_features(changed),
        atol=0.0,
        rtol=0.0,
    )


def test_conditioned_modes_require_matching_field() -> None:
    image = torch.rand(1, 1, 17, 19)
    model = CurvilinearSegmenter(
        num_orientations=8,
        width=4,
        adapter_mode="concat",
    )
    with pytest.raises(ValueError, match="requires a StructuralField"):
        model(image)

    wrong_bins = _uniform_field(image, num_orientations=16)
    with pytest.raises(ValueError, match="orientation bins"):
        model(image, wrong_bins)

    other_shape = _uniform_field(torch.rand(1, 1, 18, 19))
    with pytest.raises(ValueError, match="batch and spatial"):
        model(image, other_shape)


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_empty_and_full_masks_have_finite_loss_and_gradient(fill: float) -> None:
    logits = torch.randn(2, 1, 13, 15, requires_grad=True)
    target = torch.full_like(logits, fill)
    loss, components = curvilinear_segmentation_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "adapter_mode",
    ["none", "concat", "film", "reliability_gated"],
)
def test_cuda_autocast_forward_backward(adapter_mode: str) -> None:
    image = torch.rand(1, 1, 17, 19, device="cuda", requires_grad=True)
    field = None if adapter_mode == "none" else _uniform_field(image.detach())
    model = CurvilinearSegmenter(
        num_orientations=8,
        width=4,
        adapter_mode=adapter_mode,
    ).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(image, field)
        target = torch.zeros_like(outputs["segmentation"])
        loss, components = curvilinear_segmentation_loss(outputs, target)
    loss.backward()
    assert outputs["segmentation"].dtype == torch.float16
    assert loss.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in components.values())
    assert torch.isfinite(outputs["segmentation"]).all()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_invalid_constructor_values_fail_fast() -> None:
    with pytest.raises(ValueError, match="positive"):
        CurvilinearSegmenter(width=0)
    with pytest.raises(ValueError, match="adapter mode"):
        CurvilinearSegmenter(adapter_mode="unknown")
    with pytest.raises(TypeError, match="integer"):
        CurvilinearSegmenter(num_orientations=True)
    with pytest.raises(ValueError, match="at least 2"):
        CurvilinearSegmenter(num_orientations=1)


def test_loss_rejects_shape_mismatch_and_invalid_weights() -> None:
    logits = torch.zeros(1, 1, 11, 13)
    with pytest.raises(ValueError, match="same shape"):
        curvilinear_segmentation_loss(logits, torch.zeros(1, 1, 11, 12))
    with pytest.raises(ValueError, match="non-negative"):
        curvilinear_segmentation_loss(logits, torch.zeros_like(logits), bce_weight=-0.1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_low_precision_loss_uses_float32_and_preserves_gradient(
    dtype: torch.dtype,
) -> None:
    logits = torch.randn(2, 1, 13, 15, dtype=dtype, requires_grad=True)
    target = torch.zeros_like(logits)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, components = curvilinear_segmentation_loss(logits, target)
    loss.backward()
    assert loss.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in components.values())
    assert logits.grad is not None
    assert logits.grad.dtype == dtype
    assert torch.isfinite(logits.grad).all()


def test_float64_loss_preserves_float64_computation_and_gradient() -> None:
    logits = torch.randn(2, 1, 13, 15, dtype=torch.float64, requires_grad=True)
    target = torch.ones_like(logits)
    loss, components = curvilinear_segmentation_loss(logits, target)
    loss.backward()
    assert loss.dtype == torch.float64
    assert all(value.dtype == torch.float64 for value in components.values())
    assert logits.grad is not None
    assert logits.grad.dtype == torch.float64
    assert torch.isfinite(logits.grad).all()
