"""Analytic contracts for same-canvas structural-field geometry."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from structfield.core import FieldMetadata, StructuralField, axial_angle_distance
from structfield.transforms import (
    GeometricTransform,
    rotate_structural_field,
    rotate_tensor,
    scale_structural_field,
    transform_structural_field,
    transform_tensor,
)


def make_constant_field(
    *,
    num_orientations: int = 18,
    angle_rad: float = math.radians(30.0),
    batch: int = 1,
    height: int = 13,
    width: int = 15,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    with_scale: bool = True,
) -> StructuralField:
    step = math.pi / num_orientations
    source_bin = round(angle_rad / step) % num_orientations
    probability = torch.zeros(
        batch,
        num_orientations,
        height,
        width,
        dtype=dtype,
        device=device,
    )
    probability[:, source_bin] = 1.0
    scalar = torch.ones(batch, 1, height, width, dtype=dtype, device=device)
    metadata = FieldMetadata(
        num_orientations=num_orientations,
        frequency_candidates=(0.05, 0.10, 0.15),
        scale_candidates=(1.0, 2.0) if with_scale else (),
        estimator="stage-7-fixture",
    )
    scale_prob = None
    selected_scale = None
    if with_scale:
        scale_prob = torch.cat((scalar * 0.25, scalar * 0.75), dim=1)
        selected_scale = scalar * 1.75
    return StructuralField(
        orientation_prob=probability,
        frequency_mean=scalar * 0.10,
        frequency_var=scalar * 0.002,
        coherence=scalar * 0.70,
        orientation_entropy=torch.zeros_like(scalar),
        structural_uncertainty=scalar * 0.20,
        metadata=metadata,
        scale_prob=scale_prob,
        selected_scale=selected_scale,
    )


def test_positive_rotation_is_displayed_counter_clockwise_on_rectangular_canvas() -> None:
    image = torch.zeros(1, 1, 5, 9)
    image[0, 0, 2, 6] = 1.0  # two pixels to the right of the canvas center
    result = rotate_tensor(image, math.pi / 2.0)
    assert float(result.tensor[0, 0, 0, 4]) == pytest.approx(1.0, abs=1e-6)
    assert torch.count_nonzero(result.tensor > 1e-6) == 1
    assert result.equivariance_valid_mask[0, 0, 0, 4]


def test_combined_rotation_and_scale_share_the_pixel_center() -> None:
    image = torch.zeros(1, 1, 9, 13)
    image[0, 0, 4, 7] = 1.0
    result = transform_tensor(image, GeometricTransform(rotation_rad=math.pi / 2.0, scale=2.0))
    assert float(result.tensor[0, 0, 2, 6]) == pytest.approx(1.0, abs=1e-6)


def test_combined_field_transform_updates_orientation_and_frequency_together() -> None:
    field = make_constant_field(angle_rad=math.radians(30.0))
    result = transform_structural_field(
        field,
        GeometricTransform(rotation_rad=math.radians(20.0), scale=2.0),
    )
    center = (field.spatial_shape[0] // 2, field.spatial_shape[1] // 2)
    assert result.equivariance_valid_mask[0, 0, *center]
    assert float(result.field.topk_orientations(k=1).angles[0, 0, *center]) == pytest.approx(
        math.radians(50.0),
        abs=2e-6,
    )
    assert float(result.field.frequency_mean[0, 0, *center]) == pytest.approx(
        0.05,
        abs=1e-7,
    )


def test_thirty_degree_orientation_plus_twenty_degree_rotation_is_fifty() -> None:
    field = make_constant_field()
    result = rotate_structural_field(field, math.radians(20.0))
    center = (field.spatial_shape[0] // 2, field.spatial_shape[1] // 2)
    angle = result.field.topk_orientations(k=1).angles[0, 0, center[0], center[1]]
    assert float(angle) == pytest.approx(math.radians(50.0), abs=2e-6)
    assert int(result.field.orientation_prob[0, :, center[0], center[1]].argmax()) == 5
    assert result.equivariance_valid_mask[0, 0, center[0], center[1]]


def test_exact_bin_rotation_is_an_axial_mass_roll_with_wraparound() -> None:
    num_bins = 8
    step = math.pi / num_bins
    field = make_constant_field(num_orientations=num_bins, angle_rad=7.0 * step)
    result = rotate_structural_field(field, step)
    center_probability = result.field.orientation_prob[0, :, 6, 7]
    assert torch.equal(center_probability, torch.nn.functional.one_hot(torch.tensor(0), num_bins).float())


def test_arbitrary_rotation_linearly_transports_axial_bin_mass() -> None:
    num_bins = 8
    half_bin = 0.5 * math.pi / num_bins
    field = make_constant_field(num_orientations=num_bins, angle_rad=0.0)
    result = rotate_structural_field(field, half_bin)
    center_probability = result.field.orientation_prob[0, :, 6, 7]
    expected = torch.zeros(num_bins)
    expected[0] = 0.5
    expected[1] = 0.5
    assert torch.allclose(center_probability, expected, atol=2e-6, rtol=0.0)
    expected_entropy = math.log(2.0) / math.log(num_bins)
    assert float(result.field.orientation_entropy[0, 0, 6, 7]) == pytest.approx(
        expected_entropy,
        abs=2e-6,
    )


def test_probability_interpolation_preserves_axial_wrap() -> None:
    field = make_constant_field(num_orientations=18, angle_rad=math.radians(179.0), width=17)
    probability = torch.zeros_like(field.orientation_prob)
    probability[:, 17, :, :8] = 1.0
    probability[:, 0, :, 8:] = 1.0
    field = replace(field, orientation_prob=probability)
    # At output x=7, a 2x magnification samples source x=7.5 exactly: the
    # axial mean of 179 and 1 degrees is 0 degrees, not the linear mean 90.
    result = scale_structural_field(field, 2.0)
    interpolated = result.field.topk_orientations(k=1).angles[0, 0, 6, 7]
    distance = axial_angle_distance(interpolated, interpolated.new_zeros(()))
    assert float(distance) == pytest.approx(0.0, abs=2e-4)


def test_scale_updates_frequency_moments_and_scale_semantics() -> None:
    field = make_constant_field()
    result = scale_structural_field(field, 2.0)
    transformed = result.field
    center = (6, 7)
    assert float(transformed.frequency_mean[0, 0, *center]) == pytest.approx(0.05, abs=1e-7)
    assert float(transformed.frequency_var[0, 0, *center]) == pytest.approx(0.0005, abs=1e-7)
    assert transformed.selected_scale is not None
    assert float(transformed.selected_scale[0, 0, *center]) == pytest.approx(3.5, abs=1e-6)
    assert transformed.metadata.frequency_candidates == pytest.approx((0.025, 0.05, 0.075))
    assert transformed.metadata.scale_candidates == pytest.approx((2.0, 4.0))


def test_combined_field_transform_applies_rotation_and_scale_semantics_together() -> None:
    field = make_constant_field()
    transformed = transform_structural_field(
        field,
        GeometricTransform(rotation_rad=math.radians(20.0), scale=2.0),
    ).field
    center = (6, 7)
    assert float(transformed.topk_orientations(k=1).angles[0, 0, *center]) == pytest.approx(
        math.radians(50.0),
        abs=2e-6,
    )
    assert float(transformed.frequency_mean[0, 0, *center]) == pytest.approx(0.05, abs=1e-7)
    assert float(transformed.frequency_var[0, 0, *center]) == pytest.approx(0.0005, abs=1e-7)


def test_scale_posterior_is_warped_and_remains_normalized() -> None:
    field = make_constant_field(batch=2)
    transformed = scale_structural_field(field, 0.75).field
    assert transformed.scale_prob is not None
    assert torch.allclose(
        transformed.scale_prob.sum(dim=1),
        torch.ones_like(transformed.scale_prob[:, 0]),
        atol=2e-5,
        rtol=2e-5,
    )
    assert transformed.scale_prob.dtype == field.dtype
    assert transformed.scale_prob.device == field.device


def test_field_without_scale_posterior_stays_without_scale_posterior() -> None:
    field = make_constant_field(with_scale=False)
    transformed = transform_structural_field(
        field,
        GeometricTransform(rotation_rad=0.31, scale=1.2),
    ).field
    assert transformed.scale_prob is None
    assert transformed.selected_scale is None
    assert transformed.metadata.scale_candidates == ()


def test_valid_mask_excludes_padding_and_propagates_source_validity() -> None:
    image = torch.ones(1, 1, 9, 13)
    rotated = rotate_tensor(image, math.radians(30.0))
    mask = rotated.equivariance_valid_mask
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 1, 9, 13)
    assert mask[0, 0, 4, 6]
    assert not mask[0, 0, 0, 0]
    assert not mask.all()

    source_mask = torch.ones_like(mask)
    source_mask[0, 0, 4, 6] = False
    identity = transform_tensor(image, GeometricTransform(), source_valid_mask=source_mask)
    assert torch.equal(identity.equivariance_valid_mask, source_mask)
    rotated_hole = rotate_tensor(image, math.radians(30.0), source_valid_mask=source_mask)
    assert not rotated_hole.equivariance_valid_mask[0, 0, 4, 6]

    # A complete bilinear footprint is required: under 2x scaling, output
    # columns 1 and 3 each mix the invalid source center with a valid neighbor.
    small_image = torch.ones(1, 1, 5, 5)
    small_mask = torch.ones_like(small_image, dtype=torch.bool)
    small_mask[0, 0, 2, 2] = False
    scaled_hole = transform_tensor(
        small_image,
        GeometricTransform(scale=2.0),
        source_valid_mask=small_mask,
    ).equivariance_valid_mask
    assert torch.equal(
        scaled_hole[0, 0, 2],
        torch.tensor([True, False, False, False, True]),
    )


def test_tensor_and_field_transform_return_the_same_geometry_mask() -> None:
    field = make_constant_field(batch=2, height=10, width=17)
    transform = GeometricTransform(rotation_rad=-0.37, scale=0.9)
    tensor_result = transform_tensor(field.frequency_mean, transform)
    field_result = transform_structural_field(field, transform)
    assert torch.equal(
        tensor_result.equivariance_valid_mask,
        field_result.equivariance_valid_mask,
    )


def test_transform_does_not_mutate_source_field() -> None:
    field = make_constant_field()
    before = {name: value.clone() for name, value in vars(field).items() if isinstance(value, torch.Tensor)}
    result = transform_structural_field(field, GeometricTransform(rotation_rad=0.23, scale=1.1))
    for name, expected in before.items():
        assert torch.equal(getattr(field, name), expected)
    assert result.field is not field
    assert result.field.orientation_prob.data_ptr() != field.orientation_prob.data_ptr()


def test_tensor_transform_preserves_finite_gradients() -> None:
    image = torch.rand(2, 3, 11, 17, requires_grad=True)
    result = transform_tensor(image, GeometricTransform(rotation_rad=0.27, scale=0.91))
    loss = (result.tensor * result.equivariance_valid_mask).square().mean()
    loss.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_field_transform_preserves_finite_gradients() -> None:
    base = make_constant_field(num_orientations=8, angle_rad=0.4)
    logits = torch.randn_like(base.orientation_prob, requires_grad=True)
    frequency = base.frequency_mean.detach().clone().requires_grad_()
    field = replace(
        base,
        orientation_prob=torch.softmax(logits, dim=1),
        frequency_mean=frequency,
    )
    result = transform_structural_field(field, GeometricTransform(rotation_rad=0.19, scale=1.07))
    mask = result.equivariance_valid_mask
    loss = (
        (result.field.orientation_prob.square() * mask).mean()
        + (result.field.topk_orientations(k=2).probabilities * mask).mean()
        + (result.field.frequency_mean * mask).mean()
    )
    loss.backward()
    for gradient in (logits.grad, frequency.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_cpu_transform_is_safe_and_preserves_dtype(dtype: torch.dtype) -> None:
    field = make_constant_field(num_orientations=8, dtype=dtype)
    result = transform_structural_field(field, GeometricTransform(rotation_rad=0.21, scale=0.83))
    transformed = result.field
    for value in vars(transformed).values():
        if isinstance(value, torch.Tensor):
            assert value.dtype == dtype
            assert torch.isfinite(value).all()
    assert result.equivariance_valid_mask.dtype == torch.bool


def test_cpu_autocast_does_not_change_float32_contract() -> None:
    field = make_constant_field()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        transformed = rotate_structural_field(field, 0.17).field
    assert transformed.dtype == torch.float32
    assert torch.isfinite(transformed.orientation_prob).all()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"rotation_rad": True}, TypeError, "not bool"),
        ({"rotation_rad": float("nan")}, ValueError, "finite"),
        ({"scale": False}, TypeError, "not bool"),
        ({"scale": 0.0}, ValueError, "positive"),
        ({"scale": float("inf")}, ValueError, "finite and positive"),
    ],
)
def test_transform_configuration_fails_fast(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        GeometricTransform(**kwargs)  # type: ignore[arg-type]


def test_tensor_and_mask_input_contracts_fail_fast() -> None:
    image = torch.ones(1, 1, 5, 7)
    with pytest.raises(TypeError, match="floating-point"):
        transform_tensor(image.to(torch.int64), GeometricTransform())
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        transform_tensor(image[0], GeometricTransform())
    with pytest.raises(ValueError, match="NaN or Inf"):
        transform_tensor(image + float("nan"), GeometricTransform())
    with pytest.raises(TypeError, match="GeometricTransform"):
        transform_tensor(image, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.bool"):
        transform_tensor(image, GeometricTransform(), source_valid_mask=torch.ones_like(image))
    with pytest.raises(ValueError, match=r"\[B, 1, H, W\]"):
        transform_tensor(
            image.expand(1, 2, 5, 7),
            GeometricTransform(),
            source_valid_mask=torch.ones(1, 2, 5, 7, dtype=torch.bool),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_transform_dtype_device_schema_and_gradient(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 unavailable")
    field = make_constant_field(num_orientations=8, dtype=dtype, device="cuda")
    transformed = transform_structural_field(
        field,
        GeometricTransform(rotation_rad=0.28, scale=0.88),
    )
    assert transformed.field.device.type == "cuda"
    assert transformed.field.dtype == dtype
    assert transformed.equivariance_valid_mask.device.type == "cuda"
    assert transformed.equivariance_valid_mask.dtype == torch.bool
    assert torch.isfinite(transformed.field.orientation_prob).all()

    image = torch.rand(1, 2, 9, 13, device="cuda", dtype=dtype, requires_grad=True)
    result = rotate_tensor(image, 0.22)
    (result.tensor.float() * result.equivariance_valid_mask).mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
