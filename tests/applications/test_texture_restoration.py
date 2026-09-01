from __future__ import annotations

import inspect
import math

import pytest
import torch

from structfield import FieldMetadata, StructuralField
from structfield.applications._shared import texture_structural_features
from structfield.applications.texture_restoration import (
    TextureRestorationLoss,
    TextureRestorationLossConfig,
    TextureRestorer,
    texture_restoration_loss,
)


def _field(
    *,
    batch: int = 1,
    orientations: int = 8,
    height: int = 17,
    width: int = 19,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> StructuralField:
    shape = (batch, 1, height, width)
    orientation_prob = torch.full(
        (batch, orientations, height, width),
        1.0 / orientations,
        dtype=dtype,
        device=device,
    )
    frequency_mean = torch.full(shape, 0.08, dtype=dtype, device=device)
    frequency_var = torch.full(shape, 0.0004, dtype=dtype, device=device)
    coherence = torch.full(shape, 0.75, dtype=dtype, device=device)
    entropy = torch.full(shape, 0.5, dtype=dtype, device=device)
    uncertainty = torch.full(shape, 0.25, dtype=dtype, device=device)
    scale_prob = torch.full(
        (batch, 2, height, width),
        0.5,
        dtype=dtype,
        device=device,
    )
    selected_scale = torch.full(shape, 1.5, dtype=dtype, device=device)
    return StructuralField(
        orientation_prob=orientation_prob,
        frequency_mean=frequency_mean,
        frequency_var=frequency_var,
        coherence=coherence,
        orientation_entropy=entropy,
        structural_uncertainty=uncertainty,
        scale_prob=scale_prob,
        selected_scale=selected_scale,
        metadata=FieldMetadata(
            num_orientations=orientations,
            frequency_candidates=(0.04, 0.08, 0.16),
            scale_candidates=(1.0, 2.0),
            estimator="texture-test",
        ),
    )


def test_texture_restorer_defaults_are_explicit() -> None:
    model = TextureRestorer()
    assert model.num_orientations == 16
    assert model.width == 24
    assert model.adapter_mode == "concat"
    assert model.structural_input_channels == 24
    assert model.backbone.bottleneck[-1].__class__.__name__ == "FourierResidualBlock"


def test_texture_structural_mapping_has_exactly_k_plus_eight_channels() -> None:
    field = _field(orientations=8)
    features = texture_structural_features(field)
    assert features.shape == (1, 16, 17, 19)
    assert features.dtype == field.dtype
    assert features.device == field.device
    assert torch.isfinite(features).all()


@pytest.mark.parametrize(
    "adapter_mode",
    ["none", "concat", "film", "reliability_gated"],
)
def test_all_adapter_modes_forward_and_backward_on_odd_shape(adapter_mode: str) -> None:
    torch.manual_seed(12)
    image = torch.full((1, 1, 17, 19), 0.5)
    field = None if adapter_mode == "none" else _field()
    model = TextureRestorer(
        num_orientations=8,
        width=4,
        adapter_mode=adapter_mode,
    )

    prediction = model(image, field)
    assert prediction.shape == image.shape
    assert torch.isfinite(prediction).all()
    assert bool((prediction >= 0.0).all()) and bool((prediction <= 1.0).all())
    loss, components = TextureRestorationLoss()(prediction, torch.zeros_like(prediction))
    loss.backward()

    assert set(components) == {
        "l1",
        "spatial_gradient",
        "axial_gradient_direction",
        "fourier_amplitude",
    }
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)


def test_no_prior_mode_ignores_an_optional_valid_field_exactly() -> None:
    torch.manual_seed(23)
    image = torch.rand(1, 1, 17, 19)
    field = _field()
    model = TextureRestorer(8, 4, "none").eval()

    without_field = model(image)
    with_field = model(image, field)
    assert torch.equal(without_field, with_field)
    assert model.field_encoder is None


def test_reliability_adapter_exposes_spatial_gate_and_structural_delta() -> None:
    image = torch.rand(1, 1, 17, 19)
    model = TextureRestorer(8, 4, "reliability_gated")
    default_restoration = model(image, _field())
    restored, adapter_output = model(
        image,
        _field(),
        return_adapter_output=True,
    )

    torch.testing.assert_close(restored, default_restoration)
    assert adapter_output.features.shape == (1, 4, 17, 19)
    assert adapter_output.gate is not None
    assert adapter_output.structural_delta is not None
    assert adapter_output.gate.shape == adapter_output.structural_delta.shape == adapter_output.features.shape
    assert bool((adapter_output.gate >= 0.0).all())
    assert bool((adapter_output.gate <= 1.0).all())
    assert torch.isfinite(adapter_output.gate).all()
    assert torch.isfinite(adapter_output.structural_delta).all()


@pytest.mark.parametrize(
    ("adapter_mode", "expected_channels"),
    [("none", 4), ("concat", 8)],
)
def test_none_and_concat_adapter_diagnostics_have_explicit_semantics(
    adapter_mode: str,
    expected_channels: int,
) -> None:
    image = torch.rand(1, 1, 17, 19)
    field = None if adapter_mode == "none" else _field()
    model = TextureRestorer(8, 4, adapter_mode)
    restored, adapter_output = model(
        image,
        field,
        return_adapter_output=True,
    )

    assert restored.shape == image.shape
    assert adapter_output.features.shape == (1, expected_channels, 17, 19)
    assert adapter_output.gate is None
    assert adapter_output.structural_delta is None
    if adapter_mode == "none":
        # The no-prior adapter is an exact identity on the image-encoder map.
        torch.testing.assert_close(adapter_output.features, model.image_encoder(image))


def test_adapter_diagnostic_flag_is_keyword_only_and_boolean() -> None:
    model = TextureRestorer(8, 4, "none")
    image = torch.rand(1, 1, 17, 19)
    parameter = inspect.signature(TextureRestorer.forward).parameters["return_adapter_output"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="positional arguments"):
        model(image, None, True)  # type: ignore[misc]
    with pytest.raises(TypeError, match="must be a bool"):
        model(image, return_adapter_output=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("adapter_mode", ["concat", "film", "reliability_gated"])
def test_structural_modes_require_an_explicit_field(adapter_mode: str) -> None:
    model = TextureRestorer(8, 4, adapter_mode)
    with pytest.raises(ValueError, match="requires an explicit StructuralField"):
        model(torch.rand(1, 1, 17, 19))


def test_concat_changes_backbone_input_channels_but_other_modes_preserve_width() -> None:
    concat = TextureRestorer(8, 4, "concat")
    assert concat.fusion.out_channels == 8
    assert concat.backbone.encoder1.body[0].in_channels == 8
    for mode in ("none", "film", "reliability_gated"):
        model = TextureRestorer(8, 4, mode)
        assert model.fusion.out_channels == 4
        assert model.backbone.encoder1.body[0].in_channels == 4


def test_texture_restorer_rejects_boundary_mismatches() -> None:
    image = torch.rand(1, 1, 17, 19)
    model = TextureRestorer(8, 4, "concat")
    with pytest.raises(ValueError, match="orientation bins"):
        model(image, _field(orientations=16))
    with pytest.raises(ValueError, match="batch and spatial"):
        model(image, _field(height=16, width=19))
    with pytest.raises(ValueError, match="same dtype"):
        model(image, _field(dtype=torch.float64))
    with pytest.raises(ValueError, match=r"\[B,1,H,W\]"):
        TextureRestorer(8, 4, "none")(torch.rand(1, 3, 17, 19))
    invalid = image.clone()
    invalid[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        TextureRestorer(8, 4, "none")(invalid)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"num_orientations": True}, TypeError, "integer"),
        ({"num_orientations": 1}, ValueError, "at least 2"),
        ({"width": 0}, ValueError, "at least 1"),
        ({"adapter_mode": "attention"}, ValueError, "adapter_mode"),
    ],
)
def test_texture_restorer_rejects_invalid_configuration(
    arguments: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        TextureRestorer(**arguments)  # type: ignore[arg-type]


def test_texture_loss_reports_an_exact_weighted_decomposition() -> None:
    torch.manual_seed(31)
    prediction = torch.rand(1, 1, 8, 10, requires_grad=True)
    target = torch.rand_like(prediction)
    config = TextureRestorationLossConfig(
        l1_weight=1.2,
        spatial_gradient_weight=0.3,
        axial_direction_weight=0.4,
        fourier_amplitude_weight=0.5,
    )
    total, components = texture_restoration_loss(prediction, target, config=config)
    expected = (
        1.2 * components["l1"]
        + 0.3 * components["spatial_gradient"]
        + 0.4 * components["axial_gradient_direction"]
        + 0.5 * components["fourier_amplitude"]
    )
    torch.testing.assert_close(total.detach(), expected)
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_fourier_amplitude_can_be_isolated_and_backpropagated() -> None:
    x = torch.linspace(0.0, 2.0 * math.pi, 16)
    prediction = torch.sin(x).view(1, 1, 1, 16).expand(1, 1, 8, 16).clone()
    prediction.requires_grad_(True)
    target = torch.zeros_like(prediction)
    config = TextureRestorationLossConfig(
        l1_weight=0.0,
        spatial_gradient_weight=0.0,
        axial_direction_weight=0.0,
        fourier_amplitude_weight=1.0,
    )
    total, components = texture_restoration_loss(prediction, target, config=config)
    torch.testing.assert_close(total.detach(), components["fourier_amplitude"])
    assert float(total.detach()) > 0.0
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_identical_inputs_have_zero_l1_only_objective() -> None:
    image = torch.rand(1, 1, 8, 9)
    config = TextureRestorationLossConfig(
        l1_weight=1.0,
        spatial_gradient_weight=0.0,
        axial_direction_weight=0.0,
        fourier_amplitude_weight=0.0,
    )
    total, _ = texture_restoration_loss(image, image.clone(), config=config)
    torch.testing.assert_close(total, torch.tensor(0.0), atol=0.0, rtol=0.0)


def test_default_objective_is_zero_for_identical_flat_images() -> None:
    image = torch.full((1, 1, 8, 9), 0.4)
    total, components = texture_restoration_loss(image, image.clone())
    torch.testing.assert_close(total, torch.zeros_like(total), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        components["axial_gradient_direction"],
        torch.zeros_like(components["axial_gradient_direction"]),
        atol=0.0,
        rtol=0.0,
    )


def test_texture_loss_accepts_zero_stride_expanded_inputs() -> None:
    row = torch.linspace(0.0, 1.0, 19).view(1, 1, 1, 19)
    prediction = row.expand(1, 1, 17, 19).requires_grad_()
    target = prediction.detach().clone()
    total, _ = texture_restoration_loss(prediction, target)
    torch.testing.assert_close(total, torch.zeros_like(total), atol=0.0, rtol=0.0)
    total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_texture_loss_uses_stable_fft_dtype(
    dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    prediction = torch.rand(1, 1, 8, 9, dtype=dtype, requires_grad=True)
    target = torch.rand_like(prediction)
    total, components = texture_restoration_loss(prediction, target)
    assert total.dtype == expected_dtype
    assert all(value.dtype == expected_dtype for value in components.values())
    assert torch.isfinite(total)
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_texture_loss_validates_configuration_and_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TextureRestorationLossConfig(
            l1_weight=0.0,
            spatial_gradient_weight=0.0,
            axial_direction_weight=0.0,
            fourier_amplitude_weight=0.0,
        )
    with pytest.raises(TypeError, match="real number"):
        TextureRestorationLossConfig(l1_weight=True)
    with pytest.raises(ValueError, match="positive"):
        TextureRestorationLossConfig(eps=0.0)

    prediction = torch.rand(1, 1, 8, 9)
    with pytest.raises(ValueError, match="identical shapes"):
        texture_restoration_loss(prediction, torch.rand(1, 1, 8, 8))
    with pytest.raises(ValueError, match="same dtype"):
        texture_restoration_loss(prediction, prediction.double())
    invalid = prediction.clone()
    invalid[..., 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        texture_restoration_loss(invalid, prediction)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("adapter_mode", ["none", "concat", "film", "reliability_gated"])
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_texture_restorer_cuda_autocast_forward_backward(
    adapter_mode: str,
    autocast_dtype: torch.dtype,
) -> None:
    if autocast_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is unavailable")
    torch.manual_seed(37)
    image = torch.full((1, 1, 17, 19), 0.5, device="cuda")
    field = None if adapter_mode == "none" else _field(device="cuda")
    model = TextureRestorer(8, 4, adapter_mode).cuda()

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        prediction = model(image, field)
        total, _ = TextureRestorationLoss()(prediction, torch.zeros_like(prediction))
    assert prediction.device.type == "cuda" and prediction.shape == image.shape
    assert total.dtype == torch.float32 and torch.isfinite(total)
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_texture_restorer_rejects_cross_device_field() -> None:
    model = TextureRestorer(8, 4, "concat").cuda()
    image = torch.rand(1, 1, 17, 19, device="cuda")
    with pytest.raises(ValueError, match="same device"):
        model(image, _field())
