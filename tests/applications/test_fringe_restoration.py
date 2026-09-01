"""Tiny CPU/CUDA contracts for the V0.2 fringe-restoration application."""

from __future__ import annotations

import math

import pytest
import torch

from structfield import FieldMetadata, StructuralField
from structfield.adapters import AdapterOutput
from structfield.applications.fringe_restoration import (
    FringeRestorationLoss,
    FringeRestorationLossConfig,
    FringeRestorer,
)


def _field(
    *,
    batch: int = 1,
    num_orientations: int = 8,
    height: int = 17,
    width: int = 19,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> StructuralField:
    probabilities = torch.zeros(
        batch,
        num_orientations,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    probabilities[:, 0] = 0.6
    probabilities[:, num_orientations // 2] = 0.4
    scalar = torch.ones(batch, 1, height, width, device=device, dtype=dtype)
    scale_prob = torch.empty(batch, 2, height, width, device=device, dtype=dtype)
    scale_prob[:, 0] = 0.25
    scale_prob[:, 1] = 0.75
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(1, keepdim=True) / math.log(
        num_orientations
    )
    return StructuralField(
        orientation_prob=probabilities,
        frequency_mean=0.1 * scalar,
        frequency_var=0.001 * scalar,
        coherence=0.7 * scalar,
        orientation_entropy=entropy,
        structural_uncertainty=0.25 * scalar,
        metadata=FieldMetadata(
            num_orientations=num_orientations,
            frequency_candidates=(0.05, 0.1, 0.15),
            scale_candidates=(1.0, 2.0),
            estimator="test-fringe-application",
        ),
        scale_prob=scale_prob,
        selected_scale=1.75 * scalar,
    )


@pytest.mark.parametrize("adapter_mode", ["none", "concat", "film", "reliability_gated"])
def test_all_adapter_modes_forward_on_odd_spatial_shape(adapter_mode: str) -> None:
    image = torch.rand(2, 1, 17, 19)
    model = FringeRestorer(
        num_orientations=8,
        width=4,
        noise_dim=6,
        adapter_mode=adapter_mode,
    )
    if adapter_mode == "none":
        output = model(image)
    else:
        output = model(
            image,
            field=_field(batch=2),
            normalized_image=torch.randn_like(image),
            noise_descriptor=torch.randn(2, 6),
        )
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert output.device == image.device
    assert bool(torch.isfinite(output).all())
    assert float(output.detach().min()) >= 0.0
    assert float(output.detach().max()) <= 1.0
    output.mean().backward()
    assert model.residual_head.weight.grad is not None
    assert bool(torch.isfinite(model.residual_head.weight.grad).all())


def test_conditioned_model_and_loss_backward_reach_every_branch() -> None:
    image = torch.rand(2, 1, 17, 19, requires_grad=True)
    normalized = torch.randn_like(image, requires_grad=True)
    descriptor = torch.randn(2, 6, requires_grad=True)
    target = torch.rand_like(image)
    model = FringeRestorer(
        num_orientations=8,
        width=4,
        noise_dim=6,
        adapter_mode="concat",
    )

    prediction = model(
        image,
        field=_field(batch=2),
        normalized_image=normalized,
        noise_descriptor=descriptor,
    )
    loss_output = FringeRestorationLoss()(prediction, target)
    loss_output.total.backward()

    for value in (image, normalized, descriptor):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
    for module in (
        model.image_encoder,
        model.structure_encoder,
        model.backbone,
        model.noise_film,
        model.residual_head,
    ):
        parameters = list(module.parameters())
        assert parameters
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)


def test_none_mode_fixes_zero_second_channel_and_ignores_valid_field() -> None:
    torch.manual_seed(7)
    image = torch.rand(1, 1, 17, 19)
    model = FringeRestorer(
        num_orientations=8,
        width=4,
        adapter_mode="none",
    ).eval()
    assert model.structure_encoder is None
    assert not any(name.startswith("structure_encoder") for name, _ in model.named_parameters())
    without_field = model(image)
    with_field = model(image, field=_field())
    assert torch.equal(without_field, with_field)
    with pytest.raises(ValueError, match="fixed to zero"):
        model(image, normalized_image=torch.zeros_like(image))


def test_explicit_field_and_normalized_image_contracts() -> None:
    image = torch.rand(1, 1, 17, 19)
    model = FringeRestorer(num_orientations=8, width=4)
    field = _field()

    with pytest.raises(ValueError, match="explicit field"):
        model(image, normalized_image=torch.zeros_like(image))
    with pytest.raises(ValueError, match="requires normalized_image"):
        model(image, field=field)
    with pytest.raises(ValueError, match="same shape"):
        model(
            image,
            field=field,
            normalized_image=torch.zeros(1, 1, 16, 19),
        )
    with pytest.raises(ValueError, match="same dtype"):
        model(image, field=field, normalized_image=torch.zeros_like(image).double())
    with pytest.raises(ValueError, match="orientation bins"):
        model(
            image,
            field=_field(num_orientations=16),
            normalized_image=torch.zeros_like(image),
        )
    with pytest.raises(ValueError, match="batch and spatial"):
        model(
            image,
            field=_field(height=16),
            normalized_image=torch.zeros_like(image),
        )


def test_noise_dimension_is_configured_without_hardcoded_ten() -> None:
    image = torch.rand(1, 1, 17, 19)
    model = FringeRestorer(
        num_orientations=8,
        width=4,
        noise_dim=3,
        adapter_mode="film",
    )
    kwargs = {"field": _field(), "normalized_image": torch.zeros_like(image)}
    assert model(image, noise_descriptor=torch.zeros(1, 3), **kwargs).shape == image.shape
    assert model(image, **kwargs).shape == image.shape
    with pytest.raises(ValueError, match=r"\(1, 3\)"):
        model(image, noise_descriptor=torch.zeros(1, 10), **kwargs)


def test_constructor_and_adapter_validation() -> None:
    defaults = FringeRestorer()
    assert defaults.num_orientations == 16
    assert defaults.width == 24
    assert defaults.noise_dim == 10
    assert defaults.adapter_mode == "concat"
    with pytest.raises(TypeError, match="num_orientations"):
        FringeRestorer(num_orientations=True)
    with pytest.raises(ValueError, match="at least 2"):
        FringeRestorer(num_orientations=1)
    with pytest.raises(ValueError, match="noise_dim"):
        FringeRestorer(noise_dim=0)
    with pytest.raises(ValueError, match="adapter mode"):
        FringeRestorer(adapter_mode="unknown")


def test_reliability_adapter_output_is_explicitly_observable() -> None:
    image = torch.rand(1, 1, 17, 19)
    model = FringeRestorer(
        num_orientations=8,
        width=4,
        adapter_mode="reliability_gated",
    ).eval()
    kwargs = {
        "field": _field(),
        "normalized_image": torch.zeros_like(image),
    }
    default_output = model(image, **kwargs)
    restored, adapter_output = model(image, return_adapter_output=True, **kwargs)

    assert isinstance(default_output, torch.Tensor)
    assert torch.equal(default_output, restored)
    assert isinstance(adapter_output, AdapterOutput)
    assert adapter_output.gate is not None
    assert adapter_output.gate.shape == (1, 4, 17, 19)
    assert float(adapter_output.gate.detach().min()) >= 0.0
    assert float(adapter_output.gate.detach().max()) <= 1.0
    assert adapter_output.structural_delta is not None
    with pytest.raises(TypeError, match="must be a bool"):
        model(image, return_adapter_output=1, **kwargs)  # type: ignore[arg-type]


def test_loss_components_are_explicit_and_weighted() -> None:
    prediction = torch.rand(2, 1, 9, 11, requires_grad=True)
    target = torch.rand_like(prediction)
    config = FringeRestorationLossConfig(
        l1_weight=1.5,
        axial_gradient_weight=0.25,
        gradient_magnitude_weight=0.125,
    )
    output = FringeRestorationLoss(config)(prediction, target)
    expected = (
        config.l1_weight * output.l1
        + config.axial_gradient_weight * output.axial_gradient
        + config.gradient_magnitude_weight * output.gradient_magnitude
    )
    assert torch.allclose(output.total.detach(), expected)
    assert not output.l1.requires_grad
    assert not output.axial_gradient.requires_grad
    assert not output.gradient_magnitude.requires_grad
    assert not hasattr(output, "frequency")
    output.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_default_loss_is_zero_for_identical_flat_images() -> None:
    image = torch.full((1, 1, 9, 11), 0.4)
    output = FringeRestorationLoss()(image, image.clone())
    torch.testing.assert_close(output.total, torch.zeros_like(output.total), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        output.axial_gradient,
        torch.zeros_like(output.axial_gradient),
        atol=0.0,
        rtol=0.0,
    )


def test_loss_validates_inputs_and_config() -> None:
    prediction = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="same shape"):
        FringeRestorationLoss()(prediction, torch.zeros(1, 1, 8, 9))
    with pytest.raises(ValueError, match="same dtype"):
        FringeRestorationLoss()(prediction, prediction.double())
    with pytest.raises(ValueError, match="at least one"):
        FringeRestorationLossConfig(
            l1_weight=0.0,
            axial_gradient_weight=0.0,
            gradient_magnitude_weight=0.0,
        )
    with pytest.raises(ValueError, match="eps must be positive"):
        FringeRestorationLossConfig(eps=0.0)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_loss_uses_stable_working_dtype(
    dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    prediction = torch.rand(1, 1, 8, 9, dtype=dtype, requires_grad=True)
    target = torch.rand_like(prediction)
    output = FringeRestorationLoss()(prediction, target)
    assert output.total.dtype == expected_dtype
    assert output.l1.dtype == expected_dtype
    assert output.axial_gradient.dtype == expected_dtype
    assert output.gradient_magnitude.dtype == expected_dtype
    assert bool(torch.isfinite(output.total))
    output.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_forward_backward() -> None:
    device = torch.device("cuda")
    image = torch.rand(1, 1, 17, 19, device=device, requires_grad=True)
    normalized = torch.randn_like(image, requires_grad=True)
    descriptor = torch.randn(1, 6, device=device, requires_grad=True)
    model = FringeRestorer(
        num_orientations=8,
        width=4,
        noise_dim=6,
        adapter_mode="reliability_gated",
    ).to(device)
    prediction = model(
        image,
        field=_field(device=device),
        normalized_image=normalized,
        noise_descriptor=descriptor,
    )
    loss = FringeRestorationLoss()(prediction, torch.rand_like(prediction)).total
    loss.backward()
    assert prediction.device.type == "cuda"
    assert bool(torch.isfinite(prediction).all())
    for value in (image, normalized, descriptor):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("adapter_mode", ["none", "concat", "film", "reliability_gated"])
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast_forward_backward(
    adapter_mode: str,
    autocast_dtype: torch.dtype,
) -> None:
    if autocast_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is unavailable")
    device = torch.device("cuda")
    image = torch.full((1, 1, 17, 19), 0.5, device=device)
    model = FringeRestorer(
        num_orientations=8,
        width=4,
        noise_dim=6,
        adapter_mode=adapter_mode,
    ).to(device)
    kwargs: dict[str, object] = {"noise_descriptor": torch.zeros(1, 6, device=device)}
    if adapter_mode != "none":
        kwargs.update(
            field=_field(device=device),
            normalized_image=torch.zeros_like(image),
        )

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        prediction = model(image, **kwargs)  # type: ignore[arg-type]
        if not isinstance(prediction, torch.Tensor):  # pragma: no cover - default contract
            raise AssertionError("default forward must return a tensor")
        loss = FringeRestorationLoss()(prediction, torch.zeros_like(prediction)).total
    assert prediction.shape == image.shape
    assert prediction.device.type == "cuda"
    assert loss.dtype == torch.float32
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)
