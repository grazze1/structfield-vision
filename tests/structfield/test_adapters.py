"""Contracts for Stage 5 structural-feature adapters."""

from __future__ import annotations

import pytest
import torch

from structfield.adapters import (
    AdapterOutput,
    ConcatStructuralAdapter,
    FiLMStructuralAdapter,
    NoPriorAdapter,
    ReliabilityGatedAdapter,
)

FEATURE_CHANNELS = 4
STRUCTURAL_CHANNELS = 3


def _inputs(
    *,
    batch: int = 2,
    height: int = 5,
    width: int = 7,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.randn(
        batch,
        FEATURE_CHANNELS,
        height,
        width,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    structural_features = torch.randn(
        batch,
        STRUCTURAL_CHANNELS,
        height,
        width,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    structural_uncertainty = torch.rand(
        batch,
        1,
        height,
        width,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return features, structural_features, structural_uncertainty


def _assert_finite_gradients(module: torch.nn.Module, *inputs: torch.Tensor) -> None:
    for value in inputs:
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


def test_adapter_output_validates_gate_and_structural_delta_contracts() -> None:
    features = torch.randn(2, 4, 3, 5)
    gate = torch.rand_like(features)
    delta = torch.randn_like(features)
    output = AdapterOutput(features=features, gate=gate, structural_delta=delta)
    assert output.features is features
    assert output.gate is gate
    assert output.structural_delta is delta

    with pytest.raises(ValueError, match="same shape"):
        AdapterOutput(features=features, gate=gate[:, :1])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        AdapterOutput(features=features, gate=torch.full_like(features, 1.01))
    with pytest.raises(ValueError, match="same shape"):
        AdapterOutput(features=features, structural_delta=delta[:, :1])
    with pytest.raises(ValueError, match="dtype and device"):
        AdapterOutput(features=features, structural_delta=delta.double())


def test_adapter_boundary_rejects_unsupported_float8() -> None:
    float8_dtype = getattr(torch, "float8_e4m3fn", None)
    if float8_dtype is None:
        pytest.skip("float8_e4m3fn is unavailable")
    features = torch.empty(1, FEATURE_CHANNELS, 3, 5, dtype=float8_dtype)
    with pytest.raises(TypeError, match="supported floating-point"):
        NoPriorAdapter()(features)
    with pytest.raises(TypeError, match="supported floating-point"):
        AdapterOutput(features=features)


def test_no_prior_is_exact_identity_for_a_batch() -> None:
    features, structural_features, _ = _inputs(batch=3)
    adapter = NoPriorAdapter()
    output = adapter(features, structural_features)

    assert output.features is features
    assert output.gate is None
    assert output.structural_delta is None
    assert adapter.output_channels(FEATURE_CHANNELS, STRUCTURAL_CHANNELS) == FEATURE_CHANNELS


def test_no_prior_supports_an_explicit_no_prior_call() -> None:
    features, _, _ = _inputs()
    output = NoPriorAdapter()(features)
    assert output.features is features


def test_concat_is_a_fixed_channelwise_structural_prior() -> None:
    features, structural_features, _ = _inputs(batch=3)
    adapter = ConcatStructuralAdapter()
    output = adapter(features, structural_features)

    assert output.features.shape == (3, FEATURE_CHANNELS + STRUCTURAL_CHANNELS, 5, 7)
    assert torch.equal(output.features[:, :FEATURE_CHANNELS], features)
    assert torch.equal(output.features[:, FEATURE_CHANNELS:], structural_features)
    assert output.gate is None
    assert output.structural_delta is None
    assert adapter.output_channels(FEATURE_CHANNELS, STRUCTURAL_CHANNELS) == 7


def test_film_preserves_channels_and_exposes_its_additive_delta() -> None:
    torch.manual_seed(5)
    features, structural_features, _ = _inputs(batch=3, requires_grad=True)
    adapter = FiLMStructuralAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS)
    output = adapter(features, structural_features)

    assert output.features.shape == features.shape
    assert output.structural_delta is not None
    assert output.structural_delta.shape == features.shape
    assert output.gate is None
    assert torch.equal(output.features, features + output.structural_delta)
    assert adapter.output_channels(FEATURE_CHANNELS, STRUCTURAL_CHANNELS) == FEATURE_CHANNELS

    output.features.square().mean().backward()
    _assert_finite_gradients(adapter, features, structural_features)


def test_reliability_gate_is_spatial_channel_and_distinct_from_uncertainty() -> None:
    torch.manual_seed(11)
    features, structural_features, structural_uncertainty = _inputs(batch=3, requires_grad=True)
    adapter = ReliabilityGatedAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS)
    output = adapter(features, structural_features, structural_uncertainty)

    assert output.features.shape == features.shape
    assert output.gate is not None
    assert output.gate.shape == features.shape
    assert output.gate.shape[1] == FEATURE_CHANNELS
    assert structural_uncertainty.shape[1] == 1
    assert output.structural_delta is not None
    assert output.structural_delta.shape == features.shape
    assert bool(((0.0 <= output.gate) & (output.gate <= 1.0)).all())
    assert torch.equal(output.features, features + output.gate * output.structural_delta)
    assert adapter.output_channels(FEATURE_CHANNELS, STRUCTURAL_CHANNELS) == FEATURE_CHANNELS

    output.features.square().mean().backward()
    _assert_finite_gradients(adapter, features, structural_features, structural_uncertainty)


def test_zero_gate_is_torch_equal_residual_identity() -> None:
    features, _, _ = _inputs(batch=3)
    structural_delta = torch.randn_like(features)
    gate = torch.zeros_like(features)

    output = ReliabilityGatedAdapter.fuse(features, structural_delta, gate)
    assert torch.equal(output, features)


@pytest.mark.parametrize("adapter", [NoPriorAdapter(), ConcatStructuralAdapter()])
def test_parameter_free_adapters_backward(adapter: torch.nn.Module) -> None:
    features, structural_features, _ = _inputs(requires_grad=True)
    output = adapter(features, structural_features)
    output.features.square().mean().backward()

    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())
    if isinstance(adapter, ConcatStructuralAdapter):
        assert structural_features.grad is not None
        assert bool(torch.isfinite(structural_features.grad).all())
    else:
        assert structural_features.grad is None


@pytest.mark.parametrize(
    "feature_channels, structural_channels, error",
    [
        (0, 3, "positive"),
        (4, 0, "positive"),
        (True, 3, "integer"),
        (4, False, "integer"),
    ],
)
def test_output_channels_rejects_invalid_counts(
    feature_channels: int,
    structural_channels: int,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        ConcatStructuralAdapter().output_channels(feature_channels, structural_channels)


@pytest.mark.parametrize(
    "adapter",
    [
        FiLMStructuralAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS),
        ReliabilityGatedAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS),
    ],
)
def test_learned_output_channels_rejects_configuration_mismatch(adapter: torch.nn.Module) -> None:
    with pytest.raises(ValueError, match="configured value"):
        adapter.output_channels(FEATURE_CHANNELS + 1, STRUCTURAL_CHANNELS)
    with pytest.raises(ValueError, match="configured value"):
        adapter.output_channels(FEATURE_CHANNELS, STRUCTURAL_CHANNELS + 1)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda value: value[:1], "batch size"),
        (lambda value: value[..., :-1, :], "spatial shape"),
        (lambda value: value.double(), "same dtype"),
    ],
)
def test_adapters_reject_incompatible_structural_features_without_resize_or_cast(
    mutate: object,
    error: str,
) -> None:
    features, structural_features, _ = _inputs()
    with pytest.raises(ValueError, match=error):
        ConcatStructuralAdapter()(features, mutate(structural_features))  # type: ignore[operator]


def test_learned_adapters_reject_wrong_runtime_channel_counts() -> None:
    features, structural_features, structural_uncertainty = _inputs()
    wrong_features = features[:, :-1]
    wrong_structural = structural_features[:, :-1]

    with pytest.raises(ValueError, match="features must have"):
        FiLMStructuralAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS)(wrong_features, structural_features)
    with pytest.raises(ValueError, match="structural_features must have"):
        ReliabilityGatedAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS)(
            features,
            wrong_structural,
            structural_uncertainty,
        )


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda value: value.expand(-1, 2, -1, -1), "1 channels"),
        (lambda value: value[:1], "batch size"),
        (lambda value: value[..., :-1], "spatial shape"),
        (lambda value: value.double(), "same dtype"),
        (lambda value: value + 1.0, r"\[0,1\]"),
    ],
)
def test_reliability_rejects_invalid_structural_uncertainty(
    mutate: object,
    error: str,
) -> None:
    features, structural_features, structural_uncertainty = _inputs()
    adapter = ReliabilityGatedAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS)
    with pytest.raises(ValueError, match=error):
        adapter(
            features,
            structural_features,
            mutate(structural_uncertainty),  # type: ignore[operator]
        )


def test_fuse_rejects_non_spatial_channel_or_unbounded_gate() -> None:
    features, _, _ = _inputs()
    delta = torch.randn_like(features)
    with pytest.raises(ValueError, match="identical shapes"):
        ReliabilityGatedAdapter.fuse(features, delta, torch.zeros_like(features[:, :1]))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        ReliabilityGatedAdapter.fuse(features, delta, torch.full_like(features, -0.1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_low_precision_preserves_dtype_and_backward(dtype: torch.dtype) -> None:
    features, structural_features, structural_uncertainty = _inputs(dtype=dtype, requires_grad=True)
    adapter = ReliabilityGatedAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS).to(dtype=dtype)
    output = adapter(features, structural_features, structural_uncertainty)

    assert output.features.dtype == dtype
    assert output.gate is not None and output.gate.dtype == dtype
    assert output.structural_delta is not None and output.structural_delta.dtype == dtype
    output.features.float().square().mean().backward()
    _assert_finite_gradients(adapter, features, structural_features, structural_uncertainty)


@pytest.mark.parametrize("adapter_type", [FiLMStructuralAdapter, ReliabilityGatedAdapter])
def test_learned_adapters_cpu_autocast_preserves_public_dtype_and_full_backward(
    adapter_type: type[torch.nn.Module],
) -> None:
    features, structural_features, structural_uncertainty = _inputs(requires_grad=True)
    adapter = adapter_type(FEATURE_CHANNELS, STRUCTURAL_CHANNELS)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        if isinstance(adapter, ReliabilityGatedAdapter):
            output = adapter(features, structural_features, structural_uncertainty)
        else:
            output = adapter(features, structural_features)
    assert output.features.dtype == torch.float32
    assert output.structural_delta is not None and output.structural_delta.dtype == torch.float32
    if output.gate is not None:
        assert output.gate.dtype == torch.float32
    output.features.square().mean().backward()
    if isinstance(adapter, ReliabilityGatedAdapter):
        _assert_finite_gradients(adapter, features, structural_features, structural_uncertainty)
    else:
        _assert_finite_gradients(adapter, features, structural_features)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_adapters_reject_cross_device_structural_inputs() -> None:
    features, structural_features, structural_uncertainty = _inputs()
    with pytest.raises(ValueError, match="same device"):
        ConcatStructuralAdapter()(features, structural_features.cuda())
    with pytest.raises(ValueError, match="same device"):
        ReliabilityGatedAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS)(
            features,
            structural_features,
            structural_uncertainty.cuda(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("adapter_name", ["no_prior", "concat", "film", "reliability"])
def test_adapters_cuda_shape_dtype_device_and_backward(
    adapter_name: str,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is not supported")
    device = torch.device("cuda")
    features, structural_features, structural_uncertainty = _inputs(
        batch=3,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    if adapter_name == "no_prior":
        adapter = NoPriorAdapter().to(device=device, dtype=dtype)
        output = adapter(features, structural_features)
    elif adapter_name == "concat":
        adapter = ConcatStructuralAdapter().to(device=device, dtype=dtype)
        output = adapter(features, structural_features)
    elif adapter_name == "film":
        adapter = FiLMStructuralAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS).to(device=device, dtype=dtype)
        output = adapter(features, structural_features)
    else:
        adapter = ReliabilityGatedAdapter(FEATURE_CHANNELS, STRUCTURAL_CHANNELS).to(device=device, dtype=dtype)
        output = adapter(features, structural_features, structural_uncertainty)

    assert output.features.device.type == "cuda"
    assert output.features.dtype == dtype
    assert output.features.shape[0] == 3
    output.features.float().square().mean().backward()
    assert features.grad is not None and bool(torch.isfinite(features.grad).all())
    if adapter_name != "no_prior":
        assert structural_features.grad is not None
        assert bool(torch.isfinite(structural_features.grad).all())
    if adapter_name == "reliability":
        assert structural_uncertainty.grad is not None
        assert bool(torch.isfinite(structural_uncertainty.grad).all())
        assert output.gate is not None and output.gate.shape == features.shape
        assert output.structural_delta is not None and output.structural_delta.shape == features.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("adapter_type", [FiLMStructuralAdapter, ReliabilityGatedAdapter])
def test_learned_adapters_preserve_public_float32_under_cuda_autocast(
    adapter_type: type[torch.nn.Module],
) -> None:
    features, structural_features, structural_uncertainty = _inputs(device="cuda", requires_grad=True)
    adapter = adapter_type(FEATURE_CHANNELS, STRUCTURAL_CHANNELS).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        if isinstance(adapter, ReliabilityGatedAdapter):
            output = adapter(features, structural_features, structural_uncertainty)
        else:
            output = adapter(features, structural_features)

    assert output.features.dtype == torch.float32
    assert output.structural_delta is not None and output.structural_delta.dtype == torch.float32
    if output.gate is not None:
        assert output.gate.dtype == torch.float32
    output.features.square().mean().backward()
    assert features.grad is not None and bool(torch.isfinite(features.grad).all())
    assert structural_features.grad is not None and bool(torch.isfinite(structural_features.grad).all())
    for parameter in adapter.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    if isinstance(adapter, ReliabilityGatedAdapter):
        assert structural_uncertainty.grad is not None
        assert bool(torch.isfinite(structural_uncertainty.grad).all())
