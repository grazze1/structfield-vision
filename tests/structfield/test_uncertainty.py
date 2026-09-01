"""Contracts for the parameter-free Stage 4 uncertainty baseline."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from structfield import FieldMetadata, StructuralField
from structfield.estimators import AnalyticEstimatorConfig, AnalyticStructuralFieldEstimator
from structfield.uncertainty import HeuristicStructuralUncertainty, normalized_categorical_entropy
from tests.structfield.test_analytic_estimator import crossing_pattern, estimator, synthetic_grating


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(probability.dtype).tiny
    value = -(probability * probability.clamp_min(tiny).log()).sum(dim=1, keepdim=True)
    return value / math.log(probability.shape[1])


def _make_field(
    orientation_prob: torch.Tensor,
    *,
    frequency_var: float | torch.Tensor = 0.0,
    coherence: float | torch.Tensor = 1.0,
    scale_prob: torch.Tensor | None = None,
) -> StructuralField:
    batch, bins, height, width = orientation_prob.shape
    dtype = orientation_prob.dtype
    device = orientation_prob.device

    def scalar_map(value: float | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype).expand(batch, 1, height, width)
        return torch.full((batch, 1, height, width), value, device=device, dtype=dtype)

    frequencies = (0.05, 0.10, 0.15)
    scales: tuple[float, ...] = ()
    selected_scale = None
    if scale_prob is not None:
        scales = tuple(float(index + 1) for index in range(scale_prob.shape[1]))
        scale_values = scale_prob.new_tensor(scales).view(1, -1, 1, 1)
        selected_scale = (scale_prob * scale_values).sum(dim=1, keepdim=True)
    metadata = FieldMetadata(
        num_orientations=bins,
        frequency_candidates=frequencies,
        scale_candidates=scales,
        estimator="stage4-test",
    )
    return StructuralField(
        orientation_prob=orientation_prob,
        frequency_mean=scalar_map(0.10),
        frequency_var=scalar_map(frequency_var),
        coherence=scalar_map(coherence),
        orientation_entropy=_entropy(orientation_prob),
        structural_uncertainty=scalar_map(0.0),
        metadata=metadata,
        scale_prob=scale_prob,
        selected_scale=selected_scale,
    )


def test_default_and_configurable_weights_have_no_learned_parameters() -> None:
    module = HeuristicStructuralUncertainty()
    assert module.weights == pytest.approx((0.25, 0.20, 0.20, 0.25, 0.10))
    assert tuple(module.parameters()) == ()
    assert module.state_dict() == {}

    probability = torch.zeros(1, 4, 2, 3)
    probability[:, 0] = 1.0
    field = _make_field(probability, coherence=0.3)
    coherence_only = HeuristicStructuralUncertainty(weights=(0.0, 0.0, 0.0, 1.0, 0.0))
    assert torch.allclose(coherence_only(field), 1.0 - field.coherence)


def test_shared_normalized_entropy_handles_boundaries_and_low_precision() -> None:
    one_hot = torch.tensor((1.0, 0.0, 0.0, 0.0)).view(1, 4, 1, 1)
    uniform = torch.full((1, 4, 2, 3), 0.25, dtype=torch.float16)
    single = torch.ones(2, 1, 2, 2)
    assert torch.equal(normalized_categorical_entropy(one_hot), torch.zeros(1, 1, 1, 1))
    uniform_entropy = normalized_categorical_entropy(uniform)
    assert uniform_entropy.dtype == torch.float32
    assert torch.equal(uniform_entropy, torch.ones_like(uniform_entropy))
    assert torch.equal(normalized_categorical_entropy(single), torch.zeros(2, 1, 2, 2))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1])
def test_shared_normalized_entropy_rejects_invalid_probability(invalid: float) -> None:
    probability = torch.tensor((1.0, invalid)).view(1, 2, 1, 1)
    with pytest.raises(ValueError):
        normalized_categorical_entropy(probability)


@pytest.mark.parametrize(
    "weights, error",
    [
        ((1.0, 2.0), "exactly five"),
        ((0.0, 0.0, 0.0, 0.0, 1.0), "non-scale"),
        ((0.25, 0.20, -0.20, 0.25, 0.10), "non-negative"),
        ((0.25, 0.20, float("nan"), 0.25, 0.10), "finite"),
    ],
)
def test_invalid_weights_are_rejected(weights: tuple[float, ...], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        HeuristicStructuralUncertainty(weights)


def test_missing_scale_is_omitted_and_remaining_weights_are_renormalized() -> None:
    probability = torch.zeros(2, 4, 3, 5)
    probability[:, 0] = 0.7
    probability[:, 1] = 0.2
    probability[:, 2] = 0.1
    field = _make_field(probability, frequency_var=0.001, coherence=0.7)

    actual = HeuristicStructuralUncertainty()(field)
    relative_margin_uncertainty = 1.0 - (0.7 - 0.2) / (0.7 + 0.2)
    normalized_frequency_variance = 0.001 / ((0.15 - 0.05) ** 2 / 4.0)
    expected = (
        0.25 * field.orientation_entropy
        + 0.20 * relative_margin_uncertainty
        + 0.20 * normalized_frequency_variance
        + 0.25 * (1.0 - field.coherence)
    ) / 0.90
    assert actual.shape == (2, 1, 3, 5)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert bool(((0.0 <= actual) & (actual <= 1.0)).all())


def test_flat_field_has_high_uncertainty_with_uniform_frequency_posterior() -> None:
    probability = torch.full((2, 8, 4, 6), 1.0 / 8.0)
    scale_prob = torch.full((2, 3, 4, 6), 1.0 / 3.0)
    frequencies = torch.tensor((0.05, 0.10, 0.15))
    uniform_frequency_variance = float(frequencies.var(unbiased=False))
    field = _make_field(
        probability,
        frequency_var=uniform_frequency_variance,
        coherence=0.0,
        scale_prob=scale_prob,
    )
    uncertainty = HeuristicStructuralUncertainty()(field)
    normalized_frequency_variance = 2.0 / 3.0
    expected = 0.25 + 0.20 + 0.20 * normalized_frequency_variance + 0.25 + 0.10
    assert torch.allclose(uncertainty, torch.full_like(uncertainty, expected), atol=1e-6)
    assert float(uncertainty.min()) > 0.9


def test_frequency_variance_reaches_and_clamps_at_bounded_support_maximum() -> None:
    probability = torch.zeros(1, 4, 2, 3)
    probability[:, 0] = 1.0
    maximum_variance = (0.15 - 0.05) ** 2 / 4.0
    frequency_only = HeuristicStructuralUncertainty(weights=(0.0, 0.0, 1.0, 0.0, 0.0))
    zero = frequency_only(_make_field(probability, frequency_var=0.0))
    maximum = frequency_only(_make_field(probability, frequency_var=maximum_variance))
    above_maximum = frequency_only(_make_field(probability, frequency_var=2.0 * maximum_variance))
    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.allclose(maximum, torch.ones_like(maximum), atol=1e-6)
    assert torch.equal(above_maximum, torch.ones_like(above_maximum))


def test_ambiguous_crossing_is_more_uncertain_than_a_single_orientation() -> None:
    single = torch.zeros(1, 8, 4, 5)
    single[:, 0] = 1.0
    crossing = torch.zeros_like(single)
    crossing[:, 0] = 0.5
    crossing[:, 4] = 0.5
    scale_prob = torch.zeros(1, 2, 4, 5)
    scale_prob[:, 0] = 1.0

    module = HeuristicStructuralUncertainty()
    single_uncertainty = module(_make_field(single, scale_prob=scale_prob))
    crossing_uncertainty = module(_make_field(crossing, scale_prob=scale_prob))
    assert torch.equal(single_uncertainty, torch.zeros_like(single_uncertainty))
    assert bool((crossing_uncertainty > single_uncertainty).all())


def test_margin_uses_separated_modes_not_adjacent_bins() -> None:
    adjacent_single_mode = torch.zeros(1, 16, 2, 2)
    adjacent_single_mode[:, 0] = 0.55
    adjacent_single_mode[:, 1] = 0.45
    separated_crossing = torch.zeros_like(adjacent_single_mode)
    separated_crossing[:, 0] = 0.5
    separated_crossing[:, 8] = 0.5
    margin_only = HeuristicStructuralUncertainty(weights=(0.0, 1.0, 0.0, 0.0, 0.0))

    single_uncertainty = margin_only(_make_field(adjacent_single_mode))
    crossing_uncertainty = margin_only(_make_field(separated_crossing))
    assert torch.equal(single_uncertainty, torch.zeros_like(single_uncertainty))
    assert torch.equal(crossing_uncertainty, torch.ones_like(crossing_uncertainty))


def test_margin_is_defined_when_no_separated_runner_up_is_feasible() -> None:
    probability = torch.zeros(2, 8, 2, 3)
    probability[:, 0] = 1.0
    field = _make_field(probability)
    extreme_separation = replace(
        field,
        metadata=FieldMetadata(
            num_orientations=8,
            frequency_candidates=(0.05, 0.10, 0.15),
            default_min_separation_rad=3.0,
            estimator="stage4-test",
        ),
    )
    margin_only = HeuristicStructuralUncertainty(weights=(0.0, 1.0, 0.0, 0.0, 0.0))
    assert torch.equal(margin_only(extreme_separation), torch.zeros_like(field.coherence))


def test_valid_single_frequency_estimator_with_extreme_separation_is_supported() -> None:
    config = AnalyticEstimatorConfig(
        num_orientations=8,
        frequencies=(0.1,),
        tensor_scales=(1.0,),
        min_separation_rad=3.0,
    )
    field = AnalyticStructuralFieldEstimator(config)(torch.rand(1, 1, 17, 19))
    assert isinstance(field, StructuralField)
    uncertainty = HeuristicStructuralUncertainty()(field)
    assert uncertainty.shape == (1, 1, 17, 19)
    assert torch.isfinite(uncertainty).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_analytic_estimator_writes_canonical_uncertainty_and_metadata(dtype: torch.dtype) -> None:
    module = AnalyticStructuralFieldEstimator(
        num_orientations=8,
        frequencies=(0.05, 0.10, 0.15),
        tensor_scales=(1.0, 2.0),
    )
    field = module(torch.full((2, 1, 24, 26), 0.5, dtype=dtype))
    assert isinstance(field, StructuralField)
    assert field.dtype == dtype
    recomputed = HeuristicStructuralUncertainty()(field)
    assert torch.equal(field.structural_uncertainty, recomputed)
    assert float(field.structural_uncertainty.min()) > 0.9
    extras = field.metadata.to_dict()["extras"]
    assert extras["structural_uncertainty"] == "baseline-heuristic-v1"
    assert extras["structural_uncertainty_weights"] == pytest.approx((0.25, 0.20, 0.20, 0.25, 0.10))


def test_analytic_flat_and_crossing_are_more_uncertain_than_single_orientation() -> None:
    module = estimator()
    flat = module(torch.full((1, 1, 64, 64), 0.5))
    single = module(synthetic_grating(math.radians(37.0)))
    crossing = module(crossing_pattern())
    assert isinstance(flat, StructuralField)
    assert isinstance(single, StructuralField)
    assert isinstance(crossing, StructuralField)
    interior = (slice(None), slice(None), slice(16, -16), slice(16, -16))
    flat_mean = flat.structural_uncertainty[interior].mean()
    single_mean = single.structural_uncertainty[interior].mean()
    crossing_mean = crossing.structural_uncertainty[interior].mean()
    assert float(flat_mean) > 0.9
    assert float(crossing_mean - single_mean) > 0.25
    assert float(flat_mean - crossing_mean) > 0.20


def test_single_scale_is_valid_and_has_zero_scale_entropy() -> None:
    probability = torch.zeros(1, 4, 2, 2)
    probability[:, 0] = 1.0
    single_scale = torch.ones(1, 1, 2, 2)
    field = _make_field(probability, coherence=0.5, scale_prob=single_scale)
    missing_scale = _make_field(probability, coherence=0.5)
    module = HeuristicStructuralUncertainty(weights=(0.0, 0.0, 0.0, 0.9, 0.1))
    assert torch.allclose(module(field), torch.full_like(field.coherence, 0.45))
    assert torch.allclose(module(missing_scale), torch.full_like(field.coherence, 0.50))


def test_frequency_support_handles_single_candidate_and_rejects_missing_support() -> None:
    probability = torch.zeros(1, 4, 2, 2)
    probability[:, 0] = 1.0
    field = _make_field(probability)
    one_frequency = replace(
        field,
        metadata=FieldMetadata(num_orientations=4, frequency_candidates=(0.1,), estimator="stage4-test"),
    )
    assert torch.equal(HeuristicStructuralUncertainty()(one_frequency), torch.zeros_like(field.coherence))

    missing_frequency = replace(
        field,
        metadata=FieldMetadata(num_orientations=4, estimator="stage4-test"),
    )
    with pytest.raises(ValueError, match="frequency_candidates are required"):
        HeuristicStructuralUncertainty()(missing_frequency)
    zero_span = replace(
        field,
        metadata=FieldMetadata(
            num_orientations=4,
            frequency_candidates=(0.1, 0.1),
            estimator="stage4-test",
        ),
    )
    with pytest.raises(ValueError, match="positive range"):
        HeuristicStructuralUncertainty()(zero_span)
    without_frequency = HeuristicStructuralUncertainty(weights=(0.3, 0.2, 0.0, 0.5, 0.0))
    assert torch.equal(without_frequency(missing_frequency), torch.zeros_like(field.coherence))


def test_output_is_deterministic_and_preserves_dtype() -> None:
    probability = torch.tensor([0.55, 0.25, 0.15, 0.05], dtype=torch.float64).view(1, 4, 1, 1)
    scale_prob = torch.tensor([0.8, 0.2], dtype=torch.float64).view(1, 2, 1, 1)
    field = _make_field(probability, frequency_var=0.0007, coherence=0.6, scale_prob=scale_prob)
    module = HeuristicStructuralUncertainty()
    first = module(field)
    second = module(field)
    assert torch.equal(first, second)
    assert first.dtype == torch.float64
    assert first.device == field.device
    assert torch.count_nonzero(field.structural_uncertainty) == 0


def test_all_five_signals_preserve_finite_gradient_flow() -> None:
    orientation_logits = torch.randn(2, 6, 3, 4, requires_grad=True)
    orientation_prob = orientation_logits.softmax(dim=1)
    frequency_logit = torch.tensor(0.2, requires_grad=True)
    frequency_var = frequency_logit.sigmoid() * 0.002
    coherence_logit = torch.tensor(-0.3, requires_grad=True)
    coherence = coherence_logit.sigmoid()
    scale_logits = torch.randn(2, 3, 3, 4, requires_grad=True)
    scale_prob = scale_logits.softmax(dim=1)
    field = _make_field(
        orientation_prob,
        frequency_var=frequency_var,
        coherence=coherence,
        scale_prob=scale_prob,
    )

    uncertainty = HeuristicStructuralUncertainty()(field)
    uncertainty.mean().backward()
    for value in (orientation_logits, frequency_logit, coherence_logit, scale_logits):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert int(torch.count_nonzero(value.grad)) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_autocast_preserves_dtype_and_uncertainty_gradient(dtype: torch.dtype) -> None:
    orientation_logits = torch.randn(1, 8, 3, 4, dtype=dtype, requires_grad=True)
    scale_logits = torch.randn(1, 3, 3, 4, dtype=dtype, requires_grad=True)
    field = _make_field(
        orientation_logits.softmax(dim=1),
        frequency_var=0.001,
        coherence=0.6,
        scale_prob=scale_logits.softmax(dim=1),
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        uncertainty = HeuristicStructuralUncertainty()(field)
    assert uncertainty.dtype == dtype
    assert bool(((0.0 <= uncertainty) & (uncertainty <= 1.0)).all())
    uncertainty.mean().backward()
    for value in (orientation_logits, scale_logits):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert int(torch.count_nonzero(value.grad)) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_autocast_preserves_dtype_device_and_gradient(dtype: torch.dtype) -> None:
    orientation_logits = torch.randn(1, 8, 3, 4, device="cuda", dtype=dtype, requires_grad=True)
    scale_logits = torch.randn(1, 3, 3, 4, device="cuda", dtype=dtype, requires_grad=True)
    field = _make_field(
        orientation_logits.softmax(dim=1),
        frequency_var=0.001,
        coherence=0.6,
        scale_prob=scale_logits.softmax(dim=1),
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        uncertainty = HeuristicStructuralUncertainty()(field)
    assert uncertainty.dtype == dtype
    assert uncertainty.device.type == "cuda"
    assert bool(((0.0 <= uncertainty) & (uncertainty <= 1.0)).all())

    uncertainty.mean().backward()
    for value in (orientation_logits, scale_logits):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
