"""Contracts for Stage 6 structural-prior corruption functions."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields, replace
from typing import Callable

import pytest
import torch

from structfield import FieldMetadata, StructuralField
from structfield.corruption import (
    CorruptionMetadata,
    coherence_noise,
    field_dropout,
    frequency_noise,
    frequency_scale,
    mode_swap,
    orientation_noise,
    orientation_shift,
    scale_shift,
    spatial_shift,
)

CorruptionCall = Callable[[StructuralField], tuple[StructuralField, CorruptionMetadata]]


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    working = probability.float() if probability.dtype in {torch.float16, torch.bfloat16} else probability
    tiny = torch.finfo(working.dtype).tiny
    entropy = -(working * working.clamp_min(tiny).log()).sum(dim=1, keepdim=True)
    return (entropy / math.log(probability.shape[1])).to(dtype=probability.dtype)


def _make_field(
    *,
    batch: int = 4,
    height: int = 3,
    width: int = 5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
    with_scale: bool = True,
) -> StructuralField:
    bins = 8
    probability = torch.zeros(batch, bins, height, width, device=device, dtype=dtype)
    probability[:, 0] = 0.50
    probability[:, 4] = 0.25
    probability[:, 2] = 0.125
    probability[:, 6] = 0.125
    grid = torch.linspace(0.0, 1.0, height * width, device=device, dtype=dtype).view(1, 1, height, width)
    grid = grid.expand(batch, 1, height, width).clone()
    frequency_mean = 0.08 + 0.02 * grid
    frequency_var = torch.full_like(grid, 0.001)
    coherence = 0.25 + 0.50 * grid
    structural_uncertainty = 0.10 + 0.20 * grid
    orientation_entropy = _entropy(probability)

    scale_candidates: tuple[float, ...] = ()
    scale_probability = None
    selected_scale = None
    if with_scale:
        scale_candidates = (1.0, 2.0, 4.0)
        scale_probability = torch.empty(batch, 3, height, width, device=device, dtype=dtype)
        scale_probability[:, 0] = 0.25
        scale_probability[:, 1] = 0.50
        scale_probability[:, 2] = 0.25
        selected_scale = torch.full_like(grid, 2.25)

    tensor_values = [
        probability,
        frequency_mean,
        frequency_var,
        coherence,
        orientation_entropy,
        structural_uncertainty,
        scale_probability,
        selected_scale,
    ]
    if requires_grad:
        for value in tensor_values:
            if value is not None:
                value.requires_grad_(True)

    return StructuralField(
        orientation_prob=probability,
        frequency_mean=frequency_mean,
        frequency_var=frequency_var,
        coherence=coherence,
        orientation_entropy=orientation_entropy,
        structural_uncertainty=structural_uncertainty,
        metadata=FieldMetadata(
            num_orientations=bins,
            frequency_candidates=(0.05, 0.10, 0.15),
            scale_candidates=scale_candidates,
            estimator="stage6-test",
        ),
        scale_prob=scale_probability,
        selected_scale=selected_scale,
    )


def _tensor_items(field: StructuralField) -> dict[str, torch.Tensor]:
    return {
        item.name: value for item in fields(field) if isinstance((value := getattr(field, item.name)), torch.Tensor)
    }


def _assert_same_field(first: StructuralField, second: StructuralField) -> None:
    assert first.metadata == second.metadata
    assert _tensor_items(first).keys() == _tensor_items(second).keys()
    for name, value in _tensor_items(first).items():
        assert torch.equal(value, getattr(second, name)), name


CORRUPTIONS: list[tuple[str, CorruptionCall]] = [
    ("orientation_shift", lambda field: orientation_shift(field, 0.2)),
    ("orientation_noise", lambda field: orientation_noise(field, 0.1, seed=7)),
    ("frequency_scale", lambda field: frequency_scale(field, 1.25)),
    ("frequency_noise", lambda field: frequency_noise(field, 0.01, seed=7)),
    ("spatial_shift", lambda field: spatial_shift(field, 1, -2)),
    ("scale_shift", lambda field: scale_shift(field, 1)),
    ("coherence_noise", lambda field: coherence_noise(field, 0.1, seed=7)),
    ("mode_swap", mode_swap),
    ("field_dropout", lambda field: field_dropout(field, 0.5, seed=7)),
]


@pytest.mark.parametrize(("expected_name", "corrupt"), CORRUPTIONS)
def test_corruptions_are_pure_valid_and_independently_stored(
    expected_name: str,
    corrupt: CorruptionCall,
) -> None:
    field = _make_field(batch=8)
    snapshot = {name: value.clone() for name, value in _tensor_items(field).items()}
    corrupted, metadata = corrupt(field)

    assert corrupted.validate() is corrupted
    assert metadata.name == expected_name
    assert metadata.schema_version == "0.2"
    for name, original in _tensor_items(field).items():
        assert torch.equal(original, snapshot[name]), name
        result = getattr(corrupted, name)
        assert result.dtype == original.dtype
        assert result.device == original.device
        assert result.data_ptr() != original.data_ptr()


def test_corruption_metadata_is_frozen_typed_and_json_compatible() -> None:
    _, metadata = orientation_noise(_make_field(), 0.125, seed=19)
    assert metadata.name == "orientation_noise"
    assert metadata.seed == 19
    assert metadata.parameter_dict["std_radians"] == pytest.approx(0.125)
    assert metadata.parameter_dict["distribution"] == "gaussian"
    assert metadata.to_dict()["parameters"] == metadata.parameter_dict
    with pytest.raises(FrozenInstanceError):
        setattr(metadata, "seed", 20)
    copied_parameters = metadata.parameter_dict
    copied_parameters["std_radians"] = 2.0
    assert metadata.parameter_dict["std_radians"] == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("corrupt", "tensor_name"),
    [
        (lambda field, seed: orientation_noise(field, 0.2, seed=seed), "orientation_prob"),
        (lambda field, seed: frequency_noise(field, 0.02, seed=seed), "frequency_mean"),
        (lambda field, seed: coherence_noise(field, 0.2, seed=seed), "coherence"),
        (lambda field, seed: field_dropout(field, 0.5, seed=seed), "orientation_prob"),
    ],
)
def test_stochastic_corruptions_are_seed_deterministic_and_seed_sensitive(
    corrupt: Callable[[StructuralField, int], tuple[StructuralField, CorruptionMetadata]],
    tensor_name: str,
) -> None:
    field = _make_field(batch=64)
    first, first_metadata = corrupt(field, 123)
    repeated, repeated_metadata = corrupt(field, 123)
    different, different_metadata = corrupt(field, 456)

    _assert_same_field(first, repeated)
    assert first_metadata == repeated_metadata
    assert not torch.equal(getattr(first, tensor_name), getattr(different, tensor_name))
    assert different_metadata.seed == 456


def test_local_generators_do_not_pollute_global_cpu_rng() -> None:
    field = _make_field(batch=16)
    torch.manual_seed(2026)
    initial_state = torch.random.get_rng_state().clone()
    orientation_noise(field, 0.2, seed=1)
    frequency_noise(field, 0.02, seed=2)
    coherence_noise(field, 0.2, seed=3)
    field_dropout(field, 0.5, seed=4)
    assert torch.equal(torch.random.get_rng_state(), initial_state)


def test_orientation_shift_is_axially_periodic_and_has_expected_direction() -> None:
    field = _make_field()
    identity, _ = orientation_shift(field, math.pi)
    _assert_same_field(identity, field)

    shift = math.pi / field.metadata.num_orientations
    shifted, metadata = orientation_shift(field, shift)
    assert torch.allclose(shifted.orientation_prob, torch.roll(field.orientation_prob, 1, dims=1), atol=1e-6)
    shifted_top1 = shifted.topk_orientations(k=1).angles[:, 0]
    assert torch.allclose(shifted_top1, torch.full_like(shifted_top1, shift))
    assert metadata.parameter_dict["period_radians"] == pytest.approx(math.pi)

    arbitrary, _ = orientation_shift(field, 0.217)
    periodic, _ = orientation_shift(field, 0.217 + math.pi)
    _assert_same_field(arbitrary, periodic)


def test_frequency_scale_transforms_mean_and_variance_in_consistent_units() -> None:
    field = _make_field()
    corrupted, metadata = frequency_scale(field, 1.5)
    assert torch.allclose(corrupted.frequency_mean, field.frequency_mean * 1.5)
    assert torch.allclose(corrupted.frequency_var, field.frequency_var * 1.5**2)
    assert metadata.parameter_dict == {"factor": 1.5, "variance_factor": 2.25}


def test_spatial_shift_rolls_every_tensor_with_one_boundary_rule() -> None:
    field = _make_field()
    corrupted, metadata = spatial_shift(field, 1, -2)
    for name, value in _tensor_items(field).items():
        assert torch.equal(getattr(corrupted, name), torch.roll(value, (1, -2), dims=(-2, -1)))
    assert metadata.parameter_dict["boundary"] == "circular"


def test_scale_shift_clamps_mass_at_endpoints_and_recomputes_selected_scale() -> None:
    field = _make_field()
    upward, metadata = scale_shift(field, 1)
    expected_upward = field.scale_prob.new_tensor((0.0, 0.25, 0.75)).view(1, 3, 1, 1)
    assert torch.allclose(upward.scale_prob, expected_upward.expand_as(upward.scale_prob))
    assert torch.allclose(upward.selected_scale, torch.full_like(upward.selected_scale, 3.5))
    assert metadata.parameter_dict["boundary"] == "clamp-accumulate"

    downward, _ = scale_shift(field, -1)
    expected_downward = field.scale_prob.new_tensor((0.75, 0.25, 0.0)).view(1, 3, 1, 1)
    assert torch.allclose(downward.scale_prob, expected_downward.expand_as(downward.scale_prob))
    assert torch.allclose(downward.selected_scale, torch.full_like(downward.selected_scale, 1.25))


def test_scale_shift_requires_an_available_scale_posterior() -> None:
    with pytest.raises(ValueError, match="requires scale_prob"):
        scale_shift(_make_field(with_scale=False), 1)


def test_noise_clamping_preserves_schema_bounds() -> None:
    field = _make_field(batch=16)
    frequency, _ = frequency_noise(field, 100.0, seed=1)
    coherence, _ = coherence_noise(field, 100.0, seed=2)
    assert float(frequency.frequency_mean.min()) >= 0.0
    assert bool(((0.0 <= coherence.coherence) & (coherence.coherence <= 1.0)).all())


def test_mode_swap_exchanges_top_modes_and_corresponding_probability_mass() -> None:
    field = _make_field()
    swapped, metadata = mode_swap(field)
    assert torch.equal(swapped.orientation_prob[:, 0], field.orientation_prob[:, 4])
    assert torch.equal(swapped.orientation_prob[:, 4], field.orientation_prob[:, 0])
    assert torch.equal(swapped.topk_orientations(k=2).angles[:, 0], field.topk_orientations(k=2).angles[:, 1])
    assert torch.equal(swapped.topk_orientations(k=2).angles[:, 1], field.topk_orientations(k=2).angles[:, 0])
    assert torch.equal(swapped.orientation_entropy, field.orientation_entropy)
    assert metadata.parameter_dict == {"first_mode": 0, "second_mode": 1}


def test_full_field_dropout_produces_a_neutral_schema_valid_prior() -> None:
    field = _make_field(batch=8)
    dropped, metadata = field_dropout(field, 1.0, seed=17)
    assert torch.allclose(
        dropped.orientation_prob,
        torch.full_like(dropped.orientation_prob, 1.0 / dropped.metadata.num_orientations),
    )
    assert torch.equal(dropped.frequency_mean, torch.zeros_like(dropped.frequency_mean))
    assert torch.equal(dropped.frequency_var, torch.zeros_like(dropped.frequency_var))
    assert torch.equal(dropped.coherence, torch.zeros_like(dropped.coherence))
    assert torch.equal(dropped.orientation_entropy, torch.ones_like(dropped.orientation_entropy))
    assert torch.equal(dropped.structural_uncertainty, torch.ones_like(dropped.structural_uncertainty))
    assert dropped.scale_prob is not None
    assert torch.allclose(dropped.scale_prob, torch.full_like(dropped.scale_prob, 1.0 / 3.0))
    assert dropped.selected_scale is not None
    assert torch.allclose(dropped.selected_scale, torch.full_like(dropped.selected_scale, 7.0 / 3.0))
    assert metadata.parameter_dict["granularity"] == "sample"
    assert metadata.parameter_dict["replacement"] == "neutral-valid-field"

    retained, _ = field_dropout(field, 0.0, seed=17)
    _assert_same_field(retained, field)


@pytest.mark.parametrize(("expected_name", "corrupt"), CORRUPTIONS)
def test_corruptions_preserve_autograd_where_meaningful(
    expected_name: str,
    corrupt: CorruptionCall,
) -> None:
    field = _make_field(batch=8, requires_grad=True)
    corrupted, metadata = corrupt(field)
    loss = sum(value.float().square().mean() for value in _tensor_items(corrupted).values())
    loss.backward()

    assert metadata.name == expected_name
    replaced_derived_inputs = {
        "orientation_shift": {"orientation_entropy"},
        "orientation_noise": {"orientation_entropy"},
        "scale_shift": {"selected_scale"},
    }.get(expected_name, set())
    for name, value in _tensor_items(field).items():
        if name in replaced_derived_inputs:
            assert value.grad is None, name
            continue
        assert value.grad is not None, name
        assert bool(torch.isfinite(value.grad).all()), name


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_corruptions_preserve_cpu_low_precision_contract(dtype: torch.dtype) -> None:
    field = _make_field(dtype=dtype, batch=8)
    for _, corrupt in CORRUPTIONS:
        corrupted, _ = corrupt(field)
        assert corrupted.validate() is corrupted
        assert all(value.dtype == dtype for value in _tensor_items(corrupted).values())


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda field: orientation_noise(field, -0.1, seed=1), "non-negative"),
        (lambda field: frequency_scale(field, 0.0), "positive"),
        (lambda field: frequency_noise(field, -0.1, seed=1), "non-negative"),
        (lambda field: coherence_noise(field, -0.1, seed=1), "non-negative"),
        (lambda field: field_dropout(field, 1.1, seed=1), r"\[0,1\]"),
        (lambda field: mode_swap(field, 0, 0), "distinct"),
    ],
)
def test_invalid_corruption_parameters_fail_fast(call: CorruptionCall, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        call(_make_field())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_corruptions_cuda_dtype_device_rng_and_backward(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is not supported")
    field = _make_field(device="cuda", dtype=dtype, batch=8, requires_grad=True)
    initial_rng_state = torch.cuda.get_rng_state().clone()
    outputs = [corrupt(field)[0] for _, corrupt in CORRUPTIONS]
    assert torch.equal(torch.cuda.get_rng_state(), initial_rng_state)

    for corrupted in outputs:
        assert corrupted.validate() is corrupted
        for value in _tensor_items(corrupted).values():
            assert value.device.type == "cuda"
            assert value.dtype == dtype
    loss = sum(value.float().square().mean() for corrupted in outputs for value in _tensor_items(corrupted).values())
    loss.backward()
    for name, value in _tensor_items(field).items():
        assert value.grad is not None, name
        assert bool(torch.isfinite(value.grad).all()), name


def test_scale_shift_rejects_missing_scale_even_if_metadata_lists_candidates() -> None:
    field = _make_field()
    without_scale = replace(field, scale_prob=None, selected_scale=None)
    with pytest.raises(ValueError, match="requires scale_prob"):
        scale_shift(without_scale, 1)
