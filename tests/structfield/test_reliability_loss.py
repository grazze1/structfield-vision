"""Tests for the Stage-6 reference reliability objective."""

from __future__ import annotations

import math

import pytest
import torch

from structfield.losses import ReliabilityLoss, ReliabilityLossConfig, ReliabilityTargets


def _maps(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.full(
        (2, 3, 4, 5),
        0.6,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    orientation = torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
    frequency = torch.rand(2, 1, 4, 5, device=device, dtype=dtype)
    return prediction, orientation, frequency


def test_targets_implement_the_documented_soft_formula_and_detach_errors() -> None:
    config = ReliabilityLossConfig(
        orientation_error_weight=2.0,
        frequency_error_weight=0.5,
        tau=0.4,
        threshold=0.3,
    )
    module = ReliabilityLoss(config)
    orientation = torch.tensor([[[[0.1, 0.4]]]], requires_grad=True)
    frequency = torch.tensor([[[[0.2, 0.6]]]], requires_grad=True)

    targets = module.targets(orientation, frequency)
    combined = (2.0 * (orientation.detach() / (math.pi / 2.0)) + 0.5 * frequency.detach()) / 2.5
    expected = torch.exp(-combined / 0.4)

    assert isinstance(targets, ReliabilityTargets)
    assert torch.equal(targets.combined_error, combined)
    assert torch.equal(targets.soft_reliability, expected)
    assert torch.equal(targets.is_reliable, expected >= 0.3)
    assert not targets.combined_error.requires_grad
    assert not targets.soft_reliability.requires_grad


def test_soft_reliability_decreases_monotonically_with_error() -> None:
    error = torch.tensor([[[[0.0, 0.1, 0.3, 1.0]]]])
    target = ReliabilityLoss().targets(error, torch.zeros_like(error)).soft_reliability
    assert target[0, 0, 0, 0] == 1.0
    assert bool((target[..., 1:] < target[..., :-1]).all())
    assert bool(((0.0 < target) & (target <= 1.0)).all())


def test_pi_over_two_is_unit_normalized_axial_orientation_error() -> None:
    config = ReliabilityLossConfig(
        orientation_error_weight=1.0,
        frequency_error_weight=0.0,
        tau=0.5,
    )
    orientation = torch.tensor([[[[math.pi / 2.0]]]])
    targets = ReliabilityLoss(config).targets(orientation, torch.zeros_like(orientation))
    assert torch.allclose(targets.combined_error, torch.ones_like(targets.combined_error))
    assert torch.allclose(
        targets.soft_reliability,
        torch.full_like(targets.soft_reliability, math.exp(-2.0)),
    )


def test_threshold_controls_region_weighting_without_hardening_target() -> None:
    orientation_for_half_reliability = math.log(2.0) * (math.pi / 2.0)
    config = ReliabilityLossConfig(
        orientation_error_weight=1.0,
        frequency_error_weight=0.0,
        reliable_sample_weight=2.0,
        unreliable_sample_weight=5.0,
        tau=1.0,
        threshold=0.75,
    )
    prediction = torch.tensor([[[[0.25, 0.25]], [[0.75, 0.75]]]])
    orientation = torch.tensor([[[[0.0, orientation_for_half_reliability]]]])
    frequency = torch.zeros_like(orientation)

    loss = ReliabilityLoss(config)(prediction, orientation, frequency)
    soft = torch.tensor([[[[1.0, 0.5]]]]).expand_as(prediction)
    weights = torch.tensor([[[[2.0, 5.0]]]]).expand_as(prediction)
    expected = (weights * (prediction - soft).square()).sum() / weights.sum()
    assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-6)

    all_reliable = ReliabilityLossConfig(
        orientation_error_weight=1.0,
        frequency_error_weight=0.0,
        reliable_sample_weight=2.0,
        unreliable_sample_weight=5.0,
        tau=1.0,
        threshold=0.4,
    )
    changed_threshold_module = ReliabilityLoss(all_reliable)
    changed_threshold_loss = changed_threshold_module(prediction, orientation, frequency)
    assert not torch.isclose(loss, changed_threshold_loss)
    original_targets = ReliabilityLoss(config).targets(orientation, frequency)
    changed_targets = changed_threshold_module.targets(orientation, frequency)
    assert torch.equal(original_targets.soft_reliability, changed_targets.soft_reliability)
    assert not torch.equal(original_targets.is_reliable, changed_targets.is_reliable)


def test_common_error_weight_scaling_does_not_change_target() -> None:
    orientation = torch.tensor([[[[0.2, 0.7]]]])
    frequency = torch.tensor([[[[0.4, 0.1]]]])
    first = ReliabilityLoss(ReliabilityLossConfig(orientation_error_weight=2.0, frequency_error_weight=3.0)).targets(
        orientation, frequency
    )
    second = ReliabilityLoss(ReliabilityLossConfig(orientation_error_weight=20.0, frequency_error_weight=30.0)).targets(
        orientation, frequency
    )
    assert torch.allclose(first.combined_error, second.combined_error)
    assert torch.allclose(first.soft_reliability, second.soft_reliability)


def test_matching_soft_target_has_exactly_zero_loss() -> None:
    _, orientation, frequency = _maps()
    module = ReliabilityLoss()
    prediction = module.targets(orientation, frequency).soft_reliability.expand(-1, 3, -1, -1)
    assert torch.equal(module(prediction, orientation, frequency), torch.zeros(()))


def test_loss_is_parameter_free_and_backpropagates_to_prediction_only() -> None:
    prediction, orientation, frequency = _maps(requires_grad=True)
    orientation.requires_grad_(True)
    frequency.requires_grad_(True)
    module = ReliabilityLoss()

    loss = module(prediction, orientation, frequency)
    loss.backward()

    assert loss.ndim == 0
    assert list(module.parameters()) == []
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert orientation.grad is None
    assert frequency.grad is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_cpu_inputs_use_float32_working_loss(dtype: torch.dtype) -> None:
    prediction, orientation, frequency = _maps(dtype=dtype, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = ReliabilityLoss()(prediction, orientation, frequency)
    assert loss.dtype is torch.float32
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert prediction.grad is not None


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"orientation_error_weight": -1.0}, "non-negative"),
        ({"orientation_error_weight": 0.0, "frequency_error_weight": 0.0}, "positive"),
        ({"reliable_sample_weight": 0.0}, "positive"),
        ({"unreliable_sample_weight": -1.0}, "positive"),
        ({"tau": 0.0}, "positive"),
        ({"threshold": 1.1}, r"\[0,1\]"),
        ({"eps": 0.0}, "positive"),
        ({"tau": float("inf")}, "finite"),
        ({"threshold": True}, "not bool"),
    ],
)
def test_config_rejects_invalid_constants(kwargs: dict[str, float], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        ReliabilityLossConfig(**kwargs)


def test_constructor_requires_typed_config() -> None:
    with pytest.raises(TypeError, match="ReliabilityLossConfig"):
        ReliabilityLoss({})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda value: value.long(), "floating-point"),
        (lambda value: value[:, 0], r"\[B,1,H,W\]"),
        (lambda value: value.expand(-1, 2, -1, -1), r"\[B,1,H,W\]"),
        (lambda value: value - 2.0, "non-negative"),
        (lambda value: value + float("nan"), "NaN or Inf"),
    ],
)
def test_error_map_contract_fails_fast(mutate: object, error: str) -> None:
    prediction, orientation, frequency = _maps()
    with pytest.raises((TypeError, ValueError), match=error):
        ReliabilityLoss()(prediction, mutate(orientation), frequency)  # type: ignore[operator]


def test_axial_orientation_error_rejects_values_above_pi_over_two() -> None:
    prediction, orientation, frequency = _maps()
    orientation[0, 0, 0, 0] = math.pi / 2.0 + 0.1
    with pytest.raises(ValueError, match="must not exceed"):
        ReliabilityLoss()(prediction, orientation, frequency)


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda value: value.long(), "floating-point"),
        (lambda value: value[:, 0], r"\[B,C,H,W\]"),
        (lambda value: value + 1.0, r"\[0,1\]"),
        (lambda value: value + float("inf"), "NaN or Inf"),
    ],
)
def test_prediction_contract_fails_fast(mutate: object, error: str) -> None:
    prediction, orientation, frequency = _maps()
    with pytest.raises((TypeError, ValueError), match=error):
        ReliabilityLoss()(mutate(prediction), orientation, frequency)  # type: ignore[operator]


def test_cross_input_contracts_fail_fast() -> None:
    prediction, orientation, frequency = _maps()
    with pytest.raises(ValueError, match="identical shapes"):
        ReliabilityLoss()(prediction, orientation[..., :-1], frequency)
    with pytest.raises(ValueError, match="same batch size"):
        ReliabilityLoss()(prediction[:1], orientation, frequency)
    with pytest.raises(ValueError, match="same spatial shape"):
        ReliabilityLoss()(prediction[..., :-1], orientation, frequency)
    with pytest.raises(ValueError, match="same dtype"):
        ReliabilityLoss()(prediction.double(), orientation, frequency)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_forward_backward_and_autocast(dtype: torch.dtype) -> None:
    prediction, orientation, frequency = _maps(device="cuda", dtype=dtype, requires_grad=True)
    autocast_dtype = torch.float16 if dtype is torch.float16 else torch.bfloat16
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        loss = ReliabilityLoss()(prediction, orientation, frequency)
    expected_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    assert loss.device.type == "cuda"
    assert loss.dtype is expected_dtype
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
