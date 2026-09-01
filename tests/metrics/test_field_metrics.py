"""Artificially small, hand-computable tests for Stage-9 field metrics."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from structfield.metrics.angular import AxialAngularError, OrientationRecallAtK
from structfield.metrics.field import (
    EquivarianceError,
    FrequencyMAE,
    OrientationDistributionJS,
)


def _mask(
    width: int = 1,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.ones(1, 1, 1, width, dtype=torch.bool, device=device)


def test_axial_angular_error_treats_zero_pi_and_opposite_axes_as_equal() -> None:
    prediction = torch.tensor([[[[0.0, math.pi, -math.pi, 3.0 * math.pi]]]])
    target = torch.zeros_like(prediction)

    result = AxialAngularError()(prediction, target, _mask(4))

    assert result.item() == pytest.approx(0.0, abs=5e-7)


def test_axial_angular_error_has_pi_periodic_shortest_path_and_masked_mean() -> None:
    prediction = torch.tensor([[[[math.pi - 0.1, math.pi / 2.0, 1.2]]]])
    target = torch.zeros_like(prediction)
    valid_mask = torch.tensor([[[[True, True, False]]]])

    result = AxialAngularError()(prediction, target, valid_mask)

    assert result.item() == pytest.approx((0.1 + math.pi / 2.0) / 2.0, abs=1e-6)


def test_orientation_recall_at_k_has_hand_computable_hit_and_miss() -> None:
    prediction = torch.tensor([[[[0.00, 0.00]], [[0.80, 0.80]]]])
    target = torch.tensor([[[[math.pi + 0.01, 0.40]]]])

    recall = OrientationRecallAtK(angular_threshold_rad=0.05)(
        prediction,
        target,
        _mask(2),
    )

    assert recall.item() == pytest.approx(0.5)


def test_orientation_recall_counts_active_ground_truth_modes_not_predictions() -> None:
    prediction = torch.tensor([[[[0.0]], [[1.0]]]])
    target = torch.tensor([[[[0.01]], [[0.50]], [[float("nan")]]]])
    target_count = torch.tensor([[[[2]]]])

    recall = OrientationRecallAtK(angular_threshold_rad=0.05)(
        prediction,
        target,
        _mask(),
        target_count,
    )

    assert recall.item() == pytest.approx(0.5)


def test_orientation_recall_uses_first_target_count_modes_per_pixel() -> None:
    prediction = torch.tensor([[[[0.0, 0.0]]]])
    target = torch.tensor(
        [
            [
                [[0.0, 0.0]],
                [[0.7, 0.7]],
            ]
        ]
    )
    target_count = torch.tensor([[[[1, 2]]]])

    recall = OrientationRecallAtK(angular_threshold_rad=0.01)(
        prediction,
        target,
        _mask(2),
        target_count,
    )

    assert recall.item() == pytest.approx(0.75)


def test_frequency_mae_averages_channels_before_valid_pixels() -> None:
    prediction = torch.tensor([[[[1.0, 2.0]], [[3.0, 6.0]]]])
    target = torch.zeros_like(prediction)
    valid_mask = torch.tensor([[[[True, False]]]])

    result = FrequencyMAE()(prediction, target, valid_mask)

    assert result.item() == pytest.approx(2.0)


def test_equivariance_error_matches_tiny_manual_channel_and_pixel_mean() -> None:
    prediction = torch.tensor([[[[1.0, 2.0]], [[5.0, -1.0]]]])
    target = torch.tensor([[[[0.0, 4.0]], [[3.0, 2.0]]]])

    result = EquivarianceError()(prediction, target, _mask(2))

    # Channel means are (1 + 2) / 2 and (2 + 3) / 2.
    assert result.item() == pytest.approx(2.0)


def test_orientation_distribution_js_is_zero_for_identical_distributions() -> None:
    probability = torch.tensor([[[[0.25, 0.75]], [[0.75, 0.25]]]])

    result = OrientationDistributionJS()(probability, probability.clone(), _mask(2))

    assert torch.equal(result, torch.zeros((), dtype=probability.dtype))


def test_orientation_distribution_js_is_ln2_for_disjoint_distributions() -> None:
    prediction = torch.tensor([[[[1.0]], [[0.0]]]])
    target = torch.tensor([[[[0.0]], [[1.0]]]])

    result = OrientationDistributionJS()(prediction, target, _mask())

    assert result.item() == pytest.approx(math.log(2.0), abs=1e-7)
    assert 0.0 <= result.item() <= math.log(2.0)


@pytest.mark.parametrize(
    "metric,prediction,target",
    [
        (
            AxialAngularError(),
            torch.tensor([[[[0.0, float("inf")]]]]),
            torch.tensor([[[[0.0, float("nan")]]]]),
        ),
        (
            FrequencyMAE(),
            torch.tensor([[[[1.0, torch.finfo(torch.float32).max]]]]),
            torch.tensor([[[[1.0, -torch.finfo(torch.float32).max]]]]),
        ),
        (
            EquivarianceError(),
            torch.tensor([[[[1.0, torch.finfo(torch.float32).max]]]]),
            torch.tensor([[[[1.0, -torch.finfo(torch.float32).max]]]]),
        ),
        (
            OrientationDistributionJS(),
            torch.tensor([[[[1.0, float("inf")]], [[0.0, float("nan")]]]]),
            torch.tensor([[[[1.0, float("nan")]], [[0.0, float("inf")]]]]),
        ),
    ],
)
def test_invalid_extreme_or_nonfinite_values_are_excluded_before_arithmetic(
    metric: torch.nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    valid_mask = torch.tensor([[[[True, False]]]])
    result = metric(prediction, target, valid_mask)
    assert torch.equal(result, torch.zeros((), dtype=torch.float32))


def test_orientation_recall_excludes_invalid_pixels_before_pairwise_difference() -> None:
    prediction = torch.tensor([[[[0.0, float("inf")]]]])
    target = torch.tensor([[[[math.pi, float("nan")]]]])
    valid_mask = torch.tensor([[[[True, False]]]])
    target_count = torch.tensor([[[[1, -999]]]])

    result = OrientationRecallAtK(angular_threshold_rad=0.0)(
        prediction,
        target,
        valid_mask,
        target_count,
    )

    assert torch.equal(result, torch.ones((), dtype=torch.float32))


@pytest.mark.parametrize(
    "module",
    [
        AxialAngularError(),
        OrientationRecallAtK(angular_threshold_rad=0.1),
        FrequencyMAE(),
        OrientationDistributionJS(),
        EquivarianceError(),
    ],
)
def test_metrics_are_state_free_and_require_an_explicit_valid_mask(
    module: torch.nn.Module,
) -> None:
    assert list(module.parameters()) == []
    assert list(module.buffers()) == []
    signature = inspect.signature(module.forward)
    assert signature.parameters["valid_mask"].default is inspect.Parameter.empty


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_low_precision_is_accumulated_in_float32(dtype: torch.dtype) -> None:
    valid_mask = _mask()
    angle = torch.tensor([[[[0.2]]]], dtype=dtype, requires_grad=True)
    zero_angle = torch.zeros_like(angle)
    frequency = torch.tensor([[[[1.0]], [[3.0]]]], dtype=dtype, requires_grad=True)
    zero_frequency = torch.zeros_like(frequency)
    first_probability = torch.tensor([[[[0.25]], [[0.75]]]], dtype=dtype, requires_grad=True)
    second_probability = torch.tensor([[[[0.75]], [[0.25]]]], dtype=dtype, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        angular = AxialAngularError()(angle, zero_angle, valid_mask)
        recall = OrientationRecallAtK(angular_threshold_rad=0.25)(
            angle,
            zero_angle,
            valid_mask,
        )
        frequency_error = FrequencyMAE()(frequency, zero_frequency, valid_mask)
        equivariance = EquivarianceError()(frequency, zero_frequency, valid_mask)
        js = OrientationDistributionJS()(
            first_probability,
            second_probability,
            valid_mask,
        )

    for result in (angular, recall, frequency_error, equivariance, js):
        assert result.dtype is torch.float32
        assert bool(torch.isfinite(result))
    (angular + frequency_error + equivariance + js).backward()
    for value in (angle, frequency, first_probability, second_probability):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())


def test_float64_is_preserved_by_every_metric() -> None:
    valid_mask = _mask()
    angle = torch.tensor([[[[0.2]]]], dtype=torch.float64)
    probability = torch.tensor([[[[0.25]], [[0.75]]]], dtype=torch.float64)
    reverse_probability = probability.flip(1)
    field = torch.tensor([[[[1.0]], [[3.0]]]], dtype=torch.float64)

    results = (
        AxialAngularError()(angle, torch.zeros_like(angle), valid_mask),
        OrientationRecallAtK(angular_threshold_rad=0.1)(angle, angle, valid_mask),
        FrequencyMAE()(field, torch.zeros_like(field), valid_mask),
        OrientationDistributionJS()(probability, reverse_probability, valid_mask),
        EquivarianceError()(field, torch.zeros_like(field), valid_mask),
    )
    assert all(result.dtype is torch.float64 for result in results)


def test_differentiable_metrics_have_finite_nonzero_gradients() -> None:
    valid_mask = _mask()
    angle = torch.tensor([[[[0.2]]]], requires_grad=True)
    angle_target = torch.tensor([[[[0.0]]]], requires_grad=True)
    field = torch.tensor([[[[1.0]], [[3.0]]]], requires_grad=True)
    field_target = torch.tensor([[[[0.0]], [[1.0]]]], requires_grad=True)
    probability = torch.tensor([[[[0.25]], [[0.75]]]], requires_grad=True)
    probability_target = torch.tensor([[[[0.75]], [[0.25]]]], requires_grad=True)

    loss = (
        AxialAngularError()(angle, angle_target, valid_mask)
        + FrequencyMAE()(field, field_target, valid_mask)
        + EquivarianceError()(field, field_target, valid_mask)
        + OrientationDistributionJS()(probability, probability_target, valid_mask)
    )
    loss.backward()

    for value in (
        angle,
        angle_target,
        field,
        field_target,
        probability,
        probability_target,
    ):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
        assert bool((value.grad != 0.0).any())


def test_recall_is_deliberately_nondifferentiable_but_returns_tensor_metric() -> None:
    prediction = torch.tensor([[[[0.2]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    result = OrientationRecallAtK(angular_threshold_rad=0.3)(
        prediction,
        target,
        _mask(),
    )
    assert torch.equal(result, torch.ones(()))
    assert not result.requires_grad


@pytest.mark.parametrize(
    "threshold,error",
    [
        (True, "not bool"),
        (float("nan"), "finite"),
        (-0.1, r"\[0, pi/2\]"),
        (math.pi / 2.0 + 0.01, r"\[0, pi/2\]"),
    ],
)
def test_recall_threshold_contract_fails_fast(threshold: object, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        OrientationRecallAtK(angular_threshold_rad=threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target_count,error",
    [
        (torch.ones(1, 1, 1), r"\[B,1,H,W\]"),
        (torch.ones(1, 1, 1, 1), "integer"),
        (torch.zeros(1, 1, 1, 1, dtype=torch.int64), r"\[1,2\]"),
        (torch.full((1, 1, 1, 1), 3, dtype=torch.int64), r"\[1,2\]"),
    ],
)
def test_recall_target_count_contract_fails_fast(
    target_count: torch.Tensor,
    error: str,
) -> None:
    prediction = torch.zeros(1, 1, 1, 1)
    target = torch.zeros(1, 2, 1, 1)
    with pytest.raises((TypeError, ValueError), match=error):
        OrientationRecallAtK(angular_threshold_rad=0.1)(
            prediction,
            target,
            _mask(),
            target_count,
        )


@pytest.mark.parametrize(
    "valid_mask,error",
    [
        (torch.ones(1, 1, 1, 1), "bool dtype"),
        (torch.ones(1, 1, 1, dtype=torch.bool), r"\[B,1,H,W\]"),
        (torch.zeros(1, 1, 1, 1, dtype=torch.bool), "at least one valid"),
    ],
)
def test_valid_mask_contract_is_strict(valid_mask: torch.Tensor, error: str) -> None:
    value = torch.zeros(1, 1, 1, 1)
    with pytest.raises((TypeError, ValueError), match=error):
        AxialAngularError()(value, value, valid_mask)


def test_tensor_alignment_dtype_and_valid_finiteness_fail_fast() -> None:
    value = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="same dtype"):
        FrequencyMAE()(value, value.double(), _mask())
    with pytest.raises(ValueError, match="same channel count"):
        FrequencyMAE()(value, value.expand(1, 2, 1, 1), _mask())
    with pytest.raises(ValueError, match="NaN or Inf on a valid pixel"):
        AxialAngularError()(value + float("nan"), value, _mask())
    with pytest.raises(TypeError, match="float16"):
        AxialAngularError()(value.long(), value.long(), _mask())


@pytest.mark.parametrize(
    "prediction,error",
    [
        (torch.tensor([[[[-0.1]], [[1.1]]]]), "non-negative"),
        (torch.tensor([[[[0.2]], [[0.2]]]]), "sum to one"),
    ],
)
def test_js_probability_contract_fails_fast(
    prediction: torch.Tensor,
    error: str,
) -> None:
    target = torch.tensor([[[[0.5]], [[0.5]]]])
    with pytest.raises(ValueError, match=error):
        OrientationDistributionJS()(prediction, target, _mask())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_all_metrics_run_on_cuda_with_autocast_and_expected_gradients(
    dtype: torch.dtype,
) -> None:
    device = torch.device("cuda")
    valid_mask = _mask(device=device)
    angle = torch.tensor([[[[0.2]]]], device=device, dtype=dtype, requires_grad=True)
    angle_target = torch.zeros_like(angle)
    field = torch.tensor([[[[1.0]], [[3.0]]]], device=device, dtype=dtype, requires_grad=True)
    field_target = torch.zeros_like(field)
    probability = torch.tensor([[[[0.25]], [[0.75]]]], device=device, dtype=dtype, requires_grad=True)
    probability_target = torch.tensor([[[[0.75]], [[0.25]]]], device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        angular = AxialAngularError()(angle, angle_target, valid_mask)
        recall = OrientationRecallAtK(angular_threshold_rad=0.25)(
            angle,
            angle_target,
            valid_mask,
        )
        frequency = FrequencyMAE()(field, field_target, valid_mask)
        equivariance = EquivarianceError()(field, field_target, valid_mask)
        js = OrientationDistributionJS()(probability, probability_target, valid_mask)

    expected_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    for result in (angular, recall, frequency, equivariance, js):
        assert result.device.type == "cuda"
        assert result.dtype is expected_dtype
        assert bool(torch.isfinite(result))
    (angular + frequency + equivariance + js).backward()
    for value in (angle, field, probability, probability_target):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cross_device_contract_fails_fast() -> None:
    cpu = torch.zeros(1, 1, 1, 1)
    cuda = cpu.cuda()
    with pytest.raises(ValueError, match="same device"):
        FrequencyMAE()(cpu, cuda, _mask())
    with pytest.raises(ValueError, match="same device"):
        FrequencyMAE()(cpu, cpu, _mask(device="cuda"))
