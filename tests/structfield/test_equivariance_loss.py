"""Tests for the Stage-7 masked equivariance reference losses."""

from __future__ import annotations

import inspect

import pytest
import torch

from structfield.losses.equivariance import (
    FrequencyEquivarianceLoss,
    OrientationEquivarianceLoss,
    UncertaintyEquivarianceLoss,
)

LOSS_TYPES = (
    OrientationEquivarianceLoss,
    FrequencyEquivarianceLoss,
    UncertaintyEquivarianceLoss,
)
L1_LOSS_TYPES = (FrequencyEquivarianceLoss, UncertaintyEquivarianceLoss)


def _inputs(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.rand(
        2,
        3,
        4,
        5,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    target = torch.rand(
        2,
        3,
        4,
        5,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    mask = torch.ones(2, 1, 4, 5, device=device, dtype=torch.bool)
    return prediction, target, mask


def test_orientation_loss_matches_tiny_channel_then_mask_manual_reduction() -> None:
    prediction = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.5]]]])

    loss = OrientationEquivarianceLoss()(prediction, target, mask)
    pixel_error = (prediction - target).square().mean(dim=1, keepdim=True)
    expected = (pixel_error * mask).sum() / mask.sum()

    assert torch.equal(pixel_error, torch.tensor([[[[5.0, 10.0]]]]))
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("loss_type", L1_LOSS_TYPES)
def test_l1_losses_match_tiny_channel_then_mask_manual_reduction(
    loss_type: type[torch.nn.Module],
) -> None:
    prediction = torch.tensor([[[[-1.0, 2.0]], [[3.0, -4.0]]]])
    target = torch.tensor([[[[1.0, 1.0]], [[2.0, 0.0]]]])
    mask = torch.tensor([[[[0.25, 1.0]]]])

    loss = loss_type()(prediction, target, mask)
    pixel_error = (prediction - target).abs().mean(dim=1, keepdim=True)
    expected = (pixel_error * mask).sum() / mask.sum()

    assert torch.equal(pixel_error, torch.tensor([[[[1.5, 2.5]]]]))
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_perfect_match_is_exact_zero(loss_type: type[torch.nn.Module]) -> None:
    prediction, _, mask = _inputs()
    loss = loss_type()(prediction, prediction.clone(), mask)
    assert torch.equal(loss, torch.zeros((), dtype=prediction.dtype))


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_mask_excludes_arbitrarily_large_invalid_error(
    loss_type: type[torch.nn.Module],
) -> None:
    extreme = torch.finfo(torch.float32).max
    prediction = torch.tensor([[[[0.0, extreme]], [[0.0, -extreme]]]])
    target = torch.zeros_like(prediction)
    target[..., 1] = -prediction[..., 1]
    mask = torch.tensor([[[[True, False]]]])
    assert torch.equal(loss_type()(prediction, target, mask), torch.zeros(()))


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(1, 1, 1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 1, 1),
    ],
)
def test_all_invalid_mask_fails_fast(
    loss_type: type[torch.nn.Module],
    mask: torch.Tensor,
) -> None:
    value = torch.zeros(1, 2, 1, 1)
    with pytest.raises(ValueError, match="at least one valid"):
        loss_type()(value, value, mask)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_both_estimator_branches_receive_gradients_and_mask_does_not(
    loss_type: type[torch.nn.Module],
) -> None:
    prediction, target, _ = _inputs(requires_grad=True)
    mask = torch.ones(2, 1, 4, 5, requires_grad=True)

    module = loss_type()
    loss = module(prediction, target, mask)
    loss.backward()

    assert loss.ndim == 0
    assert list(module.parameters()) == []
    assert prediction.grad is not None
    assert target.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(torch.isfinite(target.grad).all())
    assert bool((prediction.grad != 0.0).any())
    assert bool((target.grad != 0.0).any())
    assert mask.grad is None


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_batch_channel_reduction_matches_explicit_reference(
    loss_type: type[torch.nn.Module],
) -> None:
    prediction, target, _ = _inputs()
    mask = torch.tensor(
        [
            [[[1.0] * 5] * 4],
            [[[0.25] * 5] * 4],
        ]
    )
    difference = prediction - target
    element_error = difference.square() if loss_type is OrientationEquivarianceLoss else difference.abs()
    expected = (element_error.mean(dim=1, keepdim=True) * mask).sum() / mask.sum()
    assert torch.allclose(loss_type()(prediction, target, mask), expected)


def test_every_forward_requires_explicit_equivariance_valid_mask() -> None:
    prediction, target, _ = _inputs()
    for loss_type in LOSS_TYPES:
        signature = inspect.signature(loss_type.forward)
        assert signature.parameters["equivariance_valid_mask"].default is inspect.Parameter.empty
        with pytest.raises(TypeError, match="equivariance_valid_mask"):
            loss_type()(prediction, target)  # type: ignore[call-arg]


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_cpu_uses_float32_and_backpropagates(
    loss_type: type[torch.nn.Module],
    dtype: torch.dtype,
) -> None:
    prediction, target, mask = _inputs(dtype=dtype, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = loss_type()(prediction, target, mask)
    assert loss.dtype is torch.float32
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert prediction.grad is not None
    assert target.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(torch.isfinite(target.grad).all())


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_float64_is_preserved(loss_type: type[torch.nn.Module]) -> None:
    prediction, target, mask = _inputs(dtype=torch.float64)
    assert loss_type()(prediction, target, mask).dtype is torch.float64


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_tiny_float64_mask_does_not_underflow_for_float32_inputs(
    loss_type: type[torch.nn.Module],
) -> None:
    prediction, target, _ = _inputs()
    tiny_mask = torch.full((2, 1, 4, 5), 1e-300, dtype=torch.float64)
    reference = loss_type()(prediction, target, torch.ones_like(tiny_mask))
    result = loss_type()(prediction, target, tiny_mask)
    assert torch.isfinite(result)
    assert torch.allclose(result, reference)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda value: value.long(), "floating-point"),
        (lambda value: value[:, 0], r"\[B,C,H,W\]"),
        (lambda value: value[..., :-1], "identical BCHW shapes"),
        (lambda value: value.double(), "same dtype"),
        (lambda value: value + float("nan"), "NaN or Inf"),
    ],
)
def test_prediction_target_contract_fails_fast(
    loss_type: type[torch.nn.Module],
    mutate: object,
    error: str,
) -> None:
    prediction, target, mask = _inputs()
    with pytest.raises((TypeError, ValueError), match=error):
        loss_type()(prediction, mutate(target), mask)  # type: ignore[operator]


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize(
    "mask,error",
    [
        (torch.ones(2, 4, 5, dtype=torch.bool), r"\[B,1,H,W\]"),
        (torch.ones(2, 2, 4, 5, dtype=torch.bool), r"\[B,1,H,W\]"),
        (torch.ones(2, 1, 4, 5, dtype=torch.int64), "bool or floating-point"),
        (torch.full((2, 1, 4, 5), -0.1), r"\[0,1\]"),
        (torch.full((2, 1, 4, 5), 1.1), r"\[0,1\]"),
        (torch.full((2, 1, 4, 5), float("inf")), "NaN or Inf"),
    ],
)
def test_mask_contract_fails_fast(
    loss_type: type[torch.nn.Module],
    mask: torch.Tensor,
    error: str,
) -> None:
    prediction, target, _ = _inputs()
    with pytest.raises((TypeError, ValueError), match=error):
        loss_type()(prediction, target, mask)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_cross_device_contract_fails_before_computation(loss_type: type[torch.nn.Module]) -> None:
    prediction, target, mask = _inputs()
    with pytest.raises(ValueError, match="same device"):
        loss_type()(prediction, target.to(device="meta"), mask)
    with pytest.raises(ValueError, match="same device"):
        loss_type()(prediction, target, mask.to(device="meta"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_forward_backward_and_autocast(
    loss_type: type[torch.nn.Module],
    dtype: torch.dtype,
) -> None:
    prediction, target, mask = _inputs(device="cuda", dtype=dtype, requires_grad=True)
    autocast_dtype = torch.float16 if dtype is torch.float16 else torch.bfloat16
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        loss = loss_type()(prediction, target, mask)
    expected_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    assert loss.device.type == "cuda"
    assert loss.dtype is expected_dtype
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert prediction.grad is not None
    assert target.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(torch.isfinite(target.grad).all())
