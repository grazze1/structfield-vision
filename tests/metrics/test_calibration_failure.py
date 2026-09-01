from __future__ import annotations

import inspect
import math

import pytest
import torch

from structfield.metrics.calibration import OrientationCalibrationECE
from structfield.metrics.failure import FailureDetectionAUPRC, FailureDetectionAUROC


def _bchw(
    values: list[float] | list[int] | list[bool],
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device).reshape(1, 1, 1, -1)


def _mask(length: int, *, device: str = "cpu") -> torch.Tensor:
    return torch.ones(1, 1, 1, length, dtype=torch.bool, device=device)


def test_calibration_constructor_and_forward_require_explicit_choices() -> None:
    constructor = inspect.signature(OrientationCalibrationECE.__init__)
    assert constructor.parameters["angular_threshold_rad"].default is inspect.Parameter.empty
    assert constructor.parameters["num_bins"].default is inspect.Parameter.empty
    forward = inspect.signature(OrientationCalibrationECE.forward)
    assert forward.parameters["valid_mask"].default is inspect.Parameter.empty

    with pytest.raises(TypeError, match="angular_threshold_rad"):
        OrientationCalibrationECE()  # type: ignore[call-arg]
    metric = OrientationCalibrationECE(angular_threshold_rad=0.1, num_bins=2)
    with pytest.raises(TypeError, match="valid_mask"):
        metric(_bchw([0.0]), _bchw([0.0]), _bchw([1.0]))  # type: ignore[call-arg]


def test_orientation_calibration_ece_matches_hand_calculation() -> None:
    metric = OrientationCalibrationECE(angular_threshold_rad=0.1, num_bins=2)
    predicted = _bchw([0.0, 0.0, 0.0, 0.0])
    target = _bchw([0.0, 0.3, math.pi, 0.3])
    confidence = _bchw([0.9, 0.8, 0.4, 0.2])

    # Lower bin: |0.30 - 0.50| * 2/4 = 0.10.
    # Upper bin: |0.85 - 0.50| * 2/4 = 0.175.
    result = metric(predicted, target, confidence, _mask(4))
    torch.testing.assert_close(result, torch.tensor(0.275))


def test_orientation_calibration_treats_zero_and_pi_as_axially_equivalent() -> None:
    metric = OrientationCalibrationECE(angular_threshold_rad=0.0, num_bins=4)
    predicted = _bchw([0.0, math.pi + 0.125, -math.pi + 0.25])
    target = _bchw([math.pi, 0.125, 0.25])
    confidence = _bchw([1.0, 1.0, 1.0])

    torch.testing.assert_close(
        metric(predicted, target, confidence, _mask(3)),
        torch.tensor(0.0),
        atol=0.0,
        rtol=0.0,
    )


def test_orientation_calibration_uses_inclusive_threshold() -> None:
    metric = OrientationCalibrationECE(angular_threshold_rad=0.25, num_bins=1)
    result = metric(_bchw([0.0]), _bchw([0.25]), _bchw([1.0]), _mask(1))
    torch.testing.assert_close(result, torch.tensor(0.0), atol=0.0, rtol=0.0)


def test_orientation_calibration_masks_invalid_extremes_before_arithmetic() -> None:
    metric = OrientationCalibrationECE(angular_threshold_rad=0.1, num_bins=2)
    predicted = _bchw([0.0, torch.finfo(torch.float32).max, float("nan")])
    target = _bchw([0.0, -torch.finfo(torch.float32).max, float("inf")])
    confidence = _bchw([1.0, float("inf"), -10.0])
    valid = _bchw([True, False, False], dtype=torch.bool)

    result = metric(predicted, target, confidence, valid)
    torch.testing.assert_close(result, torch.tensor(0.0), atol=0.0, rtol=0.0)


def test_orientation_calibration_wraps_valid_extremes_before_subtraction() -> None:
    limit = torch.finfo(torch.float32).max
    result = OrientationCalibrationECE(0.1, 2)(
        _bchw([limit]),
        _bchw([-limit]),
        _bchw([0.5]),
        _mask(1),
    )
    assert torch.isfinite(result)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_orientation_calibration_accumulates_in_stable_dtype(
    dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    metric = OrientationCalibrationECE(angular_threshold_rad=0.2, num_bins=3)
    result = metric(
        _bchw([0.0, 0.0], dtype=dtype),
        _bchw([0.0, 0.5], dtype=dtype),
        _bchw([0.75, 0.25], dtype=dtype),
        _mask(2),
    )
    assert result.dtype == expected_dtype
    assert torch.isfinite(result)


@pytest.mark.parametrize(
    ("threshold", "bins", "error", "message"),
    [
        (True, 2, TypeError, "real number"),
        (float("nan"), 2, ValueError, "finite"),
        (-0.1, 2, ValueError, r"\[0, pi/2\]"),
        (math.pi, 2, ValueError, r"\[0, pi/2\]"),
        (0.1, True, TypeError, "integer"),
        (0.1, 0, ValueError, "positive"),
    ],
)
def test_orientation_calibration_rejects_invalid_configuration(
    threshold: float,
    bins: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        OrientationCalibrationECE(threshold, bins)


def test_orientation_calibration_validates_inputs() -> None:
    metric = OrientationCalibrationECE(0.1, 2)
    angle = _bchw([0.0, 0.0])
    confidence = _bchw([0.5, 0.5])

    with pytest.raises(ValueError, match="1 channel"):
        metric(angle.expand(1, 2, 1, 2), angle.expand(1, 2, 1, 2), confidence.expand(1, 2, 1, 2), _mask(2))
    with pytest.raises(ValueError, match="same dtype"):
        metric(angle.double(), angle.double(), confidence, _mask(2))
    with pytest.raises(ValueError, match="at least one valid"):
        metric(angle, angle, confidence, torch.zeros_like(_mask(2)))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        metric(angle, angle, _bchw([0.5, 1.1]), _mask(2))
    with pytest.raises(ValueError, match="NaN or Inf"):
        metric(angle, _bchw([0.0, float("nan")]), confidence, _mask(2))
    with pytest.raises(TypeError, match="bool dtype"):
        metric(angle, angle, confidence, _bchw([1.0, 1.0]))


def test_failure_metrics_perfect_ranking_is_one() -> None:
    scores = _bchw([0.05, 0.2, 0.8, 0.95])
    target = _bchw([0, 0, 1, 1], dtype=torch.int64)
    valid = _mask(4)

    torch.testing.assert_close(FailureDetectionAUROC()(scores, target, valid), torch.tensor(1.0))
    torch.testing.assert_close(FailureDetectionAUPRC()(scores, target, valid), torch.tensor(1.0))


def test_failure_auroc_random_ranking_sanity_is_half() -> None:
    scores = _bchw([0.4, 0.1, 0.3, 0.2])
    target = _bchw([0, 0, 1, 1], dtype=torch.int64)
    result = FailureDetectionAUROC()(scores, target, _mask(4))
    torch.testing.assert_close(result, torch.tensor(0.5))


def test_failure_metrics_treat_ties_as_one_group() -> None:
    scores = _bchw([0.5, 0.5, 0.5, 0.5])
    target = _bchw([1, 0, 1, 0], dtype=torch.int64)
    valid = _mask(4)

    # Every positive/negative score pair ties, and AP is the prevalence.
    torch.testing.assert_close(FailureDetectionAUROC()(scores, target, valid), torch.tensor(0.5))
    torch.testing.assert_close(FailureDetectionAUPRC()(scores, target, valid), torch.tensor(0.5))


def test_failure_metrics_match_tiny_manual_values() -> None:
    scores = _bchw([0.9, 0.8, 0.7, 0.6])
    target = _bchw([1, 0, 1, 0], dtype=torch.int64)
    valid = _mask(4)

    # Three of four positive-negative pairs are favorable.
    torch.testing.assert_close(FailureDetectionAUROC()(scores, target, valid), torch.tensor(0.75))
    # AP = 1 * 1/2 + (2/3) * 1/2.
    torch.testing.assert_close(FailureDetectionAUPRC()(scores, target, valid), torch.tensor(5.0 / 6.0))


def test_failure_metrics_mask_invalid_extremes_and_labels_first() -> None:
    scores = _bchw([0.1, 0.2, 0.8, 0.9, float("inf"), float("nan")])
    target = _bchw([0.0, 0.0, 1.0, 1.0, 7.0, float("nan")])
    valid = _bchw([True, True, True, True, False, False], dtype=torch.bool)

    torch.testing.assert_close(FailureDetectionAUROC()(scores, target, valid), torch.tensor(1.0))
    torch.testing.assert_close(FailureDetectionAUPRC()(scores, target, valid), torch.tensor(1.0))


def test_failure_target_is_caller_defined_and_required() -> None:
    scores = _bchw([0.1, 0.2, 0.8, 0.9])
    target = _bchw([0, 0, 1, 1], dtype=torch.int64)
    reversed_target = 1 - target
    valid = _mask(4)

    metric = FailureDetectionAUROC()
    torch.testing.assert_close(metric(scores, target, valid), torch.tensor(1.0))
    torch.testing.assert_close(metric(scores, reversed_target, valid), torch.tensor(0.0))
    signature = inspect.signature(FailureDetectionAUROC.forward)
    assert signature.parameters["failure_target"].default is inspect.Parameter.empty
    assert signature.parameters["valid_mask"].default is inspect.Parameter.empty
    assert "threshold" not in signature.parameters
    with pytest.raises(TypeError, match="valid_mask"):
        metric(scores, target)  # type: ignore[call-arg]


def test_auprc_all_positive_behavior_is_explicitly_defined() -> None:
    scores = _bchw([0.1, 0.4, 0.9])
    target = _bchw([1, 1, 1], dtype=torch.int64)
    valid = _mask(3)
    torch.testing.assert_close(FailureDetectionAUPRC()(scores, target, valid), torch.tensor(1.0))
    with pytest.raises(ValueError, match="positive and one negative"):
        FailureDetectionAUROC()(scores, target, valid)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_failure_metrics_accumulate_in_stable_dtype(
    dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    scores = _bchw([0.1, 0.2, 0.8, 0.9], dtype=dtype)
    target = _bchw([0, 0, 1, 1], dtype=torch.int64)
    valid = _mask(4)
    for metric in (FailureDetectionAUROC(), FailureDetectionAUPRC()):
        result = metric(scores, target, valid)
        assert result.dtype == expected_dtype
        torch.testing.assert_close(result, torch.ones((), dtype=expected_dtype))


def test_failure_metrics_validate_labels_classes_and_mask() -> None:
    scores = _bchw([0.1, 0.9])
    valid = _mask(2)
    with pytest.raises(ValueError, match="only explicit 0/1"):
        FailureDetectionAUROC()(scores, _bchw([0, 2], dtype=torch.int64), valid)
    with pytest.raises(ValueError, match="positive and one negative"):
        FailureDetectionAUROC()(scores, _bchw([0, 0], dtype=torch.int64), valid)
    with pytest.raises(ValueError, match="at least one positive"):
        FailureDetectionAUPRC()(scores, _bchw([0, 0], dtype=torch.int64), valid)
    with pytest.raises(ValueError, match="at least one valid"):
        FailureDetectionAUPRC()(scores, _bchw([0, 1], dtype=torch.int64), torch.zeros_like(valid))
    with pytest.raises(ValueError, match="NaN or Inf"):
        FailureDetectionAUROC()(_bchw([0.1, float("nan")]), _bchw([0, 1]), valid)
    with pytest.raises(ValueError, match=r"\[B,1,H,W\]"):
        FailureDetectionAUROC()(scores, torch.tensor([0, 1]), valid)
    float8_dtype = getattr(torch, "float8_e4m3fn", None)
    if float8_dtype is not None:
        with pytest.raises(TypeError, match="float16, bfloat16, float32, or float64"):
            FailureDetectionAUROC()(
                scores,
                torch.empty_like(scores, dtype=float8_dtype),
                valid,
            )


def test_metrics_are_parameter_free_and_have_no_accumulated_state() -> None:
    metrics = (
        OrientationCalibrationECE(0.1, 4),
        FailureDetectionAUROC(),
        FailureDetectionAUPRC(),
    )
    for metric in metrics:
        assert list(metric.parameters()) == []
        assert metric.state_dict() == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_calibration_and_failure_metrics_on_cuda_with_autocast(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 is unavailable")
    device = "cuda"
    predicted = _bchw([0.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    target_angle = _bchw([0.0, 0.3, math.pi, 0.3], dtype=dtype, device=device)
    confidence = _bchw([0.9, 0.8, 0.4, 0.2], dtype=dtype, device=device)
    scores = _bchw([0.1, 0.2, 0.8, 0.9], dtype=dtype, device=device)
    failure_target = _bchw([0, 0, 1, 1], dtype=torch.int64, device=device)
    valid = _mask(4, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        ece = OrientationCalibrationECE(0.1, 2)(predicted, target_angle, confidence, valid)
        auroc = FailureDetectionAUROC()(scores, failure_target, valid)
        auprc = FailureDetectionAUPRC()(scores, failure_target, valid)
    expected_dtype = torch.float32
    assert ece.device.type == auroc.device.type == auprc.device.type == "cuda"
    assert ece.dtype == auroc.dtype == auprc.dtype == expected_dtype
    assert torch.isfinite(ece)
    torch.testing.assert_close(auroc, torch.ones((), device=device))
    torch.testing.assert_close(auprc, torch.ones((), device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_metrics_reject_cross_device_inputs() -> None:
    cpu = _bchw([0.0, 0.0])
    gpu = cpu.cuda()
    valid_gpu = _mask(2, device="cuda")
    with pytest.raises(ValueError, match="same device"):
        OrientationCalibrationECE(0.1, 2)(gpu, cpu, gpu, valid_gpu)
    with pytest.raises(ValueError, match="same device"):
        FailureDetectionAUROC()(gpu, _bchw([0, 1]), valid_gpu)
