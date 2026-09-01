"""Installed-style public API smoke tests for Stage-9 metrics."""

from __future__ import annotations

import torch

from structfield.metrics import (
    AxialAngularError,
    EquivarianceError,
    FailureDetectionAUPRC,
    FailureDetectionAUROC,
    FrequencyMAE,
    OrientationCalibrationECE,
    OrientationDistributionJS,
    OrientationRecallAtK,
)
from structfield.metrics import AxialAngularError as MetricsAxialAngularError


def test_metrics_are_exported_from_stable_submodule() -> None:
    assert AxialAngularError is MetricsAxialAngularError
    modules = (
        AxialAngularError(),
        OrientationRecallAtK(angular_threshold_rad=0.1),
        FrequencyMAE(),
        OrientationDistributionJS(),
        EquivarianceError(),
        OrientationCalibrationECE(angular_threshold_rad=0.1, num_bins=5),
        FailureDetectionAUROC(),
        FailureDetectionAUPRC(),
    )
    assert all(isinstance(module, torch.nn.Module) for module in modules)
