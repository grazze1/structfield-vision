"""Base interfaces and input conversion for structural-field estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
from torch import nn

from structfield.core import StructuralField, StructuralFieldDiagnostics
from structfield.validation import ValidationMode, is_strict_validation

EstimatorOutput: TypeAlias = StructuralField | tuple[StructuralField, StructuralFieldDiagnostics]

LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


def validate_estimator_input(
    image: torch.Tensor,
    *,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> None:
    """Validate a grayscale or RGB floating-point BCHW image tensor."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("estimator input must be a torch.Tensor")
    if image.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise TypeError(
            "estimator input must use a supported floating-point dtype: float16, bfloat16, float32, or float64"
        )
    if image.ndim != 4 or image.shape[1] not in {1, 3}:
        raise ValueError("estimator input must have shape [B,1,H,W] or [B,3,H,W]")
    if image.shape[0] < 1 or image.shape[-2] < 1 or image.shape[-1] < 1:
        raise ValueError("estimator input dimensions must be non-empty")
    if is_strict_validation(validation_mode) and not bool(torch.isfinite(image).all()):
        raise ValueError("estimator input contains NaN or Inf")


def to_luminance(
    image: torch.Tensor,
    *,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> torch.Tensor:
    """Return grayscale input unchanged or convert RGB with fixed linear weights."""
    validate_estimator_input(image, validation_mode=validation_mode)
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor(LUMINANCE_WEIGHTS).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


class BaseStructuralFieldEstimator(nn.Module, ABC):
    """Abstract interface shared by analytic and future structural estimators."""

    @abstractmethod
    def forward(self, image: torch.Tensor, *, return_diagnostics: bool = False) -> EstimatorOutput:
        """Estimate a structural field, optionally returning analytic diagnostics."""

    def estimate_with_diagnostics(
        self,
        image: torch.Tensor,
    ) -> tuple[StructuralField, StructuralFieldDiagnostics]:
        """Typed convenience wrapper that always requests diagnostics."""
        result = self.forward(image, return_diagnostics=True)
        if not isinstance(result, tuple):
            raise RuntimeError("estimator did not return requested diagnostics")
        return result
