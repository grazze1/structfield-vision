"""Shared contracts for lightweight structural-feature adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from structfield.validation import ValidationMode, is_strict_validation

_SUPPORTED_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def validate_channels(name: str, value: int) -> int:
    """Return a validated positive channel count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def validate_feature_map(
    name: str,
    value: torch.Tensor,
    *,
    channels: int | None = None,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> torch.Tensor:
    """Validate a finite floating feature map with shape ``[B,C,H,W]``."""
    mode = ValidationMode.normalize(validation_mode)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(f"{name} must use a supported floating-point dtype: float16, bfloat16, float32, or float64")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,H,W]")
    if any(size < 1 for size in value.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got {value.shape[1]}")
    if mode is ValidationMode.strict and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def validate_feature_pair(
    features: torch.Tensor,
    structural_features: torch.Tensor,
    *,
    feature_channels: int | None = None,
    structural_channels: int | None = None,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> None:
    """Validate the strict adapter boundary without silently resizing or casting."""
    validate_feature_map("features", features, channels=feature_channels, validation_mode=validation_mode)
    validate_feature_map(
        "structural_features",
        structural_features,
        channels=structural_channels,
        validation_mode=validation_mode,
    )
    if features.shape[0] != structural_features.shape[0]:
        raise ValueError("features and structural_features must have the same batch size")
    if features.shape[-2:] != structural_features.shape[-2:]:
        raise ValueError("features and structural_features must have the same spatial shape")
    if features.device != structural_features.device:
        raise ValueError("features and structural_features must use the same device")
    if features.dtype != structural_features.dtype:
        raise ValueError("features and structural_features must use the same dtype")


def validate_structural_uncertainty(
    structural_uncertainty: torch.Tensor,
    features: torch.Tensor,
    *,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> None:
    """Validate scalar structural uncertainty, distinct from an adapter gate."""
    validate_feature_map(
        "structural_uncertainty",
        structural_uncertainty,
        channels=1,
        validation_mode=validation_mode,
    )
    if structural_uncertainty.shape[0] != features.shape[0]:
        raise ValueError("structural_uncertainty and features must have the same batch size")
    if structural_uncertainty.shape[-2:] != features.shape[-2:]:
        raise ValueError("structural_uncertainty and features must have the same spatial shape")
    if structural_uncertainty.device != features.device:
        raise ValueError("structural_uncertainty and features must use the same device")
    if structural_uncertainty.dtype != features.dtype:
        raise ValueError("structural_uncertainty and features must use the same dtype")
    mode = ValidationMode.normalize(validation_mode)
    if mode is ValidationMode.strict and (
        float(structural_uncertainty.detach().min()) < 0.0 or float(structural_uncertainty.detach().max()) > 1.0
    ):
        raise ValueError("structural_uncertainty must lie in [0,1]")


@dataclass(frozen=True)
class AdapterOutput:
    """Uniform output of a structural adapter.

    ``gate`` is downstream reliability and, when present, is spatial-channel
    ``[B,C,H,W]``. It is not the scalar ``structural_uncertainty`` supplied to
    :class:`ReliabilityGatedAdapter`. ``structural_delta`` is the raw, ungated
    residual contribution in feature space.
    """

    features: torch.Tensor
    gate: torch.Tensor | None = None
    structural_delta: torch.Tensor | None = None
    validation_mode: ValidationMode = ValidationMode.strict

    def __post_init__(self) -> None:
        mode = ValidationMode.normalize(self.validation_mode)
        validate_feature_map("features", self.features, validation_mode=mode)
        if self.gate is not None:
            validate_feature_map("gate", self.gate, validation_mode=mode)
            if self.gate.shape != self.features.shape:
                raise ValueError("gate must have the same shape as features")
            if self.gate.device != self.features.device or self.gate.dtype != self.features.dtype:
                raise ValueError("gate must share the features dtype and device")
            if is_strict_validation(mode) and (
                float(self.gate.detach().min()) < 0.0 or float(self.gate.detach().max()) > 1.0
            ):
                raise ValueError("gate must lie in [0,1]")
        if self.structural_delta is not None:
            validate_feature_map("structural_delta", self.structural_delta, validation_mode=mode)
            if self.structural_delta.shape != self.features.shape:
                raise ValueError("structural_delta must have the same shape as features")
            if (
                self.structural_delta.device != self.features.device
                or self.structural_delta.dtype != self.features.dtype
            ):
                raise ValueError("structural_delta must share the features dtype and device")


class StructuralAdapter(nn.Module):
    """Base type for the task-neutral tensor adapter boundary."""

    def output_channels(self, feature_channels: int, structural_channels: int) -> int:
        """Return output channels for the declared input channel counts."""
        raise NotImplementedError


__all__ = [
    "AdapterOutput",
    "StructuralAdapter",
    "validate_channels",
    "validate_feature_map",
    "validate_feature_pair",
    "validate_structural_uncertainty",
]
