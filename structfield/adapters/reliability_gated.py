"""Lightweight spatial-channel reliability-gated structural adapter."""

from __future__ import annotations

import torch
from torch import nn

from structfield.validation import ValidationMode, is_strict_validation

from .base import (
    AdapterOutput,
    StructuralAdapter,
    validate_channels,
    validate_feature_map,
    validate_feature_pair,
    validate_structural_uncertainty,
)


class ReliabilityGatedAdapter(StructuralAdapter):
    """Fuse a structural residual through learned downstream reliability.

    Structural uncertainty is a scalar input map ``[B,1,H,W]``. It conditions
    the gate network but is not itself the gate. The learned downstream gate is
    spatial-channel ``[B,C,H,W]`` and the exact residual contract is::

        output = features + gate * raw_structural_delta

    Both the gate and the ungated structural delta are returned for inspection.
    """

    def __init__(self, feature_channels: int, structural_channels: int) -> None:
        super().__init__()
        self.feature_channels = validate_channels("feature_channels", feature_channels)
        self.structural_channels = validate_channels("structural_channels", structural_channels)
        self.structural_adapter = nn.Conv2d(
            self.structural_channels,
            self.feature_channels,
            kernel_size=1,
        )
        self.gate_network = nn.Conv2d(
            self.feature_channels + self.structural_channels + 1,
            self.feature_channels,
            kernel_size=1,
        )

    def output_channels(self, feature_channels: int, structural_channels: int) -> int:
        feature_channels = validate_channels("feature_channels", feature_channels)
        structural_channels = validate_channels("structural_channels", structural_channels)
        if feature_channels != self.feature_channels:
            raise ValueError(
                f"feature_channels must match configured value {self.feature_channels}, got {feature_channels}"
            )
        if structural_channels != self.structural_channels:
            raise ValueError(
                f"structural_channels must match configured value {self.structural_channels}, got {structural_channels}"
            )
        return self.feature_channels

    @staticmethod
    def fuse(
        features: torch.Tensor,
        structural_delta: torch.Tensor,
        gate: torch.Tensor,
        *,
        validation_mode: ValidationMode | str = ValidationMode.strict,
    ) -> torch.Tensor:
        """Apply the residual formula, including exact identity for a zero gate."""
        mode = ValidationMode.normalize(validation_mode)
        validate_feature_map("features", features, validation_mode=mode)
        validate_feature_map("structural_delta", structural_delta, validation_mode=mode)
        validate_feature_map("gate", gate, validation_mode=mode)
        if structural_delta.shape != features.shape or gate.shape != features.shape:
            raise ValueError("features, structural_delta, and gate must have identical shapes")
        if structural_delta.device != features.device or gate.device != features.device:
            raise ValueError("features, structural_delta, and gate must use the same device")
        if structural_delta.dtype != features.dtype or gate.dtype != features.dtype:
            raise ValueError("features, structural_delta, and gate must use the same dtype")
        if is_strict_validation(mode):
            gate_values = gate.detach()
            if float(gate_values.min()) < 0.0 or float(gate_values.max()) > 1.0:
                raise ValueError("gate must lie in [0,1]")
        return features + gate * structural_delta

    def forward(
        self,
        features: torch.Tensor,
        structural_features: torch.Tensor,
        structural_uncertainty: torch.Tensor,
        *,
        validation_mode: ValidationMode | str = ValidationMode.strict,
    ) -> AdapterOutput:
        validated_mode = ValidationMode.normalize(validation_mode)
        validate_feature_pair(
            features,
            structural_features,
            feature_channels=self.feature_channels,
            structural_channels=self.structural_channels,
            validation_mode=validated_mode,
        )
        validate_structural_uncertainty(structural_uncertainty, features, validation_mode=validated_mode)

        structural_delta = self.structural_adapter(structural_features).to(dtype=features.dtype)
        gate_inputs = torch.cat((features, structural_features, structural_uncertainty), dim=1)
        gate = torch.sigmoid(self.gate_network(gate_inputs)).to(dtype=features.dtype)
        if is_strict_validation(validated_mode):
            gate_values = gate.detach()
            if float(gate_values.min()) < 0.0 or float(gate_values.max()) > 1.0:
                raise ValueError("gate must lie in [0,1]")
        output = self.fuse(features, structural_delta, gate, validation_mode=validated_mode)
        return AdapterOutput(
            features=output,
            gate=gate,
            structural_delta=structural_delta,
            validation_mode=validated_mode,
        )


__all__ = ["ReliabilityGatedAdapter"]
