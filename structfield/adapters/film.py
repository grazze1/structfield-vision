"""Feature-wise linear modulation by structural features."""

from __future__ import annotations

import torch
from torch import nn

from structfield.validation import ValidationMode

from .base import AdapterOutput, StructuralAdapter, validate_channels, validate_feature_pair


class FiLMStructuralAdapter(StructuralAdapter):
    """Apply lightweight spatial FiLM while preserving backbone channels.

    A ``1x1`` projection predicts per-pixel scale and bias. The returned
    ``structural_delta`` satisfies ``output = features + structural_delta``.
    """

    def __init__(self, feature_channels: int, structural_channels: int) -> None:
        super().__init__()
        self.feature_channels = validate_channels("feature_channels", feature_channels)
        self.structural_channels = validate_channels("structural_channels", structural_channels)
        self.modulation = nn.Conv2d(self.structural_channels, 2 * self.feature_channels, kernel_size=1)

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

    def forward(
        self,
        features: torch.Tensor,
        structural_features: torch.Tensor,
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
        modulation = self.modulation(structural_features).to(dtype=features.dtype)
        scale_delta, bias = modulation.chunk(2, dim=1)
        structural_delta = features * scale_delta + bias
        output = features + structural_delta
        return AdapterOutput(
            features=output,
            structural_delta=structural_delta,
            validation_mode=validated_mode,
        )


__all__ = ["FiLMStructuralAdapter"]
