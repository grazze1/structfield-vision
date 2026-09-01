"""Fixed concatenation structural adapter."""

from __future__ import annotations

import torch

from structfield.validation import ValidationMode

from .base import AdapterOutput, StructuralAdapter, validate_channels, validate_feature_pair


class ConcatStructuralAdapter(StructuralAdapter):
    """Concatenate structural features with backbone features channel-wise."""

    def output_channels(self, feature_channels: int, structural_channels: int) -> int:
        feature_channels = validate_channels("feature_channels", feature_channels)
        structural_channels = validate_channels("structural_channels", structural_channels)
        return feature_channels + structural_channels

    def forward(
        self,
        features: torch.Tensor,
        structural_features: torch.Tensor,
        *,
        validation_mode: ValidationMode | str = ValidationMode.strict,
    ) -> AdapterOutput:
        validate_feature_pair(features, structural_features, validation_mode=validation_mode)
        return AdapterOutput(
            features=torch.cat((features, structural_features), dim=1),
            validation_mode=ValidationMode.normalize(validation_mode),
        )


__all__ = ["ConcatStructuralAdapter"]
