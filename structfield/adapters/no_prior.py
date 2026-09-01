"""No-structural-prior baseline adapter."""

from __future__ import annotations

import torch

from structfield.validation import ValidationMode

from .base import AdapterOutput, StructuralAdapter, validate_channels, validate_feature_map, validate_feature_pair


class NoPriorAdapter(StructuralAdapter):
    """Return backbone features unchanged for the no-prior ablation."""

    def output_channels(self, feature_channels: int, structural_channels: int = 1) -> int:
        feature_channels = validate_channels("feature_channels", feature_channels)
        validate_channels("structural_channels", structural_channels)
        return feature_channels

    def forward(
        self,
        features: torch.Tensor,
        structural_features: torch.Tensor | None = None,
        *,
        validation_mode: ValidationMode | str = ValidationMode.strict,
    ) -> AdapterOutput:
        """Return ``features`` exactly; an optional prior is validated then ignored."""
        validated_mode = ValidationMode.normalize(validation_mode)
        if structural_features is None:
            validate_feature_map("features", features, validation_mode=validated_mode)
        else:
            validate_feature_pair(features, structural_features, validation_mode=validated_mode)
        return AdapterOutput(features=features, validation_mode=validated_mode)


__all__ = ["NoPriorAdapter"]
