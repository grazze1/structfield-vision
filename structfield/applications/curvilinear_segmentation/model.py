"""Curvilinear segmentation with optional StructuralField conditioning.

The default model is a one-channel, three-level U-Net baseline.  Conditioned
variants inject the fixed, public eight-channel curvilinear representation at
the image boundary through the task-neutral Stage-5 adapters.
"""

from __future__ import annotations

import torch
from torch import nn

from structfield.applications._shared import (
    CURVILINEAR_FEATURE_SCHEMA,
    StructuralFusion,
    UNetBackbone,
    curvilinear_structural_features,
    validate_image_and_field,
)
from structfield.core import StructuralField
from structfield.validation import ValidationMode, is_strict_validation


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_image(
    image: torch.Tensor,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if not torch.is_floating_point(image):
        raise TypeError("image must use a floating-point dtype")
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image must have shape [B,1,H,W]")
    if image.shape[0] < 1 or image.shape[-2] < 8 or image.shape[-1] < 8:
        raise ValueError("image must have non-empty batch and spatial dimensions at least 8x8")
    if is_strict_validation(validation_mode) and not bool(torch.isfinite(image).all()):
        raise ValueError("image contains NaN or Inf")


class CurvilinearSegmenter(nn.Module):
    """Return full-resolution binary-segmentation logits.

    ``adapter_mode='none'`` preserves the plain one-channel U-Net input and is
    the only mode that accepts ``field=None``.  The model never estimates a
    field internally: callers of conditioned variants must make that cost and
    provenance explicit by supplying a validated :class:`StructuralField`.
    """

    feature_schema = CURVILINEAR_FEATURE_SCHEMA
    structural_channels = 8

    def __init__(
        self,
        *,
        num_orientations: int = 16,
        width: int = 24,
        adapter_mode: str = "none",
        validation_mode: ValidationMode | str = ValidationMode.strict,
    ) -> None:
        super().__init__()
        self.num_orientations = _positive_int("num_orientations", num_orientations)
        if self.num_orientations < 2:
            raise ValueError("num_orientations must be at least 2")
        self.width = _positive_int("width", width)
        self.validation_mode = ValidationMode.normalize(validation_mode)
        self.fusion = StructuralFusion(
            feature_channels=1,
            structural_channels=self.structural_channels,
            mode=adapter_mode,
            validation_mode=self.validation_mode,
        )
        self.adapter_mode = self.fusion.mode
        self.backbone = UNetBackbone(
            self.fusion.out_channels,
            self.width,
            residual_blocks=False,
            fourier_bottleneck=False,
        )
        self.segmentation_head = nn.Conv2d(self.width, 1, kernel_size=1)

    def forward(
        self,
        image: torch.Tensor,
        field: StructuralField | None = None,
    ) -> dict[str, torch.Tensor]:
        """Segment ``image`` without implicitly constructing a structural field."""
        structural_features: torch.Tensor | None = None
        structural_uncertainty: torch.Tensor | None = None
        if field is None:
            _validate_image(image, self.validation_mode)
            if self.adapter_mode != "none":
                raise ValueError(f"adapter mode {self.adapter_mode!r} requires a StructuralField")
        else:
            validate_image_and_field(
                image,
                field,
                num_orientations=self.num_orientations,
                validation_mode=self.validation_mode,
            )
            structural_features = curvilinear_structural_features(field)
            structural_uncertainty = field.structural_uncertainty

        fused = self.fusion(
            image,
            structural_features,
            structural_uncertainty,
        )
        logits = self.segmentation_head(self.backbone(fused.features))
        outputs = {"segmentation": logits}
        if fused.gate is not None:
            outputs["structural_gate"] = fused.gate
        if fused.structural_delta is not None:
            outputs["structural_delta"] = fused.structural_delta
        return outputs

    def extra_repr(self) -> str:
        return (
            f"num_orientations={self.num_orientations}, width={self.width}, "
            f"adapter_mode={self.adapter_mode!r}, feature_schema={self.feature_schema!r}"
        )


__all__ = ["CurvilinearSegmenter"]
