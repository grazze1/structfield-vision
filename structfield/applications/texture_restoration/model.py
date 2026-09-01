"""Texture restoration with an explicit StructField application boundary.

This is an independent V0.2 rewrite of the project-specific dual-encoder
restorer.  It retains the high-level residual U-Net and Fourier-bottleneck
design while deliberately excluding the provenance-uncertain historical
NAFNet-like baseline.  Structural fields are always supplied by the caller;
the model never hides analytic estimation inside ``forward``.
"""

from __future__ import annotations

import torch
from torch import nn

from structfield.adapters import AdapterOutput
from structfield.applications._shared import (
    ADAPTER_MODES,
    TEXTURE_FEATURE_SCHEMA,
    StructuralFusion,
    UNetBackbone,
    texture_structural_features,
    validate_image_and_field,
)
from structfield.core import StructuralField
from structfield.validation import ValidationMode, is_strict_validation


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_image_without_field(
    image: torch.Tensor,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> None:
    """Validate the no-prior input using the shared application contract."""
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


def _feature_encoder(in_channels: int, width: int) -> nn.Sequential:
    """Encode one full-resolution input branch without changing its geometry."""
    return nn.Sequential(
        nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(width, width, kernel_size=3, padding=1),
    )


class TextureRestorer(nn.Module):
    """Restore a grayscale texture using an optional explicit structural prior.

    Parameters
    ----------
    num_orientations:
        Expected orientation-bin count of every supplied ``StructuralField``.
    width:
        Base channel width of both encoders and the three-level residual U-Net.
    adapter_mode:
        One of ``none``, ``concat``, ``film``, or ``reliability_gated``.  Only
        ``none`` permits ``field=None``.  A supplied field is validated even in
        no-prior mode, but it is ignored by the numerical path.

    Notes
    -----
    ``texture_structural_features`` provides the versioned ``K+8`` field
    mapping.  The downstream reliability gate, when selected, is learned by
    the adapter and is not equal to ``1 - structural_uncertainty``.
    """

    feature_schema = TEXTURE_FEATURE_SCHEMA

    def __init__(
        self,
        num_orientations: int = 16,
        width: int = 24,
        adapter_mode: str = "concat",
        validation_mode: ValidationMode | str = ValidationMode.strict,
    ) -> None:
        super().__init__()
        self.num_orientations = _positive_int(
            "num_orientations",
            num_orientations,
            minimum=2,
        )
        self.width = _positive_int("width", width)
        if adapter_mode not in ADAPTER_MODES:
            raise ValueError(f"adapter_mode must be one of {ADAPTER_MODES}")
        self.adapter_mode = adapter_mode
        self.validation_mode = ValidationMode.normalize(validation_mode)
        self.structural_input_channels = self.num_orientations + 8

        self.image_encoder = _feature_encoder(1, self.width)
        self.field_encoder: nn.Module | None
        if self.adapter_mode == "none":
            # The no-prior baseline has no unused trainable field branch.
            self.field_encoder = None
        else:
            self.field_encoder = _feature_encoder(
                self.structural_input_channels,
                self.width,
            )

        self.fusion = StructuralFusion(
            self.width,
            self.width,
            mode=self.adapter_mode,
            validation_mode=self.validation_mode,
        )
        self.backbone = UNetBackbone(
            self.fusion.out_channels,
            self.width,
            residual_blocks=True,
            fourier_bottleneck=True,
        )
        self.residual_head = nn.Conv2d(self.width, 1, kernel_size=1)

    def _encode_field(
        self,
        image: torch.Tensor,
        field: StructuralField,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        validate_image_and_field(
            image,
            field,
            num_orientations=self.num_orientations,
            validation_mode=self.validation_mode,
        )
        if self.adapter_mode == "none":
            return None, None
        if self.field_encoder is None:  # pragma: no cover - constructor invariant
            raise AssertionError("field encoder is missing for a structural adapter")

        structural_inputs = texture_structural_features(field)
        if structural_inputs.shape[1] != self.structural_input_channels:
            raise RuntimeError(
                f"texture structural mapping must produce exactly K+8={self.structural_input_channels} channels"
            )
        encoded = self.field_encoder(structural_inputs)
        uncertainty = field.structural_uncertainty.to(dtype=encoded.dtype)
        return encoded, uncertainty

    def forward(
        self,
        image: torch.Tensor,
        field: StructuralField | None = None,
        *,
        return_adapter_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AdapterOutput]:
        """Return a restoration and, optionally, inspectable adapter diagnostics."""
        if not isinstance(return_adapter_output, bool):
            raise TypeError("return_adapter_output must be a bool")
        if field is None:
            if self.adapter_mode != "none":
                raise ValueError(f"adapter_mode={self.adapter_mode!r} requires an explicit StructuralField")
            _validate_image_without_field(image, self.validation_mode)
            structural_features = None
            structural_uncertainty = None
        else:
            structural_features, structural_uncertainty = self._encode_field(
                image,
                field,
            )

        image_features = self.image_encoder(image)
        if structural_features is not None and structural_features.dtype != image_features.dtype:
            # Autocast may choose an encoder output dtype different from the
            # caller-owned field dtype.  Align explicitly at the adapter boundary.
            structural_features = structural_features.to(dtype=image_features.dtype)
        if structural_uncertainty is not None:
            structural_uncertainty = structural_uncertainty.to(dtype=image_features.dtype)
        adapter_output = self.fusion(
            image_features,
            structural_features,
            structural_uncertainty,
        )
        residual = self.residual_head(self.backbone(adapter_output.features))
        restored = (image + residual).clamp(0.0, 1.0)
        if return_adapter_output:
            return restored, adapter_output
        return restored

    def extra_repr(self) -> str:
        return f"num_orientations={self.num_orientations}, width={self.width}, adapter_mode={self.adapter_mode!r}"


__all__ = ["TextureRestorer"]
