"""StructField-conditioned restoration model for interference fringes.

The model never estimates a structural field implicitly.  Conditioned adapter
modes require both a validated :class:`structfield.StructuralField` and the
explicit normalized image that produced it.  The ``none`` ablation fixes the
second image-encoder channel to zero and does not consume a structural prior.
"""

from __future__ import annotations

import torch
from torch import nn

from structfield.adapters import AdapterOutput
from structfield.applications._shared import (
    FRINGE_FEATURE_SCHEMA,
    StructuralFusion,
    UNetBackbone,
    fringe_structural_features,
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


def _validate_normalized_image(
    normalized_image: torch.Tensor,
    image: torch.Tensor,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> None:
    if not isinstance(normalized_image, torch.Tensor):
        raise TypeError("normalized_image must be a torch.Tensor")
    if not torch.is_floating_point(normalized_image):
        raise TypeError("normalized_image must use a floating-point dtype")
    if normalized_image.shape != image.shape:
        raise ValueError("normalized_image must have the same shape as image")
    if normalized_image.device != image.device:
        raise ValueError("normalized_image and image must use the same device")
    if normalized_image.dtype != image.dtype:
        raise ValueError("normalized_image and image must use the same dtype")
    if is_strict_validation(validation_mode) and not bool(torch.isfinite(normalized_image).all()):
        raise ValueError("normalized_image contains NaN or Inf")


class FringeRestorer(nn.Module):
    """Restore one-channel fringes from explicit V0.2 structural conditioning.

    ``adapter_mode`` may be ``none``, ``concat``, ``film``, or
    ``reliability_gated``.  The conditioned modes encode the versioned ``K+5``
    fringe feature schema before applying the selected Stage-5 adapter.  In the
    ``none`` ablation, ``field`` may be omitted and the image encoder receives
    ``cat(image, zeros_like(image))``.  Supplying ``normalized_image`` in that
    mode is rejected so the no-prior semantics cannot silently change.

    A missing noise descriptor means a zero descriptor of the configured
    ``noise_dim``.  FiLM is applied only to the final full-resolution decoder
    feature, matching the historical model's high-level data flow without
    promising checkpoint compatibility.
    """

    feature_schema = FRINGE_FEATURE_SCHEMA

    def __init__(
        self,
        *,
        num_orientations: int = 16,
        width: int = 24,
        noise_dim: int = 10,
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
        self.noise_dim = _positive_int("noise_dim", noise_dim)
        self.adapter_mode = adapter_mode
        self.validation_mode = ValidationMode.normalize(validation_mode)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(2, self.width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.structure_encoder: nn.Sequential | None
        if self.adapter_mode == "none":
            self.structure_encoder = None
        else:
            self.structure_encoder = nn.Sequential(
                nn.Conv2d(
                    self.num_orientations + 5,
                    self.width,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(self.width, self.width, kernel_size=3, padding=1),
                nn.GELU(),
            )
        self.fusion = StructuralFusion(
            self.width,
            self.width,
            mode=adapter_mode,
            validation_mode=self.validation_mode,
        )
        self.backbone = UNetBackbone(
            self.fusion.out_channels,
            self.width,
            residual_blocks=True,
            fourier_bottleneck=False,
        )
        self.noise_film = nn.Sequential(
            nn.Linear(self.noise_dim, 2 * self.width),
            nn.GELU(),
            nn.Linear(2 * self.width, 2 * self.width),
        )
        self.residual_head = nn.Conv2d(self.width, 1, kernel_size=1)

    def _noise_descriptor(
        self,
        image: torch.Tensor,
        descriptor: torch.Tensor | None,
    ) -> torch.Tensor:
        if descriptor is None:
            return image.new_zeros((image.shape[0], self.noise_dim))
        if not isinstance(descriptor, torch.Tensor):
            raise TypeError("noise_descriptor must be a torch.Tensor")
        if not torch.is_floating_point(descriptor):
            raise TypeError("noise_descriptor must use a floating-point dtype")
        expected = (image.shape[0], self.noise_dim)
        if descriptor.shape != expected:
            raise ValueError(f"noise_descriptor must have shape {expected}")
        if descriptor.device != image.device:
            raise ValueError("noise_descriptor and image must use the same device")
        if descriptor.dtype != image.dtype:
            raise ValueError("noise_descriptor and image must use the same dtype")
        if is_strict_validation(self.validation_mode) and not bool(torch.isfinite(descriptor).all()):
            raise ValueError("noise_descriptor contains NaN or Inf")
        return descriptor

    def forward(
        self,
        image: torch.Tensor,
        *,
        field: StructuralField | None = None,
        normalized_image: torch.Tensor | None = None,
        noise_descriptor: torch.Tensor | None = None,
        return_adapter_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AdapterOutput]:
        """Return the restoration and, optionally, the inspectable adapter output."""
        if not isinstance(return_adapter_output, bool):
            raise TypeError("return_adapter_output must be a bool")
        _validate_image(image, self.validation_mode)

        structural_features: torch.Tensor | None
        structural_uncertainty: torch.Tensor | None
        if self.adapter_mode == "none":
            if normalized_image is not None:
                raise ValueError(
                    "normalized_image must be omitted for adapter_mode='none'; "
                    "the second image channel is fixed to zero"
                )
            if field is not None:
                validate_image_and_field(
                    image,
                    field,
                    num_orientations=self.num_orientations,
                    validation_mode=self.validation_mode,
                )
            image_pair = torch.cat((image, torch.zeros_like(image)), dim=1)
            structural_features = None
            structural_uncertainty = None
        else:
            if field is None:
                raise ValueError(f"adapter_mode {self.adapter_mode!r} requires an explicit field")
            validate_image_and_field(
                image,
                field,
                num_orientations=self.num_orientations,
                validation_mode=self.validation_mode,
            )
            if normalized_image is None:
                raise ValueError(f"adapter_mode {self.adapter_mode!r} requires normalized_image")
            _validate_normalized_image(normalized_image, image, self.validation_mode)
            image_pair = torch.cat((image, normalized_image), dim=1)
            if self.structure_encoder is None:  # pragma: no cover - constructor invariant
                raise AssertionError("structure encoder is missing for a structural adapter")
            structural_features = self.structure_encoder(fringe_structural_features(field))
            structural_uncertainty = field.structural_uncertainty

        image_features = self.image_encoder(image_pair)
        if structural_features is not None:
            structural_features = structural_features.to(dtype=image_features.dtype)
        if structural_uncertainty is not None:
            structural_uncertainty = structural_uncertainty.to(dtype=image_features.dtype)
        adapter_output = self.fusion(
            image_features,
            structural_features,
            structural_uncertainty,
        )
        decoder_features = self.backbone(adapter_output.features)

        descriptor = self._noise_descriptor(image, noise_descriptor)
        modulation = self.noise_film(descriptor).to(dtype=decoder_features.dtype)
        gamma, beta = modulation.chunk(2, dim=1)
        decoder_features = decoder_features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        residual = self.residual_head(decoder_features)
        restored = (image + residual).clamp(0.0, 1.0)
        if return_adapter_output:
            return restored, adapter_output
        return restored


__all__ = ["FringeRestorer"]
