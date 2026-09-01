"""Independently written application blocks and StructuralField projections.

The convolutional encoder/decoder follows the high-level U-Net design.  This
module is deliberately small and excludes the provenance-uncertain historical
NAFNet-like implementation.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from structfield.adapters import (
    AdapterOutput,
    ConcatStructuralAdapter,
    FiLMStructuralAdapter,
    NoPriorAdapter,
    ReliabilityGatedAdapter,
    StructuralAdapter,
)
from structfield.core import StructuralField
from structfield.validation import ValidationMode

FRINGE_FEATURE_SCHEMA = "fringe-structfield-k-plus-5-v0.2"
TEXTURE_FEATURE_SCHEMA = "texture-structfield-k-plus-8-v0.2"
CURVILINEAR_FEATURE_SCHEMA = "curvilinear-structfield-8-v0.2"
ADAPTER_MODES = ("none", "concat", "film", "reliability_gated")


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def validate_image_and_field(
    image: torch.Tensor,
    field: StructuralField,
    *,
    num_orientations: int,
    validation_mode: ValidationMode | str = ValidationMode.strict,
) -> None:
    """Validate the strict application boundary without implicit estimation."""
    mode = ValidationMode.normalize(validation_mode)
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if not torch.is_floating_point(image):
        raise TypeError("image must use a floating-point dtype")
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image must have shape [B,1,H,W]")
    if image.shape[0] < 1 or image.shape[-2] < 8 or image.shape[-1] < 8:
        raise ValueError("image must have non-empty batch and spatial dimensions at least 8x8")
    if mode is ValidationMode.strict and not bool(torch.isfinite(image).all()):
        raise ValueError("image contains NaN or Inf")
    if not isinstance(field, StructuralField):
        raise TypeError("field must be a structfield.StructuralField")
    if field.metadata.num_orientations != num_orientations:
        raise ValueError("field orientation bins must match the configured num_orientations")
    if field.batch_size != image.shape[0] or field.spatial_shape != image.shape[-2:]:
        raise ValueError("field batch and spatial dimensions must match image")
    if field.device != image.device:
        raise ValueError("field and image must use the same device")
    if field.dtype != image.dtype:
        raise ValueError("field and image must use the same dtype")


def _doubled_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cos(2.0 * angle), torch.sin(2.0 * angle)), dim=1)


def _normalized_scale(field: StructuralField) -> torch.Tensor:
    if field.selected_scale is None:
        return torch.zeros_like(field.frequency_mean)
    candidates = field.metadata.scale_candidates
    if not candidates:
        raise ValueError("selected_scale requires scale candidates in field metadata")
    return field.selected_scale / max(candidates)


def _scale_entropy(field: StructuralField) -> torch.Tensor:
    if field.scale_prob is None or field.scale_prob.shape[1] == 1:
        return torch.zeros_like(field.frequency_mean)
    probability = field.scale_prob
    tiny = torch.finfo(probability.dtype).tiny
    entropy = -(probability * probability.clamp_min(tiny).log()).sum(dim=1, keepdim=True)
    return entropy / math.log(probability.shape[1])


def fringe_structural_features(field: StructuralField) -> torch.Tensor:
    """Encode the versioned Q1 ``K+5`` semantic feature mapping."""
    top1 = _doubled_angle(field.topk_orientations(k=1).angles)
    return torch.cat(
        (
            field.orientation_prob,
            top1,
            field.frequency_mean,
            _normalized_scale(field),
            field.coherence,
        ),
        dim=1,
    )


def texture_structural_features(field: StructuralField) -> torch.Tensor:
    """Encode the versioned Q2 ``K+8`` multimodal feature mapping."""
    angles = field.topk_orientations(k=2, min_separation=0.0).angles
    first = _doubled_angle(angles[:, :1])
    second = _doubled_angle(angles[:, 1:2])
    frequency_std = field.frequency_var.clamp_min(0.0).sqrt()
    return torch.cat(
        (
            field.orientation_prob,
            first,
            second,
            field.frequency_mean,
            frequency_std,
            field.coherence,
            _normalized_scale(field),
        ),
        dim=1,
    )


def curvilinear_structural_features(field: StructuralField) -> torch.Tensor:
    """Encode a fixed eight-channel public-field representation for Q3."""
    centers = field.orientation_prob.new_tensor(field.metadata.orientation_bin_centers).view(1, -1, 1, 1)
    axial_x = (field.orientation_prob * torch.cos(2.0 * centers)).sum(dim=1, keepdim=True)
    axial_y = (field.orientation_prob * torch.sin(2.0 * centers)).sum(dim=1, keepdim=True)
    frequency_scale = max(field.metadata.frequency_candidates, default=0.5)
    frequency_scale = max(float(frequency_scale), torch.finfo(field.dtype).eps)
    frequency_mean = field.frequency_mean / frequency_scale
    frequency_std = field.frequency_var.clamp_min(0.0).sqrt() / frequency_scale
    return torch.cat(
        (
            axial_x,
            axial_y,
            frequency_mean,
            frequency_std,
            field.coherence,
            field.orientation_entropy,
            _scale_entropy(field),
            _normalized_scale(field),
        ),
        dim=1,
    )


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("one always divides a positive channel count")


class ConvBlock(nn.Module):
    """Two normalized convolutions with an optional learned residual path."""

    def __init__(self, in_channels: int, out_channels: int, *, residual: bool) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        groups = _group_count(out_channels)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.residual = bool(residual)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if self.residual and in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.body(inputs)
        return result + self.skip(inputs) if self.residual else result


class StructuralFusion(nn.Module):
    """Select one Stage-5 adapter behind a uniform application boundary."""

    def __init__(
        self,
        feature_channels: int,
        structural_channels: int,
        *,
        mode: str,
        validation_mode: ValidationMode | str = ValidationMode.strict,
    ) -> None:
        super().__init__()
        self.validation_mode = ValidationMode.normalize(validation_mode)
        self.feature_channels = _positive_int("feature_channels", feature_channels)
        self.structural_channels = _positive_int("structural_channels", structural_channels)
        if mode not in ADAPTER_MODES:
            raise ValueError(f"adapter mode must be one of {ADAPTER_MODES}")
        self.mode = mode
        self.adapter: StructuralAdapter
        if mode == "none":
            self.adapter = NoPriorAdapter()
        elif mode == "concat":
            self.adapter = ConcatStructuralAdapter()
        elif mode == "film":
            self.adapter = FiLMStructuralAdapter(
                self.feature_channels,
                self.structural_channels,
            )
        else:
            self.adapter = ReliabilityGatedAdapter(
                self.feature_channels,
                self.structural_channels,
            )
        self.out_channels = self.adapter.output_channels(
            self.feature_channels,
            self.structural_channels,
        )

    def forward(
        self,
        features: torch.Tensor,
        structural_features: torch.Tensor | None,
        structural_uncertainty: torch.Tensor | None,
    ) -> AdapterOutput:
        if self.mode == "none":
            return self.adapter(
                features,
                structural_features,
                validation_mode=self.validation_mode,
            )
        if structural_features is None:
            raise ValueError(f"adapter mode {self.mode!r} requires a StructuralField")
        if self.mode == "reliability_gated":
            if structural_uncertainty is None:
                raise ValueError("reliability_gated adapter requires structural_uncertainty")
            return self.adapter(
                features,
                structural_features,
                structural_uncertainty,
                validation_mode=self.validation_mode,
            )
        return self.adapter(
            features,
            structural_features,
            validation_mode=self.validation_mode,
        )

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, validation_mode={self.validation_mode.value!r}"


class FourierResidualBlock(nn.Module):
    """Project a global FFT representation and add it as a residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = _positive_int("channels", channels)
        self.real_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.imaginary_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        work_dtype = torch.float32 if inputs.dtype in {torch.float16, torch.bfloat16} else inputs.dtype
        autocast_context = (
            torch.autocast(device_type=inputs.device.type, enabled=False)
            if inputs.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_context:
            work = inputs.to(dtype=work_dtype)
            spectrum = torch.fft.rfft2(work, norm="ortho")
            transformed = torch.complex(
                self.real_projection(spectrum.real),
                self.imaginary_projection(spectrum.imag),
            )
            global_features = torch.fft.irfft2(
                transformed,
                s=inputs.shape[-2:],
                norm="ortho",
            )
            residual = self.output_projection(global_features)
        return inputs + residual.to(dtype=inputs.dtype)


class UNetBackbone(nn.Module):
    """Three-level U-Net returning a full-resolution decoder feature map."""

    def __init__(
        self,
        in_channels: int,
        width: int,
        *,
        residual_blocks: bool = True,
        fourier_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        width = _positive_int("width", width)
        self.encoder1 = ConvBlock(in_channels, width, residual=residual_blocks)
        self.encoder2 = ConvBlock(width, 2 * width, residual=residual_blocks)
        self.encoder3 = ConvBlock(2 * width, 4 * width, residual=residual_blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        bottleneck: list[nn.Module] = [ConvBlock(4 * width, 8 * width, residual=residual_blocks)]
        if fourier_bottleneck:
            bottleneck.append(FourierResidualBlock(8 * width))
        self.bottleneck = nn.Sequential(*bottleneck)
        self.up3 = nn.ConvTranspose2d(8 * width, 4 * width, 2, stride=2)
        self.decoder3 = ConvBlock(8 * width, 4 * width, residual=residual_blocks)
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.decoder2 = ConvBlock(4 * width, 2 * width, residual=residual_blocks)
        self.up1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.decoder1 = ConvBlock(2 * width, width, residual=residual_blocks)
        self.out_channels = width

    @staticmethod
    def _match_size(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] == reference.shape[-2:]:
            return value
        return F.interpolate(
            value,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        center = self.bottleneck(self.pool(encoder3))
        decoder3 = self.decoder3(torch.cat((self._match_size(self.up3(center), encoder3), encoder3), dim=1))
        decoder2 = self.decoder2(torch.cat((self._match_size(self.up2(decoder3), encoder2), encoder2), dim=1))
        return self.decoder1(torch.cat((self._match_size(self.up1(decoder2), encoder1), encoder1), dim=1))


__all__ = [
    "ADAPTER_MODES",
    "CURVILINEAR_FEATURE_SCHEMA",
    "FRINGE_FEATURE_SCHEMA",
    "TEXTURE_FEATURE_SCHEMA",
    "ConvBlock",
    "FourierResidualBlock",
    "StructuralFusion",
    "UNetBackbone",
    "curvilinear_structural_features",
    "fringe_structural_features",
    "texture_structural_features",
    "validate_image_and_field",
]
