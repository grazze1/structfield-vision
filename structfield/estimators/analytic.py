"""Task-neutral analytic structural-field estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from structfield.core import FieldMetadata, StructuralField, StructuralFieldDiagnostics
from structfield.uncertainty import HeuristicStructuralUncertainty, normalized_categorical_entropy
from structfield.validation import ValidationMode

from .base import LUMINANCE_WEIGHTS, BaseStructuralFieldEstimator, EstimatorOutput, to_luminance
from .frequency import frequency_moments
from .gabor import estimate_gabor_posterior
from .hessian import HessianMaps, estimate_hessian_maps
from .structure_tensor import estimate_structure_tensor, gaussian_blur


def _positive_tuple(name: str, values: tuple[float, ...], *, upper_bound: float | None = None) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    if upper_bound is not None and any(value >= upper_bound for value in result):
        raise ValueError(f"{name} values must be smaller than {upper_bound}")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return result


@dataclass(frozen=True)
class AnalyticEstimatorConfig:
    """Validated configuration for the deterministic analytic baseline."""

    num_orientations: int = 16
    frequencies: tuple[float, ...] = (0.04, 0.08, 0.16)
    tensor_scales: tuple[float, ...] = (1.2, 2.4, 4.8)
    hessian_scales: tuple[float, ...] = (1.0, 2.0, 4.0)
    weak_denoise_type: str = "gaussian"
    weak_denoise_sigma: float = 0.8
    local_radius: float = 9.0
    normalization_eps: float = 1e-3
    normalization_clip: float = 6.0
    gabor_sigma: float = 3.0
    orientation_temperature: float = 0.15
    scale_temperature: float = 0.1
    analytic_eps: float = 1e-6
    min_separation_rad: float = math.pi / 8.0
    include_hessian: bool = False
    hessian_beta: float = 0.5
    hessian_sensitivity: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequencies", _positive_tuple("frequencies", self.frequencies, upper_bound=0.5))
        object.__setattr__(self, "tensor_scales", _positive_tuple("tensor_scales", self.tensor_scales))
        object.__setattr__(self, "hessian_scales", _positive_tuple("hessian_scales", self.hessian_scales))
        self.validate()

    def validate(self) -> AnalyticEstimatorConfig:
        if isinstance(self.num_orientations, bool) or not isinstance(self.num_orientations, int):
            raise TypeError("num_orientations must be an integer")
        if self.num_orientations < 2:
            raise ValueError("num_orientations must be at least 2")
        if self.weak_denoise_type not in {"none", "gaussian"}:
            raise ValueError("weak_denoise_type must be 'none' or 'gaussian'")
        positive_parameters = {
            "weak_denoise_sigma": self.weak_denoise_sigma,
            "local_radius": self.local_radius,
            "normalization_eps": self.normalization_eps,
            "normalization_clip": self.normalization_clip,
            "gabor_sigma": self.gabor_sigma,
            "orientation_temperature": self.orientation_temperature,
            "scale_temperature": self.scale_temperature,
            "analytic_eps": self.analytic_eps,
            "hessian_beta": self.hessian_beta,
            "hessian_sensitivity": self.hessian_sensitivity,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_parameters.values()):
            raise ValueError("analytic estimator scalar parameters must be finite and positive")
        if not math.isfinite(self.min_separation_rad) or not 0.0 <= self.min_separation_rad < math.pi:
            raise ValueError("min_separation_rad must lie in [0, pi)")
        if not isinstance(self.include_hessian, bool):
            raise TypeError("include_hessian must be a bool")
        return self


def _local_normalize(image: torch.Tensor, config: AnalyticEstimatorConfig) -> torch.Tensor:
    if config.weak_denoise_type == "gaussian":
        denoised = gaussian_blur(image, config.weak_denoise_sigma)
    else:
        denoised = image
    local_sigma = max(config.local_radius / 3.0, 0.5)
    local_mean = gaussian_blur(denoised, local_sigma)
    local_second_moment = gaussian_blur(denoised.square(), local_sigma)
    local_variance = (local_second_moment - local_mean.square()).clamp_min(0.0)
    local_std = (torch.sqrt(local_variance + config.analytic_eps**2) - config.analytic_eps).clamp_min(0.0)
    normalized = (denoised - local_mean) / (local_std + config.normalization_eps)
    normalized = normalized.clamp(-config.normalization_clip, config.normalization_clip)
    informative = local_variance > config.normalization_eps**2
    return torch.where(informative, normalized, torch.zeros_like(normalized))


class AnalyticStructuralFieldEstimator(BaseStructuralFieldEstimator):
    """Deterministic multi-scale tensor/Gabor estimator with optional Hessian evidence."""

    def __init__(
        self,
        config: AnalyticEstimatorConfig | None = None,
        *,
        validation_mode: ValidationMode | str = ValidationMode.strict,
        **config_overrides: Any,
    ) -> None:
        super().__init__()
        if config is not None and config_overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        self.config = config if config is not None else AnalyticEstimatorConfig(**config_overrides)
        if not isinstance(self.config, AnalyticEstimatorConfig):
            raise TypeError("config must be AnalyticEstimatorConfig")
        self.config.validate()
        self.validation_mode = ValidationMode.normalize(validation_mode)
        self.uncertainty_estimator = HeuristicStructuralUncertainty()

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(self, image: torch.Tensor, *, return_diagnostics: bool = False) -> EstimatorOutput:
        input_channels = int(image.shape[1]) if isinstance(image, torch.Tensor) and image.ndim >= 2 else -1
        luminance = to_luminance(image, validation_mode=self.validation_mode)
        output_dtype = luminance.dtype
        working_luminance = (
            luminance.to(dtype=torch.float32) if output_dtype in {torch.float16, torch.bfloat16} else luminance
        )
        normalized = _local_normalize(working_luminance, self.config)
        tensor_maps = estimate_structure_tensor(
            normalized,
            self.config.tensor_scales,
            scale_temperature=self.config.scale_temperature,
            eps=self.config.analytic_eps,
        )
        gabor = estimate_gabor_posterior(
            normalized,
            self.config.frequencies,
            self.config.num_orientations,
            sigma=self.config.gabor_sigma,
            orientation_temperature=self.config.orientation_temperature,
            eps=self.config.analytic_eps,
        )
        frequency_mean, frequency_var = frequency_moments(gabor.frequency_prob, self.config.frequencies)
        entropy = normalized_categorical_entropy(
            gabor.orientation_prob,
            dim=1,
            validation_mode=self.validation_mode,
        )
        uncertainty_placeholder = torch.zeros_like(entropy)
        metadata = FieldMetadata(
            num_orientations=self.config.num_orientations,
            frequency_candidates=self.config.frequencies,
            scale_candidates=self.config.tensor_scales,
            default_min_separation_rad=self.config.min_separation_rad,
            estimator="analytic-v1",
            extras={
                "input_channels": input_channels,
                "luminance_weights_rgb": LUMINANCE_WEIGHTS,
                "rgb_gamma_linearization": "none",
                "structural_uncertainty": "baseline-heuristic-v1",
                "structural_uncertainty_weights": self.uncertainty_estimator.weights,
            },
        )
        field = StructuralField(
            orientation_prob=gabor.orientation_prob,
            frequency_mean=frequency_mean,
            frequency_var=frequency_var,
            coherence=tensor_maps.coherence,
            orientation_entropy=entropy,
            structural_uncertainty=uncertainty_placeholder,
            metadata=metadata,
            scale_prob=tensor_maps.scale_prob,
            selected_scale=tensor_maps.selected_scale,
            validation_mode=self.validation_mode,
        )

        hessian: HessianMaps | None = None
        if return_diagnostics and self.config.include_hessian:
            hessian = estimate_hessian_maps(
                normalized,
                self.config.hessian_scales,
                beta=self.config.hessian_beta,
                sensitivity=self.config.hessian_sensitivity,
                eps=self.config.analytic_eps,
            )
        if field.dtype != output_dtype:
            field = field.to(dtype=output_dtype)
        field = replace(field, structural_uncertainty=self.uncertainty_estimator(field))
        if not return_diagnostics:
            return field
        diagnostics = StructuralFieldDiagnostics(
            metadata=metadata,
            normalized_image=normalized,
            orientation_frequency_response=gabor.joint_response,
            orientation_frequency_prob=gabor.joint_prob,
            orientation_response=gabor.orientation_response,
            frequency_prob=gabor.frequency_prob,
            scale_response=tensor_maps.scale_response,
            tensor_orientation=tensor_maps.orientation,
            tensor_eigenvalue_max=tensor_maps.eigenvalue_max,
            tensor_eigenvalue_min=tensor_maps.eigenvalue_min,
            hessian_vesselness=None if hessian is None else hessian.vesselness,
            hessian_orientation=None if hessian is None else hessian.orientation,
            hessian_line_confidence=None if hessian is None else hessian.line_confidence,
            hessian_blob_response=None if hessian is None else hessian.blob_response,
        )
        if diagnostics.normalized_image is not None and diagnostics.normalized_image.dtype != output_dtype:
            diagnostics = diagnostics.to(dtype=output_dtype)
        return field, diagnostics


StructuralFieldEstimator = AnalyticStructuralFieldEstimator


def estimate_structural_field(
    image: torch.Tensor,
    *,
    config: AnalyticEstimatorConfig | None = None,
    return_diagnostics: bool = False,
    validation_mode: ValidationMode | str = ValidationMode.strict,
    **config_overrides: Any,
) -> EstimatorOutput:
    """Functional convenience API for one analytic estimation call."""
    estimator = AnalyticStructuralFieldEstimator(
        config,
        validation_mode=validation_mode,
        **config_overrides,
    )
    return estimator(image, return_diagnostics=return_diagnostics)
