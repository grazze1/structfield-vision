"""Deterministic image and geometry degradations for StructField-Bench-v0.1.

All operators are project-authored wrappers around PyTorch/Pillow primitives.
They return independently stored samples and record their resolved parameters.
Photometric operators preserve ground truth; geometric and occlusion operators
update every affected ground-truth component.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from io import BytesIO
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from structfield.core import FieldMetadata, StructuralField
from structfield.transforms import GeometricTransform, transform_structural_field, transform_tensor
from structfield.uncertainty import normalized_categorical_entropy

from .schema import (
    CANONICAL_DEGRADATIONS,
    BenchmarkDegradation,
    BenchmarkMetadata,
    BenchmarkSample,
)

DIFFICULTIES = ("easy", "medium", "hard")

_DEFAULTS: dict[str, dict[str, dict[str, float | int]]] = {
    "none": {level: {} for level in DIFFICULTIES},
    "gaussian": {
        "easy": {"sigma": 0.02},
        "medium": {"sigma": 0.05},
        "hard": {"sigma": 0.10},
    },
    "poisson": {
        "easy": {"peak": 80.0},
        "medium": {"peak": 30.0},
        "hard": {"peak": 10.0},
    },
    "speckle": {
        "easy": {"sigma": 0.04},
        "medium": {"sigma": 0.10},
        "hard": {"sigma": 0.20},
    },
    "salt_and_pepper": {
        "easy": {"probability": 0.01},
        "medium": {"probability": 0.04},
        "hard": {"probability": 0.10},
    },
    "gaussian_blur": {
        "easy": {"sigma": 0.5},
        "medium": {"sigma": 1.0},
        "hard": {"sigma": 2.0},
    },
    "motion_blur": {
        "easy": {"kernel_size": 3, "angle_rad": 0.0},
        "medium": {"kernel_size": 5, "angle_rad": 0.0},
        "hard": {"kernel_size": 9, "angle_rad": 0.0},
    },
    "contrast": {
        "easy": {"factor": 0.85},
        "medium": {"factor": 0.60},
        "hard": {"factor": 0.35},
    },
    "brightness": {
        "easy": {"delta": 0.05},
        "medium": {"delta": 0.12},
        "hard": {"delta": 0.25},
    },
    "rotation": {
        "easy": {"angle_rad": math.radians(5.0)},
        "medium": {"angle_rad": math.radians(15.0)},
        "hard": {"angle_rad": math.radians(30.0)},
    },
    "scaling": {
        "easy": {"scale": 1.10},
        "medium": {"scale": 1.25},
        "hard": {"scale": 1.50},
    },
    "jpeg": {
        "easy": {"quality": 90},
        "medium": {"quality": 60},
        "hard": {"quality": 30},
    },
    "occlusion": {
        "easy": {"height_fraction": 0.12, "width_fraction": 0.12},
        "medium": {"height_fraction": 0.25, "width_fraction": 0.25},
        "hard": {"height_fraction": 0.40, "width_fraction": 0.40},
    },
}


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    if not 0 <= value <= 2**63 - 1:
        raise ValueError("seed must lie in [0, 2**63 - 1]")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DegradationConfig:
    """One canonical degradation with transparent difficulty defaults."""

    name: BenchmarkDegradation
    difficulty: str = "medium"
    parameters: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in CANONICAL_DEGRADATIONS:
            raise ValueError(f"name must be one of {CANONICAL_DEGRADATIONS}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        validated: dict[str, float | int] = {}
        for key, value in self.parameters.items():
            if not isinstance(key, str) or not key:
                raise TypeError("parameter names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("degradation parameters must be real numbers, not bool")
            converted: float | int = int(value) if isinstance(value, int) else float(value)
            if isinstance(converted, float) and not math.isfinite(converted):
                raise ValueError("degradation parameters must be finite")
            validated[key] = converted
        object.__setattr__(self, "parameters", MappingProxyType(validated))

    def resolved_parameters(self) -> dict[str, float | int]:
        """Return difficulty defaults with explicit overrides applied."""
        resolved = dict(_DEFAULTS[self.name][self.difficulty])
        unknown = set(self.parameters) - set(resolved)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown parameters for {self.name}: {names}")
        resolved.update(self.parameters)
        return resolved


def _generator(sample: BenchmarkSample, seed: int) -> torch.Generator:
    generator = torch.Generator(device=sample.device)
    generator.manual_seed(_seed(seed))
    return generator


def _clone_sample(
    sample: BenchmarkSample,
    *,
    image: torch.Tensor | None = None,
    orientation_distribution_gt: torch.Tensor | None = None,
    orientation_count: torch.Tensor | None = None,
    component_frequency_gt: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    junction_mask: torch.Tensor | None = None,
    metadata: BenchmarkMetadata | None = None,
) -> BenchmarkSample:
    return BenchmarkSample(
        image=(sample.image if image is None else image).clone(),
        orientation_distribution_gt=(
            sample.orientation_distribution_gt if orientation_distribution_gt is None else orientation_distribution_gt
        ).clone(),
        orientation_count=(sample.orientation_count if orientation_count is None else orientation_count).clone(),
        component_frequency_gt=(
            sample.component_frequency_gt if component_frequency_gt is None else component_frequency_gt
        ).clone(),
        valid_mask=(sample.valid_mask if valid_mask is None else valid_mask).clone(),
        junction_mask=(sample.junction_mask if junction_mask is None else junction_mask).clone(),
        metadata=sample.metadata if metadata is None else metadata,
    )


def _metadata(
    sample: BenchmarkSample,
    config: DegradationConfig,
    seed: int,
    realized: Mapping[str, Any],
) -> BenchmarkMetadata:
    parameters = dict(sample.metadata.parameters)
    parameters["source_difficulty"] = sample.metadata.difficulty
    parameters["source_sample_id"] = sample.metadata.sample_id
    parameters["degradation"] = {
        "name": config.name,
        "difficulty": config.difficulty,
        "seed": seed,
        **realized,
    }
    canonical_parameters = json.dumps(realized, sort_keys=True, separators=(",", ":"))
    parameter_digest = hashlib.sha256(canonical_parameters.encode("utf-8")).hexdigest()[:12]
    sample_id = f"{sample.metadata.sample_id}:deg-{config.name}-{config.difficulty}:s{seed}:p{parameter_digest}"
    return replace(
        sample.metadata,
        sample_id=sample_id,
        degradation=config.name,
        difficulty=config.difficulty,
        parameters=parameters,
    )


def _randn(sample: BenchmarkSample, seed: int) -> torch.Tensor:
    working_dtype = torch.float32 if sample.dtype in {torch.float16, torch.bfloat16} else sample.dtype
    return torch.randn(
        sample.image.shape,
        device=sample.device,
        dtype=working_dtype,
        generator=_generator(sample, seed),
    )


def _rand(sample: BenchmarkSample, seed: int, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    working_dtype = torch.float32 if sample.dtype in {torch.float16, torch.bfloat16} else sample.dtype
    return torch.rand(
        sample.image.shape if shape is None else shape,
        device=sample.device,
        dtype=working_dtype,
        generator=_generator(sample, seed),
    )


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image.clone()
    working_dtype = torch.float32 if image.dtype in {torch.float16, torch.bfloat16} else image.dtype
    radius = max(1, math.ceil(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=working_dtype)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    working = image.to(dtype=working_dtype).unsqueeze(0)
    padded = F.pad(working, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel).squeeze(0).to(dtype=image.dtype)


def _motion_blur(image: torch.Tensor, kernel_size: int, angle_rad: float) -> torch.Tensor:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    angle_rad = _finite_scalar("angle_rad", angle_rad)
    working_dtype = torch.float32 if image.dtype in {torch.float16, torch.bfloat16} else image.dtype
    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=working_dtype)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    # Public positive angles are counter-clockwise on the displayed image,
    # while tensor row coordinates ``yy`` increase downward.  The displayed
    # direction vector is therefore (cos(angle), -sin(angle)) in (x, row).
    normal_distance = (math.sin(angle_rad) * xx + math.cos(angle_rad) * yy).abs()
    along = (math.cos(angle_rad) * xx - math.sin(angle_rad) * yy).abs()
    kernel = ((normal_distance <= 0.5) & (along <= radius + 0.25)).to(dtype=working_dtype)
    if float(kernel.sum()) <= 0.0:  # pragma: no cover - center always participates
        kernel[radius, radius] = 1.0
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)
    working = image.to(dtype=working_dtype).unsqueeze(0)
    padded = F.pad(working, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel).squeeze(0).to(dtype=image.dtype)


def _jpeg_roundtrip(image: torch.Tensor, quality: int) -> torch.Tensor:
    if isinstance(quality, bool) or not isinstance(quality, int):
        raise TypeError("quality must be an integer")
    if not 1 <= quality <= 100:
        raise ValueError("quality must lie in [1,100]")
    height, width = image.shape[-2:]
    array = (
        image.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)[0]
        .numpy()
    )
    buffer = BytesIO()
    Image.fromarray(array).save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded_array = np.array(decoded.convert("L"), dtype=np.float32, copy=True)
    tensor = torch.from_numpy(decoded_array).view(1, height, width) / 255.0
    return tensor.to(device=image.device, dtype=image.dtype)


def _temporary_field(sample: BenchmarkSample) -> StructuralField:
    probability = sample.orientation_distribution_gt.unsqueeze(0)
    scalar_shape = (1, 1, *sample.spatial_shape)
    zeros = probability.new_zeros(scalar_shape)
    coherence = sample.valid_mask.unsqueeze(0).to(dtype=sample.dtype)
    entropy = normalized_categorical_entropy(probability, dim=1).to(dtype=sample.dtype)
    uncertainty = (~sample.valid_mask).unsqueeze(0).to(dtype=sample.dtype)
    return StructuralField(
        orientation_prob=probability,
        frequency_mean=zeros,
        frequency_var=zeros.clone(),
        coherence=coherence,
        orientation_entropy=entropy,
        structural_uncertainty=uncertainty,
        metadata=FieldMetadata(
            num_orientations=sample.metadata.num_orientations,
            orientation_bin_centers=sample.metadata.orientation_bin_centers_rad,
            estimator="StructField-Bench-v0.1-gt-transform",
        ),
    )


def _geometric(
    sample: BenchmarkSample,
    transform: GeometricTransform,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    batched_source_mask = sample.valid_mask.unsqueeze(0)
    image_result = transform_tensor(sample.image.unsqueeze(0), transform)
    field_result = transform_structural_field(
        _temporary_field(sample),
        transform,
        source_valid_mask=batched_source_mask,
    )
    count_result = transform_tensor(
        sample.orientation_count.to(dtype=sample.dtype).unsqueeze(0),
        transform,
    ).tensor[0]
    frequency_result = (
        transform_tensor(
            sample.component_frequency_gt.unsqueeze(0),
            transform,
        ).tensor[0]
        / transform.scale
    )
    valid = image_result.equivariance_valid_mask[0] & field_result.equivariance_valid_mask[0]
    count = count_result.round().to(dtype=torch.int64).clamp(0, sample.metadata.max_components)
    count = torch.where(valid, count.clamp_min(1), torch.zeros_like(count))
    component_index = torch.arange(sample.metadata.max_components, device=sample.device).view(-1, 1, 1)
    active = component_index < count[0].unsqueeze(0)
    component_frequency = torch.where(active, frequency_result.clamp_min(0.0), 0.0)
    # Canonical v0.1 junction semantics are exactly multi-component pixels.
    # Derive this from the transformed discrete component count rather than
    # independently thresholding a warped bool mask, which can disagree at
    # non-integer interpolation footprints.
    junction = (count >= 2) & valid

    orientation = field_result.field.orientation_prob[0]
    uniform = torch.full_like(orientation, 1.0 / sample.metadata.num_orientations)
    orientation = torch.where(valid.expand_as(orientation), orientation, uniform)
    values = {
        "image": image_result.tensor[0].clamp(0.0, 1.0),
        "orientation_distribution_gt": orientation,
        "orientation_count": count,
        "component_frequency_gt": component_frequency,
        "valid_mask": valid,
        "junction_mask": junction,
    }
    realized = {
        "angle_rad": transform.rotation_rad,
        "scale": transform.scale,
        "boundary": "same-canvas-zero-padding",
        "validity": "complete-bilinear-footprint",
    }
    return values, realized


def _occlude(
    sample: BenchmarkSample,
    parameters: Mapping[str, float | int],
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    height_fraction = _finite_scalar("height_fraction", parameters["height_fraction"])
    width_fraction = _finite_scalar("width_fraction", parameters["width_fraction"])
    if not 0.0 < height_fraction <= 1.0 or not 0.0 < width_fraction <= 1.0:
        raise ValueError("occlusion fractions must lie in (0,1]")
    height, width = sample.spatial_shape
    box_height = min(height, max(1, round(height * height_fraction)))
    box_width = min(width, max(1, round(width * width_fraction)))
    generator = _generator(sample, seed)
    top = int(torch.randint(0, height - box_height + 1, (), device=sample.device, generator=generator))
    left = int(torch.randint(0, width - box_width + 1, (), device=sample.device, generator=generator))
    bottom, right = top + box_height, left + box_width

    image = sample.image.clone()
    orientation = sample.orientation_distribution_gt.clone()
    count = sample.orientation_count.clone()
    frequency = sample.component_frequency_gt.clone()
    valid = sample.valid_mask.clone()
    junction = sample.junction_mask.clone()
    image[:, top:bottom, left:right] = 0.0
    orientation[:, top:bottom, left:right] = 1.0 / sample.metadata.num_orientations
    count[:, top:bottom, left:right] = 0
    frequency[:, top:bottom, left:right] = 0.0
    valid[:, top:bottom, left:right] = False
    junction[:, top:bottom, left:right] = False
    values = {
        "image": image,
        "orientation_distribution_gt": orientation,
        "orientation_count": count,
        "component_frequency_gt": frequency,
        "valid_mask": valid,
        "junction_mask": junction,
    }
    realized: dict[str, float | int] = {
        "height_fraction": height_fraction,
        "width_fraction": width_fraction,
        "top": top,
        "left": left,
        "height": box_height,
        "width": box_width,
        "fill": 0.0,
    }
    return values, realized


def apply_degradation(
    sample: BenchmarkSample,
    config: DegradationConfig,
    *,
    seed: int,
) -> BenchmarkSample:
    """Apply one pure, deterministic degradation with explicit provenance."""
    if not isinstance(sample, BenchmarkSample):
        raise TypeError("sample must be a BenchmarkSample")
    if not isinstance(config, DegradationConfig):
        raise TypeError("config must be a DegradationConfig")
    seed = _seed(seed)
    parameters = config.resolved_parameters()
    image = sample.image
    realized: dict[str, Any] = dict(parameters)
    values: dict[str, torch.Tensor] = {}

    if config.name == "none":
        values["image"] = image.clone()
    elif config.name == "gaussian":
        sigma = _finite_scalar("sigma", parameters["sigma"])
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        noise = _randn(sample, seed)
        values["image"] = (
            (image.to(dtype=noise.dtype) + sigma * noise)
            .clamp(
                0.0,
                1.0,
            )
            .to(dtype=sample.dtype)
        )
    elif config.name == "poisson":
        peak = _finite_scalar("peak", parameters["peak"])
        if peak <= 0.0:
            raise ValueError("peak must be positive")
        working = image.to(dtype=torch.float32 if sample.dtype in {torch.float16, torch.bfloat16} else sample.dtype)
        values["image"] = (
            (torch.poisson(working.clamp(0.0, 1.0) * peak, generator=_generator(sample, seed)) / peak)
            .clamp(0.0, 1.0)
            .to(dtype=sample.dtype)
        )
    elif config.name == "speckle":
        sigma = _finite_scalar("sigma", parameters["sigma"])
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        noise = _randn(sample, seed)
        working = image.to(dtype=noise.dtype)
        values["image"] = (working + sigma * working * noise).clamp(0.0, 1.0).to(dtype=sample.dtype)
    elif config.name == "salt_and_pepper":
        probability = _finite_scalar("probability", parameters["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0,1]")
        random = _rand(sample, seed)
        working = image.to(dtype=random.dtype)
        salted = torch.where(random < probability / 2.0, 0.0, working)
        salted = torch.where(random > 1.0 - probability / 2.0, 1.0, salted)
        values["image"] = salted.to(dtype=sample.dtype)
    elif config.name == "gaussian_blur":
        sigma = _finite_scalar("sigma", parameters["sigma"])
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        values["image"] = _gaussian_blur(image, sigma).clamp(0.0, 1.0)
    elif config.name == "motion_blur":
        kernel_size_value = parameters["kernel_size"]
        if isinstance(kernel_size_value, bool) or not isinstance(kernel_size_value, int):
            raise TypeError("kernel_size must be an integer")
        angle = _finite_scalar("angle_rad", parameters["angle_rad"])
        values["image"] = _motion_blur(image, kernel_size_value, angle).clamp(0.0, 1.0)
    elif config.name == "contrast":
        factor = _finite_scalar("factor", parameters["factor"])
        if factor < 0.0:
            raise ValueError("factor must be non-negative")
        values["image"] = ((image - 0.5) * factor + 0.5).clamp(0.0, 1.0)
    elif config.name == "brightness":
        delta = _finite_scalar("delta", parameters["delta"])
        values["image"] = (image + delta).clamp(0.0, 1.0)
    elif config.name == "rotation":
        angle = _finite_scalar("angle_rad", parameters["angle_rad"])
        values, realized = _geometric(sample, GeometricTransform(rotation_rad=angle))
    elif config.name == "scaling":
        scale = _finite_scalar("scale", parameters["scale"])
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        values, realized = _geometric(sample, GeometricTransform(scale=scale))
    elif config.name == "jpeg":
        quality_value = parameters["quality"]
        if isinstance(quality_value, bool) or not isinstance(quality_value, int):
            raise TypeError("quality must be an integer")
        values["image"] = _jpeg_roundtrip(image, quality_value)
        realized["codec"] = "Pillow-JPEG-in-memory"
    elif config.name == "occlusion":
        values, realized = _occlude(sample, parameters, seed)
    else:  # pragma: no cover - dataclass vocabulary is exhaustive
        raise AssertionError(f"unhandled degradation {config.name}")

    metadata = _metadata(sample, config, seed, realized)
    return _clone_sample(sample, metadata=metadata, **values)


__all__ = ["DIFFICULTIES", "DegradationConfig", "apply_degradation"]
