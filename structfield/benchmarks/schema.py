"""Typed sample schema for StructField-Bench-v0.1.

This module is a project-authored benchmark boundary.  It deliberately keeps
the procedural generator, degradations, and model-facing ``StructuralField``
schema separate: a benchmark sample is an unbatched image plus dense ground
truth and immutable provenance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import torch

BENCHMARK_VERSION = "StructField-Bench-v0.1"
GENERATOR_VERSION = "structfield-procedural-v0.1"

_SUPPORTED_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}

CANONICAL_FAMILIES = (
    "straight",
    "parallel",
    "curved",
    "sinusoidal",
    "radial",
    "concentric",
    "crossing",
    "dual_crossing",
    "bifurcation",
    "junction",
    "varying_frequency",
    "varying_thickness",
    "domain_boundary",
    "occlusion",
)
"""Frozen family vocabulary for the v0.1 benchmark."""

CANONICAL_DEGRADATIONS = (
    "none",
    "gaussian",
    "poisson",
    "speckle",
    "salt_and_pepper",
    "gaussian_blur",
    "motion_blur",
    "contrast",
    "brightness",
    "rotation",
    "scaling",
    "jpeg",
    "occlusion",
)
"""Frozen degradation vocabulary: twelve operators plus ``none``."""

CANONICAL_SPLITS = ("train", "validation", "test")

BenchmarkFamily: TypeAlias = Literal[
    "straight",
    "parallel",
    "curved",
    "sinusoidal",
    "radial",
    "concentric",
    "crossing",
    "dual_crossing",
    "bifurcation",
    "junction",
    "varying_frequency",
    "varying_thickness",
    "domain_boundary",
    "occlusion",
]
BenchmarkDegradation: TypeAlias = Literal[
    "none",
    "gaussian",
    "poisson",
    "speckle",
    "salt_and_pepper",
    "gaussian_blur",
    "motion_blur",
    "contrast",
    "brightness",
    "rotation",
    "scaling",
    "jpeg",
    "occlusion",
]
BenchmarkSplit: TypeAlias = Literal["train", "validation", "test"]

JSONScalar: TypeAlias = str | int | float | bool | None
FrozenJSONValue: TypeAlias = JSONScalar | tuple["FrozenJSONValue", ...] | Mapping[str, "FrozenJSONValue"]


def _freeze_json(value: Any, *, path: str) -> FrozenJSONValue:
    """Validate and recursively freeze a JSON-compatible value."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: FrozenJSONValue) -> Any:
    """Return plain ``dict``/``list`` containers accepted by ``json``."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_exact_integer(name: str, value: object, *, minimum: int, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f"[{minimum}, infinity)"
        raise ValueError(f"{name} must lie in {interval}")


def _probability_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 2e-2
    if dtype == torch.float16:
        return 5e-3
    if dtype == torch.float32:
        return 2e-5
    return 1e-7


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkMetadata:
    """Immutable identity, split, and generation provenance for one sample."""

    sample_id: str
    seed: int
    split: BenchmarkSplit
    family: BenchmarkFamily
    difficulty: str
    num_orientations: int
    orientation_bin_centers_rad: tuple[float, ...]
    max_components: int
    degradation: BenchmarkDegradation = "none"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    benchmark_version: str = BENCHMARK_VERSION
    generator_version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise TypeError("sample_id must be a non-empty string")
        _require_exact_integer("seed", self.seed, minimum=0, maximum=2**63 - 1)
        if self.split not in CANONICAL_SPLITS:
            raise ValueError(f"split must be one of {CANONICAL_SPLITS}")
        if self.family not in CANONICAL_FAMILIES:
            raise ValueError(f"family must be one of {CANONICAL_FAMILIES}")
        if not isinstance(self.difficulty, str) or not self.difficulty.strip():
            raise TypeError("difficulty must be a non-empty string")
        if self.degradation not in CANONICAL_DEGRADATIONS:
            raise ValueError(f"degradation must be one of {CANONICAL_DEGRADATIONS}")
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError(f"benchmark_version must be {BENCHMARK_VERSION!r}")
        if not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise TypeError("generator_version must be a non-empty string")

        _require_exact_integer("num_orientations", self.num_orientations, minimum=2)
        _require_exact_integer("max_components", self.max_components, minimum=1)
        if not isinstance(self.orientation_bin_centers_rad, Sequence) or isinstance(
            self.orientation_bin_centers_rad,
            (str, bytes, bytearray),
        ):
            raise TypeError("orientation_bin_centers_rad must be a sequence of real numbers")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in self.orientation_bin_centers_rad):
            raise TypeError("orientation_bin_centers_rad must contain only real numbers")
        centers = tuple(float(value) for value in self.orientation_bin_centers_rad)
        if len(centers) != self.num_orientations:
            raise ValueError("orientation_bin_centers_rad length must match num_orientations")
        if not all(math.isfinite(value) and 0.0 <= value < math.pi for value in centers):
            raise ValueError("orientation_bin_centers_rad must be finite and lie in [0, pi)")
        if any(right <= left for left, right in zip(centers, centers[1:])):
            raise ValueError("orientation_bin_centers_rad must be strictly increasing")
        object.__setattr__(self, "orientation_bin_centers_rad", centers)

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        frozen_parameters = _freeze_json(self.parameters, path="parameters")
        if not isinstance(frozen_parameters, Mapping):  # pragma: no cover - guarded by the check above
            raise AssertionError("parameters did not freeze to a mapping")
        object.__setattr__(self, "parameters", frozen_parameters)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable metadata dictionary."""
        return {
            "benchmark_version": self.benchmark_version,
            "generator_version": self.generator_version,
            "sample_id": self.sample_id,
            "seed": self.seed,
            "split": self.split,
            "family": self.family,
            "difficulty": self.difficulty,
            "degradation": self.degradation,
            "parameters": _thaw_json(self.parameters),
            "num_orientations": self.num_orientations,
            "orientation_bin_centers_rad": list(self.orientation_bin_centers_rad),
            "max_components": self.max_components,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """One unbatched StructField-Bench sample with validated dense ground truth.

    Tensor attributes retain normal PyTorch mutability.  The object itself is
    structurally immutable, and :meth:`to` clones every tensor so its result is
    safe to modify independently.
    """

    image: torch.Tensor
    orientation_distribution_gt: torch.Tensor
    orientation_count: torch.Tensor
    component_frequency_gt: torch.Tensor
    valid_mask: torch.Tensor
    junction_mask: torch.Tensor
    metadata: BenchmarkMetadata

    def __post_init__(self) -> None:
        self.validate()

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.image.shape[-2]), int(self.image.shape[-1])

    @property
    def dtype(self) -> torch.dtype:
        return self.image.dtype

    @property
    def device(self) -> torch.device:
        return self.image.device

    def validate(self) -> BenchmarkSample:
        """Validate shape, type, device, ranges, and cross-field semantics."""
        if not isinstance(self.metadata, BenchmarkMetadata):
            raise TypeError("metadata must be BenchmarkMetadata")

        floating = {
            "image": self.image,
            "orientation_distribution_gt": self.orientation_distribution_gt,
            "component_frequency_gt": self.component_frequency_gt,
        }
        for name, value in floating.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
                raise TypeError(
                    f"{name} must use a floating-point dtype supported by StructField-Bench: "
                    "float16, bfloat16, float32, or float64"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or Inf")

        if self.image.ndim != 3 or self.image.shape[0] != 1:
            raise ValueError("image must have shape [1, H, W]")
        _, height, width = self.image.shape
        if height < 1 or width < 1:
            raise ValueError("image spatial dimensions must be non-empty")
        if float(self.image.amin()) < 0.0 or float(self.image.amax()) > 1.0:
            raise ValueError("image must lie in [0, 1]")

        if self.orientation_distribution_gt.shape != (
            self.metadata.num_orientations,
            height,
            width,
        ):
            raise ValueError("orientation_distribution_gt must have shape [K, H, W] matching metadata")
        if self.component_frequency_gt.shape != (self.metadata.max_components, height, width):
            raise ValueError("component_frequency_gt must have shape [M, H, W] matching metadata")
        for name, value in floating.items():
            if value.dtype != self.dtype or value.device != self.device:
                raise ValueError(f"{name} must share image dtype and device")

        if float(self.orientation_distribution_gt.amin()) < 0.0:
            raise ValueError("orientation_distribution_gt contains negative probability mass")
        tolerance = _probability_tolerance(self.dtype)
        probability_sum = self.orientation_distribution_gt.sum(dim=0)
        if not torch.allclose(
            probability_sum,
            torch.ones_like(probability_sum),
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("orientation_distribution_gt must sum to one at every pixel")
        if float(self.component_frequency_gt.amin()) < 0.0:
            raise ValueError("component_frequency_gt must be non-negative")

        integer_and_masks = {
            "orientation_count": self.orientation_count,
            "valid_mask": self.valid_mask,
            "junction_mask": self.junction_mask,
        }
        for name, value in integer_and_masks.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.shape != (1, height, width):
                raise ValueError(f"{name} must have shape [1, H, W]")
            if value.device != self.device:
                raise ValueError(f"{name} must share image device")
        if self.orientation_count.dtype != torch.int64:
            raise TypeError("orientation_count must use torch.int64")
        if self.valid_mask.dtype != torch.bool or self.junction_mask.dtype != torch.bool:
            raise TypeError("valid_mask and junction_mask must use torch.bool")
        if int(self.orientation_count.amin()) < 0 or int(self.orientation_count.amax()) > self.metadata.max_components:
            raise ValueError("orientation_count must lie in [0, max_components]")

        counted_valid = self.orientation_count > 0
        if not torch.equal(self.valid_mask, counted_valid):
            raise ValueError("valid_mask must equal orientation_count > 0")
        if bool((self.junction_mask & ~self.valid_mask).any()):
            raise ValueError("junction_mask must be a subset of valid_mask")

        invalid = ~self.valid_mask[0]
        if bool(invalid.any()):
            invalid_orientation = self.orientation_distribution_gt[:, invalid]
            uniform = torch.full_like(invalid_orientation, 1.0 / self.metadata.num_orientations)
            if not torch.allclose(invalid_orientation, uniform, atol=tolerance, rtol=tolerance):
                raise ValueError("invalid pixels must have a uniform orientation distribution")
            invalid_frequency = self.component_frequency_gt[:, invalid]
            if not torch.allclose(
                invalid_frequency,
                torch.zeros_like(invalid_frequency),
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError("invalid pixels must have zero component frequency")

        component_index = torch.arange(self.metadata.max_components, device=self.device).view(-1, 1, 1)
        padding = component_index >= self.orientation_count[0].unsqueeze(0)
        padded_frequency = self.component_frequency_gt.masked_select(padding)
        if padded_frequency.numel() and not torch.allclose(
            padded_frequency,
            torch.zeros_like(padded_frequency),
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError("frequency channels beyond orientation_count must be zero")
        return self

    def to(self, *args: Any, **kwargs: Any) -> BenchmarkSample:
        """Return a fully independent tensor copy on the requested dtype/device."""
        image = self.image.to(*args, **kwargs).clone()
        if image.dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError(
                "BenchmarkSample.to must preserve or request a supported floating-point dtype: "
                "float16, bfloat16, float32, or float64"
            )
        orientation_distribution_gt = self.orientation_distribution_gt.to(*args, **kwargs).clone()
        component_frequency_gt = self.component_frequency_gt.to(*args, **kwargs).clone()
        orientation_count = self.orientation_count.to(device=image.device).clone()
        valid_mask = self.valid_mask.to(device=image.device).clone()
        junction_mask = self.junction_mask.to(device=image.device).clone()

        # A low-precision distribution can satisfy its dtype-aware tolerance
        # while its values, promoted to float64, no longer meet float64's much
        # tighter tolerance.  Renormalize after conversion so chained ``to``
        # calls remain valid; restore the schema's exact neutral convention at
        # invalid pixels as well.
        probability_sum = orientation_distribution_gt.sum(dim=0, keepdim=True)
        orientation_distribution_gt = orientation_distribution_gt / probability_sum
        invalid = ~valid_mask[0]
        if bool(invalid.any()):
            uniform = torch.full_like(orientation_distribution_gt, 1.0 / self.metadata.num_orientations)
            orientation_distribution_gt = torch.where(invalid.unsqueeze(0), uniform, orientation_distribution_gt)
        return BenchmarkSample(
            image=image,
            orientation_distribution_gt=orientation_distribution_gt,
            orientation_count=orientation_count,
            component_frequency_gt=component_frequency_gt,
            valid_mask=valid_mask,
            junction_mask=junction_mask,
            metadata=self.metadata,
        )
