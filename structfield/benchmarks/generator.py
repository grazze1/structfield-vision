"""Deterministic procedural generator for StructField-Bench-v0.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch

from .families import FAMILY_NAMES, FamilyScene, render_family
from .schema import (
    BENCHMARK_VERSION,
    CANONICAL_FAMILIES,
    GENERATOR_VERSION,
    BenchmarkFamily,
    BenchmarkMetadata,
    BenchmarkSample,
    BenchmarkSplit,
)
from .splits import MAX_BASE_SEED, derive_sample_seed, make_sample_id


@dataclass(frozen=True, slots=True)
class StructFieldBenchConfig:
    """Configuration shared by every v0.1 procedural family.

    ``base_seed`` selects a disjoint block inside each split seed domain.  A
    concrete sample remains fully identified by ``(split, family, index,
    base_seed)``; no Python process-randomized hash participates in generation.
    """

    height: int = 96
    width: int = 96
    num_orientations: int = 18
    max_components: int = 3
    base_frequency_cpp: float = 0.08
    base_seed: int = 0
    background_intensity: float = 0.05

    def __post_init__(self) -> None:
        integer_fields = {
            "height": self.height,
            "width": self.width,
            "num_orientations": self.num_orientations,
            "max_components": self.max_components,
            "base_seed": self.base_seed,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.height < 24 or self.width < 24:
            raise ValueError("height and width must each be at least 24 pixels")
        if self.num_orientations < 2:
            raise ValueError("num_orientations must be at least 2")
        if self.max_components < 3:
            raise ValueError("max_components must be at least 3 for the canonical junction families")
        if not 0 <= self.base_seed <= MAX_BASE_SEED:
            raise ValueError(f"base_seed must lie in [0, {MAX_BASE_SEED}]")
        if isinstance(self.base_frequency_cpp, bool) or not isinstance(self.base_frequency_cpp, (int, float)):
            raise TypeError("base_frequency_cpp must be a real number")
        if not math.isfinite(float(self.base_frequency_cpp)) or not 0.0 < self.base_frequency_cpp < 0.25:
            raise ValueError("base_frequency_cpp must be finite and lie in (0, 0.25)")
        if isinstance(self.background_intensity, bool) or not isinstance(self.background_intensity, (int, float)):
            raise TypeError("background_intensity must be a real number")
        if not math.isfinite(float(self.background_intensity)) or not 0.0 <= self.background_intensity < 1.0:
            raise ValueError("background_intensity must be finite and lie in [0, 1)")


def _orientation_bin_centers(num_orientations: int) -> tuple[float, ...]:
    return tuple(index * math.pi / num_orientations for index in range(num_orientations))


def _assemble_ground_truth(
    scene: FamilyScene,
    *,
    num_orientations: int,
    max_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack analytic components and linearly splat axial orientation bins."""
    height, width = scene.junction_mask.shape
    orientation_mass = torch.zeros(num_orientations, height, width, dtype=torch.float32)
    orientation_count = torch.zeros(height, width, dtype=torch.int64)
    component_frequency = torch.zeros(max_components, height, width, dtype=torch.float32)

    for component in scene.components:
        active = component.support
        if bool((active & (orientation_count >= max_components)).any()):
            raise ValueError("family scene exceeds configured max_components at one or more pixels")

        positions = component.orientation_rad * (num_orientations / math.pi)
        lower_unwrapped = torch.floor(positions).to(torch.int64)
        lower = torch.remainder(lower_unwrapped, num_orientations)
        upper = torch.remainder(lower + 1, num_orientations)
        upper_weight = positions - lower_unwrapped.to(positions.dtype)
        lower_weight = 1.0 - upper_weight
        orientation_mass.scatter_add_(
            0,
            lower.unsqueeze(0),
            torch.where(active, lower_weight, torch.zeros_like(lower_weight)).unsqueeze(0),
        )
        orientation_mass.scatter_add_(
            0,
            upper.unsqueeze(0),
            torch.where(active, upper_weight, torch.zeros_like(upper_weight)).unsqueeze(0),
        )

        for slot in range(max_components):
            occupies_slot = active & (orientation_count == slot)
            component_frequency[slot] = torch.where(
                occupies_slot,
                component.frequency_cpp,
                component_frequency[slot],
            )
        orientation_count = orientation_count + active.to(torch.int64)

    valid_mask = orientation_count > 0
    divisor = orientation_count.clamp_min(1).to(torch.float32).unsqueeze(0)
    orientation_distribution = orientation_mass / divisor
    uniform = torch.full_like(orientation_distribution, 1.0 / num_orientations)
    orientation_distribution = torch.where(valid_mask.unsqueeze(0), orientation_distribution, uniform)
    return orientation_distribution, orientation_count, component_frequency, valid_mask


def _compose_image(
    scene: FamilyScene,
    valid_mask: torch.Tensor,
    *,
    difficulty: str,
    background_intensity: float,
) -> tuple[torch.Tensor, float]:
    signal_sum = torch.zeros_like(scene.components[0].signal)
    signal_count = torch.zeros_like(signal_sum)
    for component in scene.components:
        signal_sum = signal_sum + torch.where(component.support, component.signal, torch.zeros_like(component.signal))
        signal_count = signal_count + component.support.to(signal_sum.dtype)
    mean_signal = signal_sum / signal_count.clamp_min(1.0)
    contrast = {"easy": 0.90, "medium": 0.82, "hard": 0.72}[difficulty]
    contrast = min(contrast, 1.0 - background_intensity)
    rendered = background_intensity + contrast * mean_signal
    image = torch.where(valid_mask, rendered, torch.full_like(rendered, background_intensity))
    return image.clamp(0.0, 1.0).unsqueeze(0), contrast


class StructFieldBenchGenerator:
    """Generate deterministic, schema-validated benchmark samples."""

    def __init__(self, config: StructFieldBenchConfig | None = None) -> None:
        self.config = StructFieldBenchConfig() if config is None else config
        if not isinstance(self.config, StructFieldBenchConfig):
            raise TypeError("config must be a StructFieldBenchConfig")
        if FAMILY_NAMES != CANONICAL_FAMILIES:
            raise RuntimeError("family renderer vocabulary does not match the frozen benchmark schema")

    def generate_sample(
        self,
        family: BenchmarkFamily | str,
        *,
        split: BenchmarkSplit | str = "train",
        index: int = 0,
        difficulty: str = "medium",
    ) -> BenchmarkSample:
        """Generate one sample identified by an explicit split and index."""
        if family not in CANONICAL_FAMILIES:
            raise ValueError(f"family must be one of {CANONICAL_FAMILIES}")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be 'easy', 'medium', or 'hard'")

        typed_family = cast(BenchmarkFamily, family)
        typed_split = cast(BenchmarkSplit, split)

        seed = derive_sample_seed(typed_split, typed_family, index, self.config.base_seed)
        sample_id = make_sample_id(
            typed_split,
            typed_family,
            index,
            self.config.base_seed,
            difficulty=difficulty,
        )
        local_generator = torch.Generator(device="cpu")
        local_generator.manual_seed(seed)
        scene = render_family(
            typed_family,
            height=self.config.height,
            width=self.config.width,
            difficulty=difficulty,
            base_frequency=float(self.config.base_frequency_cpp),
            generator=local_generator,
        )
        orientation_distribution, orientation_count, component_frequency, valid_mask = _assemble_ground_truth(
            scene,
            num_orientations=self.config.num_orientations,
            max_components=self.config.max_components,
        )
        junction_mask = scene.junction_mask & valid_mask
        image, image_contrast = _compose_image(
            scene,
            valid_mask,
            difficulty=difficulty,
            background_intensity=float(self.config.background_intensity),
        )
        parameters = {
            **scene.parameters,
            "height": self.config.height,
            "width": self.config.width,
            "base_seed": self.config.base_seed,
            "sample_seed": seed,
            "sample_index": index,
            "difficulty": difficulty,
            "requested_base_frequency_cpp": float(self.config.base_frequency_cpp),
            "background_intensity": float(self.config.background_intensity),
            "image_contrast": image_contrast,
        }
        metadata = BenchmarkMetadata(
            benchmark_version=BENCHMARK_VERSION,
            generator_version=GENERATOR_VERSION,
            sample_id=sample_id,
            seed=seed,
            split=typed_split,
            family=typed_family,
            difficulty=difficulty,
            degradation="none",
            parameters=parameters,
            num_orientations=self.config.num_orientations,
            orientation_bin_centers_rad=_orientation_bin_centers(self.config.num_orientations),
            max_components=self.config.max_components,
        )
        return BenchmarkSample(
            image=image,
            orientation_distribution_gt=orientation_distribution,
            orientation_count=orientation_count.unsqueeze(0),
            component_frequency_gt=component_frequency,
            valid_mask=valid_mask.unsqueeze(0),
            junction_mask=junction_mask.unsqueeze(0),
            metadata=metadata,
        )


def generate_sample(
    family: BenchmarkFamily | str,
    *,
    split: BenchmarkSplit | str = "train",
    index: int = 0,
    difficulty: str = "medium",
    config: StructFieldBenchConfig | None = None,
) -> BenchmarkSample:
    """Convenience wrapper around :class:`StructFieldBenchGenerator`."""
    return StructFieldBenchGenerator(config).generate_sample(
        family,
        split=split,
        index=index,
        difficulty=difficulty,
    )
