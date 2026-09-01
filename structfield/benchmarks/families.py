"""Project-authored analytic families for StructField-Bench-v0.1.

The renderers in this module intentionally favour transparent ground truth over
photorealism.  Every visible component carries an analytic axial orientation,
spatial frequency, and support mask; :mod:`.generator` converts those
components to the public benchmark schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

ParameterValue: TypeAlias = str | int | float | bool


@dataclass(frozen=True)
class AnalyticComponent:
    """One locally present orientation/frequency component."""

    orientation_rad: torch.Tensor
    frequency_cpp: torch.Tensor
    support: torch.Tensor
    signal: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.orientation_rad.shape
        if self.orientation_rad.ndim != 2:
            raise ValueError("component tensors must have shape [H, W]")
        if self.frequency_cpp.shape != shape or self.support.shape != shape or self.signal.shape != shape:
            raise ValueError("all component tensors must have identical shapes")
        if not torch.is_floating_point(self.orientation_rad) or not torch.is_floating_point(self.frequency_cpp):
            raise TypeError("component orientation and frequency must be floating point")
        if self.signal.dtype != self.orientation_rad.dtype or self.frequency_cpp.dtype != self.orientation_rad.dtype:
            raise TypeError("component floating tensors must share a dtype")
        if self.support.dtype != torch.bool:
            raise TypeError("component support must use torch.bool")
        if not bool(torch.isfinite(self.orientation_rad).all()):
            raise ValueError("component orientation contains NaN or Inf")
        if not bool(torch.isfinite(self.frequency_cpp).all()) or bool((self.frequency_cpp < 0.0).any()):
            raise ValueError("component frequency must be finite and non-negative")
        if not bool(torch.isfinite(self.signal).all()):
            raise ValueError("component signal contains NaN or Inf")


@dataclass(frozen=True)
class FamilyScene:
    """Internal analytic scene returned by every canonical family renderer."""

    components: tuple[AnalyticComponent, ...]
    junction_mask: torch.Tensor
    parameters: dict[str, ParameterValue]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a family scene must contain at least one component")
        shape = self.components[0].support.shape
        if self.junction_mask.shape != shape or self.junction_mask.dtype != torch.bool:
            raise ValueError("junction_mask must be bool with component spatial shape")
        if any(component.support.shape != shape for component in self.components):
            raise ValueError("family components must share spatial shape")


@dataclass(frozen=True)
class _Coordinates:
    x: torch.Tensor
    y: torch.Tensor
    x_normalized: torch.Tensor
    y_normalized: torch.Tensor
    min_size: float


def _coordinates(height: int, width: int) -> _Coordinates:
    rows = torch.arange(height, dtype=torch.float32)
    columns = torch.arange(width, dtype=torch.float32)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    x = column_grid - (width - 1.0) / 2.0
    # Displayed-image positive y points upward even though tensor rows increase downward.
    y = (height - 1.0) / 2.0 - row_grid
    half_width = max((width - 1.0) / 2.0, 1.0)
    half_height = max((height - 1.0) / 2.0, 1.0)
    return _Coordinates(
        x=x,
        y=y,
        x_normalized=x / half_width,
        y_normalized=y / half_height,
        min_size=float(min(height, width)),
    )


def _rand(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def _difficulty_value(difficulty: str) -> float:
    return {"easy": 0.25, "medium": 0.55, "hard": 0.85}[difficulty]


def _constant(reference: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full_like(reference, float(value))


def _wrap_axial(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle, math.pi)


def _line_coordinates(coordinates: _Coordinates, directed_angle_rad: float) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = math.cos(directed_angle_rad)
    sine = math.sin(directed_angle_rad)
    tangent = cosine * coordinates.x + sine * coordinates.y
    normal = -sine * coordinates.x + cosine * coordinates.y
    return tangent, normal


def _plane_signal(normal: torch.Tensor, frequency_cpp: float, phase_cycles: float) -> torch.Tensor:
    return 0.5 + 0.5 * torch.cos(2.0 * math.pi * (frequency_cpp * normal + phase_cycles))


def _component(
    orientation: torch.Tensor,
    frequency: torch.Tensor,
    support: torch.Tensor,
    signal: torch.Tensor,
) -> AnalyticComponent:
    return AnalyticComponent(
        orientation_rad=_wrap_axial(orientation).to(torch.float32),
        frequency_cpp=frequency.to(torch.float32),
        support=support,
        signal=signal.clamp(0.0, 1.0).to(torch.float32),
    )


def _straight(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    angle = _rand(generator, 0.08 * math.pi, 0.42 * math.pi)
    frequency = base_frequency * _rand(generator, 0.90, 1.10)
    phase = _rand(generator, 0.0, 1.0)
    _, normal = _line_coordinates(c, angle)
    reference = c.x
    component = _component(
        _constant(reference, angle),
        _constant(reference, frequency),
        torch.ones_like(reference, dtype=torch.bool),
        _plane_signal(normal, frequency, phase),
    )
    return FamilyScene(
        (component,),
        torch.zeros_like(reference, dtype=torch.bool),
        {
            "orientation_rad": angle,
            "frequency_cpp": frequency,
            "phase_cycles": phase,
            "difficulty_level": level,
        },
    )


def _parallel(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    angle = _rand(generator, 0.0, 0.35 * math.pi)
    frequency = base_frequency * _rand(generator, 0.75, 1.15)
    phase = _rand(generator, 0.0, 1.0)
    duty = 0.34 - 0.16 * level
    _, normal = _line_coordinates(c, angle)
    cosine = torch.cos(2.0 * math.pi * (frequency * normal + phase))
    threshold = math.cos(math.pi * duty)
    signal = torch.sigmoid((cosine - threshold) / 0.06)
    reference = c.x
    component = _component(
        _constant(reference, angle),
        _constant(reference, frequency),
        torch.ones_like(reference, dtype=torch.bool),
        signal,
    )
    return FamilyScene(
        (component,),
        torch.zeros_like(reference, dtype=torch.bool),
        {
            "orientation_rad": angle,
            "frequency_cpp": frequency,
            "band_duty_cycle": duty,
            "phase_cycles": phase,
            "difficulty_level": level,
        },
    )


def _curved(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    sign = -1.0 if _rand(generator, 0.0, 1.0) < 0.5 else 1.0
    curvature = sign * (0.08 + 0.20 * level) / c.min_size
    frequency = base_frequency * _rand(generator, 0.80, 1.05)
    phase = _rand(generator, 0.0, 1.0)
    slope = 2.0 * curvature * c.x
    level_coordinate = c.y - curvature * c.x.square()
    orientation = torch.atan2(slope, torch.ones_like(slope))
    local_frequency = frequency * torch.sqrt(1.0 + slope.square())
    component = _component(
        orientation,
        local_frequency,
        torch.ones_like(c.x, dtype=torch.bool),
        _plane_signal(level_coordinate, frequency, phase),
    )
    return FamilyScene(
        (component,),
        torch.zeros_like(c.x, dtype=torch.bool),
        {
            "curvature_per_pixel": curvature,
            "base_frequency_cpp": frequency,
            "phase_cycles": phase,
            "difficulty_level": level,
        },
    )


def _sinusoidal(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    amplitude = (0.04 + 0.10 * level) * c.min_size
    wavelength = (0.85 - 0.22 * level) * max(float(c.x.shape[1]), 1.0)
    offset = _rand(generator, 0.0, 2.0 * math.pi)
    frequency = base_frequency * _rand(generator, 0.72, 0.96)
    centerline = amplitude * torch.sin(2.0 * math.pi * c.x / wavelength + offset)
    slope = amplitude * (2.0 * math.pi / wavelength) * torch.cos(2.0 * math.pi * c.x / wavelength + offset)
    level_coordinate = c.y - centerline
    orientation = torch.atan2(slope, torch.ones_like(slope))
    local_frequency = frequency * torch.sqrt(1.0 + slope.square())
    component = _component(
        orientation,
        local_frequency,
        torch.ones_like(c.x, dtype=torch.bool),
        _plane_signal(level_coordinate, frequency, _rand(generator, 0.0, 1.0)),
    )
    return FamilyScene(
        (component,),
        torch.zeros_like(c.x, dtype=torch.bool),
        {
            "amplitude_pixels": amplitude,
            "wavelength_pixels": wavelength,
            "base_frequency_cpp": frequency,
            "phase_offset_rad": offset,
            "difficulty_level": level,
        },
    )


def _radial(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    del base_frequency
    level = _difficulty_value(difficulty)
    spokes = int(round(7.0 + 8.0 * level))
    phase = _rand(generator, 0.0, 2.0 * math.pi)
    radius = torch.sqrt(c.x.square() + c.y.square())
    polar_angle = torch.atan2(c.y, c.x)
    safe_radius = radius.clamp_min(1.0)
    local_frequency = spokes / (2.0 * math.pi * safe_radius)
    singular_radius = spokes / (2.0 * math.pi * 0.36)
    support = radius >= singular_radius
    signal = 0.5 + 0.5 * torch.cos(spokes * polar_angle + phase)
    component = _component(polar_angle, local_frequency, support, signal)
    return FamilyScene(
        (component,),
        torch.zeros_like(c.x, dtype=torch.bool),
        {
            "num_spokes": spokes,
            "center_invalid_radius_pixels": singular_radius,
            "phase_offset_rad": phase,
            "difficulty_level": level,
        },
    )


def _concentric(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    center_x = _rand(generator, -0.08, 0.08) * c.min_size
    center_y = _rand(generator, -0.08, 0.08) * c.min_size
    x = c.x - center_x
    y = c.y - center_y
    radius = torch.sqrt(x.square() + y.square())
    polar_angle = torch.atan2(y, x)
    frequency = base_frequency * _rand(generator, 0.78, 1.08)
    phase = _rand(generator, 0.0, 1.0)
    support = radius >= 1.5
    component = _component(
        polar_angle + math.pi / 2.0,
        _constant(c.x, frequency),
        support,
        _plane_signal(radius, frequency, phase),
    )
    return FamilyScene(
        (component,),
        torch.zeros_like(c.x, dtype=torch.bool),
        {
            "center_x_pixels": center_x,
            "center_y_pixels": center_y,
            "frequency_cpp": frequency,
            "phase_cycles": phase,
            "difficulty_level": level,
        },
    )


def _corridor_component(
    c: _Coordinates,
    angle: float,
    frequency: float,
    phase: float,
    width: float,
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
    window_radius: float | None = None,
) -> AnalyticComponent:
    shifted = _Coordinates(
        x=c.x - center_x,
        y=c.y - center_y,
        x_normalized=c.x_normalized,
        y_normalized=c.y_normalized,
        min_size=c.min_size,
    )
    _, normal = _line_coordinates(shifted, angle)
    support = normal.abs() <= width
    if window_radius is not None:
        support = support & (shifted.x.square() + shifted.y.square() <= window_radius**2)
    return _component(
        _constant(c.x, angle),
        _constant(c.x, frequency),
        support,
        _plane_signal(normal, frequency, phase),
    )


def _crossing(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    first_angle = _rand(generator, 0.05 * math.pi, 0.25 * math.pi)
    separation = (0.28 + 0.18 * level) * math.pi
    second_angle = (first_angle + separation) % math.pi
    width = (0.10 - 0.025 * level) * c.min_size
    first_frequency = base_frequency * _rand(generator, 0.78, 1.02)
    second_frequency = base_frequency * _rand(generator, 1.08, 1.32)
    first = _corridor_component(c, first_angle, first_frequency, _rand(generator, 0.0, 1.0), width)
    second = _corridor_component(c, second_angle, second_frequency, _rand(generator, 0.0, 1.0), width)
    junction = first.support & second.support
    return FamilyScene(
        (first, second),
        junction,
        {
            "orientation_1_rad": first_angle,
            "orientation_2_rad": second_angle,
            "frequency_1_cpp": first_frequency,
            "frequency_2_cpp": second_frequency,
            "corridor_half_width_pixels": width,
            "difficulty_level": level,
        },
    )


def _dual_crossing(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    center_offset = 0.27 * float(c.x.shape[1])
    radius = 0.21 * c.min_size
    width = (0.065 - 0.012 * level) * c.min_size
    components: list[AnalyticComponent] = []
    junction = torch.zeros_like(c.x, dtype=torch.bool)
    parameters: dict[str, ParameterValue] = {
        "center_offset_pixels": center_offset,
        "window_radius_pixels": radius,
        "corridor_half_width_pixels": width,
        "difficulty_level": level,
    }
    for location, center_x in enumerate((-center_offset, center_offset), start=1):
        first_angle = _rand(generator, 0.02 * math.pi, 0.22 * math.pi)
        second_angle = (first_angle + _rand(generator, 0.30 * math.pi, 0.46 * math.pi)) % math.pi
        first_frequency = base_frequency * _rand(generator, 0.76, 1.00)
        second_frequency = base_frequency * _rand(generator, 1.05, 1.30)
        first = _corridor_component(
            c,
            first_angle,
            first_frequency,
            _rand(generator, 0.0, 1.0),
            width,
            center_x=center_x,
            window_radius=radius,
        )
        second = _corridor_component(
            c,
            second_angle,
            second_frequency,
            _rand(generator, 0.0, 1.0),
            width,
            center_x=center_x,
            window_radius=radius,
        )
        components.extend((first, second))
        junction |= first.support & second.support
        parameters[f"crossing_{location}_orientation_1_rad"] = first_angle
        parameters[f"crossing_{location}_orientation_2_rad"] = second_angle
        parameters[f"crossing_{location}_frequency_1_cpp"] = first_frequency
        parameters[f"crossing_{location}_frequency_2_cpp"] = second_frequency
    return FamilyScene(tuple(components), junction, parameters)


def _ray_component(
    c: _Coordinates,
    directed_angle: float,
    frequency: float,
    phase: float,
    width: float,
    overlap_backtrack: float,
) -> AnalyticComponent:
    tangent, normal = _line_coordinates(c, directed_angle)
    support = (tangent >= -overlap_backtrack) & (normal.abs() <= width)
    return _component(
        _constant(c.x, directed_angle),
        _constant(c.x, frequency),
        support,
        _plane_signal(normal, frequency, phase),
    )


def _bifurcation(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    branch_angle = (0.16 + 0.13 * level) * math.pi
    width = (0.055 + 0.012 * level) * c.min_size
    overlap = 0.75 * width
    frequency = base_frequency * _rand(generator, 0.82, 1.05)
    directed_angles = (-math.pi / 2.0, math.pi / 2.0 - branch_angle, math.pi / 2.0 + branch_angle)
    components = tuple(
        _ray_component(c, angle, frequency, _rand(generator, 0.0, 1.0), width, overlap) for angle in directed_angles
    )
    count = sum(
        (component.support.to(torch.int64) for component in components),
        start=torch.zeros_like(c.x, dtype=torch.int64),
    )
    junction = count >= 2
    return FamilyScene(
        components,
        junction,
        {
            "parent_orientation_rad": math.pi / 2.0,
            "left_branch_orientation_rad": (math.pi / 2.0 - branch_angle) % math.pi,
            "right_branch_orientation_rad": (math.pi / 2.0 + branch_angle) % math.pi,
            "branch_angle_rad": branch_angle,
            "frequency_cpp": frequency,
            "corridor_half_width_pixels": width,
            "difficulty_level": level,
        },
    )


def _junction(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    rotation = _rand(generator, 0.0, math.pi)
    width = (0.050 + 0.012 * level) * c.min_size
    overlap = 0.85 * width
    directed_angles = tuple(rotation + offset for offset in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0))
    components = tuple(
        _ray_component(
            c,
            angle,
            base_frequency * (0.86 + 0.10 * index),
            _rand(generator, 0.0, 1.0),
            width,
            overlap,
        )
        for index, angle in enumerate(directed_angles)
    )
    count = sum(
        (component.support.to(torch.int64) for component in components),
        start=torch.zeros_like(c.x, dtype=torch.int64),
    )
    junction = count >= 2
    return FamilyScene(
        components,
        junction,
        {
            "junction_rotation_rad": rotation,
            "branch_1_orientation_rad": directed_angles[0] % math.pi,
            "branch_2_orientation_rad": directed_angles[1] % math.pi,
            "branch_3_orientation_rad": directed_angles[2] % math.pi,
            "corridor_half_width_pixels": width,
            "difficulty_level": level,
        },
    )


def _varying_frequency(
    c: _Coordinates,
    generator: torch.Generator,
    difficulty: str,
    base_frequency: float,
) -> FamilyScene:
    level = _difficulty_value(difficulty)
    angle = _rand(generator, 0.0, math.pi)
    _, normal = _line_coordinates(c, angle)
    normal_scale = max(c.min_size / 2.0, 1.0)
    variation = 0.18 + 0.46 * level
    frequency = base_frequency * (1.0 + variation * normal / normal_scale)
    frequency = frequency.clamp_min(base_frequency * 0.25)
    phase = base_frequency * (normal + 0.5 * variation * normal.square() / normal_scale)
    component = _component(
        _constant(c.x, angle),
        frequency,
        torch.ones_like(c.x, dtype=torch.bool),
        0.5 + 0.5 * torch.cos(2.0 * math.pi * (phase + _rand(generator, 0.0, 1.0))),
    )
    return FamilyScene(
        (component,),
        torch.zeros_like(c.x, dtype=torch.bool),
        {
            "orientation_rad": angle,
            "base_frequency_cpp": base_frequency,
            "relative_frequency_variation": variation,
            "frequency_min_cpp": float(frequency.min()),
            "frequency_max_cpp": float(frequency.max()),
            "difficulty_level": level,
        },
    )


def _varying_thickness(
    c: _Coordinates,
    generator: torch.Generator,
    difficulty: str,
    base_frequency: float,
) -> FamilyScene:
    level = _difficulty_value(difficulty)
    angle = _rand(generator, 0.0, math.pi)
    tangent, normal = _line_coordinates(c, angle)
    tangent_scale = max(c.min_size / 2.0, 1.0)
    normalized_tangent = (tangent / tangent_scale).clamp(-1.0, 1.0)
    thin_duty = 0.13 + 0.03 * (1.0 - level)
    thick_duty = 0.52 + 0.16 * level
    duty = thin_duty + (thick_duty - thin_duty) * (normalized_tangent + 1.0) / 2.0
    phase = base_frequency * normal + _rand(generator, 0.0, 1.0)
    cosine = torch.cos(2.0 * math.pi * phase)
    threshold = torch.cos(math.pi * duty)
    signal = torch.sigmoid((cosine - threshold) / 0.055)
    component = _component(
        _constant(c.x, angle),
        _constant(c.x, base_frequency),
        torch.ones_like(c.x, dtype=torch.bool),
        signal,
    )
    return FamilyScene(
        (component,),
        torch.zeros_like(c.x, dtype=torch.bool),
        {
            "orientation_rad": angle,
            "frequency_cpp": base_frequency,
            "thin_duty_cycle": thin_duty,
            "thick_duty_cycle": thick_duty,
            "difficulty_level": level,
        },
    )


def _domain_boundary(
    c: _Coordinates,
    generator: torch.Generator,
    difficulty: str,
    base_frequency: float,
) -> FamilyScene:
    level = _difficulty_value(difficulty)
    first_angle = _rand(generator, 0.02 * math.pi, 0.26 * math.pi)
    second_angle = (first_angle + (0.28 + 0.18 * level) * math.pi) % math.pi
    boundary_amplitude = (0.03 + 0.08 * level) * c.min_size
    boundary_phase = _rand(generator, 0.0, 2.0 * math.pi)
    boundary = boundary_amplitude * torch.sin(math.pi * c.y_normalized + boundary_phase)
    signed_distance = c.x - boundary
    gap = 1.25
    left_support = signed_distance < -gap
    right_support = signed_distance > gap
    first_frequency = base_frequency * _rand(generator, 0.80, 1.00)
    second_frequency = base_frequency * _rand(generator, 1.05, 1.28)
    _, first_normal = _line_coordinates(c, first_angle)
    _, second_normal = _line_coordinates(c, second_angle)
    first = _component(
        _constant(c.x, first_angle),
        _constant(c.x, first_frequency),
        left_support,
        _plane_signal(first_normal, first_frequency, _rand(generator, 0.0, 1.0)),
    )
    second = _component(
        _constant(c.x, second_angle),
        _constant(c.x, second_frequency),
        right_support,
        _plane_signal(second_normal, second_frequency, _rand(generator, 0.0, 1.0)),
    )
    return FamilyScene(
        (first, second),
        torch.zeros_like(c.x, dtype=torch.bool),
        {
            "left_orientation_rad": first_angle,
            "right_orientation_rad": second_angle,
            "left_frequency_cpp": first_frequency,
            "right_frequency_cpp": second_frequency,
            "boundary_amplitude_pixels": boundary_amplitude,
            "boundary_phase_rad": boundary_phase,
            "invalid_boundary_half_width_pixels": gap,
            "difficulty_level": level,
        },
    )


def _occlusion(c: _Coordinates, generator: torch.Generator, difficulty: str, base_frequency: float) -> FamilyScene:
    level = _difficulty_value(difficulty)
    angle = _rand(generator, 0.0, math.pi)
    frequency = base_frequency * _rand(generator, 0.84, 1.14)
    center_x = _rand(generator, -0.18, 0.18)
    center_y = _rand(generator, -0.18, 0.18)
    radius_x = 0.18 + 0.13 * level
    radius_y = 0.14 + 0.11 * level
    ellipse = ((c.x_normalized - center_x) / radius_x).square() + ((c.y_normalized - center_y) / radius_y).square()
    occluded = ellipse <= 1.0
    _, normal = _line_coordinates(c, angle)
    component = _component(
        _constant(c.x, angle),
        _constant(c.x, frequency),
        ~occluded,
        _plane_signal(normal, frequency, _rand(generator, 0.0, 1.0)),
    )
    return FamilyScene(
        (component,),
        torch.zeros_like(c.x, dtype=torch.bool),
        {
            "orientation_rad": angle,
            "frequency_cpp": frequency,
            "occlusion_center_x_normalized": center_x,
            "occlusion_center_y_normalized": center_y,
            "occlusion_radius_x_normalized": radius_x,
            "occlusion_radius_y_normalized": radius_y,
            "difficulty_level": level,
        },
    )


_FAMILY_RENDERERS: dict[str, Callable[[_Coordinates, torch.Generator, str, float], FamilyScene]] = {
    "straight": _straight,
    "parallel": _parallel,
    "curved": _curved,
    "sinusoidal": _sinusoidal,
    "radial": _radial,
    "concentric": _concentric,
    "crossing": _crossing,
    "dual_crossing": _dual_crossing,
    "bifurcation": _bifurcation,
    "junction": _junction,
    "varying_frequency": _varying_frequency,
    "varying_thickness": _varying_thickness,
    "domain_boundary": _domain_boundary,
    "occlusion": _occlusion,
}

FAMILY_NAMES: tuple[str, ...] = tuple(_FAMILY_RENDERERS)


def render_family(
    family: str,
    *,
    height: int,
    width: int,
    difficulty: str,
    base_frequency: float,
    generator: torch.Generator,
) -> FamilyScene:
    """Render one canonical family using only the supplied local generator."""
    if family not in _FAMILY_RENDERERS:
        raise ValueError(f"unknown benchmark family {family!r}; expected one of {FAMILY_NAMES}")
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty must be 'easy', 'medium', or 'hard'")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    return _FAMILY_RENDERERS[family](_coordinates(height, width), generator, difficulty, base_frequency)
