"""Pure, reproducible corruptions for structural-prior robustness tests."""

from .functional import (
    coherence_noise,
    field_dropout,
    frequency_noise,
    frequency_scale,
    mode_swap,
    orientation_noise,
    orientation_shift,
    scale_shift,
    spatial_shift,
)
from .metadata import CorruptionMetadata, CorruptionName, CorruptionParameters, CorruptionValue

__all__ = [
    "CorruptionMetadata",
    "CorruptionName",
    "CorruptionParameters",
    "CorruptionValue",
    "coherence_noise",
    "field_dropout",
    "frequency_noise",
    "frequency_scale",
    "mode_swap",
    "orientation_noise",
    "orientation_shift",
    "scale_shift",
    "spatial_shift",
]
