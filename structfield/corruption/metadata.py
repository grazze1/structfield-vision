"""Typed immutable provenance for structural-field corruptions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

CorruptionName: TypeAlias = Literal[
    "orientation_shift",
    "orientation_noise",
    "frequency_scale",
    "frequency_noise",
    "spatial_shift",
    "scale_shift",
    "coherence_noise",
    "mode_swap",
    "field_dropout",
]
CorruptionValue: TypeAlias = str | int | float | bool
CorruptionParameters: TypeAlias = tuple[tuple[str, CorruptionValue], ...]

_CORRUPTION_NAMES = {
    "orientation_shift",
    "orientation_noise",
    "frequency_scale",
    "frequency_noise",
    "spatial_shift",
    "scale_shift",
    "coherence_noise",
    "mode_swap",
    "field_dropout",
}


@dataclass(frozen=True)
class CorruptionMetadata:
    """Reproducibility metadata returned with one corrupted field.

    Parameters are stored as an immutable ordered tuple rather than a mutable
    dictionary. ``seed`` is present only for stochastic corruptions and refers
    to a function-local :class:`torch.Generator`.
    """

    name: CorruptionName
    parameters: CorruptionParameters = ()
    seed: int | None = None
    schema_version: str = "0.2"

    def __post_init__(self) -> None:
        if self.name not in _CORRUPTION_NAMES:
            raise ValueError(f"unsupported corruption name: {self.name!r}")
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters must be an immutable tuple")
        keys: list[str] = []
        for item in self.parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each corruption parameter must be a (name, value) tuple")
            key, value = item
            if not isinstance(key, str) or not key:
                raise TypeError("corruption parameter names must be non-empty strings")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("corruption parameter values must be finite")
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError("corruption parameter values must be scalar")
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError("corruption parameter names must be unique")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an integer or None")
            if not 0 <= self.seed <= 2**63 - 1:
                raise ValueError("seed must lie in [0, 2**63 - 1]")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise TypeError("schema_version must be a non-empty string")

    @property
    def parameter_dict(self) -> dict[str, CorruptionValue]:
        """Return a mutable convenience copy of the immutable parameters."""
        return dict(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible corruption provenance."""
        return {
            "name": self.name,
            "parameters": self.parameter_dict,
            "seed": self.seed,
            "schema_version": self.schema_version,
        }


__all__ = [
    "CorruptionMetadata",
    "CorruptionName",
    "CorruptionParameters",
    "CorruptionValue",
]
