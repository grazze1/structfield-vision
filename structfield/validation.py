"""Validation strictness controls for runtime boundaries."""

from __future__ import annotations

from enum import Enum


class ValidationMode(str, Enum):
    """Runtime validation profile used by contracts and boundaries."""

    strict = "strict"
    trusted = "trusted"

    @classmethod
    def normalize(cls, mode: "ValidationMode | str") -> "ValidationMode":
        if isinstance(mode, ValidationMode):
            return mode
        if isinstance(mode, str):
            normalized = mode.strip().lower()
            try:
                return cls(normalized)
            except ValueError as error:
                raise ValueError(f"validation_mode must be 'strict' or 'trusted', got {mode!r}") from error
        raise TypeError("validation_mode must be a ValidationMode or one of {'strict', 'trusted'}")


def is_strict_validation(mode: ValidationMode | str) -> bool:
    """Return whether a mode still performs potentially synchronizing value checks."""

    return ValidationMode.normalize(mode) is ValidationMode.strict


__all__ = ["ValidationMode", "is_strict_validation"]
