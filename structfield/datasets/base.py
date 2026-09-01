"""Typed, side-effect-free contracts for local dataset discovery."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

ReservedDatasetIdentifier: TypeAlias = Literal[
    "DRIVE",
    "CHASE_DB1",
    "HRF",
    "Crack500",
    "DeepGlobe",
]
RESERVED_DATASET_IDENTIFIERS: tuple[ReservedDatasetIdentifier, ...] = (
    "DRIVE",
    "CHASE_DB1",
    "HRF",
    "Crack500",
    "DeepGlobe",
)
"""Identifiers reserved for future dataset-specific adapters, not current support claims."""

StrPath: TypeAlias = str | os.PathLike[str]
JSONScalar: TypeAlias = str | int | float | bool | None
FrozenJSONValue: TypeAlias = JSONScalar | tuple["FrozenJSONValue", ...] | Mapping[str, "FrozenJSONValue"]


def _freeze_json(value: Any, *, path: str) -> FrozenJSONValue:
    if value is None or isinstance(value, (str, bool, int)):
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
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty string without surrounding whitespace")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL")
    return value


def _validate_optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _validate_text(name, value)


def _coerce_local_path(name: str, value: object) -> Path:
    if isinstance(value, (bool, bytes, bytearray)):
        raise TypeError(f"{name} must be a local string or path-like value")
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a local string or path-like value")
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip():
        raise TypeError(f"{name} must be a non-empty local path")
    lowered = raw.lstrip().lower()
    if "://" in raw or lowered.startswith("file:"):
        raise ValueError(f"{name} must be a local filesystem path, not a URL")
    return Path(raw).expanduser()


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetRecord:
    """Immutable local-file identity for one discovered dataset sample.

    Construction validates only structure. Use :func:`validate_dataset_record`
    when on-disk existence and root containment must be checked.
    """

    dataset_id: str
    sample_id: str
    image_path: Path
    target_path: Path | None = None
    valid_mask_path: Path | None = None
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _validate_text("dataset_id", self.dataset_id))
        object.__setattr__(self, "sample_id", _validate_text("sample_id", self.sample_id))
        object.__setattr__(self, "split", _validate_optional_text("split", self.split))

        image_path = _coerce_local_path("image_path", self.image_path)
        target_path = None if self.target_path is None else _coerce_local_path("target_path", self.target_path)
        valid_mask_path = (
            None if self.valid_mask_path is None else _coerce_local_path("valid_mask_path", self.valid_mask_path)
        )
        paths = [path for path in (image_path, target_path, valid_mask_path) if path is not None]
        if len(set(paths)) != len(paths):
            raise ValueError("image, target, and valid-mask paths must be distinct")
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "target_path", target_path)
        object.__setattr__(self, "valid_mask_path", valid_mask_path)

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        frozen_metadata = _freeze_json(self.metadata, path="metadata")
        if not isinstance(frozen_metadata, Mapping):  # pragma: no cover - guarded above
            raise AssertionError("metadata did not freeze to a mapping")
        object.__setattr__(self, "metadata", frozen_metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable record description."""
        return {
            "dataset_id": self.dataset_id,
            "sample_id": self.sample_id,
            "image_path": os.fspath(self.image_path),
            "target_path": None if self.target_path is None else os.fspath(self.target_path),
            "valid_mask_path": None if self.valid_mask_path is None else os.fspath(self.valid_mask_path),
            "split": self.split,
            "metadata": _thaw_json(self.metadata),
        }


@runtime_checkable
class DatasetAdapter(Protocol):
    """Minimal protocol implemented by read-only local dataset adapters."""

    @property
    def dataset_id(self) -> str:
        """Stable adapter-level dataset identifier."""

    @property
    def root(self) -> Path:
        """Caller-provided local dataset root."""

    def records(self) -> tuple[DatasetRecord, ...]:
        """Discover and validate records without decoding or copying files."""


__all__ = [
    "RESERVED_DATASET_IDENTIFIERS",
    "DatasetAdapter",
    "DatasetRecord",
    "FrozenJSONValue",
    "ReservedDatasetIdentifier",
    "StrPath",
]
