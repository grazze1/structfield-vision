"""Shared type aliases for the StructField core schema."""

from __future__ import annotations

from typing import TypeAlias

MetadataScalar: TypeAlias = str | int | float | bool | None
MetadataValue: TypeAlias = MetadataScalar | tuple[MetadataScalar, ...]
MetadataItems: TypeAlias = tuple[tuple[str, MetadataValue], ...]
