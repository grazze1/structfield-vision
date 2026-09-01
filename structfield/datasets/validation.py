"""Fail-fast validation utilities for local dataset records."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .base import DatasetAdapter, DatasetRecord, StrPath, _coerce_local_path, _validate_text


def _validate_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def validate_dataset_root(root: StrPath) -> Path:
    """Resolve and validate a caller-provided local dataset directory."""
    candidate = _coerce_local_path("root", root)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"dataset root does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise ValueError(f"dataset root must be a directory: {candidate}")
    return resolved


def _resolve_record_file(name: str, path: Path, *, root: Path | None) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} escapes dataset root: {path}") from error
    return resolved


def _validate_record(
    record: DatasetRecord,
    *,
    root: Path | None,
    require_target: bool,
    require_valid_mask: bool,
) -> tuple[Path, Path | None, Path | None]:
    if not isinstance(record, DatasetRecord):
        raise TypeError("record must be a DatasetRecord")
    if require_target and record.target_path is None:
        raise ValueError(f"record {record.sample_id!r} requires a target_path")
    if require_valid_mask and record.valid_mask_path is None:
        raise ValueError(f"record {record.sample_id!r} requires a valid_mask_path")

    image = _resolve_record_file("image_path", record.image_path, root=root)
    target = None
    if record.target_path is not None:
        target = _resolve_record_file("target_path", record.target_path, root=root)
        if target == image:
            raise ValueError("image_path and target_path resolve to the same file")
    valid_mask = None
    if record.valid_mask_path is not None:
        valid_mask = _resolve_record_file("valid_mask_path", record.valid_mask_path, root=root)
        if valid_mask == image or valid_mask == target:
            raise ValueError("valid_mask_path must resolve to a distinct file")
    return image, target, valid_mask


def validate_dataset_record(
    record: DatasetRecord,
    *,
    root: StrPath | None = None,
    require_target: bool = False,
    require_valid_mask: bool = False,
) -> DatasetRecord:
    """Validate one record's files and optional root-containment contract."""
    _validate_bool("require_target", require_target)
    _validate_bool("require_valid_mask", require_valid_mask)
    resolved_root = None if root is None else validate_dataset_root(root)
    _validate_record(
        record,
        root=resolved_root,
        require_target=require_target,
        require_valid_mask=require_valid_mask,
    )
    return record


def validate_dataset_records(
    records: Iterable[DatasetRecord],
    *,
    root: StrPath | None = None,
    require_target: bool = False,
    require_valid_mask: bool = False,
    allow_empty: bool = False,
) -> tuple[DatasetRecord, ...]:
    """Validate a finite record collection and reject ambiguous identities."""
    _validate_bool("require_target", require_target)
    _validate_bool("require_valid_mask", require_valid_mask)
    _validate_bool("allow_empty", allow_empty)
    try:
        result = tuple(records)
    except TypeError as error:
        raise TypeError("records must be an iterable of DatasetRecord objects") from error
    if not result and not allow_empty:
        raise ValueError("records must not be empty")

    resolved_root = None if root is None else validate_dataset_root(root)
    identities: set[tuple[str, str]] = set()
    images: set[Path] = set()
    targets: set[Path] = set()
    valid_masks: set[Path] = set()
    for record in result:
        image, target, valid_mask = _validate_record(
            record,
            root=resolved_root,
            require_target=require_target,
            require_valid_mask=require_valid_mask,
        )
        identity = (record.dataset_id, record.sample_id)
        if identity in identities:
            raise ValueError(f"duplicate dataset/sample identity: {identity!r}")
        if image in images:
            raise ValueError(f"duplicate image file: {image}")
        if image in targets or image in valid_masks:
            raise ValueError(f"image file is reused in another record role: {image}")
        if target is not None and target in targets:
            raise ValueError(f"duplicate target file: {target}")
        if target is not None and (target in images or target in valid_masks):
            raise ValueError(f"target file is reused in another record role: {target}")
        if valid_mask is not None and (valid_mask in images or valid_mask in targets):
            raise ValueError(f"valid-mask file is reused in another record role: {valid_mask}")
        identities.add(identity)
        images.add(image)
        if target is not None:
            targets.add(target)
        if valid_mask is not None:
            valid_masks.add(valid_mask)
    return result


def validate_dataset_adapter(
    adapter: DatasetAdapter,
    *,
    require_target: bool = False,
    require_valid_mask: bool = False,
) -> tuple[DatasetRecord, ...]:
    """Discover and validate records from a protocol-compatible adapter."""
    if not isinstance(adapter, DatasetAdapter):
        raise TypeError("adapter must satisfy DatasetAdapter")
    dataset_id = _validate_text("adapter.dataset_id", adapter.dataset_id)
    root = validate_dataset_root(adapter.root)
    try:
        records = tuple(adapter.records())
    except TypeError as error:
        raise TypeError("adapter.records() must return an iterable of DatasetRecord objects") from error
    if any(not isinstance(record, DatasetRecord) for record in records):
        raise TypeError("adapter.records() must contain only DatasetRecord objects")
    mismatches = sorted({record.dataset_id for record in records if record.dataset_id != dataset_id})
    if mismatches:
        raise ValueError(f"adapter records must use dataset_id {dataset_id!r}; found {mismatches!r}")
    return validate_dataset_records(
        records,
        root=root,
        require_target=require_target,
        require_valid_mask=require_valid_mask,
    )


__all__ = [
    "validate_dataset_adapter",
    "validate_dataset_record",
    "validate_dataset_records",
    "validate_dataset_root",
]
