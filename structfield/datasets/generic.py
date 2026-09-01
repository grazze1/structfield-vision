"""Deterministic adapter for caller-provided, paired local folders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .base import DatasetRecord, _coerce_local_path, _validate_optional_text, _validate_text
from .validation import validate_dataset_records, validate_dataset_root

COMMON_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".npy")


def _validate_relative_directory(name: str, value: object) -> Path:
    path = _coerce_local_path(name, value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative directory within root")
    return path


def _normalize_extensions(name: str, values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of extensions")
    candidates = tuple(values)
    if not candidates:
        raise ValueError(f"{name} must not be empty")
    normalized: list[str] = []
    for value in candidates:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain strings")
        extension = value.lower()
        if value != value.strip() or not extension.startswith(".") or not extension[1:].isalnum():
            raise ValueError(f"invalid extension in {name}: {value!r}")
        if extension in normalized:
            raise ValueError(f"duplicate extension in {name}: {value!r}")
        normalized.append(extension)
    return tuple(normalized)


def _resolve_directory(root: Path, relative: Path, *, name: str) -> Path:
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a directory: {candidate}")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes dataset root: {candidate}") from error
    return resolved


def _file_index(
    directory: Path,
    *,
    root: Path,
    extensions: tuple[str, ...],
    recursive: bool,
    name: str,
) -> dict[str, Path]:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    index: dict[str, Path] = {}
    for candidate in sorted(candidates, key=lambda path: path.as_posix()):
        if candidate.suffix.lower() not in extensions or not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} contains a file that escapes dataset root: {candidate}") from error
        key = candidate.relative_to(directory).with_suffix("").as_posix()
        if key in index:
            raise ValueError(f"{name} contains multiple files for relative stem {key!r}")
        index[key] = resolved
    return index


def _require_same_keys(reference: dict[str, Path], candidate: dict[str, Path], *, name: str) -> None:
    missing = sorted(reference.keys() - candidate.keys())
    orphan = sorted(candidate.keys() - reference.keys())
    if missing:
        raise ValueError(f"{name} is missing relative stems: {missing!r}")
    if orphan:
        raise ValueError(f"{name} contains orphan relative stems: {orphan!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class GenericFolderDatasetAdapter:
    """Discover a strict relative-stem pairing from local folders.

    The adapter enumerates paths only. It never downloads, decodes, copies,
    extracts, caches, or otherwise mutates dataset content.
    """

    root: Path
    dataset_id: str = "generic-folder"
    image_dir: Path = Path("images")
    target_dir: Path | None = None
    valid_mask_dir: Path | None = None
    image_extensions: tuple[str, ...] = COMMON_IMAGE_EXTENSIONS
    target_extensions: tuple[str, ...] = COMMON_IMAGE_EXTENSIONS
    valid_mask_extensions: tuple[str, ...] = COMMON_IMAGE_EXTENSIONS
    recursive: bool = True
    split: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _coerce_local_path("root", self.root))
        object.__setattr__(self, "dataset_id", _validate_text("dataset_id", self.dataset_id))
        object.__setattr__(self, "split", _validate_optional_text("split", self.split))
        object.__setattr__(self, "image_dir", _validate_relative_directory("image_dir", self.image_dir))
        if self.target_dir is not None:
            object.__setattr__(self, "target_dir", _validate_relative_directory("target_dir", self.target_dir))
        if self.valid_mask_dir is not None:
            object.__setattr__(
                self,
                "valid_mask_dir",
                _validate_relative_directory("valid_mask_dir", self.valid_mask_dir),
            )
        directories = [path for path in (self.image_dir, self.target_dir, self.valid_mask_dir) if path is not None]
        if len(set(directories)) != len(directories):
            raise ValueError("image, target, and valid-mask directories must be distinct")
        object.__setattr__(
            self,
            "image_extensions",
            _normalize_extensions("image_extensions", self.image_extensions),
        )
        object.__setattr__(
            self,
            "target_extensions",
            _normalize_extensions("target_extensions", self.target_extensions),
        )
        object.__setattr__(
            self,
            "valid_mask_extensions",
            _normalize_extensions("valid_mask_extensions", self.valid_mask_extensions),
        )
        if not isinstance(self.recursive, bool):
            raise TypeError("recursive must be a bool")

    def records(self) -> tuple[DatasetRecord, ...]:
        """Return a deterministic, strictly paired local record tuple."""
        root = validate_dataset_root(self.root)
        image_directory = _resolve_directory(root, self.image_dir, name="image_dir")
        images = _file_index(
            image_directory,
            root=root,
            extensions=self.image_extensions,
            recursive=self.recursive,
            name="image_dir",
        )
        if not images:
            raise ValueError("image_dir contains no files with configured extensions")

        targets: dict[str, Path] | None = None
        if self.target_dir is not None:
            target_directory = _resolve_directory(root, self.target_dir, name="target_dir")
            targets = _file_index(
                target_directory,
                root=root,
                extensions=self.target_extensions,
                recursive=self.recursive,
                name="target_dir",
            )
            _require_same_keys(images, targets, name="target_dir")

        valid_masks: dict[str, Path] | None = None
        if self.valid_mask_dir is not None:
            valid_mask_directory = _resolve_directory(root, self.valid_mask_dir, name="valid_mask_dir")
            valid_masks = _file_index(
                valid_mask_directory,
                root=root,
                extensions=self.valid_mask_extensions,
                recursive=self.recursive,
                name="valid_mask_dir",
            )
            _require_same_keys(images, valid_masks, name="valid_mask_dir")

        records = tuple(
            DatasetRecord(
                dataset_id=self.dataset_id,
                sample_id=key,
                image_path=images[key],
                target_path=None if targets is None else targets[key],
                valid_mask_path=None if valid_masks is None else valid_masks[key],
                split=self.split,
            )
            for key in sorted(images)
        )
        return validate_dataset_records(
            records,
            root=root,
            require_target=targets is not None,
            require_valid_mask=valid_masks is not None,
        )


__all__ = ["COMMON_IMAGE_EXTENSIONS", "GenericFolderDatasetAdapter"]
