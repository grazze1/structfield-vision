"""Filesystem-only tests for deterministic generic-folder discovery."""

from __future__ import annotations

import socket
import urllib.request
from pathlib import Path

import pytest

from structfield.datasets import (
    DatasetAdapter,
    GenericFolderDatasetAdapter,
    validate_dataset_adapter,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_recursive_pairing_is_case_insensitive_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _touch(root / "images" / "z_scene" / "second.PNG")
    _touch(root / "targets" / "z_scene" / "second.TIFF")
    _touch(root / "masks" / "z_scene" / "second.GiF")
    _touch(root / "images" / "a_scene" / "first.JpG")
    _touch(root / "targets" / "a_scene" / "first.BMP")
    _touch(root / "masks" / "a_scene" / "first.NpY")

    adapter = GenericFolderDatasetAdapter(
        root=root,
        dataset_id="mixed-layout",
        image_dir="images",  # type: ignore[arg-type]
        target_dir="targets",  # type: ignore[arg-type]
        valid_mask_dir="masks",  # type: ignore[arg-type]
        image_extensions=(".png", ".jpg"),
        target_extensions=(".tiff", ".bmp"),
        valid_mask_extensions=(".gif", ".npy"),
        recursive=True,
        split="validation",
    )

    first_pass = adapter.records()
    second_pass = adapter.records()
    assert isinstance(adapter, DatasetAdapter)
    assert first_pass == second_pass
    assert tuple(record.sample_id for record in first_pass) == (
        "a_scene/first",
        "z_scene/second",
    )
    assert all(record.dataset_id == "mixed-layout" for record in first_pass)
    assert all(record.split == "validation" for record in first_pass)
    assert all(record.image_path.is_absolute() for record in first_pass)
    assert all(record.target_path is not None for record in first_pass)
    assert all(record.valid_mask_path is not None for record in first_pass)
    assert first_pass[0].target_path == (root / "targets" / "a_scene" / "first.BMP").resolve()
    assert first_pass[1].valid_mask_path == (root / "masks" / "z_scene" / "second.GiF").resolve()
    assert (
        validate_dataset_adapter(
            adapter,
            require_target=True,
            require_valid_mask=True,
        )
        == first_pass
    )


def test_unconfigured_optional_roles_remain_absent(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _touch(root / "images" / "sample.PNG")

    records = GenericFolderDatasetAdapter(
        root=root,
        image_extensions=(".png",),
    ).records()

    assert len(records) == 1
    assert records[0].sample_id == "sample"
    assert records[0].target_path is None
    assert records[0].valid_mask_path is None


def test_nonrecursive_mode_ignores_nested_candidates(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _touch(root / "images" / "top.png")
    _touch(root / "images" / "nested" / "hidden.png")

    records = GenericFolderDatasetAdapter(
        root=root,
        image_extensions=(".png",),
        recursive=False,
    ).records()

    assert tuple(record.sample_id for record in records) == ("top",)


@pytest.mark.parametrize("role", ("target", "valid_mask"))
@pytest.mark.parametrize("issue", ("missing", "orphan"))
def test_configured_roles_require_an_exact_relative_stem_bijection(
    tmp_path: Path,
    role: str,
    issue: str,
) -> None:
    root = tmp_path / f"{role}-{issue}"
    _touch(root / "images" / "nested" / "sample.png")
    role_dir = "targets" if role == "target" else "masks"
    (root / role_dir).mkdir(parents=True)
    if issue == "orphan":
        _touch(root / role_dir / "nested" / "sample.png")
        _touch(root / role_dir / "nested" / "orphan.png")

    options = {
        "target_dir": "targets" if role == "target" else None,
        "valid_mask_dir": "masks" if role == "valid_mask" else None,
    }
    adapter = GenericFolderDatasetAdapter(
        root=root,
        image_extensions=(".png",),
        target_extensions=(".png",),
        valid_mask_extensions=(".png",),
        **options,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError):
        adapter.records()


@pytest.mark.parametrize("role", ("image", "target", "valid_mask"))
def test_duplicate_relative_stems_are_rejected(tmp_path: Path, role: str) -> None:
    root = tmp_path / role
    _touch(root / "images" / "sample.png")
    options: dict[str, object] = {}
    if role == "image":
        _touch(root / "images" / "sample.jpg")
    elif role == "target":
        _touch(root / "targets" / "sample.tif")
        _touch(root / "targets" / "sample.bmp")
        options["target_dir"] = "targets"
    else:
        _touch(root / "masks" / "sample.tif")
        _touch(root / "masks" / "sample.bmp")
        options["valid_mask_dir"] = "masks"

    adapter = GenericFolderDatasetAdapter(
        root=root,
        image_extensions=(".png", ".jpg"),
        target_extensions=(".tif", ".bmp"),
        valid_mask_extensions=(".tif", ".bmp"),
        **options,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError):
        adapter.records()


@pytest.mark.parametrize(
    ("field", "extensions"),
    (
        ("image_extensions", ()),
        ("image_extensions", ("png",)),
        ("image_extensions", (".png", ".PNG")),
        ("target_extensions", ".png"),
        ("target_extensions", (".png/child",)),
        ("valid_mask_extensions", (".png", 1)),
    ),
)
def test_invalid_extension_configuration_fails_fast(
    tmp_path: Path,
    field: str,
    extensions: object,
) -> None:
    options = {field: extensions}
    with pytest.raises((TypeError, ValueError)):
        GenericFolderDatasetAdapter(root=tmp_path, **options)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options",
    (
        {"dataset_id": ""},
        {"split": " padded "},
        {"recursive": 1},
        {"image_dir": "../outside"},
        {"target_dir": "/absolute"},
        {"image_dir": "same", "target_dir": "same"},
    ),
)
def test_invalid_adapter_configuration_fails_fast(
    tmp_path: Path,
    options: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        GenericFolderDatasetAdapter(root=tmp_path, **options)  # type: ignore[arg-type]


def test_missing_empty_and_non_directory_inputs_fail_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GenericFolderDatasetAdapter(root=tmp_path / "missing").records()

    root_file = _touch(tmp_path / "root-file")
    with pytest.raises((NotADirectoryError, ValueError)):
        GenericFolderDatasetAdapter(root=root_file).records()

    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        GenericFolderDatasetAdapter(root=root).records()

    (root / "images").mkdir()
    with pytest.raises(ValueError):
        GenericFolderDatasetAdapter(root=root).records()

    image_dir_file = _touch(root / "not-a-directory")
    with pytest.raises((NotADirectoryError, ValueError)):
        GenericFolderDatasetAdapter(root=root, image_dir=image_dir_file.name).records()  # type: ignore[arg-type]

    _touch(root / "images" / "sample.png")
    with pytest.raises(FileNotFoundError):
        GenericFolderDatasetAdapter(root=root, target_dir="missing-targets").records()  # type: ignore[arg-type]


def test_directory_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside-images"
    _touch(outside / "sample.png")
    link = root / "images"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError):
        GenericFolderDatasetAdapter(root=root).records()


def test_file_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    images = root / "images"
    images.mkdir(parents=True)
    outside = _touch(tmp_path / "outside.png")
    link = images / "sample.png"
    try:
        link.symlink_to(outside)
    except OSError as error:  # pragma: no cover - platform capability guard
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError):
        GenericFolderDatasetAdapter(root=root).records()


def test_records_are_discovered_without_decoding_writes_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    image = _touch(root / "images" / "empty.png")
    target = _touch(root / "targets" / "empty.png")
    before = {path: (path.stat().st_mode, path.stat().st_size, path.stat().st_mtime_ns) for path in (image, target)}

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("dataset adapters must not use the network")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(urllib.request, "urlopen", reject_network)
    records = GenericFolderDatasetAdapter(
        root=root,
        target_dir="targets",  # type: ignore[arg-type]
    ).records()

    after = {path: (path.stat().st_mode, path.stat().st_size, path.stat().st_mtime_ns) for path in (image, target)}
    assert len(records) == 1
    assert before == after
    assert image.read_bytes() == b""
    assert target.read_bytes() == b""
