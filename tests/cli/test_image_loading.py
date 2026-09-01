"""Input decoding and device-selection contracts for the estimate CLI."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from structfield.cli.estimate import load_image, resolve_device


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def _assert_input_metadata(
    metadata: dict[str, object],
    *,
    source_format: str,
    source_dtype: str,
    source_shape: tuple[int, ...],
    tensor_shape: tuple[int, ...],
) -> None:
    assert metadata["format"] == source_format
    assert metadata["source_dtype"] == source_dtype
    assert metadata["source_shape"] == list(source_shape)
    assert metadata["tensor_shape"] == list(tensor_shape)
    assert isinstance(metadata["sha256"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"])
    value_range = metadata["value_range"]
    assert isinstance(value_range, list)
    assert len(value_range) == 2
    assert 0.0 <= float(value_range[0]) <= float(value_range[1]) <= 1.0


@pytest.mark.parametrize("channels", (1, 3))
def test_png_load_roundtrip_preserves_pixels_and_channels(tmp_path: Path, channels: int) -> None:
    height, width = 7, 11
    values = np.arange(height * width * channels, dtype=np.uint8)
    if channels == 1:
        source = values.reshape(height, width)
        expected = torch.from_numpy(source).unsqueeze(0)
    else:
        source = values.reshape(height, width, channels)
        expected = torch.from_numpy(source).permute(2, 0, 1)
    path = tmp_path / f"input-{channels}ch.png"
    _save_png(path, source)

    tensor, metadata = load_image(path)

    expected = expected.unsqueeze(0).to(dtype=torch.float32) / 255.0
    assert tensor.shape == (1, channels, height, width)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor.is_contiguous()
    torch.testing.assert_close(tensor, expected, rtol=0.0, atol=0.0)
    _assert_input_metadata(
        metadata,
        source_format="png",
        source_dtype="uint8",
        source_shape=source.shape,
        tensor_shape=tuple(tensor.shape),
    )


def test_png_i16_preserves_single_channel_and_uint16_normalization(tmp_path: Path) -> None:
    source = np.array([[0, 32768, 65535]], dtype=np.uint16)
    path = tmp_path / "input-i16.png"
    Image.fromarray(source).save(path)
    with Image.open(path) as opened:
        assert opened.mode == "I;16"

    tensor, metadata = load_image(path)

    expected = torch.from_numpy(source.astype(np.float32) / 65535.0)[None, None]
    assert tensor.shape == (1, 1, 1, 3)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, expected, rtol=1e-6, atol=1e-7)
    assert tensor[0, 0, 0, 1].item() == pytest.approx(32768.0 / 65535.0)
    _assert_input_metadata(
        metadata,
        source_format="png",
        source_dtype="uint16",
        source_shape=source.shape,
        tensor_shape=tuple(tensor.shape),
    )


@pytest.mark.parametrize("layout", ("hw", "hwc", "chw", "bchw"))
def test_npy_load_accepts_documented_unambiguous_layouts(tmp_path: Path, layout: str) -> None:
    height, width = 7, 11
    hwc = np.linspace(0.0, 1.0, height * width * 3, dtype=np.float32).reshape(height, width, 3)
    if layout == "hw":
        source = hwc[..., 0]
        expected = torch.from_numpy(source).unsqueeze(0)
    elif layout == "hwc":
        source = hwc
        expected = torch.from_numpy(source).permute(2, 0, 1)
    elif layout == "chw":
        source = np.transpose(hwc, (2, 0, 1))
        expected = torch.from_numpy(source.copy())
    else:
        source = np.transpose(hwc, (2, 0, 1))[None, ...]
        expected = torch.from_numpy(source[0].copy())
    path = tmp_path / f"input-{layout}.npy"
    np.save(path, source)

    tensor, metadata = load_image(path)

    expected = expected.unsqueeze(0).to(dtype=torch.float32)
    assert tensor.shape == expected.shape
    assert tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    torch.testing.assert_close(tensor, expected, rtol=1e-6, atol=1e-7)
    _assert_input_metadata(
        metadata,
        source_format="npy",
        source_dtype="float32",
        source_shape=source.shape,
        tensor_shape=tuple(tensor.shape),
    )


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32))
def test_npy_unsigned_integer_values_use_dtype_range_normalization(
    tmp_path: Path,
    dtype: type[np.unsignedinteger],
) -> None:
    maximum = np.iinfo(dtype).max
    source = np.array([[0, maximum // 2, maximum]], dtype=dtype)
    path = tmp_path / f"unsigned-{np.dtype(dtype).name}.npy"
    np.save(path, source)

    tensor, metadata = load_image(path)

    expected = torch.from_numpy(source.astype(np.float32) / float(maximum))[None, None]
    torch.testing.assert_close(tensor, expected, rtol=1e-6, atol=1e-7)
    _assert_input_metadata(
        metadata,
        source_format="npy",
        source_dtype=np.dtype(dtype).name,
        source_shape=source.shape,
        tensor_shape=tuple(tensor.shape),
    )


@pytest.mark.parametrize(
    "source",
    (
        np.array([[True, False]], dtype=np.bool_),
        np.array([[0, 1]], dtype=np.int16),
        np.array([[0.0 + 0.0j]], dtype=np.complex64),
        np.array([[object()]], dtype=object),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[np.inf]], dtype=np.float32),
        np.array([[-0.01]], dtype=np.float32),
        np.array([[1.01]], dtype=np.float32),
    ),
    ids=("bool", "signed-int", "complex", "object", "nan", "inf", "negative", "above-one"),
)
def test_npy_rejects_unsafe_dtypes_and_invalid_float_values(tmp_path: Path, source: np.ndarray) -> None:
    path = tmp_path / "invalid-values.npy"
    np.save(path, source)

    with pytest.raises((TypeError, ValueError)):
        load_image(path)


@pytest.mark.parametrize(
    "shape",
    (
        (),
        (7,),
        (0, 7),
        (7, 11, 2),
        (3, 7, 3),
        (2, 1, 7, 11),
        (1, 7, 11, 3),
        (1, 1, 1, 7, 11),
    ),
    ids=("scalar", "vector", "empty", "two-channels", "ambiguous-3d", "batch-two", "nhwc", "rank-five"),
)
def test_npy_rejects_unsupported_or_ambiguous_shapes(tmp_path: Path, shape: tuple[int, ...]) -> None:
    path = tmp_path / "invalid-shape.npy"
    np.save(path, np.zeros(shape, dtype=np.float32))

    with pytest.raises(ValueError):
        load_image(path)


def test_load_image_rejects_missing_and_unsupported_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.npy")

    unsupported = tmp_path / "input.jpg"
    unsupported.write_bytes(b"not-an-image")
    with pytest.raises(ValueError):
        load_image(unsupported)

    directory = tmp_path / "input.npy"
    directory.mkdir()
    with pytest.raises((IsADirectoryError, ValueError)):
        load_image(directory)


def test_resolve_device_cpu_and_auto_are_explicit() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    expected = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert resolve_device("auto") == expected


def test_resolve_device_cuda_contract_matches_runtime_availability() -> None:
    if torch.cuda.is_available():
        assert resolve_device("cuda") == torch.device("cuda")
    else:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


@pytest.mark.parametrize("device_request", ("gpu", "", 1, None))
def test_resolve_device_rejects_unknown_or_non_string_requests(device_request: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_device(device_request)  # type: ignore[arg-type]
