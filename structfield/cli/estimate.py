"""Checkpoint-free CLI for analytic structural-field estimation."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import secrets
import stat
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from structfield import __version__
from structfield.configs import read_estimator_config
from structfield.estimators import AnalyticStructuralFieldEstimator

SCHEMA_VERSION = "structfield-estimate-v1"
CURRENT_SCHEMA_VERSION = "structfield-estimate-current-v1"
GENERATION_RE = re.compile(r"^[0-9a-f]{32}$")
OUTPUT_FILENAMES = {
    "orientation_probability": "orientation_probability.npy",
    "orientation": "orientation.npy",
    "entropy": "entropy.npy",
    "frequency": "frequency.npy",
    "coherence": "coherence.npy",
    "uncertainty": "uncertainty.npy",
}
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
VISUALIZATION_NAMES = (
    "orientation_wheel",
    "orientation_probability",
    "orientation_overlay",
    "uncertainty_heatmap",
    "frequency_heatmap",
    "coherence_heatmap",
)
_LEGACY_OUTPUT_NAMES = {
    *OUTPUT_FILENAMES.values(),
    "metadata.json",
    *(f"{name}.{suffix}" for name in VISUALIZATION_NAMES for suffix in ("svg", "png")),
}


def _require_no_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    current = absolute if absolute.exists() else absolute.parent
    while True:
        if current.is_symlink():
            raise ValueError(f"path must not traverse a symbolic link: {current}")
        if current == current.parent:
            break
        current = current.parent


def _read_regular_bytes(path: Path, *, max_bytes: int | None = None) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"expected a regular non-symlink file: {path}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"expected a singly linked regular file: {path}")
        if max_bytes is not None and info.st_size > max_bytes:
            raise ValueError(f"file exceeds safety limit: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
        ):
            raise ValueError(f"file changed while being read: {path}")
        return b"".join(chunks), info
    finally:
        os.close(descriptor)


def resolve_current_bundle(output_dir: str | Path) -> Path:
    """Validate and return the currently selected immutable output bundle."""
    output_root = Path(output_dir).absolute()
    _require_no_symlink_ancestors(output_root)
    if output_root.is_symlink():
        raise ValueError("output root must not be a symlink")
    if not output_root.is_dir():
        raise FileNotFoundError(f"current bundle metadata is missing: {output_root}")

    pointer = output_root / "current.json"
    if pointer.is_symlink() or not pointer.is_file():
        raise FileNotFoundError(f"missing current pointer: {pointer}")
    try:
        pointer_bytes, _ = _read_regular_bytes(pointer, max_bytes=4096)
        payload = json.loads(pointer_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("current pointer is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("current pointer must be a mapping")
    if set(payload) != {"schema_version", "generation"}:
        raise ValueError("current pointer fields are invalid")
    if payload.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ValueError("unsupported current pointer schema")
    generation = payload.get("generation")
    if not isinstance(generation, str) or not GENERATION_RE.fullmatch(generation):
        raise ValueError("invalid generation in current pointer")

    bundles = output_root / "bundles"
    if bundles.is_symlink() or not bundles.is_dir():
        raise FileNotFoundError("bundles directory is unavailable")
    bundle = bundles / generation
    if not bundle.is_dir() or bundle.is_symlink():
        raise FileNotFoundError(f"active bundle is unavailable: {bundle}")
    metadata_path = bundle / "metadata.json"
    try:
        metadata_bytes, _ = _read_regular_bytes(metadata_path, max_bytes=1 << 20)
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("bundle metadata is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported bundle metadata schema")
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(OUTPUT_FILENAMES):
        raise ValueError("bundle output manifest is incomplete")
    expected = {"metadata.json"}
    for name, expected_filename in OUTPUT_FILENAMES.items():
        description = outputs[name]
        if not isinstance(description, dict) or description.get("file") != expected_filename:
            raise ValueError(f"invalid output description for {name}")
        if set(("bytes", "sha256")) - set(description):
            raise ValueError(f"missing integrity metadata for {name}")
        candidate = bundle / expected_filename
        payload_bytes, info = _read_regular_bytes(candidate)
        if description["bytes"] != info.st_size:
            raise ValueError(f"size mismatch for {expected_filename}")
        if description["sha256"] != hashlib.sha256(payload_bytes).hexdigest():
            raise ValueError(f"SHA-256 mismatch for {expected_filename}")
        expected.add(expected_filename)
    visualizations = metadata.get("visualizations", {})
    if not isinstance(visualizations, dict):
        raise ValueError("visualizations manifest must be an object")
    for filenames in visualizations.values():
        if not isinstance(filenames, list):
            raise ValueError("visualization filenames must be a list")
        for filename in filenames:
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or Path(filename).suffix not in {".svg", ".png"}
            ):
                raise ValueError("unsafe visualization filename")
            _read_regular_bytes(bundle / filename)
            expected.add(filename)
    actual = {entry.name for entry in os.scandir(bundle)}
    if actual != expected:
        raise ValueError("bundle contains missing or undeclared files")
    return bundle


def _find_flat_legacy_output(output_root: Path) -> set[str]:
    return {name for name in _LEGACY_OUTPUT_NAMES if os.path.lexists(output_root / name)}


def _fsync_path(path: Path) -> None:
    open_flags = getattr(os, "O_RDONLY", 0)
    if path.is_dir():
        open_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    else:
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, open_flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_input_bytes(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _normalize_values(array: np.ndarray, *, source: str) -> np.ndarray:
    if array.dtype.kind in "OcSV":
        raise TypeError(f"{source} must use a real numeric dtype")
    if array.dtype.kind == "b":
        raise TypeError(f"{source} cannot be boolean")
    if array.dtype.kind == "c":
        raise TypeError(f"{source} cannot be complex")
    if array.dtype.kind == "u":
        maximum = np.iinfo(array.dtype).max
        result = array.astype(np.float32) / float(maximum)
    elif array.dtype.kind == "i":
        raise TypeError(f"{source} signed integer arrays have ambiguous intensity scaling")
    elif array.dtype.kind == "f":
        result = array.astype(np.float32)
    else:
        raise TypeError(f"{source} must use a real numeric dtype")
    if result.size == 0 or any(size < 1 for size in result.shape):
        raise ValueError(f"{source} must have non-empty dimensions")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{source} contains NaN or Inf")
    if float(result.min()) < -1e-7 or float(result.max()) > 1.0 + 1e-7:
        raise ValueError(f"{source} values must lie in [0,1]")
    return np.clip(result, 0.0, 1.0)


def _npy_to_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None]
    if array.ndim == 3:
        first_is_channel = array.shape[0] in {1, 3}
        last_is_channel = array.shape[-1] in {1, 3}
        if first_is_channel == last_is_channel:
            raise ValueError("3D NPY shape is ambiguous; use unambiguous CHW or HWC")
        return array if first_is_channel else np.moveaxis(array, -1, 0)
    if array.ndim == 4:
        if array.shape[0] != 1 or array.shape[1] not in {1, 3}:
            raise ValueError("4D NPY input must have shape [1,1,H,W] or [1,3,H,W]")
        return array[0]
    raise ValueError("NPY input must have shape HW, unambiguous CHW/HWC, or BCHW with B=1")


def load_image(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a local PNG/NPY image as finite float32 BCHW in ``[0,1]``."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"input does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"input must be a file: {input_path}")
    payload, input_sha256 = _read_input_bytes(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".png":
        with Image.open(io.BytesIO(payload)) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode == "L":
                source_array = np.asarray(image)
                chw = source_array[None]
            elif image.mode.startswith("I;16") or image.mode == "I":
                decoded = np.asarray(image)
                if decoded.size == 0 or int(decoded.min()) < 0 or int(decoded.max()) > 65535:
                    raise ValueError("16-bit PNG values must lie in [0,65535]")
                source_array = decoded.astype(np.uint16, copy=False)
                chw = source_array[None]
            else:
                source_array = np.asarray(image.convert("RGB"))
                chw = np.moveaxis(source_array, -1, 0)
        input_format = "png"
    elif suffix == ".npy":
        source_array = np.load(io.BytesIO(payload), allow_pickle=False)
        if not isinstance(source_array, np.ndarray):  # pragma: no cover - defensive
            raise TypeError("NPY input did not contain an ndarray")
        chw = _npy_to_chw(source_array)
        input_format = "npy"
    else:
        raise ValueError("input extension must be .png or .npy")

    normalized = _normalize_values(np.asarray(chw), source=input_format.upper())
    if normalized.shape[0] not in {1, 3}:
        raise ValueError("input must contain one grayscale or three RGB channels")
    contiguous = np.ascontiguousarray(normalized)
    tensor = torch.from_numpy(contiguous).unsqueeze(0)
    metadata = {
        "format": input_format,
        "sha256": input_sha256,
        "source_dtype": str(source_array.dtype),
        "source_shape": list(source_array.shape),
        "tensor_shape": list(tensor.shape),
        "value_range": [float(tensor.min()), float(tensor.max())],
    }
    return tensor, metadata


def resolve_device(device: str) -> torch.device:
    """Resolve exactly ``cpu``, ``cuda``, or automatic CUDA preference."""
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    normalized = device.lower()
    if normalized not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if normalized == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _output_description(name: str, array: np.ndarray) -> dict[str, Any]:
    semantics = {
        "orientation_probability": (
            "probability",
            "full axial orientation probability distribution ordered by field bin centers",
        ),
        "orientation": ("radians", "top-1 axial tangent angle in [0,pi)"),
        "entropy": ("unitless", "normalized orientation-distribution entropy"),
        "frequency": ("cycles/pixel", "posterior mean local spatial frequency"),
        "coherence": ("unitless", "structure-tensor coherence in [0,1]"),
        "uncertainty": ("unitless", "heuristic structural uncertainty in [0,1]"),
    }
    unit, meaning = semantics[name]
    return {
        "file": OUTPUT_FILENAMES[name],
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "unit": unit,
        "meaning": meaning,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _expected_filenames(*, visualize: bool, has_gate: bool) -> tuple[str, ...]:
    names = [*OUTPUT_FILENAMES.values()]
    if visualize:
        figure_names = list(VISUALIZATION_NAMES)
        if has_gate:
            figure_names.append("gate_heatmap")
        for name in figure_names:
            names.extend((f"{name}.svg", f"{name}.png"))
    names.append("metadata.json")
    return tuple(names)


def _field_arrays(field: Any) -> dict[str, np.ndarray]:
    modes = field.topk_orientations(k=1)
    tensors = {
        "orientation_probability": field.orientation_prob,
        "orientation": modes.angles,
        "entropy": field.orientation_entropy,
        "frequency": field.frequency_mean,
        "coherence": field.coherence,
        "uncertainty": field.structural_uncertainty,
    }
    return {
        name: tensor[0].detach().cpu().numpy().copy()
        if name == "orientation_probability"
        else tensor[0, 0].detach().cpu().numpy().copy()
        for name, tensor in tensors.items()
    }


def _open_output_directory(destination: Path) -> None:
    _require_no_symlink_ancestors(destination)
    if destination.exists() and not destination.is_dir():
        raise ValueError("output must be a directory")
    if destination.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    created = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    if created:
        _fsync_path(destination.parent)
        _fsync_path(destination)


@contextlib.contextmanager
def _publish_lock(destination: Path) -> Iterator[None]:
    """Serialize cooperating publishers without creating crash-stale lock files."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_output_bundle(
    root: Path,
    generation: str,
    *,
    arrays: dict[str, np.ndarray],
    visualize: bool,
    gate: np.ndarray | None,
    metadata_payload: dict[str, Any],
    image: torch.Tensor,
) -> None:
    bundles = root / "bundles"
    if bundles.is_symlink():
        raise ValueError("bundles directory must not be a symbolic link")
    bundles.mkdir(parents=False, exist_ok=True)
    if not bundles.is_dir():
        raise ValueError("bundles must be a directory")
    _fsync_path(root)

    stage_dir = Path(tempfile.mkdtemp(prefix=".structfield-stage-", dir=bundles))
    try:
        for name, array in arrays.items():
            output_path = stage_dir / OUTPUT_FILENAMES[name]
            np.save(output_path, array, allow_pickle=False)
            _fsync_path(output_path)
            payload, info = _read_regular_bytes(output_path)
            metadata_payload["outputs"][name]["bytes"] = info.st_size
            metadata_payload["outputs"][name]["sha256"] = hashlib.sha256(payload).hexdigest()

        visualizations: dict[str, list[str]] | None = None
        if visualize:
            visualizations = _write_visualizations(
                image=image,
                arrays=arrays,
                output_dir=stage_dir,
                gate=gate,
            )
            metadata_payload["visualizations"] = visualizations
            for paths in visualizations.values():
                for filename in paths:
                    _fsync_path(stage_dir / filename)

        metadata_path = stage_dir / "metadata.json"
        with metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata_payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        for filename in _expected_filenames(visualize=visualize, has_gate=gate is not None):
            candidate = stage_dir / filename
            if not candidate.is_file():
                raise RuntimeError(f"missing staged output {candidate.name}")

        _fsync_path(stage_dir)
        target = root / "bundles" / generation
        if os.path.lexists(target):
            raise FileExistsError(f"generation already exists: {generation}")
        os.rename(stage_dir, target)
        _fsync_path(target)
        _fsync_path(bundles)
    except Exception:
        for path in stage_dir.rglob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
        if stage_dir.exists():
            try:
                stage_dir.rmdir()
            except OSError:
                pass
        raise


def _write_current_pointer(output_dir: Path, generation: str, *, overwrite: bool) -> None:
    pointer = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "generation": generation,
    }
    temporary = output_dir / f".current-{generation}.tmp"
    canonical = output_dir / "current.json"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(pointer, handle, ensure_ascii=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, canonical)
        else:
            try:
                os.link(temporary, canonical, follow_symlinks=False)
            except FileExistsError as exc:
                raise FileExistsError(
                    "current output already exists; use --overwrite to publish a new generation"
                ) from exc
            temporary.unlink()
        _fsync_path(canonical)
        _fsync_path(output_dir)
    finally:
        temporary.unlink(missing_ok=True)


def _load_gate(path: Path | None) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    if not path.is_file() or path.suffix.lower() != ".npy":
        raise ValueError("gate_path must be an existing .npy file")
    payload, gate_sha256 = _read_input_bytes(path)
    gate = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(gate, np.ndarray):  # pragma: no cover - np.load contract
        raise TypeError("gate NPY did not contain an ndarray")
    if gate.dtype.kind not in "buif":
        raise TypeError("gate must use a real numeric dtype")
    if gate.size == 0 or any(size < 1 for size in gate.shape):
        raise ValueError("gate must have non-empty dimensions")
    if gate.ndim == 4:
        if gate.shape[0] != 1:
            raise ValueError("gate batch dimension must equal one")
    elif gate.ndim not in {2, 3}:
        raise ValueError("gate must have shape [H,W], [C,H,W], or [1,C,H,W]")
    if not bool(np.isfinite(gate).all()):
        raise ValueError("gate contains NaN or Inf")
    if float(gate.min()) < -1e-7 or float(gate.max()) > 1.0 + 1e-7:
        raise ValueError("gate must lie in [0,1]")
    metadata = {
        "sha256": gate_sha256,
        "source_dtype": str(gate.dtype),
        "source_shape": list(gate.shape),
        "value_range": [float(gate.min()), float(gate.max())],
        "reduction": "mean",
    }
    return np.array(gate, copy=True), metadata


def _write_visualizations(
    *,
    image: torch.Tensor,
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    gate: np.ndarray | None,
) -> dict[str, list[str]]:
    import matplotlib.pyplot as plt

    from structfield.visualization import (
        plot_coherence_heatmap,
        plot_frequency_heatmap,
        plot_gate_heatmap,
        plot_orientation_overlay,
        plot_orientation_probability,
        plot_orientation_wheel,
        plot_uncertainty_heatmap,
        save_figure,
    )

    figures: dict[str, tuple[Any, Any]] = {}
    try:
        figures["orientation_wheel"] = plot_orientation_wheel()
        figures["orientation_probability"] = plot_orientation_probability(arrays["orientation_probability"])
        figures["orientation_overlay"] = plot_orientation_overlay(
            image,
            arrays["orientation"],
            coherence=arrays["coherence"],
        )
        figures["uncertainty_heatmap"] = plot_uncertainty_heatmap(arrays["uncertainty"])
        figures["frequency_heatmap"] = plot_frequency_heatmap(arrays["frequency"])
        figures["coherence_heatmap"] = plot_coherence_heatmap(arrays["coherence"])
        if gate is not None:
            figures["gate_heatmap"] = plot_gate_heatmap(gate)

        written: dict[str, list[str]] = {}
        for name, (figure, _) in figures.items():
            paths = save_figure(figure, output_dir / name)
            written[name] = [path.name for path in paths]
        return written
    finally:
        for figure, _ in figures.values():
            plt.close(figure)


def _stage_or_root_conflict(output_dir: Path, *, overwrite: bool) -> None:
    _open_output_directory(output_dir)
    current = output_dir / "current.json"
    if os.path.lexists(current) and not overwrite:
        raise FileExistsError("current output already exists; use --overwrite to publish a new generation")
    bundles = output_dir / "bundles"
    if bundles.is_symlink() or (bundles.exists() and not bundles.is_dir()):
        raise ValueError("bundles must be a non-symlink directory")
    legacy = _find_flat_legacy_output(output_dir)
    if legacy and not overwrite:
        raise FileExistsError(
            "existing legacy flat outputs detected; use --overwrite to create a new immutable generation"
        )


def _cleanup_legacy_outputs(output_dir: Path) -> None:
    for name in _find_flat_legacy_output(output_dir):
        path = output_dir / name
        if path.is_dir():
            # never expected; keep unknown directories untouched
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def run_estimation(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    config: str = "analytic_default.yaml",
    num_orientations: int | None = None,
    overwrite: bool = False,
    visualize: bool = False,
    gate_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate one image and write a versioned generation-aware output bundle."""
    if not isinstance(overwrite, bool) or not isinstance(visualize, bool):
        raise TypeError("overwrite and visualize must be bool values")
    if gate_path is not None and not visualize:
        raise ValueError("gate_path requires visualize=True")

    gate_path_value = None if gate_path is None else Path(gate_path)
    gate, gate_metadata = _load_gate(gate_path_value)
    destination = Path(output_dir).absolute()
    _open_output_directory(destination)
    with _publish_lock(destination):
        _stage_or_root_conflict(destination, overwrite=overwrite)
        image, input_metadata = load_image(input_path)
        resolved_device = resolve_device(device)
        estimator_config = read_estimator_config(config)
        if num_orientations is not None:
            estimator_config = replace(estimator_config, num_orientations=num_orientations)
        estimator = AnalyticStructuralFieldEstimator(estimator_config).to(resolved_device)

        with torch.inference_mode():
            field = estimator(image.to(resolved_device))

        arrays = _field_arrays(field)
        generation = secrets.token_hex(16)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "input": input_metadata,
            "device": {
                "requested": device.lower(),
                "resolved": resolved_device.type,
                "cuda_available": bool(torch.cuda.is_available()),
            },
            "estimator": {
                "name": "analytic-v1",
                "config_resource": config,
                "num_orientations": estimator_config.num_orientations,
            },
            "field": field.metadata.to_dict(),
            "outputs": {name: _output_description(name, array) for name, array in arrays.items()},
        }
        if gate_metadata is not None:
            payload["gate"] = gate_metadata

        payload = json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))
        _write_output_bundle(
            destination,
            generation,
            arrays=arrays,
            visualize=visualize,
            gate=gate,
            metadata_payload=payload,
            image=image,
        )
        _write_current_pointer(destination, generation, overwrite=overwrite)

        if _find_flat_legacy_output(destination):
            _cleanup_legacy_outputs(destination)

        return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structfield-estimate",
        description="Estimate a probabilistic structural field from one PNG or NPY image.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--input", required=True, type=Path, help="Input .png or .npy image")
    parser.add_argument("--output", required=True, type=Path, help="Output root directory")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--config",
        choices=(
            "analytic_default.yaml",
            "analytic_8dir.yaml",
            "analytic_16dir.yaml",
            "analytic_32dir.yaml",
        ),
        default="analytic_default.yaml",
        help="Packaged analytic estimator configuration",
    )
    parser.add_argument(
        "--num-orientations",
        type=int,
        default=None,
        help="Optional override for the selected packaged configuration",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Create a new generation instead of using current layout",
    )
    parser.add_argument("--visualize", action="store_true", help="Also export SVG and PNG diagnostics")
    parser.add_argument("--gate-npy", type=Path, help="Optional local gate array; requires --visualize")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``structfield-estimate`` and module execution."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        metadata = run_estimation(
            args.input,
            args.output,
            device=args.device,
            config=args.config,
            num_orientations=args.num_orientations,
            overwrite=args.overwrite,
            visualize=args.visualize,
            gate_path=args.gate_npy,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    print(
        json.dumps(
            {
                "schema_version": metadata["schema_version"],
                "device": metadata["device"]["resolved"],
                "num_outputs": len(metadata["outputs"]),
            },
            sort_keys=True,
        )
    )
    return EXIT_SUCCESS


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess smoke
    raise SystemExit(main())


__all__ = [
    "OUTPUT_FILENAMES",
    "EXIT_RUNTIME_ERROR",
    "EXIT_SUCCESS",
    "EXIT_USAGE_ERROR",
    "CURRENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "VISUALIZATION_NAMES",
    "load_image",
    "main",
    "resolve_current_bundle",
    "resolve_device",
    "run_estimation",
]
