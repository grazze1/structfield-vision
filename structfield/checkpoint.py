"""Safe, schema-bound checkpoints for the three built-in application models."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from structfield._version import VERSION
from structfield.applications import APPLICATION_SCHEMA_VERSION
from structfield.applications.curvilinear_segmentation import CurvilinearSegmenter
from structfield.applications.fringe_restoration import FringeRestorer
from structfield.applications.texture_restoration import TextureRestorer
from structfield.estimators import AnalyticEstimatorConfig, AnalyticStructuralFieldEstimator

CHECKPOINT_SCHEMA_VERSION = "structfield-application-checkpoint-v1"
STRUCTURAL_FIELD_SCHEMA_VERSION = "1.0"
LEGACY_STRUCTURAL_FIELD_SCHEMA_VERSION = "0.2"
_DISTRIBUTION_NAME = "structfield-vision"
_MANIFEST_NAME = "manifest.json"
_WEIGHTS_NAME = "weights.pt"
_MAX_MANIFEST_BYTES = 1 << 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CheckpointError(ValueError):
    """Raised when a checkpoint violates the public safety contract."""


@dataclass(frozen=True)
class LoadedApplicationCheckpoint:
    """A reconstructed built-in model and its validated provenance."""

    model: nn.Module
    estimator: AnalyticStructuralFieldEstimator
    preprocessing: dict[str, Any]
    training_provenance: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _ModelSpec:
    model_type: type[nn.Module]
    constructor_fields: tuple[str, ...]


_MODEL_SPECS: dict[str, _ModelSpec] = {
    "curvilinear-segmenter": _ModelSpec(
        CurvilinearSegmenter,
        ("num_orientations", "width", "adapter_mode", "validation_mode"),
    ),
    "fringe-restorer": _ModelSpec(
        FringeRestorer,
        ("num_orientations", "width", "noise_dim", "adapter_mode", "validation_mode"),
    ),
    "texture-restorer": _ModelSpec(
        TextureRestorer,
        ("num_orientations", "width", "adapter_mode", "validation_mode"),
    ),
}


def _distribution_version() -> str:
    return VERSION


def _json_copy(name: str, value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite JSON-compatible data") from exc
    return json.loads(encoded)


def _require_exact_keys(name: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise CheckpointError(f"{name} fields must be exactly {sorted(expected)}, got {sorted(actual)}")
    return value


def _normalise_provenance(value: Any) -> dict[str, Any]:
    provenance = _json_copy("training_provenance", value)
    if not isinstance(provenance, dict):
        raise TypeError("training_provenance must be a mapping")
    if set(provenance) != {"status", "datasets"}:
        raise ValueError("training_provenance must contain exactly status and datasets")
    status_value = provenance["status"]
    datasets = provenance["datasets"]
    if status_value not in {"trained", "untrained"}:
        raise ValueError("training_provenance.status must be 'trained' or 'untrained'")
    if not isinstance(datasets, list):
        raise TypeError("training_provenance.datasets must be a list")
    if status_value == "trained" and not datasets:
        raise ValueError("trained checkpoints must list at least one training dataset")
    if status_value == "untrained" and datasets:
        raise ValueError("untrained checkpoints must use an empty datasets list")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise TypeError(f"training_provenance.datasets[{index}] must be an object")
        for required in ("name", "source", "license"):
            if not isinstance(dataset.get(required), str) or not dataset[required].strip():
                raise ValueError(f"training_provenance.datasets[{index}].{required} must be a non-empty string")
    return provenance


def _model_spec(model: nn.Module) -> tuple[str, _ModelSpec]:
    for name, spec in _MODEL_SPECS.items():
        if type(model) is spec.model_type:
            return name, spec
    allowed = ", ".join(spec.model_type.__name__ for spec in _MODEL_SPECS.values())
    raise TypeError(f"model must be exactly one of the built-in application models: {allowed}")


def _model_constructor(model: nn.Module, spec: _ModelSpec) -> dict[str, Any]:
    constructor: dict[str, Any] = {}
    for field in spec.constructor_fields:
        value = getattr(model, field)
        constructor[field] = value.value if hasattr(value, "value") else value
    return _json_copy("model constructor", constructor)


def _validate_tensor(name: str, tensor: Any) -> torch.Tensor:
    if type(tensor) is not torch.Tensor:
        raise CheckpointError(f"state_dict[{name!r}] must be a plain torch.Tensor")
    if tensor.device.type == "meta":
        raise CheckpointError(f"state_dict[{name!r}] must not be a meta tensor")
    if tensor.layout is not torch.strided or tensor.is_quantized:
        raise CheckpointError(f"state_dict[{name!r}] must be dense and non-quantized")
    if tensor.is_complex():
        raise CheckpointError(f"state_dict[{name!r}] must not use a complex dtype")
    if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
        raise CheckpointError(f"state_dict[{name!r}] contains NaN or Inf")
    return tensor


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("state_dict keys must be non-empty strings")
        tensor = _validate_tensor(name, value)
        result[name] = tensor.detach().to(device="cpu").contiguous()
    if not result:
        raise CheckpointError("application state_dict must not be empty")
    return result


def _tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _state_description(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "count": len(state_dict),
        "tensors": {name: _tensor_description(state_dict[name]) for name in sorted(state_dict)},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_symlink(path: Path) -> bool:
    return path.is_symlink() or (not path.exists() and os.path.lexists(path))


def _require_safe_parent(path: Path) -> Path:
    parent = path.parent
    if not parent.exists() or not parent.is_dir() or _is_symlink(parent):
        raise CheckpointError("checkpoint parent must be an existing non-symlink directory")
    current = parent
    while current != current.parent:
        if current.is_symlink():
            raise CheckpointError("checkpoint path must not traverse a symlink")
        current = current.parent
    return parent


def _application_manifest(model: nn.Module) -> tuple[dict[str, Any], dict[str, Any]]:
    model_name, spec = _model_spec(model)
    constructor = _model_constructor(model, spec)
    application = {
        "name": model_name,
        "schema_version": APPLICATION_SCHEMA_VERSION,
        "feature_schema": str(model.feature_schema),
        "constructor": constructor,
    }
    return application, {"mode": constructor["adapter_mode"]}


def save_application_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    *,
    estimator: AnalyticStructuralFieldEstimator,
    preprocessing: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish a new immutable application-checkpoint directory."""

    destination = Path(path).absolute()
    if destination.exists() or _is_symlink(destination):
        raise FileExistsError(f"checkpoint path already exists: {destination}")
    parent = _require_safe_parent(destination)
    if type(estimator) is not AnalyticStructuralFieldEstimator:
        raise TypeError("estimator must be exactly AnalyticStructuralFieldEstimator")
    if not isinstance(preprocessing, Mapping):
        raise TypeError("preprocessing must be a mapping")

    application, adapter = _application_manifest(model)
    if estimator.config.num_orientations != application["constructor"]["num_orientations"]:
        raise ValueError("estimator and application num_orientations must match")
    state_dict = _cpu_state_dict(model)
    estimator_config = _json_copy("estimator config", asdict(estimator.config))
    preprocessing_copy = _json_copy("preprocessing", dict(preprocessing))
    provenance = _normalise_provenance(training_provenance)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=parent))
    published = False
    try:
        weights_path = stage / _WEIGHTS_NAME
        torch.save(state_dict, weights_path)
        _fsync_file(weights_path)
        weights_bytes = weights_path.stat().st_size
        manifest: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "producer": {
                "distribution": _DISTRIBUTION_NAME,
                "version": _distribution_version(),
            },
            "application": application,
            "adapter": adapter,
            "structural_field_schema": STRUCTURAL_FIELD_SCHEMA_VERSION,
            "estimator": {
                "name": "analytic",
                "schema_version": STRUCTURAL_FIELD_SCHEMA_VERSION,
                "config": estimator_config,
            },
            "preprocessing": preprocessing_copy,
            "training_provenance": provenance,
            "weights": {
                "filename": _WEIGHTS_NAME,
                "bytes": weights_bytes,
                "sha256": _sha256(weights_path),
            },
            "state_dict": _state_description(state_dict),
        }
        manifest_path = stage / _MANIFEST_NAME
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(stage)
        os.rename(stage, destination)
        published = True
        _fsync_directory(parent)
        return manifest
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CheckpointError(f"manifest contains non-finite JSON constant {value}")


def _read_manifest(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
        raise CheckpointError("manifest.json must be a regular non-symlink file")
    if info.st_size > _MAX_MANIFEST_BYTES:
        raise CheckpointError("manifest.json exceeds the 1 MiB safety limit")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_no_duplicate_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("manifest.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointError("manifest.json root must be an object")
    return value


def _validate_manifest(
    manifest: dict[str, Any],
    max_weights_bytes: int,
    *,
    migrate_legacy_field_schema: bool,
) -> tuple[_ModelSpec, dict[str, Any], AnalyticEstimatorConfig]:
    _require_exact_keys(
        "manifest",
        manifest,
        {
            "schema_version",
            "producer",
            "application",
            "adapter",
            "structural_field_schema",
            "estimator",
            "preprocessing",
            "training_provenance",
            "weights",
            "state_dict",
        },
    )
    if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported checkpoint schema_version")
    field_schema = manifest["structural_field_schema"]
    legacy = field_schema == LEGACY_STRUCTURAL_FIELD_SCHEMA_VERSION
    if field_schema != STRUCTURAL_FIELD_SCHEMA_VERSION and not (legacy and migrate_legacy_field_schema):
        raise CheckpointError("unsupported structural-field schema")

    producer = _require_exact_keys("producer", manifest["producer"], {"distribution", "version"})
    if producer["distribution"] != _DISTRIBUTION_NAME:
        raise CheckpointError("unexpected checkpoint producer")
    if not isinstance(producer["version"], str) or not producer["version"]:
        raise CheckpointError("producer.version must be a non-empty string")

    application = _require_exact_keys(
        "application",
        manifest["application"],
        {"name", "schema_version", "feature_schema", "constructor"},
    )
    if application["schema_version"] != APPLICATION_SCHEMA_VERSION:
        raise CheckpointError("unsupported application schema")
    model_name = application["name"]
    if model_name not in _MODEL_SPECS:
        raise CheckpointError("checkpoint requests a non-built-in application model")
    spec = _MODEL_SPECS[model_name]
    constructor = _require_exact_keys(
        "application.constructor",
        application["constructor"],
        set(spec.constructor_fields),
    )
    expected_feature_schema = str(cast(Any, spec.model_type).feature_schema)
    if application["feature_schema"] != expected_feature_schema:
        raise CheckpointError("application feature schema does not match the built-in model")

    adapter = _require_exact_keys("adapter", manifest["adapter"], {"mode"})
    if adapter["mode"] != constructor["adapter_mode"]:
        raise CheckpointError("adapter mode conflicts with the model constructor")

    estimator = _require_exact_keys(
        "estimator",
        manifest["estimator"],
        {"name", "schema_version", "config"},
    )
    if estimator["name"] != "analytic" or estimator["schema_version"] != field_schema:
        raise CheckpointError("unsupported estimator")
    if not isinstance(estimator["config"], dict):
        raise CheckpointError("estimator.config must be an object")
    try:
        config_payload = dict(estimator["config"])
        if legacy:
            config_payload.pop("topk", None)
        estimator_config = AnalyticEstimatorConfig(**config_payload)
    except (TypeError, ValueError) as exc:
        raise CheckpointError("invalid analytic estimator configuration") from exc
    if estimator_config.num_orientations != constructor["num_orientations"]:
        raise CheckpointError("estimator and application num_orientations conflict")

    if not isinstance(manifest["preprocessing"], dict):
        raise CheckpointError("preprocessing must be an object")
    try:
        provenance = _normalise_provenance(manifest["training_provenance"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(str(exc)) from exc

    weights = _require_exact_keys("weights", manifest["weights"], {"filename", "bytes", "sha256"})
    if weights["filename"] != _WEIGHTS_NAME:
        raise CheckpointError("weights filename must be exactly weights.pt")
    if isinstance(weights["bytes"], bool) or not isinstance(weights["bytes"], int) or weights["bytes"] < 1:
        raise CheckpointError("weights.bytes must be a positive integer")
    if weights["bytes"] > max_weights_bytes:
        raise CheckpointError("weights file exceeds max_weights_bytes")
    if not isinstance(weights["sha256"], str) or _SHA256_RE.fullmatch(weights["sha256"]) is None:
        raise CheckpointError("weights.sha256 must be lowercase hexadecimal SHA-256")

    state = _require_exact_keys("state_dict", manifest["state_dict"], {"count", "tensors"})
    if isinstance(state["count"], bool) or not isinstance(state["count"], int) or state["count"] < 1:
        raise CheckpointError("state_dict.count must be a positive integer")
    if not isinstance(state["tensors"], dict) or len(state["tensors"]) != state["count"]:
        raise CheckpointError("state_dict tensor count is inconsistent")
    for name, description in state["tensors"].items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("state_dict keys must be non-empty strings")
        tensor_info = _require_exact_keys(
            f"state_dict.tensors[{name!r}]",
            description,
            {"shape", "dtype"},
        )
        if not isinstance(tensor_info["shape"], list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in tensor_info["shape"]
        ):
            raise CheckpointError("state_dict shapes must contain non-negative integers")
        if not isinstance(tensor_info["dtype"], str) or not tensor_info["dtype"]:
            raise CheckpointError("state_dict dtypes must be non-empty strings")
    return spec, provenance, estimator_config


def _require_checkpoint_directory(path: Path) -> None:
    _require_safe_parent(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointError(f"checkpoint directory does not exist: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CheckpointError("checkpoint path must be a non-symlink directory")
    entries = {entry.name for entry in os.scandir(path)}
    if entries != {_MANIFEST_NAME, _WEIGHTS_NAME}:
        raise CheckpointError("checkpoint directory must contain exactly manifest.json and weights.pt")


def _load_weights(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, torch.Tensor]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointError("weights.pt must be a regular non-symlink file") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CheckpointError("weights.pt must be a singly linked regular file")
        if info.st_size != expected_bytes:
            raise CheckpointError("weights.pt size does not match manifest")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise CheckpointError("weights.pt SHA-256 does not match manifest")
            stream.seek(0)
            try:
                loaded = torch.load(stream, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise CheckpointError("weights.pt is not a safe weights-only state_dict") from exc
            after = os.fstat(descriptor)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
            ):
                raise CheckpointError("weights.pt changed while it was being loaded")
    finally:
        os.close(descriptor)
    if type(loaded) is not dict:
        raise CheckpointError("weights.pt root must be a plain state_dict")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("state_dict keys must be non-empty strings")
        result[name] = _validate_tensor(name, tensor)
    return result


def load_application_checkpoint(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    max_weights_bytes: int = 1 << 30,
    migrate_legacy_field_schema: bool = False,
) -> LoadedApplicationCheckpoint:
    """Validate and reconstruct a built-in application checkpoint."""

    if not isinstance(migrate_legacy_field_schema, bool):
        raise TypeError("migrate_legacy_field_schema must be a bool")
    if isinstance(max_weights_bytes, bool) or not isinstance(max_weights_bytes, int):
        raise TypeError("max_weights_bytes must be an integer")
    if max_weights_bytes < 1:
        raise ValueError("max_weights_bytes must be positive")
    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device must identify a valid torch device") from exc
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("checkpoint device must be cpu or cuda")

    directory = Path(path).absolute()
    _require_checkpoint_directory(directory)
    manifest = _read_manifest(directory / _MANIFEST_NAME)
    spec, provenance, estimator_config = _validate_manifest(
        manifest,
        max_weights_bytes,
        migrate_legacy_field_schema=migrate_legacy_field_schema,
    )
    weights_meta = manifest["weights"]
    state_dict = _load_weights(
        directory / _WEIGHTS_NAME,
        weights_meta["bytes"],
        weights_meta["sha256"],
    )

    try:
        model = spec.model_type(**manifest["application"]["constructor"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError("invalid built-in model constructor") from exc
    expected_state = model.state_dict()
    if set(state_dict) != set(expected_state):
        raise CheckpointError("state_dict keys do not match the built-in model")
    manifest_description = manifest["state_dict"]
    if _state_description(state_dict) != manifest_description:
        raise CheckpointError("state_dict tensors do not match their manifest descriptions")
    if _state_description(expected_state) != manifest_description:
        raise CheckpointError("state_dict shapes or dtypes do not match the built-in model")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointError("state_dict cannot be loaded into the built-in model") from exc
    model.to(target_device)

    estimator = AnalyticStructuralFieldEstimator(estimator_config).to(target_device)
    returned_manifest = _json_copy("manifest", manifest)
    if returned_manifest["structural_field_schema"] == LEGACY_STRUCTURAL_FIELD_SCHEMA_VERSION:
        returned_manifest["structural_field_schema"] = STRUCTURAL_FIELD_SCHEMA_VERSION
        returned_manifest["estimator"]["schema_version"] = STRUCTURAL_FIELD_SCHEMA_VERSION
        returned_manifest["estimator"]["config"].pop("topk", None)
    return LoadedApplicationCheckpoint(
        model=model,
        estimator=estimator,
        preprocessing=_json_copy("preprocessing", manifest["preprocessing"]),
        training_provenance=provenance,
        manifest=returned_manifest,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "LoadedApplicationCheckpoint",
    "load_application_checkpoint",
    "save_application_checkpoint",
]
