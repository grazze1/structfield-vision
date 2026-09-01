"""Round-trip and adversarial tests for application checkpoint v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from structfield import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    ValidationMode,
    load_application_checkpoint,
    save_application_checkpoint,
)
from structfield.applications import (
    CurvilinearSegmenter,
    FringeRestorer,
    TextureRestorer,
)
from structfield.estimators import AnalyticEstimatorConfig, AnalyticStructuralFieldEstimator


def _estimator() -> AnalyticStructuralFieldEstimator:
    return AnalyticStructuralFieldEstimator(AnalyticEstimatorConfig(num_orientations=8))


def _provenance() -> dict[str, object]:
    return {"status": "untrained", "datasets": []}


def _model(application: str, adapter_mode: str) -> torch.nn.Module:
    common = {
        "num_orientations": 8,
        "width": 4,
        "adapter_mode": adapter_mode,
        "validation_mode": ValidationMode.trusted,
    }
    if application == "curvilinear":
        return CurvilinearSegmenter(**common)
    if application == "fringe":
        return FringeRestorer(**common, noise_dim=2)
    if application == "texture":
        return TextureRestorer(**common)
    raise AssertionError(application)


@pytest.mark.parametrize("application", ["curvilinear", "fringe", "texture"])
@pytest.mark.parametrize("adapter_mode", ["none", "concat", "film", "reliability_gated"])
def test_round_trip_all_built_in_models_and_adapters(
    tmp_path: Path,
    application: str,
    adapter_mode: str,
) -> None:
    model = _model(application, adapter_mode)
    destination = tmp_path / f"{application}-{adapter_mode}"
    manifest = save_application_checkpoint(
        destination,
        model,
        estimator=_estimator(),
        preprocessing={"range": [0.0, 1.0], "channels": 1},
        training_provenance=_provenance(),
    )

    assert {path.name for path in destination.iterdir()} == {"manifest.json", "weights.pt"}
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert manifest["adapter"] == {"mode": adapter_mode}
    assert manifest["application"]["constructor"]["validation_mode"] == "trusted"

    loaded = load_application_checkpoint(destination)
    assert type(loaded.model) is type(model)
    assert loaded.model.validation_mode is ValidationMode.trusted
    assert loaded.preprocessing["channels"] == 1
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], expected.cpu())


def _read_manifest(checkpoint: Path) -> dict[str, object]:
    return json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(checkpoint: Path, manifest: dict[str, object]) -> None:
    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    save_application_checkpoint(
        checkpoint,
        _model("texture", "concat"),
        estimator=_estimator(),
        preprocessing={},
        training_provenance=_provenance(),
    )
    return checkpoint


def _refresh_weight_identity(checkpoint: Path, manifest: dict[str, object]) -> None:
    weights_path = checkpoint / "weights.pt"
    weights = manifest["weights"]
    assert isinstance(weights, dict)
    payload = weights_path.read_bytes()
    weights["bytes"] = len(payload)
    weights["sha256"] = hashlib.sha256(payload).hexdigest()
    _write_manifest(checkpoint, manifest)


def test_rejects_existing_destination_and_invalid_training_provenance(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        save_application_checkpoint(
            existing,
            _model("texture", "none"),
            estimator=_estimator(),
            preprocessing={},
            training_provenance=_provenance(),
        )
    with pytest.raises(ValueError, match="trained checkpoints"):
        save_application_checkpoint(
            tmp_path / "trained",
            _model("texture", "none"),
            estimator=_estimator(),
            preprocessing={},
            training_provenance={"status": "trained", "datasets": []},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "structfield-application-checkpoint-v0"),
        ("structural_field_schema", "0.1"),
    ],
)
def test_rejects_old_schemas(tmp_path: Path, field: str, value: str) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    manifest = _read_manifest(checkpoint)
    manifest[field] = value
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="schema"):
        load_application_checkpoint(checkpoint)


def test_rejects_path_traversal_and_extra_files(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    manifest = _read_manifest(checkpoint)
    weights = manifest["weights"]
    assert isinstance(weights, dict)
    weights["filename"] = "../weights.pt"
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="filename"):
        load_application_checkpoint(checkpoint)

    weights["filename"] = "weights.pt"
    _write_manifest(checkpoint, manifest)
    (checkpoint / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(CheckpointError, match="exactly"):
        load_application_checkpoint(checkpoint)


def test_rejects_symlink_hash_tamper_and_size_limit(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    original = checkpoint / "weights.pt"
    target = tmp_path / "copied.pt"
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(CheckpointError, match="non-symlink"):
        load_application_checkpoint(checkpoint)

    original.unlink()
    original.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(CheckpointError, match="size"):
        load_application_checkpoint(checkpoint)

    checkpoint = tmp_path / "small-limit"
    save_application_checkpoint(
        checkpoint,
        _model("texture", "none"),
        estimator=_estimator(),
        preprocessing={},
        training_provenance=_provenance(),
    )
    with pytest.raises(CheckpointError, match="max_weights_bytes"):
        load_application_checkpoint(checkpoint, max_weights_bytes=1)


def test_rejects_shape_tamper_and_nonfinite_save(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    manifest = _read_manifest(checkpoint)
    state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    first_name = next(iter(state))
    state[first_name] = state[first_name].reshape(-1)
    torch.save(state, checkpoint / "weights.pt")
    _refresh_weight_identity(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="descriptions"):
        load_application_checkpoint(checkpoint)

    model = _model("texture", "none")
    next(model.parameters()).data.fill_(float("nan"))
    with pytest.raises(CheckpointError, match="NaN or Inf"):
        save_application_checkpoint(
            tmp_path / "nonfinite",
            model,
            estimator=_estimator(),
            preprocessing={},
            training_provenance=_provenance(),
        )


def test_malicious_pickle_is_not_executed(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            import os

            return os.system, (f"touch {marker}",)

    torch.save({"payload": Payload()}, checkpoint / "weights.pt")
    manifest = _read_manifest(checkpoint)
    _refresh_weight_identity(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="weights-only"):
        load_application_checkpoint(checkpoint)
    assert not marker.exists()


def test_rejects_hardlinks_and_explicitly_migrates_safe_field_v02(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    original = checkpoint / "weights.pt"
    linked = tmp_path / "linked.pt"
    linked.hardlink_to(original)
    with pytest.raises(CheckpointError, match="singly linked"):
        load_application_checkpoint(checkpoint)
    linked.unlink()

    manifest = _read_manifest(checkpoint)
    manifest["structural_field_schema"] = "0.2"
    estimator = manifest["estimator"]
    assert isinstance(estimator, dict)
    estimator["schema_version"] = "0.2"
    config = estimator["config"]
    assert isinstance(config, dict)
    config["topk"] = 2
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match="structural-field schema"):
        load_application_checkpoint(checkpoint)
    loaded = load_application_checkpoint(checkpoint, migrate_legacy_field_schema=True)
    assert loaded.manifest["structural_field_schema"] == "1.0"
    assert "topk" not in loaded.manifest["estimator"]["config"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("producer",), [], "JSON object"),
        (("producer", "distribution"), "other", "producer"),
        (("producer", "version"), "", "version"),
        (("application", "schema_version"), "old", "application schema"),
        (("application", "name"), "unknown", "non-built-in"),
        (("application", "feature_schema"), "wrong", "feature schema"),
        (("adapter", "mode"), "film", "adapter mode"),
        (("estimator", "name"), "other", "estimator"),
        (("estimator", "config"), [], "config"),
        (("estimator", "config", "num_orientations"), 16, "num_orientations"),
        (("preprocessing",), [], "preprocessing"),
        (("training_provenance", "status"), "unknown", "status"),
        (("weights", "bytes"), True, "positive integer"),
        (("weights", "sha256"), "ABC", "SHA-256"),
        (("state_dict", "count"), 0, "positive integer"),
        (("state_dict", "tensors"), [], "count"),
    ],
)
def test_manifest_rejects_each_typed_boundary(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    manifest = _read_manifest(checkpoint)
    target: object = manifest
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    if path:
        assert isinstance(target, dict)
        target[path[-1]] = value
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match=message):
        load_application_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", [-1], "shapes"),
        ("shape", [True], "shapes"),
        ("dtype", "", "dtypes"),
    ],
)
def test_manifest_rejects_invalid_tensor_descriptions(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    manifest = _read_manifest(checkpoint)
    tensors = manifest["state_dict"]["tensors"]
    first = next(iter(tensors.values()))
    first[field] = value
    _write_manifest(checkpoint, manifest)
    with pytest.raises(CheckpointError, match=message):
        load_application_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"schema_version": NaN}', "non-finite"),
        ('{"schema_version": 1, "schema_version": 2}', "duplicate"),
        ("[]", "root"),
        ("{", "valid UTF-8 JSON"),
    ],
)
def test_manifest_parser_rejects_noncanonical_json(tmp_path: Path, payload: str, message: str) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    (checkpoint / "manifest.json").write_text(payload, encoding="utf-8")
    with pytest.raises(CheckpointError, match=message):
        load_application_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ([], "plain state_dict"),
        ({1: torch.zeros(1)}, "keys"),
        ({"bad": 1}, "plain torch.Tensor"),
        ({"bad": torch.ones(1, dtype=torch.complex64)}, "complex"),
        ({"bad": torch.ones(2, 2).to_sparse()}, "dense"),
    ],
)
def test_weights_reject_unsafe_state_dict_values(tmp_path: Path, state: object, message: str) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    manifest = _read_manifest(checkpoint)
    torch.save(state, checkpoint / "weights.pt")
    _refresh_weight_identity(checkpoint, manifest)
    with pytest.raises(CheckpointError, match=message):
        load_application_checkpoint(checkpoint)


def test_checkpoint_argument_and_save_boundaries(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    with pytest.raises(TypeError, match="migrate"):
        load_application_checkpoint(checkpoint, migrate_legacy_field_schema=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_weights_bytes"):
        load_application_checkpoint(checkpoint, max_weights_bytes=True)
    with pytest.raises(ValueError, match="positive"):
        load_application_checkpoint(checkpoint, max_weights_bytes=0)
    with pytest.raises(ValueError, match="cpu or cuda"):
        load_application_checkpoint(checkpoint, device="meta")
    with pytest.raises(TypeError, match="estimator"):
        save_application_checkpoint(
            tmp_path / "bad-estimator",
            _model("texture", "none"),
            estimator=torch.nn.Identity(),  # type: ignore[arg-type]
            preprocessing={},
            training_provenance=_provenance(),
        )
    with pytest.raises(TypeError, match="preprocessing"):
        save_application_checkpoint(
            tmp_path / "bad-preprocessing",
            _model("texture", "none"),
            estimator=_estimator(),
            preprocessing=[],  # type: ignore[arg-type]
            training_provenance=_provenance(),
        )
    with pytest.raises(TypeError, match="built-in"):
        save_application_checkpoint(
            tmp_path / "unknown-model",
            torch.nn.Identity(),
            estimator=_estimator(),
            preprocessing={},
            training_provenance=_provenance(),
        )
    with pytest.raises(TypeError, match="JSON-compatible"):
        save_application_checkpoint(
            tmp_path / "non-json",
            _model("texture", "none"),
            estimator=_estimator(),
            preprocessing={"bad": float("nan")},
            training_provenance=_provenance(),
        )
