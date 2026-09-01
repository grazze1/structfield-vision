"""CLI input validation and atomic generation publication tests."""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from structfield.cli import estimate as estimate_cli
from structfield.cli.estimate import (
    CURRENT_SCHEMA_VERSION,
    OUTPUT_FILENAMES,
    SCHEMA_VERSION,
    resolve_current_bundle,
    run_estimation,
)


def _input(tmp_path: Path, name: str = "image.npy") -> Path:
    path = tmp_path / name
    values = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    np.save(path, values, allow_pickle=False)
    return path


def _pointer(output: Path) -> dict[str, str]:
    return json.loads((output / "current.json").read_text(encoding="utf-8"))


def _generations(output: Path) -> list[Path]:
    bundles = output / "bundles"
    return sorted(path for path in bundles.iterdir() if not path.name.startswith("."))


def test_fresh_publish_writes_complete_bundle_and_exact_pointer(tmp_path: Path) -> None:
    output = tmp_path / "output"
    metadata = run_estimation(_input(tmp_path), output, device="cpu")
    bundle = resolve_current_bundle(output)
    pointer = _pointer(output)

    assert metadata["schema_version"] == SCHEMA_VERSION
    assert pointer == {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "generation": bundle.name,
    }
    assert len(bundle.name) == 32
    assert bundle.name == bundle.name.lower()
    assert {path.name for path in bundle.iterdir()} == {
        *OUTPUT_FILENAMES.values(),
        "metadata.json",
    }
    assert json.loads((bundle / "metadata.json").read_text(encoding="utf-8")) == metadata
    assert metadata["outputs"]["orientation_probability"]["shape"] == [16, 16, 16]
    assert all("sha256" in description and "bytes" in description for description in metadata["outputs"].values())


def test_resolver_rejects_hash_tamper_and_undeclared_files(tmp_path: Path) -> None:
    output = tmp_path / "output"
    run_estimation(_input(tmp_path), output, device="cpu")
    bundle = resolve_current_bundle(output)
    probability = bundle / OUTPUT_FILENAMES["orientation_probability"]
    probability.write_bytes(probability.read_bytes() + b"x")
    with pytest.raises(ValueError, match="size mismatch"):
        resolve_current_bundle(output)

    output2 = tmp_path / "output2"
    run_estimation(_input(tmp_path, "second.npy"), output2, device="cpu")
    bundle2 = resolve_current_bundle(output2)
    (bundle2 / "undeclared.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        resolve_current_bundle(output2)


def test_no_overwrite_rejects_existing_current(tmp_path: Path) -> None:
    output = tmp_path / "output"
    input_path = _input(tmp_path)
    run_estimation(input_path, output, device="cpu")
    original = _pointer(output)
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_estimation(input_path, output, device="cpu")
    assert _pointer(output) == original
    assert len(_generations(output)) == 1


def test_overwrite_publishes_new_generation_without_mutating_old(tmp_path: Path) -> None:
    output = tmp_path / "output"
    input_path = _input(tmp_path)
    run_estimation(input_path, output, device="cpu")
    old_bundle = resolve_current_bundle(output)
    old_files = {path.name: path.read_bytes() for path in old_bundle.iterdir()}

    run_estimation(input_path, output, device="cpu", overwrite=True, num_orientations=8)
    new_bundle = resolve_current_bundle(output)
    assert new_bundle != old_bundle
    assert len(_generations(output)) == 2
    assert {path.name: path.read_bytes() for path in old_bundle.iterdir()} == old_files


def test_concurrent_fresh_writers_leave_one_pointer_and_complete_orphan(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    input_path = _input(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_estimation, input_path, output, device="cpu") for _ in range(2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except FileExistsError:
            outcomes.append("collision")

    assert sum(value == "collision" for value in outcomes) == 1
    assert resolve_current_bundle(output).is_dir()
    generations = _generations(output)
    assert len(generations) == 1
    expected = {*OUTPUT_FILENAMES.values(), "metadata.json"}
    assert all({path.name for path in bundle.iterdir()} == expected for bundle in generations)


def test_failure_before_bundle_publish_leaves_no_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("staging failed")

    monkeypatch.setattr(estimate_cli, "_write_output_bundle", fail)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="staging failed"):
        run_estimation(_input(tmp_path), output, device="cpu")
    assert not (output / "current.json").exists()


def test_failure_after_bundle_publish_preserves_old_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    input_path = _input(tmp_path)
    run_estimation(input_path, output, device="cpu")
    old_pointer = _pointer(output)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("pointer failed")

    monkeypatch.setattr(estimate_cli, "_write_current_pointer", fail)
    with pytest.raises(RuntimeError, match="pointer failed"):
        run_estimation(input_path, output, device="cpu", overwrite=True)
    assert _pointer(output) == old_pointer
    assert len(_generations(output)) == 2
    assert resolve_current_bundle(output).is_dir()


def test_legacy_flat_migration_removes_only_known_files_after_switch(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "metadata.json").write_text("legacy", encoding="utf-8")
    (output / "user-notes.txt").write_text("keep", encoding="utf-8")
    input_path = _input(tmp_path)

    with pytest.raises(FileExistsError, match="legacy"):
        run_estimation(input_path, output, device="cpu")
    run_estimation(input_path, output, device="cpu", overwrite=True)
    assert not (output / "metadata.json").exists()
    assert (output / "user-notes.txt").read_text(encoding="utf-8") == "keep"
    assert resolve_current_bundle(output).is_dir()


def test_cleanup_failure_occurs_only_after_new_pointer_is_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "metadata.json").write_text("legacy", encoding="utf-8")

    def fail_cleanup(path: Path) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(estimate_cli, "_cleanup_legacy_outputs", fail_cleanup)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        run_estimation(_input(tmp_path), output, device="cpu", overwrite=True)
    assert resolve_current_bundle(output).is_dir()
    assert (output / "metadata.json").read_text(encoding="utf-8") == "legacy"


def test_malformed_pointer_rejected_and_overwrite_can_recover(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "current.json").write_text('{"schema_version":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_current_bundle(output)
    run_estimation(_input(tmp_path), output, device="cpu", overwrite=True)
    assert resolve_current_bundle(output).is_dir()


def test_symlink_boundaries_are_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        run_estimation(_input(tmp_path), output_link, device="cpu")

    output = tmp_path / "output"
    output.mkdir()
    bundles_target = tmp_path / "bundles-target"
    bundles_target.mkdir()
    (output / "bundles").symlink_to(bundles_target, target_is_directory=True)
    with pytest.raises(ValueError, match="bundles"):
        run_estimation(_input(tmp_path, "second.npy"), output, device="cpu")


def test_existing_orphan_generation_does_not_block_fresh_publish(tmp_path: Path) -> None:
    output = tmp_path / "output"
    orphan = output / "bundles" / ("a" * 32)
    orphan.mkdir(parents=True)
    (orphan / "sentinel").write_text("orphan", encoding="utf-8")
    run_estimation(_input(tmp_path), output, device="cpu")
    assert resolve_current_bundle(output).name != orphan.name
    assert (orphan / "sentinel").read_text(encoding="utf-8") == "orphan"


def test_visualization_exports_are_warning_free_and_recorded(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metadata = run_estimation(_input(tmp_path), output, device="cpu", visualize=True)
    bundle = resolve_current_bundle(output)
    assert set(metadata["visualizations"]) == {
        "orientation_wheel",
        "orientation_probability",
        "orientation_overlay",
        "uncertainty_heatmap",
        "frequency_heatmap",
        "coherence_heatmap",
    }
    for paths in metadata["visualizations"].values():
        assert {Path(path).suffix for path in paths} == {".png", ".svg"}
        assert all((bundle / path).is_file() for path in paths)


def test_npy_external_boundary_rejects_ambiguous_or_unsafe_values(tmp_path: Path) -> None:
    signed = tmp_path / "signed.npy"
    np.save(signed, np.zeros((16, 16), dtype=np.int16), allow_pickle=False)
    with pytest.raises(TypeError, match="ambiguous"):
        run_estimation(signed, tmp_path / "signed-out", device="cpu")

    nonfinite = tmp_path / "nonfinite.npy"
    values = np.zeros((16, 16), dtype=np.float32)
    values[0, 0] = np.nan
    np.save(nonfinite, values, allow_pickle=False)
    with pytest.raises(ValueError, match="NaN or Inf"):
        run_estimation(nonfinite, tmp_path / "nonfinite-out", device="cpu")


def test_main_success_version_and_runtime_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli-output"
    assert (
        estimate_cli.main(
            [
                "--input",
                str(_input(tmp_path)),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--config",
                "analytic_8dir.yaml",
                "--num-orientations",
                "16",
            ]
        )
        == estimate_cli.EXIT_SUCCESS
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary == {"device": "cpu", "num_outputs": 6, "schema_version": SCHEMA_VERSION}

    assert (
        estimate_cli.main(
            [
                "--input",
                str(tmp_path / "missing.npy"),
                "--output",
                str(tmp_path / "bad-output"),
            ]
        )
        == estimate_cli.EXIT_RUNTIME_ERROR
    )
    assert "error:" in capsys.readouterr().err

    with pytest.raises(SystemExit) as version_exit:
        estimate_cli.main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() in {
        "structfield-estimate 1.0.0",
        "structfield-estimate 1.0.0.dev0",
    }


def test_gate_visualization_is_validated_and_recorded(tmp_path: Path) -> None:
    gate_path = tmp_path / "gate.npy"
    np.save(gate_path, np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16))
    output = tmp_path / "output"
    metadata = run_estimation(
        _input(tmp_path),
        output,
        device="cpu",
        visualize=True,
        gate_path=gate_path,
    )
    assert metadata["gate"]["reduction"] == "mean"
    assert "gate_heatmap" in metadata["visualizations"]


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((1, 1, 1, 1, 1), dtype=np.float32), "shape"),
        (np.zeros((2, 1, 16, 16), dtype=np.float32), "batch"),
        (np.full((16, 16), np.nan, dtype=np.float32), "NaN or Inf"),
        (np.full((16, 16), 2.0, dtype=np.float32), r"\[0,1\]"),
    ],
)
def test_gate_rejects_invalid_arrays(tmp_path: Path, array: np.ndarray, message: str) -> None:
    gate_path = tmp_path / "gate.npy"
    np.save(gate_path, array, allow_pickle=False)
    with pytest.raises(ValueError, match=message):
        run_estimation(
            _input(tmp_path),
            tmp_path / "output",
            device="cpu",
            visualize=True,
            gate_path=gate_path,
        )


def test_run_argument_boundaries_and_non_directory_output(tmp_path: Path) -> None:
    input_path = _input(tmp_path)
    with pytest.raises(TypeError, match="bool"):
        run_estimation(input_path, tmp_path / "output", overwrite=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires"):
        run_estimation(input_path, tmp_path / "output", gate_path=tmp_path / "gate.npy")
    occupied = tmp_path / "occupied"
    occupied.write_text("file", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        run_estimation(input_path, occupied)


@pytest.mark.parametrize(
    ("pointer", "message"),
    [
        ([], "mapping"),
        ({"schema_version": CURRENT_SCHEMA_VERSION, "generation": "a" * 32, "extra": 1}, "fields"),
        ({"schema_version": "wrong", "generation": "a" * 32}, "schema"),
        ({"schema_version": CURRENT_SCHEMA_VERSION, "generation": "../bad"}, "generation"),
    ],
)
def test_resolver_rejects_noncanonical_pointer(
    tmp_path: Path,
    pointer: object,
    message: str,
) -> None:
    output = tmp_path / "output"
    run_estimation(_input(tmp_path), output, device="cpu")
    (output / "current.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        resolve_current_bundle(output)


@pytest.mark.parametrize(
    ("array", "message", "exception"),
    [
        (np.array([], dtype=np.float32), "non-empty", ValueError),
        (np.ones((16, 16), dtype=np.complex64), "real numeric", TypeError),
    ],
)
def test_gate_rejects_empty_and_nonreal_arrays(
    tmp_path: Path,
    array: np.ndarray,
    message: str,
    exception: type[Exception],
) -> None:
    gate_path = tmp_path / "gate.npy"
    np.save(gate_path, array, allow_pickle=False)
    with pytest.raises(exception, match=message):
        run_estimation(
            _input(tmp_path),
            tmp_path / "output",
            device="cpu",
            visualize=True,
            gate_path=gate_path,
        )


def test_resolver_requires_existing_directory_and_gate_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        resolve_current_bundle(tmp_path / "missing")
    occupied = tmp_path / "occupied"
    occupied.write_text("not a directory", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        resolve_current_bundle(occupied)
    with pytest.raises(ValueError, match="existing .npy"):
        run_estimation(
            _input(tmp_path),
            tmp_path / "output",
            device="cpu",
            visualize=True,
            gate_path=tmp_path / "missing.npy",
        )
