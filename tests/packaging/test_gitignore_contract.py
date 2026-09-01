"""Regression tests for source/data separation in ``.gitignore``."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _ignored_paths(*paths: str) -> set[str]:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPOSITORY_ROOT,
        input="".join(f"{path}\n" for path in paths),
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode in {0, 1}, result.stderr
    return set(result.stdout.splitlines())


def test_dataset_adapter_sources_are_trackable_but_dataset_payloads_are_ignored() -> None:
    adapter_sources = {
        "datasets/adapters/README.md",
        "datasets/adapters/base.py",
        "datasets/adapters/generic.py",
        "datasets/adapters/validation.py",
    }
    dataset_payloads = {
        "data/private/sample.png",
        "datasets/DRIVE/images/01_test.tif",
        "datasets/local-study/sample.npy",
        "datasets/adapters/DRIVE/sample.png",
    }
    candidates = adapter_sources | dataset_payloads

    ignored = _ignored_paths(*sorted(candidates))

    assert adapter_sources.isdisjoint(ignored)
    assert dataset_payloads <= ignored


def test_checkpoints_and_generated_outputs_are_ignored() -> None:
    generated = {
        "model.ckpt",
        "weights/model.pt",
        "weights/model.pth",
        "outputs/run-001/orientation.npy",
        "results/checkpoints/best.ckpt",
        "results/raw/run-001/metrics.json",
    }

    assert generated <= _ignored_paths(*sorted(generated))


def test_build_and_distribution_artifacts_are_ignored() -> None:
    build_artifacts = {
        "build/lib/generated.py",
        "dist/structfield_vision-1.0.0-py3-none-any.whl",
        "structfield_vision.egg-info/PKG-INFO",
    }

    assert build_artifacts <= _ignored_paths(*sorted(build_artifacts))


def test_release_documentation_is_trackable_but_root_release_artifacts_are_ignored() -> None:
    release_documentation = {"docs/release/index.md"}
    root_release_artifacts = {"release/v1.0.0/SHA256SUMS"}

    ignored = _ignored_paths(*sorted(release_documentation | root_release_artifacts))

    assert release_documentation.isdisjoint(ignored)
    assert root_release_artifacts <= ignored
