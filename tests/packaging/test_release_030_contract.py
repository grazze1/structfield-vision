"""Release metadata and packaged-resource contract for 1.0.0."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from structfield._version import VERSION
from structfield.configs import available_configs, read_config_text

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_identity_and_dependencies() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["dynamic"] == ["version"]
    assert VERSION == "1.0.0"
    assert project["license"] == "Apache-2.0"
    assert not any(dependency.startswith("matplotlib") for dependency in project["dependencies"])
    assert project["optional-dependencies"]["visualization"] == ["matplotlib>=3.10.5,<4"]
    assert "Pillow>=10,<13" in project["dependencies"]
    extras = project.get("optional-dependencies", {})
    assert "equivariance" not in extras
    assert "bm3d-noncommercial" not in extras


def test_setuptools_namespace_and_resource_contract() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = data["tool"]["setuptools"]
    assert setuptools["packages"]["find"]["include"] == ["structfield*"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "structfield._version.VERSION"}
    assert data["project"]["license-files"] == ["LICENSE", "NOTICE.md"]
    assert data["tool"]["setuptools"]["package-data"]["structfield.configs"] == ["*.yaml"]
    assert data["tool"]["setuptools"]["package-data"]["structfield"] == ["py.typed"]


def test_active_config_resources_are_read_only_and_canonical() -> None:
    expected = (
        "analytic_default.yaml",
        "analytic_8dir.yaml",
        "analytic_16dir.yaml",
        "analytic_32dir.yaml",
        "baseline_matrix.yaml",
        "failure_prediction.yaml",
        "corruption_sweep.yaml",
        "equivariance_smoke.yaml",
    )
    assert available_configs() == expected
    for resource_id in expected:
        text = read_config_text(resource_id)
        assert "schema_version:" in text
    with pytest.raises(KeyError):
        read_config_text("../pyproject.toml")
    with pytest.raises(TypeError):
        read_config_text(Path("analytic_default.yaml"))  # type: ignore[arg-type]


def test_apache_identity_and_notice_boundaries() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
    assert "Copyright 2026 StructField-Vision contributors" in notice
    assert "10.5281/zenodo.18779434" in notice
    assert "10.1097/ICO.0000000000003319" in notice
    assert not (ROOT / "LICENSE-PENDING.md").exists()
