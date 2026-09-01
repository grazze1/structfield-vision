"""PEP 751, CycloneDX, and third-party license release contracts."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("profile", "torch_version", "minimum_packages"),
    [
        ("cpu", "2.13.0+cpu", 20),
        ("cu126", "2.13.0+cu126", 40),
    ],
)
def test_pep751_locks_are_hashed_and_profile_specific(
    profile: str,
    torch_version: str,
    minimum_packages: int,
) -> None:
    with (ROOT / f"pylock.{profile}.toml").open("rb") as stream:
        lock = tomllib.load(stream)
    assert lock["lock-version"] == "1.0"
    assert lock["created-by"] == "uv"
    assert lock["requires-python"] == ">=3.11"
    packages = lock["packages"]
    assert len(packages) >= minimum_packages
    versions = {package["name"]: package["version"] for package in packages}
    assert versions["torch"] == torch_version
    for package in packages:
        assert package["wheels"]
        for wheel in package["wheels"]:
            assert re.fullmatch(r"[0-9a-f]{64}", wheel["hashes"]["sha256"])
    if profile == "cpu":
        assert not any(name.startswith("nvidia-") for name in versions)
    else:
        assert "nvidia-cuda-runtime-cu12" in versions
        assert versions["triton"] == "3.7.1"


@pytest.mark.parametrize("profile", ["cpu", "cu126"])
def test_sbom_and_license_inventory_are_stable_and_complete(
    profile: str,
) -> None:
    sbom_path = ROOT / "sbom" / f"structfield-vision-{profile}.cdx.json"
    licenses_path = ROOT / "sbom" / f"third-party-licenses-{profile}.json"
    sbom_text = sbom_path.read_text(encoding="utf-8")
    licenses_text = licenses_path.read_text(encoding="utf-8")
    assert "/home/" not in sbom_text + licenses_text
    assert "UNKNOWN" not in sbom_text + licenses_text

    sbom = json.loads(sbom_text)
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.6"
    assert "serialNumber" not in sbom
    assert "timestamp" not in sbom["metadata"]
    assert sbom["metadata"]["component"]["version"] == "1.0.0"
    assert sbom["metadata"]["component"]["licenses"] == [{"expression": "Apache-2.0"}]
    with (ROOT / f"pylock.{profile}.toml").open("rb") as stream:
        count = len(tomllib.load(stream)["packages"])
    assert len(sbom["components"]) == count
    assert sbom["components"] == sorted(
        sbom["components"],
        key=lambda component: (
            re.sub(r"[-_.]+", "-", component["name"]).lower(),
            component.get("version", ""),
            component.get("purl", ""),
        ),
    )
    root_ref = sbom["metadata"]["component"]["bom-ref"]
    root_dependency = next(item for item in sbom["dependencies"] if item["ref"] == root_ref)
    assert set(root_dependency["dependsOn"]) == {component["bom-ref"] for component in sbom["components"]}

    inventory = json.loads(licenses_text)
    assert inventory["schema_version"] == "structfield-third-party-licenses-v1"
    assert inventory["profile"] == profile
    assert len(inventory["components"]) == count
    assert all(component["license_expression"] for component in inventory["components"])
