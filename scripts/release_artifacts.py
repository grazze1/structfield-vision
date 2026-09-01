#!/usr/bin/env python3
"""Generate deterministic CycloneDX and third-party license artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = str(runpy.run_path(ROOT / "structfield" / "_version.py")["VERSION"])
CYCLONEDX_VERSION = "7.3.1"
PROFILES = {
    "cpu": {
        "lock": ROOT / "pylock.cpu.toml",
        "torch": "2.13.0+cpu",
    },
    "cu126": {
        "lock": ROOT / "pylock.cu126.toml",
        "torch": "2.13.0+cu126",
    },
}

# Audited against the locked release versions and their upstream/PyPI metadata.
# A package missing from this table is a hard release failure.
LICENSE_EXPRESSIONS = {
    "babel": "BSD-3-Clause",
    "backports-tarfile": "MIT",
    "backrefs": "MIT",
    "build": "MIT",
    "certifi": "MPL-2.0",
    "cffi": "MIT-0",
    "charset-normalizer": "MIT",
    "click": "BSD-3-Clause",
    "colorama": "BSD-3-Clause",
    "contourpy": "BSD-3-Clause",
    "coverage": "Apache-2.0",
    "cryptography": "Apache-2.0 OR BSD-3-Clause",
    "cuda-bindings": "LicenseRef-NVIDIA-SOFTWARE-LICENSE",
    "cuda-pathfinder": "Apache-2.0",
    "cuda-toolkit": "LicenseRef-NVIDIA-Proprietary",
    "cycler": "BSD-3-Clause",
    "docutils": "LicenseRef-Docutils-Public-Domain-BSD-GPL",
    "filelock": "MIT",
    "fonttools": "MIT",
    "fsspec": "BSD-3-Clause",
    "ghp-import": "Apache-2.0",
    "griffelib": "ISC",
    "hypothesis": "MPL-2.0",
    "id": "Apache-2.0",
    "idna": "BSD-3-Clause",
    "importlib-metadata": "Apache-2.0",
    "iniconfig": "MIT",
    "jinja2": "BSD-3-Clause",
    "jaraco-classes": "MIT",
    "jaraco-context": "MIT",
    "jaraco-functools": "MIT",
    "jeepney": "MIT",
    "keyring": "MIT",
    "kiwisolver": "BSD-3-Clause",
    "librt": "MIT",
    "markdown": "BSD-3-Clause",
    "markdown-it-py": "MIT",
    "markupsafe": "BSD-3-Clause",
    "matplotlib": "LicenseRef-Matplotlib-1.3",
    "mdurl": "MIT",
    "mergedeep": "MIT",
    "mkdocs": "BSD-2-Clause",
    "mkdocs-autorefs": "ISC",
    "mkdocs-get-deps": "MIT",
    "mkdocs-material": "MIT",
    "mkdocs-material-extensions": "MIT",
    "mkdocstrings": "ISC",
    "mkdocstrings-python": "ISC",
    "more-itertools": "MIT",
    "mpmath": "BSD-3-Clause",
    "mypy": "MIT",
    "mypy-extensions": "MIT",
    "networkx": "BSD-3-Clause",
    "nh3": "MIT",
    "numpy": "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
    "nvidia-cublas-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cuda-cupti-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cuda-nvrtc-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cuda-runtime-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cudnn-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cufft-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cufile-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-curand-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cusolver-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cusparse-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-cusparselt-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-nccl-cu12": "BSD-3-Clause",
    "nvidia-nvjitlink-cu12": "LicenseRef-NVIDIA-Proprietary",
    "nvidia-nvshmem-cu12": "Apache-2.0 AND BSD-3-Clause",
    "nvidia-nvtx-cu12": "Apache-2.0",
    "packaging": "Apache-2.0 OR BSD-2-Clause",
    "paginate": "MIT",
    "pathspec": "MPL-2.0",
    "pillow": "MIT-CMU",
    "platformdirs": "MIT",
    "pluggy": "MIT",
    "pycparser": "BSD-3-Clause",
    "pygments": "BSD-2-Clause",
    "pymdown-extensions": "MIT",
    "pyparsing": "MIT",
    "pyproject-hooks": "MIT",
    "pytest": "MIT",
    "python-dateutil": "Apache-2.0 OR BSD-3-Clause",
    "pyyaml": "MIT",
    "pyyaml-env-tag": "MIT",
    "readme-renderer": "Apache-2.0",
    "requests": "Apache-2.0",
    "requests-toolbelt": "Apache-2.0",
    "rfc3986": "Apache-2.0",
    "rich": "MIT",
    "ruff": "MIT",
    "secretstorage": "BSD-3-Clause",
    "setuptools": "MIT",
    "six": "MIT",
    "sortedcontainers": "Apache-2.0",
    "sympy": "BSD-3-Clause",
    "tomli": "MIT",
    "torch": "BSD-3-Clause",
    "triton": "MIT",
    "twine": "Apache-2.0",
    "typing-extensions": "PSF-2.0",
    "urllib3": "MIT",
    "watchdog": "Apache-2.0",
    "wheel": "MIT",
    "zipp": "MIT",
}


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_packages(profile: str) -> tuple[Path, list[dict[str, Any]]]:
    lock_path = PROFILES[profile]["lock"]
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    if lock.get("lock-version") != "1.0" or lock.get("created-by") != "uv":
        raise RuntimeError(f"{lock_path.name} is not a uv PEP 751 lock")
    if lock.get("requires-python") != ">=3.11":
        raise RuntimeError(f"{lock_path.name} must target Python 3.11")
    packages = lock.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError(f"{lock_path.name} contains no packages")

    seen: set[str] = set()
    for package in packages:
        name = _canonical_name(package.get("name", ""))
        if not name or name in seen:
            raise RuntimeError(f"{lock_path.name} has an invalid or duplicate package")
        seen.add(name)
        if name not in LICENSE_EXPRESSIONS:
            raise RuntimeError(f"UNKNOWN third-party license for {name}")
        if LICENSE_EXPRESSIONS[name] == "UNKNOWN":
            raise RuntimeError(f"UNKNOWN third-party license for {name}")
        wheels = package.get("wheels")
        if not isinstance(wheels, list) or not wheels:
            raise RuntimeError(f"{name} has no locked wheel")
        for wheel in wheels:
            digest = wheel.get("hashes", {}).get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise RuntimeError(f"{name} has a wheel without SHA-256")

    versions = {_canonical_name(package["name"]): package["version"] for package in packages}
    if versions.get("torch") != PROFILES[profile]["torch"]:
        raise RuntimeError(f"{lock_path.name} has the wrong PyTorch build")
    return lock_path, packages


def _requirements_text(packages: list[dict[str, Any]]) -> str:
    lines = ["# Generated transiently from a checked PEP 751 lock."]
    for package in packages:
        requirement = f"{package['name']}=={package['version']}"
        marker = package.get("marker")
        if marker:
            requirement += f" ; {marker}"
        lines.append(requirement)
    return "\n".join(lines) + "\n"


def _run_cyclonedx(
    executable: str,
    requirements: Path,
    output: Path,
) -> None:
    version = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if CYCLONEDX_VERSION not in version:
        raise RuntimeError(f"cyclonedx-py {CYCLONEDX_VERSION} is required, got {version!r}")
    subprocess.run(
        [
            executable,
            "requirements",
            str(requirements),
            "--pyproject",
            str(ROOT / "pyproject.toml"),
            "--mc-type",
            "library",
            "--short-PURLs",
            "--sv",
            "1.6",
            "--output-reproducible",
            "--of",
            "JSON",
            "--output-file",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )


def _stabilise_sbom(
    profile: str,
    sbom_path: Path,
    packages: list[dict[str, Any]],
) -> None:
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.6":
        raise RuntimeError("CycloneDX generator produced an unexpected schema")
    sbom.pop("serialNumber", None)
    metadata = sbom.get("metadata", {})
    metadata.pop("timestamp", None)
    root_component = metadata.get("component", {})
    root_component["version"] = RELEASE_VERSION
    root_component["licenses"] = [{"expression": "Apache-2.0"}]

    locked = {(_canonical_name(package["name"]), package["version"]) for package in packages}
    components = sbom.get("components")
    if not isinstance(components, list):
        raise RuntimeError("CycloneDX output has no components")
    observed: set[tuple[str, str]] = set()
    for component in components:
        name = _canonical_name(component.get("name", ""))
        version = component.get("version")
        identity = (name, version)
        if identity not in locked:
            raise RuntimeError(f"unexpected CycloneDX component {identity}")
        observed.add(identity)
        component["licenses"] = [{"expression": LICENSE_EXPRESSIONS[name]}]
        properties = component.setdefault("properties", [])
        properties.append(
            {
                "name": "structfield:release-profile",
                "value": profile,
            }
        )
        properties.sort(key=lambda item: (item["name"], item["value"]))
    if observed != locked:
        missing = sorted(locked - observed)
        raise RuntimeError(f"CycloneDX output omitted locked components: {missing}")
    components.sort(
        key=lambda component: (
            _canonical_name(component["name"]),
            component.get("version", ""),
            component.get("purl", ""),
        )
    )
    component_refs = sorted(component["bom-ref"] for component in components)
    dependencies = sbom.setdefault("dependencies", [])
    root_ref = root_component.get("bom-ref")
    if not isinstance(root_ref, str) or not root_ref:
        raise RuntimeError("CycloneDX root component has no bom-ref")
    root_dependency = next(
        (dependency for dependency in dependencies if dependency.get("ref") == root_ref),
        None,
    )
    if root_dependency is None:
        root_dependency = {"ref": root_ref}
        dependencies.append(root_dependency)
    root_dependency["dependsOn"] = component_refs
    dependencies.sort(key=lambda dependency: dependency["ref"])
    sbom_path.write_text(
        json.dumps(sbom, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_license_inventory(
    profile: str,
    output: Path,
    lock_path: Path,
    packages: list[dict[str, Any]],
) -> None:
    components = [
        {
            "name": _canonical_name(package["name"]),
            "version": package["version"],
            "license_expression": LICENSE_EXPRESSIONS[_canonical_name(package["name"])],
        }
        for package in packages
    ]
    components.sort(key=lambda item: (item["name"], item["version"]))
    payload = {
        "schema_version": "structfield-third-party-licenses-v1",
        "profile": profile,
        "lock_file": lock_path.name,
        "lock_sha256": _sha256(lock_path),
        "components": components,
    }
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if "UNKNOWN" in encoded:
        raise RuntimeError("third-party license inventory contains UNKNOWN")
    output.write_text(encoded, encoding="utf-8", newline="\n")


def _generate_profile(profile: str, output_dir: Path, cyclonedx: str) -> None:
    lock_path, packages = _load_packages(profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    sbom_path = output_dir / f"structfield-vision-{profile}.cdx.json"
    license_path = output_dir / f"third-party-licenses-{profile}.json"
    with tempfile.TemporaryDirectory(prefix="structfield-sbom-") as temporary:
        requirements = Path(temporary) / "requirements.txt"
        requirements.write_text(
            _requirements_text(packages),
            encoding="utf-8",
            newline="\n",
        )
        _run_cyclonedx(cyclonedx, requirements, sbom_path)
    _stabilise_sbom(profile, sbom_path, packages)
    _write_license_inventory(profile, license_path, lock_path, packages)


def _check(cyclonedx: str) -> None:
    with tempfile.TemporaryDirectory(prefix="structfield-release-check-") as temporary:
        generated = Path(temporary)
        for profile in PROFILES:
            _generate_profile(profile, generated, cyclonedx)
        for candidate in sorted(generated.iterdir()):
            expected = ROOT / "sbom" / candidate.name
            if not expected.is_file() or candidate.read_bytes() != expected.read_bytes():
                raise RuntimeError(f"release artifact is not reproducible: {candidate.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyclonedx", default="cyclonedx-py")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        _check(args.cyclonedx)
    else:
        for profile in PROFILES:
            _generate_profile(profile, ROOT / "sbom", args.cyclonedx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
