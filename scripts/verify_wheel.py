#!/usr/bin/env python3
"""Fail when a release wheel leaks excluded namespaces or misses resources."""

from __future__ import annotations

import argparse
import email
import zipfile
from pathlib import Path

EXPECTED_CONFIGS = {
    "analytic_default.yaml",
    "analytic_8dir.yaml",
    "analytic_16dir.yaml",
    "analytic_32dir.yaml",
    "baseline_matrix.yaml",
    "failure_prediction.yaml",
    "corruption_sweep.yaml",
    "equivariance_smoke.yaml",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
        top_levels = {name.split("/", 1)[0] for name in names if "/" in name}
        forbidden = {"applications", "benchmarks", "tests", "docs", "data", "datasets"}
        leaked = top_levels & forbidden
        if leaked:
            raise RuntimeError(f"wheel leaks forbidden top-level entries: {sorted(leaked)}")
        if any(
            token in name
            for name in names
            for token in (
                "refactor_v02",
                "release_audit",
                "source_snapshot",
                "competition_submission",
                "LICENSE-PENDING",
            )
        ):
            raise RuntimeError("wheel leaks internal or pending-license material")
        if any("structfield_bench" in name for name in names):
            raise RuntimeError("wheel contains the removed nested benchmark namespace")

        configs = {
            Path(name).name for name in names if name.startswith("structfield/configs/") and name.endswith(".yaml")
        }
        if configs != EXPECTED_CONFIGS:
            raise RuntimeError(f"wheel configuration resources differ: {sorted(configs)}")
        required_modules = {
            "structfield/applications/__init__.py",
            "structfield/benchmarks/__init__.py",
            "structfield/checkpoint.py",
            "structfield/validation.py",
        }
        if not required_modules <= names:
            raise RuntimeError(f"wheel is missing modules: {sorted(required_modules - names)}")

        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))
        if metadata["Version"] != "1.0.0":
            raise RuntimeError("wheel version is not 1.0.0")
        if metadata["License-Expression"] != "Apache-2.0":
            raise RuntimeError("wheel license expression is not Apache-2.0")
        license_files = {Path(name).name for name in names if ".dist-info/licenses/" in name}
        if not {"LICENSE", "NOTICE.md"} <= license_files:
            raise RuntimeError("wheel does not contain LICENSE and NOTICE.md")
        if "structfield/py.typed" not in names:
            raise RuntimeError("wheel is missing the py.typed marker")
        if "structfield/experimental/__init__.py" not in names:
            raise RuntimeError("wheel is missing the experimental boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
