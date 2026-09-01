#!/usr/bin/env python3
"""Audit one PEP 751 release profile without installing accelerator wheels."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from release_artifacts import PROFILES, _load_packages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=tuple(PROFILES))
    parser.add_argument("--pip-audit", default="pip-audit")
    args = parser.parse_args()
    _, packages = _load_packages(args.profile)
    lines = []
    for package in packages:
        version = package["version"].split("+", 1)[0]
        requirement = f"{package['name']}=={version}"
        marker = package.get("marker")
        if marker:
            requirement += f" ; {marker}"
        lines.append(requirement)
    with tempfile.TemporaryDirectory(prefix=f"structfield-audit-{args.profile}-") as temporary:
        requirements = Path(temporary) / "requirements.txt"
        requirements.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        subprocess.run(
            [
                args.pip_audit,
                "--requirement",
                str(requirements),
                "--no-deps",
                "--disable-pip",
                "--strict",
                "--progress-spinner",
                "off",
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
