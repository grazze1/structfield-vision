#!/usr/bin/env python3
"""Create post-tag local checksums without introducing commit self-reference."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import stat
import subprocess
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
VERSION = cast(str, runpy.run_path(ROOT / "structfield/_version.py")["VERSION"])
TAG = f"v{VERSION}"
ARTIFACTS = (
    Path("pylock.cpu.toml"),
    Path("pylock.cu126.toml"),
    Path("dist") / f"structfield_vision-{VERSION}-py3-none-any.whl",
    Path("dist") / f"structfield_vision-{VERSION}.tar.gz",
    Path("sbom/structfield-vision-cpu.cdx.json"),
    Path("sbom/structfield-vision-cu126.cdx.json"),
    Path("sbom/third-party-licenses-cpu.json"),
    Path("sbom/third-party-licenses-cu126.json"),
    Path("benchmarks/v1.0.0_baseline.json"),
    Path("benchmarks/performance_cpu_reference.json"),
    Path("benchmarks/performance_cuda_reference.json"),
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tagged_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-list", "-n", "1", TAG],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _identity(relative: Path) -> dict[str, object]:
    path = ROOT / relative
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"release artifact must be a singly linked regular file: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"release artifact changed while hashing: {relative}")
    return {"path": relative.as_posix(), "bytes": info.st_size, "sha256": digest.hexdigest()}


def main() -> int:
    if _git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must be clean before generating the post-tag manifest")
    commit = _git("rev-parse", "HEAD")
    if _tagged_commit() != commit:
        raise RuntimeError(f"annotated tag {TAG} must point to HEAD")
    if _git("cat-file", "-t", TAG) != "tag":
        raise RuntimeError(f"{TAG} must be an annotated tag")

    identities = [_identity(path) for path in ARTIFACTS]
    payload = {
        "schema_version": "structfield-release-manifest-v1",
        "distribution": "structfield-vision",
        "version": VERSION,
        "git_commit": commit,
        "git_tag": TAG,
        "platform": "linux-x86_64",
        "python": "3.11",
        "artifacts": identities,
    }
    output = ROOT / "release" / TAG
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "RELEASE_MANIFEST.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    checksum_items = [*identities, _identity(manifest.relative_to(ROOT))]
    checksums = output / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in checksum_items),
        encoding="utf-8",
        newline="\n",
    )
    descriptor = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
