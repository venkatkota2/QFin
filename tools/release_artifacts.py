"""Hash built artifacts, or verify and collect the exact already-tested bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tomllib
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def version_check() -> None:
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    if os.environ.get("EXPECTED_VERSION") != version:
        raise SystemExit("requested version does not match pyproject.toml")
    tag = os.environ.get("GITHUB_REF", "")
    if tag.startswith("refs/tags/") and tag != f"refs/tags/v{version}":
        raise SystemExit("tag does not match project version")
    print(f"Validated version {version}; no publication performed")


def artifacts(directory: Path) -> list[Path]:
    return sorted([*directory.glob("*.whl"), *directory.glob("*.tar.gz")])


def record(directory: Path) -> None:
    files = artifacts(directory)
    if not files:
        raise SystemExit("no package artifacts found")
    (directory / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files))
    (directory / "build-context.json").write_text(
        json.dumps(
            {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "runner": platform.platform(),
                "python": platform.python_version(),
                "float_policy": "GCC/Clang: -fno-fast-math -ffp-contract=off; MSVC: /fp:strict",
                "wheel_builder": "cibuildwheel 4.2.1; platform default repair",
                "artifacts": {p.name: digest(p) for p in files},
            },
            indent=2,
        )
        + "\n"
    )


def collect(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    for manifest in sorted(source.rglob("SHA256SUMS")):
        for line in manifest.read_text().splitlines():
            checksum, name = line.split("  ", 1)
            if not re.fullmatch(r"[0-9a-f]{64}", checksum) or Path(name).name != name:
                raise SystemExit("invalid artifact manifest")
            file = manifest.parent / name
            if name in seen or digest(file) != checksum:
                raise SystemExit(f"duplicate or altered tested artifact: {name}")
            seen.add(name)
            shutil.copyfile(file, destination / name)
    if len(list(destination.glob("*.whl"))) != 9 or len(list(destination.glob("*.tar.gz"))) != 1:
        raise SystemExit("expected nine CPython/platform wheels and one source distribution")
    record(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--collect", type=Path)
    parser.add_argument("--check-version", action="store_true")
    args = parser.parse_args()
    if args.check_version:
        version_check()
    elif args.directory is None:
        parser.error("directory is required")
    elif args.collect:
        collect(args.directory, args.collect)
    else:
        record(args.directory)
