"""Repair with an installed MSVC redistributable at least as new as the linker.

Do not pick a stale msvcp140.dll from Python's PATH. Inspect both the extension
and bundled CRT PE headers, fail closed on missing/older runtimes, and retain
the selected source and hashes beside the repaired wheel for release review.
Uses the existing Visual Studio installation, not a downloaded runtime installer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

AMD64 = 0x8664


def pe_linker(data: bytes) -> tuple[int, tuple[int, int]]:
    """Read the documented COFF machine and optional-header linker version."""

    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError("missing DOS header")
    offset = struct.unpack_from("<I", data, 60)[0]
    if offset + 28 > len(data) or data[offset : offset + 4] != b"PE\0\0":
        raise ValueError("missing PE header")
    machine = struct.unpack_from("<H", data, offset + 4)[0]
    optional_size = struct.unpack_from("<H", data, offset + 20)[0]
    if optional_size < 4 or offset + 24 + optional_size > len(data):
        raise ValueError("truncated PE optional header")
    magic = struct.unpack_from("<H", data, offset + 24)[0]
    if magic not in (0x10B, 0x20B):
        raise ValueError("unsupported PE optional header")
    return machine, (data[offset + 26], data[offset + 27])


def wheel_extension(wheel: Path) -> tuple[str, tuple[int, int]]:
    with zipfile.ZipFile(wheel) as archive:
        names = [
            n
            for n in archive.namelist()
            if Path(n).name.startswith("_qfin_native") and n.lower().endswith(".pyd")
        ]
        if len(names) != 1:
            raise ValueError("expected exactly one QFin native Windows extension")
        machine, linker = pe_linker(archive.read(names[0]))
    if machine != AMD64:
        raise ValueError("only AMD64 Windows wheels are validated")
    return names[0], linker


def redist_directories() -> list[Path]:
    program_files = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installations = json.loads(
        subprocess.check_output(
            [str(vswhere), "-all", "-products", "*", "-format", "json", "-utf8"],
            encoding="utf-8",
        )
    )
    return sorted(
        {
            directory
            for installation in installations
            # Accept an optional toolset-family folder (for example v145), but
            # search only registered VS redist roots, never PATH/System32.
            for directory in (Path(installation["installationPath"]) / "VC/Redist/MSVC").glob(
                "*/x64/Microsoft.VC*.CRT"
            )
            if directory.is_dir()
        }
        | {
            directory
            for installation in installations
            for directory in (Path(installation["installationPath"]) / "VC/Redist/MSVC").glob(
                "v*/*/x64/Microsoft.VC*.CRT"
            )
            if directory.is_dir()
        }
    )


def select_runtime(directories: list[Path], linker: tuple[int, int]) -> Path:
    candidates = []
    for directory in directories:
        runtime = directory / "msvcp140.dll"
        if not runtime.is_file():
            continue
        machine, version = pe_linker(runtime.read_bytes())
        print(f"Runtime candidate {runtime}: machine {machine:#x}, linker {version}", flush=True)
        if machine == AMD64 and version >= linker:
            candidates.append((version, str(runtime), runtime))
    if not candidates:
        raise RuntimeError(
            f"No installed AMD64 MSVC runtime matches linker {linker}. "
            "Install a matching redistributable or select a compatible compiler; "
            "do not publish a wheel repaired with an older runtime."
        )
    return max(candidates)[2]


def verify_bundled_runtime(wheel: Path, minimum: tuple[int, int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            base = Path(name).name.lower()
            if not (base.startswith("msvcp140") and base.endswith(".dll")):
                continue
            data = archive.read(name)
            machine, version = pe_linker(data)
            if machine != AMD64 or version < minimum:
                raise RuntimeError(f"bundled {name} is incompatible with linker {minimum}")
            records.append(
                {
                    "path": name,
                    "linker_version": list(version),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    if not records:
        raise RuntimeError("repaired wheel is missing its MSVC C++ runtime")
    return records


def repair(wheel: Path, destination: Path, report_directory: Path | None = None) -> None:
    extension, linker = wheel_extension(wheel)
    runtime = select_runtime(redist_directories(), linker)
    print(f"Repairing {wheel.name}: extension linker {linker}, runtime {runtime}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-m",
            "delvewheel",
            "repair",
            "--add-path",
            str(runtime.parent),
            "-w",
            str(destination),
            str(wheel),
        ],
        check=True,
    )
    repaired = destination / wheel.name
    records = verify_bundled_runtime(repaired, linker)
    report = {
        "wheel": repaired.name,
        "wheel_sha256": hashlib.sha256(repaired.read_bytes()).hexdigest(),
        "extension": extension,
        "extension_linker_version": list(linker),
        "selected_runtime": str(runtime),
        "selected_runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "bundled_runtimes": records,
        "scope": "bundled MSVC C++ runtime; does not certify every Windows host or DLL load order",
    }
    # cibuildwheel copies only wheels out of its temporary repair directory.
    report_directory = destination if report_directory is None else report_directory
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / f"{wheel.name}.runtime.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report-directory", type=Path)
    args = parser.parse_args()
    repair(args.wheel, args.destination, args.report_directory)
