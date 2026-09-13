"""Cross-platform fixture tests for the Windows wheel's fail-closed runtime gate."""

import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "repair_windows_wheel", ROOT / "tools/repair_windows_wheel.py"
)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def pe(version=(14, 51), machine=0x8664):
    result = bytearray(256)
    result[:2] = b"MZ"
    struct.pack_into("<I", result, 60, 64)
    result[64:68] = b"PE\0\0"
    struct.pack_into("<H", result, 68, machine)
    struct.pack_into("<H", result, 84, 128)
    struct.pack_into("<H", result, 88, 0x20B)
    result[90:92] = bytes(version)
    return bytes(result)


def wheel(path, *, runtime=(14, 51), extension=(14, 51), machine=0x8664):
    with zipfile.ZipFile(path, "w") as archive:
        if extension is not None:
            archive.writestr("qfin/_qfin_native.cp312-win_amd64.pyd", pe(extension, machine))
        if runtime is not None:
            archive.writestr("qfin_quantum.libs/msvcp140-abc123.dll", pe(runtime, machine))
    return path


def test_runtime_discovery_accepts_legacy_and_nested_toolsets(tmp_path, monkeypatch):
    installations = [tmp_path / "VS2022", tmp_path / "VS2026"]
    legacy = installations[0] / "VC/Redist/MSVC/14.44.35211/x64/Microsoft.VC143.CRT"
    nested = installations[1] / "VC/Redist/MSVC/v145/14.51.36247/x64/Microsoft.VC145.CRT"
    excluded = [
        installations[1] / "VC/Redist/MSVC/v145/14.51.36247/arm64/Microsoft.VC145.CRT",
        tmp_path / "unregistered/VC/Redist/MSVC/14.60/x64/Microsoft.VC145.CRT",
    ]
    for directory in [legacy, nested, *excluded]:
        directory.mkdir(parents=True)
    (legacy / "msvcp140.dll").write_bytes(pe((14, 44)))
    (nested / "msvcp140.dll").write_bytes(pe((14, 51)))
    monkeypatch.setenv("PROGRAMFILES(X86)", str(tmp_path / "Program Files (x86)"))

    def vswhere(command, *, encoding):
        assert Path(command[0]) == (
            tmp_path / "Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
        )
        assert command[1:] == ["-all", "-products", "*", "-format", "json", "-utf8"]
        assert encoding == "utf-8"
        # Duplicate discovery should not create duplicate runtime candidates.
        return json.dumps([{"installationPath": str(path)} for path in installations * 2])

    monkeypatch.setattr(repair.subprocess, "check_output", vswhere)
    directories = repair.redist_directories()
    assert directories == sorted([legacy, nested])
    assert repair.select_runtime(directories, (14, 51)) == nested / "msvcp140.dll"


def test_header_parser_and_runtime_selection_ignore_stale_path(tmp_path):
    assert repair.pe_linker(pe()) == (0x8664, (14, 51))
    directories = []
    for label, version, machine in [
        ("old", (14, 40), 0x8664),
        ("current", (14, 51), 0x8664),
        ("wrong_arch", (14, 60), 0xAA64),
    ]:
        directory = tmp_path / label
        directory.mkdir()
        (directory / "msvcp140.dll").write_bytes(pe(version, machine))
        directories.append(directory)
    assert repair.select_runtime(directories, (14, 51)) == tmp_path / "current/msvcp140.dll"
    with pytest.raises(RuntimeError, match="No installed AMD64"):
        repair.select_runtime(directories, (14, 52))
    assert repair.select_runtime([tmp_path / "missing", *directories], (14, 51))


@pytest.mark.parametrize("data", [b"", b"MZ", b"MZ" + bytes(62), pe()[:100]])
def test_malformed_headers_fail_closed(data):
    with pytest.raises(ValueError):
        repair.pe_linker(data)


def test_repaired_wheel_requires_matching_runtime(tmp_path):
    path = wheel(tmp_path / "qfin.whl")
    assert repair.wheel_extension(path)[1] == (14, 51)
    report = repair.verify_bundled_runtime(path, (14, 51))
    assert len(report) == 1
    assert len(report[0]["sha256"]) == 64
    for version in [(14, 40), (14, 50), None]:
        path = wheel(path, runtime=version)
        with pytest.raises(RuntimeError, match=r"incompatible|missing"):
            repair.verify_bundled_runtime(path, (14, 51))


def test_wrong_architecture_and_missing_extension_fail_closed(tmp_path):
    path = wheel(tmp_path / "qfin.whl", machine=0xAA64)
    with pytest.raises(ValueError, match="AMD64"):
        repair.wheel_extension(path)
    with pytest.raises(RuntimeError, match="incompatible"):
        repair.verify_bundled_runtime(path, (14, 51))
    wheel(path, extension=None)
    with pytest.raises(ValueError, match="exactly one"):
        repair.wheel_extension(path)


def test_scipy_minimum_matches_licensing_fix():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "scipy>=1.11.1,<2" in data["project"]["dependencies"]
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "scipy==1.11.1" in workflow
    assert "scipy==1.11.0 " not in workflow
